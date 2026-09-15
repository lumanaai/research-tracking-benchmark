#!/usr/bin/env python3
"""Aggregate experiments/_findings into a comparison table.

Examples:
  .venv/bin/python scripts/analysis/compare_findings.py --detector-id gt_vehicles
  .venv/bin/python scripts/analysis/compare_findings.py --detector-id gt_vehicles --target-fps 10
  .venv/bin/python scripts/analysis/compare_findings.py --run-id-substr 20260729_085336
  .venv/bin/python scripts/analysis/compare_findings.py --detector-id gt_vehicles --out /tmp/cmp.md

Video of the same latest rows (no re-track):
  ./scripts/batch/visualize_all.sh && ./scripts/batch/serve_visualizations.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mot_pipeline.paths import FINDINGS_ROOT, PROJECT_ROOT
from mot_pipeline.tracker_meta import (
    display_tracker_name,
    preferred_detector,
    ref_ms_per_frame,
    set_ref_ms_per_frame,
    tracker_year,
)

METRIC_COLS = [
    "HOTA",
    "DetA",
    "AssA",
    "MOTA",
    "IDF1",
    "IDCons",
    "IDSW",
    "Frag",
    "MT",
    "ML",
]
# Within a benchmark, bold the best value per metric (↑ higher better, ↓ lower better).
HIGHER_IS_BETTER = {"HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDCons", "MT"}
LOWER_IS_BETTER = {"IDSW", "Frag", "ML", "ms/frame"}
BENCH_ORDER = ["fasttracker_bench", "ua_detrac", "trafficmot", "cityflow", "lumana_benchmark"]
TRACKER_ORDER = [
    "fasttracker",
    "ocsort",
    "hybridsort",
    "analytics_bytetrack",
    "analytics_bytetrack_plus",
    "analytics_bytetrack_plus_aug19",
    "botsort",
    "traffictrack",
]
# Native seqinfo / converter frameRate when tracking all frames (no --fps).
NATIVE_FPS = {
    "fasttracker_bench": 30.0,
    "ua_detrac": 25.0,
    "trafficmot": 10.0,
    "cityflow": 10.0,
    "lumana_benchmark": 20.0,  # per-seq varies ~13–30; display uses rounded mean
}

METRIC_GLOSSARY = [
    (
        "HOTA",
        "Higher Order Tracking Accuracy — geometric mean of detection (DetA) and "
        "association (AssA) accuracy over IoU thresholds; primary overall ranking metric.",
    ),
    (
        "DetA",
        "Detection Accuracy — how well predicted boxes cover GT detections "
        "(localization + presence), independent of ID quality.",
    ),
    (
        "AssA",
        "Association Accuracy — how consistently the same tracker ID stays on the "
        "same GT identity over time (identity preservation).",
    ),
    (
        "MOTA",
        "Multiple Object Tracking Accuracy — CLEAR metric: "
        "1 − (FN + FP + IDSW) / GT_dets. Sensitive to detector FP/FN; can go negative.",
    ),
    (
        "IDF1",
        "ID F1 — harmonic mean of ID precision/recall from bipartite ID matching "
        "(Identity metrics). Strong signal for ID stability.",
    ),
    (
        "IDCons",
        "ID Consistency — mean per-GT purity of the tracker IDs assigned to that object "
        "(how little a single GT is fragmented across tracker IDs).",
    ),
    (
        "IDSW",
        "ID Switches — times a GT trajectory changes which tracker ID it is matched to "
        "(↓ better).",
    ),
    (
        "Frag",
        "Fragmentations — times a tracked GT goes from matched → unmatched → matched "
        "again (trajectory breaks; ↓ better).",
    ),
    (
        "MT",
        "Mostly Tracked — GT trajectories covered for ≥80% of their lifetime (↑ better).",
    ),
    (
        "ML",
        "Mostly Lost — GT trajectories covered for ≤20% of their lifetime (↓ better).",
    ),
    (
        "FPS",
        "Effective video frame rate fed to the tracker (target --fps, else native "
        "benchmark rate). Not compute throughput.",
    ),
    (
        "ms/frame",
        "Average tracker compute time per processed frame (association only; excludes "
        "detector). Per-run when timing.json exists; otherwise a reference timing on "
        "dense GT dets (task_day_occlusion).",
    ),
    (
        "pref_det",
        "Detector the method prefers upstream / in production (YOLOX for most SORT-family "
        "papers; YOLO for in-house ByteTrack variants). This pipeline often sweeps "
        "YOLOv8m-expert or GT boxes instead.",
    ),
]


def _load_speed_refs() -> None:
    """Load optional reference timings from results/comparisons/tracker_speed.json."""
    path = PROJECT_ROOT / "results" / "comparisons" / "tracker_speed.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return
    vals = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("tracker")
        ms = row.get("avg_ms_per_frame")
        if name and ms is not None:
            try:
                vals[str(name)] = float(ms)
            except (TypeError, ValueError):
                continue
    if vals:
        set_ref_ms_per_frame(vals)


def _normalize_fps(val: object) -> Optional[float]:
    """Return float FPS or None for full-rate / missing."""
    if val is None:
        return None
    text = str(val).strip()
    if text in ("", "None", "null", "—", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fps_key(val: object) -> str:
    fps = _normalize_fps(val)
    if fps is None:
        return ""
    return f"{fps:g}"


def _fps_tag(val: object) -> str:
    key = _fps_key(val)
    return f"_fps{key.replace('.', 'p')}" if key else ""


def _load_rows(
    findings_root: Path,
    *,
    detector_id: Optional[str],
    benchmark: Optional[str],
    tracker: Optional[str],
    run_id_substr: Optional[str],
    target_fps: Optional[float],
    latest_only: bool,
) -> List[Dict[str, str]]:
    pattern = "*/*/*/findings.csv"
    rows: List[Dict[str, str]] = []
    for path in sorted(findings_root.glob(pattern)):
        # path: <bench>/<tracker>/<detector_id>/findings.csv
        det = path.parent.name
        trk = path.parent.parent.name
        bench = path.parent.parent.parent.name
        if detector_id and det != detector_id:
            continue
        if benchmark and bench != benchmark:
            continue
        if tracker and trk != tracker:
            continue
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                if run_id_substr and run_id_substr not in (r.get("run_id") or ""):
                    continue
                row_fps = _normalize_fps(r.get("target_fps"))
                if target_fps is not None:
                    # Full-rate rows (no target_fps) never match an explicit --target-fps.
                    if row_fps is None or abs(row_fps - float(target_fps)) > 1e-6:
                        continue
                else:
                    # Default (no --target-fps): full-rate runs only.
                    if row_fps is not None:
                        continue
                rows.append(r)

    if not latest_only:
        return rows

    # Keep newest evaluated_at per (benchmark, tracker, detector_id, target_fps).
    # target_fps is part of the key so subsampled runs do not replace full-rate ones.
    best: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for r in rows:
        key = (
            r.get("benchmark") or "",
            r.get("tracker") or "",
            r.get("detector_id") or "",
            _fps_key(r.get("target_fps")),
        )
        prev = best.get(key)
        if prev is None or (r.get("evaluated_at") or "") > (prev.get("evaluated_at") or ""):
            best[key] = r
    return list(best.values())


def _sort_key(r: Dict[str, str]):
    b = r.get("benchmark") or ""
    t = r.get("tracker") or ""
    fps = _normalize_fps(r.get("target_fps"))
    return (
        BENCH_ORDER.index(b) if b in BENCH_ORDER else 99,
        b,
        TRACKER_ORDER.index(t) if t in TRACKER_ORDER else 99,
        t,
        r.get("detector_id") or "",
        -1.0 if fps is None else fps,
        r.get("run_id") or "",
    )


def _fmt(col: str, val: str) -> str:
    if val in ("", None):
        return "—"
    try:
        x = float(val)
    except ValueError:
        return val
    if col in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDCons", "LocA", "MOTP", "IDP", "IDR"):
        return f"{x:.3f}"
    if col in ("IDSW", "Frag", "MT", "ML", "FP", "FN", "sequence_count"):
        return f"{x:.0f}"
    if col in ("target_fps", "FPS"):
        return f"{x:g}"
    if col in ("ms/frame", "avg_ms_per_frame"):
        return f"{x:.2f}"
    return f"{x}"


def _display_fps(r: Dict[str, str]) -> str:
    """Effective tracking FPS: target_fps if set, else native benchmark rate."""
    target = _normalize_fps(r.get("target_fps"))
    if target is not None:
        return f"{target:g}"
    native = NATIVE_FPS.get(r.get("benchmark") or "")
    if native is None:
        return "—"
    return f"{native:g}"


def _timing_from_experiment(r: Dict[str, str]) -> Optional[float]:
    exp = (r.get("experiment_dir") or "").strip()
    if not exp:
        return None
    path = Path(exp) / "timing.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    val = data.get("avg_ms_per_frame") if isinstance(data, dict) else None
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _ms_per_frame(r: Dict[str, str]) -> Optional[float]:
    raw = r.get("avg_ms_per_frame")
    if raw not in ("", None):
        try:
            return float(raw)
        except ValueError:
            pass
    from_exp = _timing_from_experiment(r)
    if from_exp is not None:
        return from_exp
    return ref_ms_per_frame(r.get("tracker") or "")


def _metric_float(val: object) -> Optional[float]:
    if val in ("", None):
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _best_metric_values(
    group: List[Dict[str, str]],
) -> Dict[str, float]:
    """Per-metric best numeric value within a benchmark group."""
    best: Dict[str, float] = {}
    for col in METRIC_COLS:
        vals = [v for v in (_metric_float(r.get(col)) for r in group) if v is not None]
        if not vals:
            continue
        if col in LOWER_IS_BETTER:
            best[col] = min(vals)
        elif col in HIGHER_IS_BETTER:
            best[col] = max(vals)
        else:
            best[col] = max(vals)
    ms_vals = [v for v in (_ms_per_frame(r) for r in group) if v is not None]
    if ms_vals:
        best["ms/frame"] = min(ms_vals)
    return best


def _fmt_metric_cell(col: str, val: str, best: Optional[float]) -> str:
    text = _fmt(col, val)
    if best is None or text == "—":
        return text
    # Compare on displayed precision so equal-looking ties are all marked.
    if text == _fmt(col, str(best)):
        return f"**{text}**"
    return text


def _glossary_markdown() -> str:
    lines = [
        "## Metric glossary",
        "",
    ]
    for name, desc in METRIC_GLOSSARY:
        lines.append(f"- **{name}** — {desc}")
    lines.append("")
    return "\n".join(lines)


def _notes_markdown(rows: Iterable[Dict[str, str]]) -> str:
    """Detector / protocol caveats that belong next to the published table."""
    ids = sorted({(r.get("detector_id") or "").strip() for r in rows if r.get("detector_id")})
    has_target_fps = any(_normalize_fps(r.get("target_fps")) is not None for r in rows)
    lines = ["## Notes", ""]
    yolo_ids = [i for i in ids if "yolov8" in i or "expert_eff" in i]
    if yolo_ids:
        lines.append(
            "- **YOLO cache.** Each `detector_id` encodes weights, `imgsz`, `conf`, and "
            "whether motorcycles are kept. The current pipeline default is "
            "`yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles`: Ultralytics "
            "`imgsz=(704, 1280)` (H×W), `conf=0.30`, keep-set bicycle/car/motorcycle/"
            "bus/truck/forklift/boat (`1,2,3,4,6,19,23`). Person is dropped. "
            "An older square-letterbox cache "
            "`…_imgsz1280_conf0.25_vehicles` (keep-set without bicycle/boat) still "
            "exists on disk for full-rate / 10 FPS tables until those are re-detected."
        )
        lines.append(
            "- **NMS / preprocess vs in-house.** This pipeline uses Ultralytics "
            "default NMS IoU `0.7` and Ultralytics letterbox. Production may differ "
            "in NMS IoU, letterbox vs stretch, package version, or post-NMS "
            "class filtering. A one-sequence replay against an in-house 5 FPS "
            "export matched boxes at mean IoU ~0.99; leftover extras were a few "
            "percent of boxes (concentrated on a couple of IDs), not a global "
            "association mismatch."
        )
        lines.append(
            "- **Bicycle / boat vs vehicle GT.** Those classes are kept to match "
            "product detections. On vehicle-only ground truth (e.g. Lumana class=1) "
            "they can count as false positives and slightly lower MOTA/DetA."
        )
    if has_target_fps:
        lines.append(
            "- **`--fps` eval.** When tracking at a target FPS, TrackEval GT is "
            "filtered to the same kept frames as the tracker (skipped frames do "
            "not exist). Full-rate eval is unchanged."
        )
    if not yolo_ids and any(i.startswith("gt_") for i in ids):
        lines.append(
            "- **GT oracle.** `gt_vehicles` feeds annotated boxes as detections "
            "(association-only). High DetA is expected; gaps are IDSW/Frag/AssA."
        )
    lines.append(
        "- **IDCons** is mean per-GT modal tracker-ID purity in this repo's "
        "TrackEval patch. Other groups may report a different identity metric "
        "under a similar name."
    )
    lines.append("")
    return "\n".join(lines)


def render_markdown(rows: Iterable[Dict[str, str]]) -> str:
    rows = sorted(rows, key=_sort_key)
    headers = [
        "benchmark",
        "tracker",
        "pref_det",
        "detector_id",
        "FPS",
        "ms/frame",
        "seqs",
        *METRIC_COLS,
        "run_id",
    ]
    n_cols = len(headers)
    # Thick visual break between benchmark blocks (GFM keeps this as a table row).
    bench_sep = "| " + " | ".join("═══" for _ in range(n_cols)) + " |"

    lines = [
        "# Tracker comparison",
        "",
        f"_{len(rows)} runs from `{FINDINGS_ROOT}`_",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]

    # Precompute best-per-metric within each contiguous benchmark group.
    groups: List[List[Dict[str, str]]] = []
    for r in rows:
        if not groups or (groups[-1][0].get("benchmark") or "") != (r.get("benchmark") or ""):
            groups.append([r])
        else:
            groups[-1].append(r)

    for gi, group in enumerate(groups):
        if gi > 0:
            lines.append(bench_sep)
        best = _best_metric_values(group)
        for r in group:
            tracker = r.get("tracker", "")
            ms = _ms_per_frame(r)
            ms_cell = _fmt_metric_cell(
                "ms/frame",
                "" if ms is None else str(ms),
                best.get("ms/frame"),
            )
            cells = [
                r.get("benchmark", ""),
                display_tracker_name(tracker),
                preferred_detector(tracker),
                r.get("detector_id", ""),
                _display_fps(r),
                ms_cell,
                _fmt("sequence_count", r.get("sequence_count", "")),
                *(_fmt_metric_cell(c, r.get(c, ""), best.get(c)) for c in METRIC_COLS),
                r.get("run_id", ""),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(_notes_markdown(rows))
    lines.append(_glossary_markdown())
    return "\n".join(lines)


def render_csv(rows: Iterable[Dict[str, str]]) -> str:
    rows = sorted(rows, key=_sort_key)
    fieldnames = [
        "benchmark",
        "tracker",
        "tracker_year",
        "preferred_detector",
        "detector_id",
        "detector",
        "FPS",
        "target_fps",
        "avg_ms_per_frame",
        "sequence_count",
        *METRIC_COLS,
        "run_id",
        "evaluated_at",
        "experiment_dir",
    ]
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        out = dict(r)
        tracker = r.get("tracker") or ""
        out["FPS"] = _display_fps(r)
        year = tracker_year(tracker)
        out["tracker_year"] = "" if year is None else str(year)
        out["preferred_detector"] = preferred_detector(tracker)
        ms = _ms_per_frame(r)
        out["avg_ms_per_frame"] = "" if ms is None else f"{ms:.4f}"
        w.writerow(out)
    return buf.getvalue()


def _default_out_stem(
    *,
    detector_id: Optional[str],
    run_id_substr: Optional[str],
    target_fps: Optional[float],
) -> str:
    tag = detector_id or run_id_substr or "all"
    if target_fps is not None:
        tag = f"{tag}{_fps_tag(target_fps)}"
    return tag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--findings-root", type=Path, default=FINDINGS_ROOT)
    p.add_argument("--detector-id", default=None, help="e.g. gt_vehicles, existing_det_vehicles")
    p.add_argument("--benchmark", default=None)
    p.add_argument("--tracker", default=None)
    p.add_argument("--run-id-substr", default=None, help="Filter run_id containing this string")
    p.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Only keep runs tracked at this target FPS. "
        "Omit to keep full-rate runs and show each benchmark's native FPS.",
    )
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="Keep every matching row (default: latest per bench×tracker×detector_id×target_fps)",
    )
    p.add_argument(
        "--format",
        choices=("md", "csv", "both"),
        default="md",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (.md/.csv). Default: print to stdout (and write SSD cmp file for md).",
    )
    args = p.parse_args()

    _load_speed_refs()

    rows = _load_rows(
        args.findings_root,
        detector_id=args.detector_id,
        benchmark=args.benchmark,
        tracker=args.tracker,
        run_id_substr=args.run_id_substr,
        target_fps=args.target_fps,
        latest_only=not args.all_rows,
    )
    if not rows:
        raise SystemExit("No matching findings rows.")

    stem = _default_out_stem(
        detector_id=args.detector_id,
        run_id_substr=args.run_id_substr,
        target_fps=args.target_fps,
    )

    if args.format in ("md", "both"):
        md = render_markdown(rows)
        if args.out and args.format == "md":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(md)
            print(f"wrote {args.out}")
        elif args.out and args.format == "both":
            out_md = args.out.with_suffix(".md")
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(md)
            print(f"wrote {out_md}")
        else:
            # Convenient default path beside experiments
            default = args.findings_root.parent / "_comparisons"
            default.mkdir(parents=True, exist_ok=True)
            path = default / f"compare_{stem}.md"
            path.write_text(md)
            print(md)
            print(f"\n# also wrote {path}", file=sys.stderr)

    if args.format in ("csv", "both"):
        text = render_csv(rows)
        if args.out and args.format == "csv":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(f"wrote {args.out}")
        elif args.out and args.format == "both":
            out_csv = args.out.with_suffix(".csv")
            out_csv.write_text(text)
            print(f"wrote {out_csv}")
        else:
            default = args.findings_root.parent / "_comparisons"
            default.mkdir(parents=True, exist_ok=True)
            path = default / f"compare_{stem}.csv"
            path.write_text(text)
            if args.format == "csv":
                print(text)
            print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
