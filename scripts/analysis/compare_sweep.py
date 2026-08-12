#!/usr/bin/env python3
"""Compare tuning-sweep runs with the varied tracker params as table columns.

Unlike compare_findings.py (one row per tracker), this joins each findings row
back to its frozen tracker_config and shows only the keys that actually differ
across the selected runs, so every row is self-describing.

Examples:
  .venv/bin/python scripts/analysis/compare_sweep.py \\
    --benchmarks cityflow ua_detrac --target-fps 5 \\
    --run-id-substr _fps5 --sort HOTA

  .venv/bin/python scripts/analysis/compare_sweep.py \\
    --benchmarks cityflow --target-fps 5 --baseline-substr n1_buf1_du1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mot_pipeline.paths import FINDINGS_ROOT

METRIC_COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "MT", "ML"]
HIGHER_IS_BETTER = {"HOTA", "DetA", "AssA", "MOTA", "IDF1", "MT"}
LOWER_IS_BETTER = {"IDSW", "Frag", "ML"}
BENCH_ORDER = ["fasttracker_bench", "ua_detrac", "trafficmot", "cityflow", "lumana_benchmark"]
DEFAULT_DETECTOR = "yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles"


def _fps_of(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "null"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_records(
    *,
    benchmarks: List[str],
    tracker: str,
    detector_id: str,
    target_fps: Optional[float],
    run_id_substr: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bench in benchmarks:
        path = FINDINGS_ROOT / bench / tracker / detector_id / "findings.json"
        if not path.is_file():
            print(f"[skip] no findings: {path}", file=sys.stderr)
            continue
        for rec in json.loads(path.read_text()):
            if not isinstance(rec, dict):
                continue
            if target_fps is not None and _fps_of(rec.get("target_fps")) != target_fps:
                continue
            if run_id_substr and run_id_substr not in str(rec.get("run_id", "")):
                continue
            rec["_benchmark"] = bench
            rows.append(rec)
    return rows


def _tracker_config(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Resolved (defaults-merged) config frozen at track time."""
    exp_dir = rec.get("experiment_dir")
    if not exp_dir:
        return {}
    cfg_path = Path(exp_dir) / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        spec = json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return {}
    cfg = spec.get("tracker_config")
    return cfg if isinstance(cfg, dict) else {}


def _varied_keys(configs: List[Dict[str, Any]]) -> List[str]:
    """Config keys taking more than one distinct value across runs."""
    keys = {k for cfg in configs for k in cfg}
    varied = []
    for key in sorted(keys):
        seen = {json.dumps(cfg.get(key), sort_keys=True) for cfg in configs}
        if len(seen) > 1:
            varied.append(key)
    return varied


def _fmt(key: str, value: object) -> str:
    if value is None or value == "":
        return "—"
    if key in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)
    if key in ("IDSW", "Frag", "MT", "ML"):
        try:
            return f"{int(round(float(value)))}"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _num(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _render_block(
    bench: str,
    rows: List[Dict[str, Any]],
    param_keys: List[str],
    sort_metric: str,
    baseline_substr: Optional[str],
) -> List[str]:
    baseline = None
    if baseline_substr:
        for row in rows:
            if baseline_substr in str(row.get("run_id", "")):
                baseline = row
                break

    reverse = sort_metric in HIGHER_IS_BETTER
    rows = sorted(
        rows,
        key=lambda r: (_num(r.get(sort_metric)) is None, _num(r.get(sort_metric)) or 0.0),
        reverse=reverse,
    )

    best: Dict[str, float] = {}
    for metric in METRIC_COLS:
        vals = [v for v in (_num(r.get(metric)) for r in rows) if v is not None]
        if not vals:
            continue
        best[metric] = max(vals) if metric in HIGHER_IS_BETTER else min(vals)

    header = ["run"] + param_keys + METRIC_COLS
    if baseline is not None:
        header.append(f"d{sort_metric}")

    lines = [
        f"### {bench}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    base_val = _num(baseline.get(sort_metric)) if baseline is not None else None

    for row in rows:
        cfg = row["_config"]
        label = str(row.get("run_id", ""))
        # Trim the shared prefix so the varying tail is visible.
        label = label.replace(f"{bench}_analytics_bytetrack_", "")
        if baseline is not None and row is baseline:
            label += " (base)"

        cells = [label]
        cells += [_fmt(k, cfg.get(k)) for k in param_keys]
        for metric in METRIC_COLS:
            text = _fmt(metric, row.get(metric))
            val = _num(row.get(metric))
            if val is not None and metric in best and abs(val - best[metric]) < 1e-12:
                text = f"**{text}**"
            cells.append(text)
        if baseline is not None:
            val = _num(row.get(sort_metric))
            cells.append(
                f"{val - base_val:+.4f}" if (val is not None and base_val is not None) else "—"
            )
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", nargs="+", default=["cityflow", "ua_detrac"])
    p.add_argument("--tracker", default="analytics_bytetrack")
    p.add_argument("--detector-id", default=DEFAULT_DETECTOR)
    p.add_argument("--target-fps", type=float, default=None)
    p.add_argument("--run-id-substr", default=None)
    p.add_argument(
        "--baseline-substr",
        default=None,
        help="Run-id substring to treat as baseline for the delta column.",
    )
    p.add_argument("--sort", default="HOTA", help="Metric to sort/delta on.")
    p.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="Force these config keys as columns (default: auto-detect varied keys).",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    records = _load_records(
        benchmarks=args.benchmarks,
        tracker=args.tracker,
        detector_id=args.detector_id,
        target_fps=args.target_fps,
        run_id_substr=args.run_id_substr,
    )
    if not records:
        sys.exit("No matching findings rows. Check --detector-id / --target-fps filters.")

    for rec in records:
        rec["_config"] = _tracker_config(rec)

    param_keys = args.params or _varied_keys([r["_config"] for r in records])
    if not param_keys:
        print("[warn] no tracker_config keys vary across runs", file=sys.stderr)

    lines = [
        "# Sweep comparison",
        "",
        f"_{len(records)} runs · tracker `{args.tracker}` · detector `{args.detector_id}`_",
        "",
        f"Varied params: {', '.join(f'`{k}`' for k in param_keys) or 'none'}",
        "",
    ]
    order = {b: i for i, b in enumerate(BENCH_ORDER)}
    benches = sorted({r["_benchmark"] for r in records}, key=lambda b: order.get(b, 99))
    for bench in benches:
        rows = [r for r in records if r["_benchmark"] == bench]
        lines += _render_block(bench, rows, param_keys, args.sort, args.baseline_substr)

    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
