#!/usr/bin/env python3
"""Compute benchmark comparison stats → SSD data root.

Writes:
  /media/7TBSSD/data/tracking/benchmarks_statistics.md
  /media/7TBSSD/data/tracking/benchmarks_statistics.json

CityFlow is read from the raw dataset (MOT convert may be partial / slow).
Other benches use normalized MOT roots under mot/<Benchmark>/<split>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mot_pipeline.class_maps import BENCHMARK_POLICIES
from mot_pipeline.mot_io import discover_mot_sequences, parse_seqinfo
from mot_pipeline.paths import MOT_ROOT, RAW_DATASETS

OUT_DIR = Path("/media/7TBSSD/data/tracking")

FT_LABELS_FALLBACK = {
    1: "person",
    2: "bus_small",
    3: "bus_big",
    4: "truck_small",
    5: "truck_big",
    6: "car",
    7: "bike",
    8: "motorbike",
    9: "ignore_region",
    10: "tractor",
    11: "trailor",
    12: "wheelchair",
    13: "heavy_equipment",
    14: "pm",
    15: "umbrella",
}

CLASS_NAMES = {
    "fasttracker_bench": FT_LABELS_FALLBACK,
    "ua_detrac": {1: "car", 2: "bus", 3: "van", 4: "others"},
    "trafficmot": {
        1: "Motor_Bike",
        2: "Bus",
        3: "LMV",
        4: "Auto",
        5: "Bike",
        6: "Pedestrian",
        7: "LCV",
        8: "E-rickshaw",
        9: "Tractor",
        10: "Truck",
    },
    "cityflow": {1: "vehicle"},
    "lumana_benchmark": {1: "vehicle"},
}

BENCH_META = [
    {
        "id": "fasttracker_bench",
        "mot_name": "FastTracker-Benchmark",
        "split": "train",
        "source_mode": "mot",
        "dataset_root": str(RAW_DATASETS["fasttracker_bench"]),
        "source": "https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark",
        "notes": [
            "Hard traffic scenes: day/night turns, occlusion, tunnel, crossing, far objects.",
            "Per-frame ignore_region (class 9) — drop before tracking.",
            "Multi-class GT beyond COCO vehicles (tractor, trailor, heavy_equipment, …).",
            "Best for association stress under occlusion / lighting extremes.",
        ],
    },
    {
        "id": "ua_detrac",
        "mot_name": "UA-DETRAC",
        "split": "train",
        "source_mode": "mot",
        "dataset_root": str(RAW_DATASETS["ua_detrac"]),
        "source": "UA-DETRAC",
        "notes": [
            "Large classic vehicle MOT; this MOT root covers the converted train split (60 seqs).",
            "Full raw set also has 40 test sequences (not in the MOT root until converted).",
            "Vehicle types only (car/bus/van/others); ignore regions exist in source XML.",
            "Good default for broad single-camera vehicle tracking at scale.",
        ],
    },
    {
        "id": "trafficmot",
        "mot_name": "TrafficMOT",
        "split": "Fully_annotate",
        "source_mode": "mot",
        "dataset_root": str(RAW_DATASETS["trafficmot"]),
        "source": "TrafficMOT",
        "notes": [
            "Very short fully-annotated clips (~30 frames).",
            "Dense / diverse classes (Auto, E-rickshaw, Tractor, …); Bike+Pedestrian in GT.",
            "FirstFrame_annotate split exists in raw data but is not in this MOT root.",
            "Good for quick smokes / class diversity; weak for long-horizon ID stability.",
        ],
    },
    {
        "id": "cityflow",
        "mot_name": "CityFlow",
        "split": "train",
        "source_mode": "cityflow_raw",
        "dataset_root": str(RAW_DATASETS["cityflow"]),
        "source": "AIC22 / CityFlowV2",
        "notes": [
            "Multi-camera vehicle tracking (MTMC); flat names Sxx_cyyy.",
            "Only vehicles seen in ≥2 cameras are annotated; per-camera ROI.",
            "Typically 10 FPS (some cams 8); GT on train + validation; test has no public GT.",
            "Stats below use the raw CityFlow tree (not the possibly-partial MOT extract).",
            "Best for multi-view / re-ID / cross-camera identity; single-cam MOT eval still valid.",
        ],
    },
    {
        "id": "lumana_benchmark",
        "mot_name": "LumanaBenchmark",
        "split": "train",
        "source_mode": "mot",
        "dataset_root": str(RAW_DATASETS["lumana_benchmark"]),
        "source": "LumanaBenchmark (internal GT)",
        "notes": [
            "Manually validated vehicle MOT GT (24 sequences); class_id always 1.",
            "MOT-native under gt_annotations_manually_validated/; mot/ train is a symlink.",
            "Per-sequence FPS varies (~13–30); dims in seqinfo.ini (typically 1920×1080).",
            "Frames/videos may be absent — GT-oracle track/eval works without img1.",
        ],
    },
]


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return float(ys[i])


def _analyze_gt_lines(
    lines,
    *,
    n_frames: int,
    fps: Optional[float],
    width: Optional[int],
    height: Optional[int],
    name: str,
    keep: Optional[Set[int]],
    drop: Set[int],
) -> Dict[str, Any]:
    frames: Set[int] = set()
    track_ids: Set[int] = set()
    class_counts: Counter = Counter()
    n_dets = 0
    n_dets_vehicle = 0
    track_ids_vehicle: Set[int] = set()
    per_frame: Counter = Counter()
    per_frame_vehicle: Counter = Counter()
    box_areas: List[float] = []
    box_areas_vehicle: List[float] = []
    track_lengths: Counter = Counter()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 6:
            continue
        fr = int(float(parts[0]))
        tid = int(float(parts[1]))
        bw = float(parts[4])
        bh = float(parts[5])
        cls = int(float(parts[7])) if len(parts) >= 8 else 1
        frames.add(fr)
        track_ids.add(tid)
        class_counts[cls] += 1
        n_dets += 1
        per_frame[fr] += 1
        track_lengths[tid] += 1
        area = bw * bh
        box_areas.append(area)

        is_keep = (keep is None) or (cls in keep)
        if cls not in drop and is_keep:
            n_dets_vehicle += 1
            track_ids_vehicle.add(tid)
            per_frame_vehicle[fr] += 1
            box_areas_vehicle.append(area)

    if n_frames <= 0:
        n_frames = max(frames) if frames else 0
    duration_s = (n_frames / fps) if fps and n_frames else None
    dens = [per_frame[fr] for fr in range(1, n_frames + 1)] if n_frames else list(per_frame.values())
    dens_v = (
        [per_frame_vehicle[fr] for fr in range(1, n_frames + 1)]
        if n_frames
        else list(per_frame_vehicle.values())
    )
    tl = list(track_lengths.values())

    return {
        "name": name,
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "duration_s": duration_s,
        "n_annotated_frames": len(frames),
        "n_dets": n_dets,
        "n_tracks": len(track_ids),
        "n_dets_vehicle_policy": n_dets_vehicle,
        "n_tracks_vehicle_policy": len(track_ids_vehicle),
        "class_counts": dict(sorted(class_counts.items())),
        "density_mean": _mean(dens),
        "density_max": max(dens) if dens else 0,
        "density_p95": _pct(dens, 95),
        "density_vehicle_mean": _mean(dens_v),
        "density_vehicle_max": max(dens_v) if dens_v else 0,
        "box_area_mean": _mean(box_areas),
        "box_area_vehicle_mean": _mean(box_areas_vehicle),
        "track_length_mean_frames": _mean([float(x) for x in tl]),
        "track_length_median_frames": _pct([float(x) for x in tl], 50),
        "track_length_max_frames": max(tl) if tl else 0,
    }


def analyze_mot_seq(
    seq_dir: Path, keep: Optional[Set[int]], drop: Set[int]
) -> Dict[str, Any]:
    meta = parse_seqinfo(seq_dir)
    fps = float(meta.get("frameRate") or 0) or None
    seq_len = int(float(meta.get("seqLength") or 0)) or 0
    w = int(float(meta.get("imWidth") or 0)) or None
    h = int(float(meta.get("imHeight") or 0)) or None
    gt = seq_dir / "gt" / "gt.txt"
    lines = gt.read_text().splitlines() if gt.is_file() else []
    return _analyze_gt_lines(
        lines,
        n_frames=seq_len,
        fps=fps,
        width=w,
        height=h,
        name=seq_dir.name,
        keep=keep,
        drop=drop,
    )


def _load_cityflow_framenum(dataset_root: Path) -> Dict[str, int]:
    """Map flat name S01_c001 → frame count from cam_framenum/."""
    out: Dict[str, int] = {}
    root = dataset_root / "cam_framenum"
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.txt")):
        scenario = path.stem  # S01
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                cam, n = parts[0], int(parts[1])
                out[f"{scenario}_{cam}"] = n
    return out


def _probe_video(vdo: Path) -> Tuple[Optional[float], Optional[int], Optional[int], int]:
    import cv2

    cap = cv2.VideoCapture(str(vdo))
    if not cap.isOpened():
        return None, None, None, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, w, h, n


def analyze_cityflow_raw(
    dataset_root: Path, split: str, keep: Optional[Set[int]], drop: Set[int]
) -> List[Dict[str, Any]]:
    frame_nums = _load_cityflow_framenum(dataset_root)
    split_root = dataset_root / split
    seqs: List[Dict[str, Any]] = []
    if not split_root.is_dir():
        return seqs
    for scenario in sorted(p for p in split_root.iterdir() if p.is_dir()):
        for cam in sorted(p for p in scenario.iterdir() if p.is_dir()):
            gt = cam / "gt" / "gt.txt"
            if not gt.is_file():
                continue
            flat = f"{scenario.name}_{cam.name}"
            vdo = cam / "vdo.avi"
            fps, w, h, n_video = (None, None, None, 0)
            if vdo.is_file():
                fps, w, h, n_video = _probe_video(vdo)
            n_frames = frame_nums.get(flat) or n_video or 0
            if fps is None:
                fps = 8.0 if cam.name == "c015" else 10.0
            lines = gt.read_text().splitlines()
            seqs.append(
                _analyze_gt_lines(
                    lines,
                    n_frames=n_frames,
                    fps=fps,
                    width=w,
                    height=h,
                    name=flat,
                    keep=keep,
                    drop=drop,
                )
            )
    return seqs


def summarize(seqs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not seqs:
        return {"n_sequences": 0}

    def ssum(key: str) -> float:
        return float(sum(s.get(key) or 0 for s in seqs))

    def smean(key: str) -> Optional[float]:
        vals = [s[key] for s in seqs if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def smin(key: str):
        vals = [s[key] for s in seqs if s.get(key) is not None]
        return min(vals) if vals else None

    def smax(key: str):
        vals = [s[key] for s in seqs if s.get(key) is not None]
        return max(vals) if vals else None

    resolutions = Counter(
        (s["width"], s["height"]) for s in seqs if s.get("width") and s.get("height")
    )
    fps_set = sorted({s["fps"] for s in seqs if s.get("fps")})
    class_tot: Counter = Counter()
    for s in seqs:
        class_tot.update(s.get("class_counts") or {})

    dens = [s["density_mean"] for s in seqs if s.get("density_mean") is not None]
    dens_v = [
        s["density_vehicle_mean"]
        for s in seqs
        if s.get("density_vehicle_mean") is not None
    ]

    return {
        "n_sequences": len(seqs),
        "n_frames_total": int(ssum("n_frames")),
        "n_dets_total": int(ssum("n_dets")),
        "n_tracks_total": int(ssum("n_tracks")),
        "n_dets_vehicle_policy_total": int(ssum("n_dets_vehicle_policy")),
        "n_tracks_vehicle_policy_total": int(ssum("n_tracks_vehicle_policy")),
        "duration_s_total": ssum("duration_s"),
        "frames_per_seq_mean": smean("n_frames"),
        "frames_per_seq_min": smin("n_frames"),
        "frames_per_seq_max": smax("n_frames"),
        "duration_s_mean": smean("duration_s"),
        "duration_s_min": smin("duration_s"),
        "duration_s_max": smax("duration_s"),
        "dets_per_seq_mean": smean("n_dets"),
        "tracks_per_seq_mean": smean("n_tracks"),
        "track_length_mean_frames": smean("track_length_mean_frames"),
        "density_mean_over_seqs": (sum(dens) / len(dens)) if dens else None,
        "density_max_over_seqs": max((s.get("density_max") or 0) for s in seqs),
        "density_vehicle_mean_over_seqs": (sum(dens_v) / len(dens_v)) if dens_v else None,
        "fps_values": fps_set,
        "resolutions": {f"{w}x{h}": c for (w, h), c in resolutions.most_common()},
        "class_counts_total": dict(sorted(class_tot.items())),
        "longest_seq": max(seqs, key=lambda s: s.get("n_frames") or 0)["name"],
        "densest_seq": max(seqs, key=lambda s: s.get("density_max") or 0)["name"],
        "most_tracks_seq": max(seqs, key=lambda s: s.get("n_tracks") or 0)["name"],
    }


def fmt_num(x, nd: int = 1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{nd}f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def fmt_dur(seconds) -> str:
    if seconds is None:
        return "—"
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {rem:.0f}s"
    h, rem_m = divmod(m, 60)
    return f"{int(h)}h {int(rem_m)}m"


def load_ft_labels(mot_root: Path) -> Dict[int, str]:
    for labels in mot_root.glob("*/gt/labels.txt"):
        names = [ln.strip() for ln in labels.read_text().splitlines() if ln.strip()]
        if names:
            return {i + 1: n for i, n in enumerate(names)}
    return dict(FT_LABELS_FALLBACK)


def collect() -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mot_root": str(MOT_ROOT),
        "out_dir": str(OUT_DIR),
        "scope": (
            "Ground-truth statistics for comparing which benchmark fits a task. "
            "FastTracker / UA-DETRAC / TrafficMOT: normalized MOT roots. "
            "CityFlow: raw dataset cameras with gt/gt.txt (avoids incomplete MOT extracts). "
            "Vehicle-policy counts use mot_pipeline default keep/drop maps (motorcycles included)."
        ),
        "benchmarks": {},
    }

    for spec in BENCH_META:
        policy = BENCHMARK_POLICIES[spec["id"]]
        keep, drop = policy.resolved(exclude_motorcycles=False)
        keep_set = set(keep) if keep is not None else None
        drop_set = set(drop)

        if spec["source_mode"] == "cityflow_raw":
            seq_stats = analyze_cityflow_raw(
                Path(spec["dataset_root"]), spec["split"], keep_set, drop_set
            )
            mot_path = str(MOT_ROOT / spec["mot_name"] / spec["split"])
        else:
            root = MOT_ROOT / spec["mot_name"] / spec["split"]
            mot_path = str(root)
            seq_dirs = discover_mot_sequences(root)
            # Skip accidental nested dirs named like the split
            seq_dirs = [p for p in seq_dirs if p.name != spec["split"]]
            seq_stats = [analyze_mot_seq(p, keep_set, drop_set) for p in seq_dirs]

        summary = summarize(seq_stats)
        names = dict(CLASS_NAMES.get(spec["id"], {}))
        if spec["id"] == "fasttracker_bench":
            names = load_ft_labels(MOT_ROOT / spec["mot_name"] / spec["split"])

        class_named = {
            f"{cid}:{names.get(cid, '?')}": cnt
            for cid, cnt in (summary.get("class_counts_total") or {}).items()
        }

        results["benchmarks"][spec["id"]] = {
            **{k: v for k, v in spec.items()},
            "mot_path": mot_path,
            "vehicle_policy": {
                "keep": sorted(keep_set) if keep_set is not None else None,
                "drop": sorted(drop_set),
            },
            "summary": summary,
            "class_counts_named": class_named,
            "sequences": seq_stats,
        }
        print(
            f"{spec['id']}: {summary.get('n_sequences', 0)} seqs, "
            f"{summary.get('n_frames_total', 0)} frames, "
            f"{summary.get('n_dets_total', 0)} dets"
        )
    return results


def render_md(results: Dict[str, Any]) -> str:
    md: List[str] = []
    md.append("# Benchmarks statistics")
    md.append("")
    md.append(f"_Generated {results['generated_at']}_")
    md.append("")
    md.append(results["scope"])
    md.append("")
    md.append("Location: next to the dataset folders under `/media/7TBSSD/data/tracking/`.")
    md.append("Regenerate: `.venv/bin/python scripts/analysis/compute_benchmark_statistics.py`")
    md.append("JSON twin: [`benchmarks_statistics.json`](benchmarks_statistics.json) (per-sequence rows).")
    md.append("")

    md.append("## Quick compare (default splits)")
    md.append("")
    md.append(
        "| Benchmark | Split | #videos | #frames | Duration | #GT boxes | #tracks | "
        "Mean dens. | Peak dens. | Mean track len (fr) | FPS | Resolutions |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for spec in BENCH_META:
        b = results["benchmarks"][spec["id"]]
        s = b["summary"]
        fps = ", ".join(
            str(int(x) if float(x).is_integer() else x) for x in (s.get("fps_values") or [])
        ) or "—"
        res = ", ".join(f"{k}×{v}" for k, v in (s.get("resolutions") or {}).items()) or "—"
        md.append(
            "| {name} | {split} | {nv} | {nf} | {dur} | {nd} | {nt} | {dens} | {peak} | {tl} | {fps} | {res} |".format(
                name=spec["id"],
                split=spec["split"],
                nv=fmt_num(s.get("n_sequences")),
                nf=fmt_num(s.get("n_frames_total")),
                dur=fmt_dur(s.get("duration_s_total")),
                nd=fmt_num(s.get("n_dets_total")),
                nt=fmt_num(s.get("n_tracks_total")),
                dens=fmt_num(s.get("density_mean_over_seqs"), 2),
                peak=fmt_num(s.get("density_max_over_seqs")),
                tl=fmt_num(s.get("track_length_mean_frames"), 1),
                fps=fps,
                res=res,
            )
        )
    md.append("")

    md.append("### Vehicle-policy subset (pipeline default: vehicles + motorcycles)")
    md.append("")
    md.append("| Benchmark | #GT boxes (kept) | #tracks (kept) | Mean dens. kept |")
    md.append("|---|---:|---:|---:|")
    for spec in BENCH_META:
        b = results["benchmarks"][spec["id"]]
        s = b["summary"]
        md.append(
            f"| {spec['id']} | {fmt_num(s.get('n_dets_vehicle_policy_total'))} | "
            f"{fmt_num(s.get('n_tracks_vehicle_policy_total'))} | "
            f"{fmt_num(s.get('density_vehicle_mean_over_seqs'), 2)} |"
        )
    md.append("")

    md.append("## Which benchmark for which task?")
    md.append("")
    md.append("| If you care about… | Prefer | Why |")
    md.append("|---|---|---|")
    md.append(
        "| Long tracks / ID stability over time | **UA-DETRAC** or **CityFlow** | "
        "Minutes-long sequences; TrafficMOT clips are ~1s. |"
    )
    md.append(
        "| Hard occlusion / night / tunnel | **FastTracker-Benchmark** | "
        "Designed stress scenes; highest peak density. |"
    )
    md.append(
        "| Scale / many sequences | **UA-DETRAC** (60) or **CityFlow** (~36 train cams) | "
        "Most videos in the analyzed roots. |"
    )
    md.append(
        "| Multi-camera / re-ID / MTMC | **CityFlow** | "
        "Global IDs across cameras; ROI; multi-cam vehicles only. |"
    )
    md.append(
        "| Unusual vehicle classes | **TrafficMOT** or **FastTracker-Benchmark** | "
        "Broader taxonomy than UA-DETRAC/CityFlow. |"
    )
    md.append(
        "| Fast iteration / smoke tests | **TrafficMOT** or 1 FastTracker seq | "
        "Few frames per clip. |"
    )
    md.append(
        "| Fair detector+tracker eval | Any + real dets (`yolov8` / `existing`) | "
        "GT-oracle mostly tests association only. |"
    )
    md.append("")

    md.append("## Per-benchmark detail")
    md.append("")
    for spec in BENCH_META:
        b = results["benchmarks"][spec["id"]]
        s = b["summary"]
        md.append(f"### `{spec['id']}`")
        md.append("")
        md.append(f"- **Dataset root:** `{spec['dataset_root']}`")
        md.append(f"- **MOT root:** `{b['mot_path']}`")
        md.append(f"- **Stats source:** `{spec['source_mode']}`")
        md.append(f"- **Source:** {spec['source']}")
        md.append(f"- **Split analyzed:** `{spec['split']}`")
        md.append("")
        md.append("| Stat | Value |")
        md.append("|---|---|")
        md.append(f"| Sequences (videos) | {fmt_num(s.get('n_sequences'))} |")
        md.append(f"| Total frames | {fmt_num(s.get('n_frames_total'))} |")
        md.append(
            f"| Frames / seq (mean · min · max) | "
            f"{fmt_num(s.get('frames_per_seq_mean'), 0)} · "
            f"{fmt_num(s.get('frames_per_seq_min'))} · "
            f"{fmt_num(s.get('frames_per_seq_max'))} |"
        )
        md.append(f"| Total duration | {fmt_dur(s.get('duration_s_total'))} |")
        md.append(
            f"| Duration / seq (mean · min · max) | "
            f"{fmt_dur(s.get('duration_s_mean'))} · "
            f"{fmt_dur(s.get('duration_s_min'))} · "
            f"{fmt_dur(s.get('duration_s_max'))} |"
        )
        md.append(f"| GT boxes (all classes) | {fmt_num(s.get('n_dets_total'))} |")
        md.append(f"| Unique tracks (all classes) | {fmt_num(s.get('n_tracks_total'))} |")
        md.append(
            f"| GT boxes (vehicle policy) | {fmt_num(s.get('n_dets_vehicle_policy_total'))} |"
        )
        md.append(
            f"| Unique tracks (vehicle policy) | {fmt_num(s.get('n_tracks_vehicle_policy_total'))} |"
        )
        md.append(
            f"| Mean / peak density (obj/frame) | "
            f"{fmt_num(s.get('density_mean_over_seqs'), 2)} / "
            f"{fmt_num(s.get('density_max_over_seqs'))} |"
        )
        md.append(
            f"| Mean density vehicle policy | "
            f"{fmt_num(s.get('density_vehicle_mean_over_seqs'), 2)} |"
        )
        md.append(
            f"| Mean track length (frames) | "
            f"{fmt_num(s.get('track_length_mean_frames'), 1)} |"
        )
        md.append(
            f"| FPS | {', '.join(map(str, s.get('fps_values') or [])) or '—'} |"
        )
        md.append(
            f"| Resolutions | "
            f"{', '.join(f'{k} ({v})' for k, v in (s.get('resolutions') or {}).items()) or '—'} |"
        )
        md.append(f"| Longest seq | `{s.get('longest_seq')}` |")
        md.append(f"| Densest seq (peak) | `{s.get('densest_seq')}` |")
        md.append(f"| Most tracks seq | `{s.get('most_tracks_seq')}` |")
        md.append("")
        md.append("**Class histogram (GT boxes):**")
        md.append("")
        for k, v in b["class_counts_named"].items():
            md.append(f"- `{k}`: {fmt_num(v)}")
        md.append("")
        md.append("**Notes:**")
        md.append("")
        for n in spec["notes"]:
            md.append(f"- {n}")
        md.append("")

        top = sorted(b["sequences"], key=lambda x: -(x.get("n_frames") or 0))
        md.append("<details><summary>Per-sequence table</summary>")
        md.append("")
        md.append(
            "| Sequence | Frames | Dur | FPS | Res | #dets | #tracks | "
            "Mean dens | Peak dens | Mean track len |"
        )
        md.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for seq in top:
            res = (
                f"{seq['width']}x{seq['height']}"
                if seq.get("width")
                else "—"
            )
            md.append(
                "| {name} | {nf} | {dur} | {fps} | {res} | {nd} | {nt} | {dens} | {peak} | {tl} |".format(
                    name=seq["name"],
                    nf=fmt_num(seq.get("n_frames")),
                    dur=fmt_dur(seq.get("duration_s")),
                    fps=fmt_num(seq.get("fps"), 0) if seq.get("fps") else "—",
                    res=res,
                    nd=fmt_num(seq.get("n_dets")),
                    nt=fmt_num(seq.get("n_tracks")),
                    dens=fmt_num(seq.get("density_mean"), 2),
                    peak=fmt_num(seq.get("density_max")),
                    tl=fmt_num(seq.get("track_length_mean_frames"), 1),
                )
            )
        md.append("")
        md.append("</details>")
        md.append("")

    md.append("## Caveats")
    md.append("")
    md.append(
        "- Reflects **what is on disk now** for the listed default splits. "
        "UA-DETRAC test / CityFlow validation / TrafficMOT FirstFrame are omitted until included."
    )
    md.append("- `#tracks` = unique GT IDs in `gt.txt` (not tracker output).")
    md.append(
        "- Density uses sequence length as denominator (empty frames still count)."
    )
    md.append(
        "- UA-DETRAC may have fewer annotated frames than `seqLength` if XML only labels a subset."
    )
    md.append("")
    return "\n".join(md) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for benchmarks_statistics.{md,json}",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect()
    json_path = out_dir / "benchmarks_statistics.json"
    md_path = out_dir / "benchmarks_statistics.md"
    json_path.write_text(json.dumps(results, indent=2))
    md_path.write_text(render_md(results))
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
