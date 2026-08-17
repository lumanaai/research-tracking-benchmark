#!/usr/bin/env python3
"""Measure average tracker compute time per frame (association only).

Runs each registered tracker on one dense sequence with GT dets, writes
``results/comparisons/tracker_speed.json``, and updates
``mot_pipeline/tracker_meta.py`` ``ref_ms_per_frame`` values so comparison
tables have a timing column before full sweeps re-record ``timing.json``.

Example:
  .venv/bin/python scripts/analysis/bench_tracker_speed.py
  .venv/bin/python scripts/analysis/bench_tracker_speed.py --sequences task_day_occlusion --repeats 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mot_pipeline.mot_io import parse_seqinfo
from mot_pipeline.paths import DETECTIONS_ROOT, PROJECT_ROOT
from mot_pipeline.registry import get_benchmark, get_tracker
from mot_pipeline.tracker_meta import TRACKER_META, set_ref_ms_per_frame
from mot_pipeline.trackers import TRACKERS
from mot_pipeline.trackers.base import load_tracker_config, resolve_tracking_schedule
from mot_pipeline.run import _default_tracker_config, _tracker_defaults

DEFAULT_BENCH = "fasttracker_bench"
DEFAULT_SEQ = "task_day_occlusion"
DEFAULT_DET_ID = "gt_vehicles"
OUT_JSON = PROJECT_ROOT / "results" / "comparisons" / "tracker_speed.json"
META_PATH = PROJECT_ROOT / "mot_pipeline" / "tracker_meta.py"

SKIP = {"traffictrack"}


def _patch_meta_file(values: Dict[str, float]) -> None:
    """Persist ref_ms_per_frame into tracker_meta.py TRACKER_META entries."""
    text = META_PATH.read_text()
    for name, ms in values.items():
        # Insert or replace ref_ms_per_frame inside the named dict block.
        pattern = rf'("{name}":\s*\{{)(.*?)(\n    \}},)'
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        body = match.group(2)
        body2 = re.sub(
            r'\n\s*"ref_ms_per_frame":\s*[0-9.]+,?',
            "",
            body,
        )
        # Place after preferred_detector line when possible.
        if '"preferred_detector"' in body2:
            body2 = re.sub(
                r'("preferred_detector":\s*"[^"]+",)',
                rf'\1\n        "ref_ms_per_frame": {ms:.4f},',
                body2,
                count=1,
            )
        else:
            body2 = body2.rstrip() + f'\n        "ref_ms_per_frame": {ms:.4f},\n'
        text = text[: match.start()] + match.group(1) + body2 + match.group(3) + text[match.end() :]
    META_PATH.write_text(text)


def _time_tracker(
    name: str,
    seq_dir: Path,
    det_path: Path,
    out_path: Path,
    *,
    repeats: int,
) -> Dict[str, Any]:
    cfg_path = _default_tracker_config(name, DEFAULT_BENCH)
    cfg = load_tracker_config(cfg_path, _tracker_defaults(name))
    meta = parse_seqinfo(seq_dir)
    seq_len = int(meta.get("seqLength", 0) or 0)
    frame_ids, _, _, _ = resolve_tracking_schedule(meta, seq_len, {})
    n_frames = len(frame_ids) if frame_ids else max(seq_len, 0)

    # Warm import / first construct outside timed loop.
    tracker = get_tracker(name)
    tracker.track_sequence(seq_dir, det_path, out_path, cfg, extra={})

    times: List[float] = []
    for _ in range(max(1, repeats)):
        tracker = get_tracker(name)
        t0 = time.perf_counter()
        tracker.track_sequence(seq_dir, det_path, out_path, cfg, extra={})
        times.append(time.perf_counter() - t0)

    best = min(times)
    avg_ms = (1000.0 * best / n_frames) if n_frames > 0 else None
    return {
        "tracker": name,
        "frames": n_frames,
        "repeats": repeats,
        "best_seconds": round(best, 6),
        "avg_ms_per_frame": None if avg_ms is None else round(avg_ms, 4),
        "all_seconds": [round(t, 6) for t in times],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default=DEFAULT_BENCH)
    p.add_argument("--sequences", nargs="+", default=[DEFAULT_SEQ])
    p.add_argument("--detector-id", default=DEFAULT_DET_ID)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--trackers", nargs="+", default=None)
    p.add_argument("--out", type=Path, default=OUT_JSON)
    p.add_argument(
        "--no-patch-meta",
        action="store_true",
        help="Do not rewrite ref_ms_per_frame into tracker_meta.py",
    )
    args = p.parse_args()

    bench = get_benchmark(args.benchmark)
    split = bench.default_split()
    seq_name = args.sequences[0]
    seq_dirs = bench.sequence_dirs(split, [seq_name])
    if not seq_dirs:
        raise SystemExit(f"Sequence not found: {seq_name}")
    seq_dir = seq_dirs[0]
    det_path = DETECTIONS_ROOT / args.benchmark / split / args.detector_id / seq_name / "det.txt"
    if not det_path.is_file():
        raise SystemExit(f"Missing detections: {det_path}")

    names = args.trackers or [n for n in TRACKERS if n not in SKIP]
    # Keep a stable order matching TRACKER_META / compare_findings.
    order = [n for n in TRACKER_META if n in names] + [n for n in names if n not in TRACKER_META]

    tmp_dir = Path("/tmp/mot_tracker_speed")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    ref: Dict[str, float] = {}
    print(f"Timing on {seq_dir.name} ({args.detector_id}), repeats={args.repeats}")
    for name in order:
        if name in SKIP:
            continue
        out_path = tmp_dir / f"{name}.txt"
        print(f"  {name} ...", flush=True)
        row = _time_tracker(name, seq_dir, det_path, out_path, repeats=args.repeats)
        results.append(row)
        if row["avg_ms_per_frame"] is not None:
            ref[name] = float(row["avg_ms_per_frame"])
            print(f"    {row['avg_ms_per_frame']:.3f} ms/frame ({row['frames']} frames)")

    payload = {
        "benchmark": args.benchmark,
        "split": split,
        "sequence": seq_name,
        "detector_id": args.detector_id,
        "repeats": args.repeats,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")

    set_ref_ms_per_frame(ref)
    if not args.no_patch_meta:
        _patch_meta_file(ref)
        print(f"updated {META_PATH}")


if __name__ == "__main__":
    main()
