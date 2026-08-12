#!/usr/bin/env python3
"""Generate analytics_bytetrack config variants and track+eval each (no detect).

Default grid = Phase-1 lifecycle: n_init × track_buffer_seconds × detector_unique.

Example (from project root):

  .venv/bin/python scripts/batch/sweep_analytics_params.py \\
    --benchmarks cityflow ua_detrac --fps 5 --dry-run

  .venv/bin/python scripts/batch/sweep_analytics_params.py \\
    --benchmarks cityflow ua_detrac --fps 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
BASE_CFG = ROOT / "mot_pipeline/configs/trackers/analytics_bytetrack/benchmark.json"
OUT_DIR = ROOT / "mot_pipeline/configs/trackers/analytics_bytetrack/sweep_fps5"
PYTHON = ROOT / ".venv/bin/python"

# Phase-1 lifecycle grid
N_INIT = (1, 2, 3)
BUFFER_S = (1.0, 2.0, 3.0, 6.0)
DETECTOR_UNIQUE = (True, False)


def _load_base() -> Dict[str, Any]:
    with BASE_CFG.open() as f:
        return json.load(f)


def _tag(n_init: int, buf: float, du: bool) -> str:
    buf_txt = f"{buf:g}".replace(".", "p")
    return f"n{n_init}_buf{buf_txt}_du{int(du)}"


def lifecycle_configs() -> List[Tuple[str, Dict[str, Any]]]:
    base = _load_base()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for n_init, buf, du in product(N_INIT, BUFFER_S, DETECTOR_UNIQUE):
        cfg = deepcopy(base)
        cfg["n_init"] = int(n_init)
        cfg["track_buffer_seconds"] = float(buf)
        cfg["detector_unique"] = bool(du)
        out.append((_tag(n_init, buf, du), cfg))
    return out


def write_configs(
    variants: Iterable[Tuple[str, Dict[str, Any]]], out_dir: Path
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for stem, cfg in variants:
        path = out_dir / f"{stem}.json"
        with path.open("w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        paths.append(path)
    return paths


def _run(cmd: List[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def track_eval(
    *,
    benchmark: str,
    cfg_path: Path,
    fps: float,
    detector: str,
    dry_run: bool,
    sequences: List[str] | None,
) -> None:
    # Stable run_id so re-runs overwrite findings for the same variant.
    run_id = f"{benchmark}_analytics_bytetrack_{detector}_{cfg_path.stem}_fps{fps:g}"
    track_cmd = [
        str(PYTHON),
        "-m",
        "mot_pipeline.run",
        "track",
        "--benchmark",
        benchmark,
        "--tracker",
        "analytics_bytetrack",
        "--detector",
        detector,
        "--tracker-config",
        str(cfg_path),
        "--fps",
        f"{fps:g}",
        "--run-id",
        run_id,
    ]
    if sequences:
        track_cmd += ["--sequences", *sequences]
    _run(track_cmd, dry_run)

    eval_cmd = [
        str(PYTHON),
        "-m",
        "mot_pipeline.run",
        "eval",
        "--run-id",
        run_id,
    ]
    _run(eval_cmd, dry_run)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=["cityflow", "ua_detrac"],
        help="Benchmarks to sweep (default: cityflow ua_detrac).",
    )
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument(
        "--detector",
        default="yolov8",
        help="Detector name for cache lookup (default yolov8 → expert weights id).",
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument(
        "--write-only",
        action="store_true",
        help="Only write JSON configs; do not track/eval.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional config stems to run (e.g. n1_buf1_du1 n3_buf6_du0).",
    )
    args = p.parse_args()

    if not PYTHON.is_file():
        sys.exit(f"Missing venv python: {PYTHON}")

    variants = lifecycle_configs()
    if args.only:
        want = set(args.only)
        variants = [(s, c) for s, c in variants if s in want]
        missing = want - {s for s, _ in variants}
        if missing:
            sys.exit(f"Unknown --only stems: {sorted(missing)}")

    paths = write_configs(variants, args.out_dir)
    print(f"Wrote {len(paths)} configs → {args.out_dir}", flush=True)
    if args.write_only:
        return

    for bench in args.benchmarks:
        for cfg_path in paths:
            print(f"\n=== {bench} / {cfg_path.stem} @ {args.fps:g} FPS ===", flush=True)
            track_eval(
                benchmark=bench,
                cfg_path=cfg_path,
                fps=args.fps,
                detector=args.detector,
                dry_run=args.dry_run,
                sequences=args.sequences,
            )


if __name__ == "__main__":
    main()
