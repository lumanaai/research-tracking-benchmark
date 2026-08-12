#!/usr/bin/env python3
"""Visualize LumanaBenchmark sequences with GT boxes and trajectories.

Dataset layout::

    gt_annotations_manually_validated/
        <seq>/seqinfo.ini
        <seq>/gt/gt.txt          # frame,id,x,y,w,h,conf,class  (vehicle=1)
        <seq>/img1/…             # optional — frames not shipped yet

Sequences without ``img1/`` frames are skipped (videos may be uploaded later).
Once frames exist under ``img1/``, this script overlays GT IDs and trails.

Example:
    .venv/bin/python scripts/visualize/visualize_lumana.py \\
        /media/7TBSSD/data/tracking/LumanaBenchmark
"""

from __future__ import annotations

import argparse
import colorsys
import configparser
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]
FrameAnn = Dict[int, dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render LumanaBenchmark videos with GT boxes and trajectories."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to LumanaBenchmark root (contains gt_annotations_manually_validated/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_parent>/LumanaBenchmark_visualizations",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence names to render.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. Default: frameRate from seqinfo.ini.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=60,
        help="Number of past centers kept for each trajectory polyline.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Stop after rendering this many sequences.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames written per sequence (smoke tests).",
    )
    return parser.parse_args()


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def parse_seqinfo(path: Path) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    sec = cfg["Sequence"] if cfg.has_section("Sequence") else cfg[cfg.sections()[0]]
    return {
        "name": sec.get("name", path.parent.name),
        "frameRate": float(sec.get("frameRate", "20")),
        "seqLength": int(float(sec.get("seqLength", "0"))),
        "imWidth": int(float(sec.get("imWidth", "0"))),
        "imHeight": int(float(sec.get("imHeight", "0"))),
    }


def load_gt(gt_path: Path) -> Dict[int, FrameAnn]:
    by_frame: Dict[int, FrameAnn] = defaultdict(dict)
    with gt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            by_frame[frame][tid] = {"box": (x, y, w, h)}
    return by_frame


def list_frames(img_dir: Path) -> List[Path]:
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(img_dir.glob("*.png"))
    return frames


def discover_sequences(
    ann_root: Path, sequences: Optional[List[str]]
) -> List[Path]:
    wanted = set(sequences) if sequences else None
    seqs: List[Path] = []
    for path in sorted(p for p in ann_root.iterdir() if p.is_dir()):
        if path.name == "seqmaps":
            continue
        if not (path / "gt" / "gt.txt").is_file():
            continue
        if wanted is not None and path.name not in wanted:
            continue
        seqs.append(path)
    return seqs


def render_sequence(
    seq_dir: Path,
    out_path: Path,
    *,
    fps_override: Optional[float],
    trail_length: int,
    max_frames: Optional[int],
) -> bool:
    img_dir = seq_dir / "img1"
    frames = list_frames(img_dir) if img_dir.is_dir() else []
    if not frames:
        print(f"  SKIP {seq_dir.name}: no img1/ frames yet")
        return False

    info = parse_seqinfo(seq_dir / "seqinfo.ini") if (seq_dir / "seqinfo.ini").is_file() else {}
    fps = fps_override or float(info.get("frameRate") or 20.0)
    anns = load_gt(seq_dir / "gt" / "gt.txt")

    sample = cv2.imread(str(frames[0]))
    if sample is None:
        print(f"  SKIP {seq_dir.name}: cannot read {frames[0]}")
        return False
    h, w = sample.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    trails: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=trail_length)
    )
    n_write = len(frames) if max_frames is None else min(len(frames), max_frames)
    for i, fp in enumerate(frames[:n_write], start=1):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        for tid, ann in anns.get(i, {}).items():
            x, y, bw, bh = ann["box"]
            color = color_for_id(tid)
            pt1 = (int(x), int(y))
            pt2 = (int(x + bw), int(y + bh))
            cv2.rectangle(frame, pt1, pt2, color, 2)
            cx, cy = x + bw / 2.0, y + bh / 2.0
            trails[tid].append((cx, cy))
            pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, color, 2)
            cv2.putText(
                frame,
                f"ID {tid}",
                (pt1[0], max(15, pt1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        writer.write(frame)
    writer.release()
    print(f"  OK {seq_dir.name} → {out_path} ({n_write} frames)")
    return True


def main() -> None:
    args = parse_args()
    ann_root = args.dataset_root / "gt_annotations_manually_validated"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"Missing annotations dir: {ann_root}")

    out_root = args.output_dir or (
        args.dataset_root.parent / "LumanaBenchmark_visualizations"
    )
    seqs = discover_sequences(ann_root, args.sequences)
    if not seqs:
        print("No sequences found.")
        return

    rendered = 0
    skipped = 0
    for seq_dir in seqs:
        if args.max_sequences is not None and rendered >= args.max_sequences:
            break
        ok = render_sequence(
            seq_dir,
            out_root / f"{seq_dir.name}.mp4",
            fps_override=args.fps,
            trail_length=args.trail_length,
            max_frames=args.max_frames,
        )
        if ok:
            rendered += 1
        else:
            skipped += 1

    print(
        f"Done. rendered={rendered} skipped={skipped} "
        f"(frames required under each seq's img1/)"
    )


if __name__ == "__main__":
    main()
