#!/usr/bin/env python3
"""Overlay a MOTChallenge results/GT file on a folder of frames -> MP4.

Generic helper to eyeball any tracker output (e.g. FastTracker) or GT in MOT
format: `frame,id,bb_left,bb_top,bb_width,bb_height,...`.

Example:
    .venv/bin/python scripts/visualize/visualize_mot_results.py \
        --frames /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/MVI_39811/img1 \
        --results /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/fasttracker_results/MVI_39811.txt \
        --out /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/MVI_39811_tracked.mp4
"""

from __future__ import annotations

import argparse
import colorsys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a MOT results file over frames.")
    p.add_argument("--frames", type=Path, required=True, help="Directory of frame images.")
    p.add_argument("--results", type=Path, required=True, help="MOT-format txt to overlay.")
    p.add_argument("--out", type=Path, required=True, help="Output .mp4 path.")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--trail-length", type=int, default=60)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth image (1-indexed MOT frames). Use with --fps to match subsampled tracking.",
    )
    p.add_argument("--label", type=str, default="", help="Optional HUD prefix.")
    return p.parse_args()


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def frame_index(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def load_mot(path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    per_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c = line.split(",")
        f = int(float(c[0]))
        tid = int(float(c[1]))
        x, y, w, h = (float(v) for v in c[2:6])
        per_frame[f].append((tid, x, y, w, h))
    return per_frame


def main() -> None:
    args = parse_args()
    frames = sorted(args.frames.glob("*.jpg"), key=frame_index)
    if not frames:
        frames = sorted(args.frames.glob("*.png"), key=frame_index)
    if not frames:
        raise SystemExit(f"No frames in {args.frames}")

    stride = max(1, int(args.frame_stride))
    if stride > 1:
        frames = [fp for fp in frames if frame_index(fp) > 0 and (frame_index(fp) - 1) % stride == 0]
        if not frames:
            raise SystemExit(f"No frames left after --frame-stride {stride}")

    per_frame = load_mot(args.results)
    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    trails: Dict[int, Deque[Tuple[int, int]]] = {}
    written = 0
    for fp in frames:
        if args.max_frames is not None and written >= args.max_frames:
            break
        img = cv2.imread(str(fp))
        if img is None:
            continue
        fidx = frame_index(fp)
        dets = per_frame.get(fidx, [])

        active = set()
        for tid, x, y, bw, bh in dets:
            active.add(tid)
            cx, cy = int(x + bw / 2), int(y + bh / 2)
            trails.setdefault(tid, deque(maxlen=args.trail_length)).append((cx, cy))
        for tid in list(trails):
            if tid not in active:
                del trails[tid]

        for tid, pts in trails.items():
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)], False,
                              color_for_id(tid), 2, cv2.LINE_AA)
        for tid, x, y, bw, bh in dets:
            c = color_for_id(tid)
            p1, p2 = (int(x), int(y)), (int(x + bw), int(y + bh))
            cv2.rectangle(img, p1, p2, c, 2)
            lbl = f"ID {tid}"
            (tw, th), bs = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 2, int(y) - 4)
            cv2.rectangle(img, (int(x), ty - th - bs), (int(x) + tw + 4, ty + 2), c, -1)
            cv2.putText(img, lbl, (int(x) + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        hud = f"{args.label} frame {fidx}  tracks {len(dets)}".strip()
        cv2.putText(img, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
        written += 1

    writer.release()
    print(f"wrote {written} frames -> {args.out}")


if __name__ == "__main__":
    main()
