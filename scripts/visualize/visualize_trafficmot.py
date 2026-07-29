#!/usr/bin/env python3
"""Visualize TrafficMOT sequences with ground-truth boxes and track trajectories.

Given the dataset root (folder containing ``image/`` and ``ground_truth/``),
renders every sequence to MP4 under a sibling output folder next to the dataset.

Fully annotated sequences get per-frame boxes + trajectory trails. First-frame
annotated sequences still render all frames, but overlays appear only where CSV
annotations exist (typically frame0).

Example:
    .venv/bin/python scripts/visualize/visualize_trafficmot.py /media/7TBSSD/data/tracking/TrafficMOT
"""

from __future__ import annotations

import argparse
import colorsys
import csv
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

CLASS_NAMES: Dict[int, str] = {
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
}

SPLITS = ("Fully_annotate", "FirstFrame_annotate")
Box = Tuple[float, float, float, float]  # x, y, w, h
FrameAnn = Dict[int, dict]  # track_id -> {box, class_id, class_name}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render TrafficMOT videos with GT boxes and trajectories."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to TrafficMOT root (contains image/ and ground_truth/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_parent>/TrafficMOT_visualizations",
    )
    parser.add_argument(
        "--split",
        choices=("all",) + SPLITS,
        default="all",
        help="Which subset to render (default: all).",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence folder names to render.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output video FPS (sequences are short ~30-frame clips).",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Number of past centers kept for each trajectory polyline.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Stop after rendering this many sequences (useful for smoke tests).",
    )
    return parser.parse_args()


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def frame_index_from_name(path: Path) -> int:
    # frame0.jpg / frame12.csv -> 0 / 12
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def list_frame_paths(seq_dir: Path) -> List[Path]:
    frames = sorted(seq_dir.glob("frame*.jpg"), key=frame_index_from_name)
    if not frames:
        frames = sorted(seq_dir.glob("*.jpg"), key=frame_index_from_name)
    return frames


def parse_csv_annotation(csv_path: Path) -> FrameAnn:
    targets: FrameAnn = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(float(row["trackID"]))
            class_id = int(float(row["classID"]))
            targets[tid] = {
                "box": (
                    float(row["x"]),
                    float(row["y"]),
                    float(row["w"]),
                    float(row["h"]),
                ),
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            }
    return targets


def load_sequence_annotations(gt_dir: Path) -> Dict[int, FrameAnn]:
    annotations: Dict[int, FrameAnn] = {}
    for csv_path in sorted(gt_dir.glob("frame*.csv"), key=frame_index_from_name):
        annotations[frame_index_from_name(csv_path)] = parse_csv_annotation(csv_path)
    return annotations


def update_trails(
    trails: Dict[int, Deque[Tuple[int, int]]],
    targets: FrameAnn,
    trail_length: int,
) -> None:
    active = set(targets)
    for tid, info in targets.items():
        x, y, w, h = info["box"]
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        if tid not in trails:
            trails[tid] = deque(maxlen=trail_length)
        trails[tid].append((cx, cy))
    for tid in list(trails):
        if tid not in active:
            del trails[tid]


def draw_annotations(
    frame: np.ndarray,
    targets: FrameAnn,
    trails: Dict[int, Deque[Tuple[int, int]]],
) -> None:
    for tid, points in trails.items():
        if len(points) < 2:
            continue
        color = color_for_id(tid)
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    h, w = frame.shape[:2]
    for tid, info in targets.items():
        x, y, bw, bh = info["box"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + bw))
        y2 = int(round(y + bh))
        # Clamp for drawing; GT can slightly leave the frame.
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        color = color_for_id(tid)
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)

        label = f"ID {tid} {info['class_name']}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + baseline + 2, y1c - 4)
        cv2.rectangle(frame, (x1c, ty - th - baseline), (x1c + tw + 4, ty + 2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1c + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def render_sequence(
    seq_name: str,
    seq_dir: Path,
    gt_dir: Path,
    out_path: Path,
    fps: float,
    trail_length: int,
) -> int:
    annotations = load_sequence_annotations(gt_dir)
    frame_paths = list_frame_paths(seq_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {seq_dir}")

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise RuntimeError(f"Failed to read {frame_paths[0]}")
    height, width = sample.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    trails: Dict[int, Deque[Tuple[int, int]]] = {}
    written = 0
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"  warning: skipping unreadable frame {frame_path}")
                continue

            frame_idx = frame_index_from_name(frame_path)
            targets = annotations.get(frame_idx, {})
            if targets:
                update_trails(trails, targets, trail_length)
            else:
                # First-frame-only sequences: clear trails off annotated frames.
                trails.clear()

            draw_annotations(frame, targets, trails)

            hud = (
                f"{seq_name}  frame {frame_idx} ({written + 1}/{len(frame_paths)})  "
                f"tracks {len(targets)}"
            )
            cv2.putText(
                frame,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    return written


def collect_jobs(
    dataset_root: Path,
    split: str,
    sequences: Optional[List[str]],
) -> List[Tuple[str, str, Path, Path]]:
    """Return list of (split_name, seq_name, image_dir, gt_dir)."""
    splits = list(SPLITS) if split == "all" else [split]
    wanted = set(sequences) if sequences else None
    jobs: List[Tuple[str, str, Path, Path]] = []

    for split_name in splits:
        image_root = dataset_root / "image" / split_name
        gt_root = dataset_root / "ground_truth" / split_name
        if not image_root.is_dir():
            print(f"warning: missing image split {image_root}, skipping")
            continue
        if not gt_root.is_dir():
            print(f"warning: missing GT split {gt_root}, skipping")
            continue

        for seq_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
            seq_name = seq_dir.name
            if wanted is not None and seq_name not in wanted:
                continue
            gt_dir = gt_root / seq_name
            if not gt_dir.is_dir():
                print(f"warning: missing GT for {split_name}/{seq_name}, skipping")
                continue
            jobs.append((split_name, seq_name, seq_dir, gt_dir))
    return jobs


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root.parent / "TrafficMOT_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(dataset_root, args.split, args.sequences)
    if args.max_sequences is not None:
        jobs = jobs[: args.max_sequences]
    if not jobs:
        raise SystemExit("No sequences found to visualize.")

    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Sequences    : {len(jobs)}")

    for i, (split_name, seq_name, seq_dir, gt_dir) in enumerate(jobs, start=1):
        out_path = output_dir / split_name / f"{seq_name}.mp4"
        print(f"[{i}/{len(jobs)}] {split_name}/{seq_name} -> {out_path}")
        n = render_sequence(
            seq_name=seq_name,
            seq_dir=seq_dir,
            gt_dir=gt_dir,
            out_path=out_path,
            fps=args.fps,
            trail_length=args.trail_length,
        )
        print(f"  wrote {n} frames")

    print("Done.")


if __name__ == "__main__":
    main()
