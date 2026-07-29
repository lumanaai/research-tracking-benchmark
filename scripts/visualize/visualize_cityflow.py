#!/usr/bin/env python3
"""Visualize CityFlow (AIC22 / CityFlowV2) cameras with GT boxes and trajectories.

Given the dataset root (folder containing ``train/``, ``validation/``, ``test/``),
reads each camera's ``vdo.avi`` and overlays MOTChallenge ground truth from
``gt/gt.txt`` (train/validation only; test has no public GT).

Example:
    .venv/bin/python scripts/visualize/visualize_cityflow.py /media/7TBSSD/data/tracking/CityFlow
"""

from __future__ import annotations

import argparse
import colorsys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

SPLITS = ("train", "validation", "test")
Box = Tuple[float, float, float, float]  # left, top, width, height
FrameAnn = Dict[int, dict]  # track_id -> {box}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render CityFlow videos with GT boxes and trajectories."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to CityFlow root (contains train/, validation/, test/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_parent>/CityFlow_visualizations",
    )
    parser.add_argument(
        "--split",
        choices=("all",) + SPLITS,
        default="all",
        help="Which split to render (default: all; cameras without GT are skipped).",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional cameras to render, e.g. c001 S01/c001 train/S01/c001.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. Default: FPS from each vdo.avi (typically 10).",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=60,
        help="Number of past centers kept for each trajectory polyline.",
    )
    parser.add_argument(
        "--show-roi",
        action="store_true",
        help="Blend the camera ROI mask (roi.jpg) onto the frame.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Stop after rendering this many cameras (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally cap frames written per camera (useful for smoke tests).",
    )
    return parser.parse_args()


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def parse_mot_gt(gt_path: Path) -> Dict[int, FrameAnn]:
    """Parse MOTChallenge gt.txt: frame,id,left,top,width,height,..."""
    frames: Dict[int, FrameAnn] = defaultdict(dict)
    with gt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left, top, width, height = map(float, parts[2:6])
            frames[frame_id][track_id] = {"box": (left, top, width, height)}
    return dict(frames)


def matches_sequence_filter(rel_path: str, sequences: Optional[List[str]]) -> bool:
    if sequences is None:
        return True
    rel = rel_path.strip("/")
    parts = rel.split("/")
    cam = parts[-1]
    scenario_cam = "/".join(parts[-2:]) if len(parts) >= 2 else cam
    for item in sequences:
        item = item.strip().strip("/")
        if item in (rel, scenario_cam, cam):
            return True
    return False


def collect_jobs(
    dataset_root: Path,
    split: str,
    sequences: Optional[List[str]],
) -> List[Tuple[str, str, str, Path, Optional[Path]]]:
    """Return (split, scenario, camera, video_path, gt_path_or_None)."""
    splits = list(SPLITS) if split == "all" else [split]
    jobs: List[Tuple[str, str, str, Path, Optional[Path]]] = []

    for split_name in splits:
        split_dir = dataset_root / split_name
        if not split_dir.is_dir():
            print(f"warning: missing split {split_dir}, skipping")
            continue
        for scenario_dir in sorted(p for p in split_dir.glob("S*") if p.is_dir()):
            for cam_dir in sorted(p for p in scenario_dir.glob("c*") if p.is_dir()):
                rel = f"{split_name}/{scenario_dir.name}/{cam_dir.name}"
                if not matches_sequence_filter(rel, sequences):
                    continue
                video_path = cam_dir / "vdo.avi"
                if not video_path.is_file():
                    print(f"warning: missing video {video_path}, skipping")
                    continue
                gt_path = cam_dir / "gt" / "gt.txt"
                jobs.append(
                    (
                        split_name,
                        scenario_dir.name,
                        cam_dir.name,
                        video_path,
                        gt_path if gt_path.is_file() else None,
                    )
                )
    return jobs


def update_trails(
    trails: Dict[int, Deque[Tuple[int, int]]],
    targets: FrameAnn,
    trail_length: int,
) -> None:
    active = set(targets)
    for tid, info in targets.items():
        left, top, width, height = info["box"]
        cx = int(round(left + width / 2.0))
        cy = int(round(top + height / 2.0))
        if tid not in trails:
            trails[tid] = deque(maxlen=trail_length)
        trails[tid].append((cx, cy))
    for tid in list(trails):
        if tid not in active:
            del trails[tid]


def blend_roi(frame: np.ndarray, roi_path: Path) -> None:
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        return
    if roi.shape[:2] != frame.shape[:2]:
        roi = cv2.resize(roi, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = roi > 0
    tint = frame.copy()
    tint[mask] = (tint[mask] * 0.65 + np.array([0, 180, 0], dtype=np.float32) * 0.35).astype(
        np.uint8
    )
    frame[:] = tint


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
        left, top, width, height = info["box"]
        x1 = int(round(left))
        y1 = int(round(top))
        x2 = int(round(left + width))
        y2 = int(round(top + height))
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        color = color_for_id(tid)
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)

        label = f"ID {tid}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + baseline + 2, y1c - 4)
        cv2.rectangle(frame, (x1c, ty - th - baseline), (x1c + tw + 4, ty + 2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1c + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def render_camera(
    label: str,
    video_path: Path,
    gt_path: Optional[Path],
    out_path: Path,
    fps_override: Optional[float],
    trail_length: int,
    show_roi: bool,
    max_frames: Optional[int],
) -> int:
    if gt_path is None:
        raise RuntimeError(f"No GT available for {label}")

    annotations = parse_mot_gt(gt_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps_override if fps_override is not None else (src_fps if src_fps > 1e-3 else 10.0)
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    roi_path = video_path.parent / "roi.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {out_path}")

    trails: Dict[int, Deque[Tuple[int, int]]] = {}
    written = 0
    frame_idx = 0  # CityFlow / MOTChallenge frames are 1-indexed in gt.txt
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames is not None and written >= max_frames:
                break

            targets = annotations.get(frame_idx, {})
            update_trails(trails, targets, trail_length)
            if show_roi and roi_path.is_file():
                blend_roi(frame, roi_path)
            draw_annotations(frame, targets, trails)

            total_str = str(total_hint) if total_hint > 0 else "?"
            hud = f"{label}  frame {frame_idx}/{total_str}  tracks {len(targets)}"
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
        cap.release()

    return written


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root.parent / "CityFlow_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(dataset_root, args.split, args.sequences)
    # Test has no public GT; skip those unless somehow present.
    skipped_no_gt = [j for j in jobs if j[4] is None]
    jobs = [j for j in jobs if j[4] is not None]
    if skipped_no_gt:
        print(f"Skipping {len(skipped_no_gt)} camera(s) without gt/gt.txt")

    if args.max_sequences is not None:
        jobs = jobs[: args.max_sequences]
    if not jobs:
        raise SystemExit("No cameras with ground truth found to visualize.")

    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Cameras      : {len(jobs)}")

    for i, (split_name, scenario, camera, video_path, gt_path) in enumerate(jobs, start=1):
        label = f"{split_name}/{scenario}/{camera}"
        out_path = output_dir / split_name / scenario / f"{camera}.mp4"
        print(f"[{i}/{len(jobs)}] {label} -> {out_path}")
        n = render_camera(
            label=label,
            video_path=video_path,
            gt_path=gt_path,
            out_path=out_path,
            fps_override=args.fps,
            trail_length=args.trail_length,
            show_roi=args.show_roi,
            max_frames=args.max_frames,
        )
        print(f"  wrote {n} frames")

    print("Done.")


if __name__ == "__main__":
    main()
