#!/usr/bin/env python3
"""Visualize UA-DETRAC sequences with ground-truth boxes and track trajectories.

Given the dataset root (the folder containing DETRAC-Images and the
DETRAC-*-Annotations-XML directories), renders every annotated sequence to MP4
under a sibling output folder next to the dataset.

Example:
    .venv/bin/python scripts/visualize/visualize_ua_detrac.py /media/7TBSSD/data/tracking/UA-DETRAC
"""

from __future__ import annotations

import argparse
import colorsys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]  # left, top, width, height
FrameAnn = Dict[int, dict]  # track_id -> {box, vehicle_type, ...}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render UA-DETRAC videos with GT boxes and trajectories."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to UA-DETRAC root (contains DETRAC-Images and annotation XMLs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_parent>/UA-DETRAC_visualizations",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="Which annotation split to render (default: all).",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence names to render, e.g. MVI_20011 MVI_39031.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output video FPS (UA-DETRAC is typically ~25).",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=60,
        help="Number of past centers kept for each trajectory polyline.",
    )
    parser.add_argument(
        "--show-ignored",
        action="store_true",
        help="Overlay ignored regions from the XML (semi-transparent gray).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Stop after rendering this many sequences (useful for smoke tests).",
    )
    return parser.parse_args()


def resolve_subdir(root: Path, *candidates: str) -> Path:
    """Resolve nested Kaggle-style folders (e.g. DETRAC-Images/DETRAC-Images)."""
    for name in candidates:
        direct = root / name
        nested = direct / name
        if nested.is_dir():
            return nested
        if direct.is_dir():
            return direct
    raise FileNotFoundError(
        f"Could not find any of {candidates} under {root}"
    )


def discover_layout(dataset_root: Path) -> Dict[str, Path]:
    images = resolve_subdir(dataset_root, "DETRAC-Images")
    train_xml = resolve_subdir(
        dataset_root,
        "DETRAC-Train-Annotations-XML",
        "DETRAC-Train-Annotations-MAT",
    )
    test_xml = resolve_subdir(
        dataset_root,
        "DETRAC-Test-Annotations-XML",
        "DETRAC-Test-Annotations-MAT",
    )
    return {"images": images, "train": train_xml, "test": test_xml}


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Stable BGR color from track id (high saturation / value)."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def parse_box(elem: ET.Element) -> Box:
    return (
        float(elem.get("left")),
        float(elem.get("top")),
        float(elem.get("width")),
        float(elem.get("height")),
    )


def parse_annotation(xml_path: Path) -> Tuple[List[Box], Dict[int, FrameAnn]]:
    """Parse UA-DETRAC XML into ignored regions and per-frame track dicts."""
    root = ET.parse(xml_path).getroot()

    ignored: List[Box] = []
    ignored_region = root.find("ignored_region")
    if ignored_region is not None:
        for box_elem in ignored_region.findall("box"):
            ignored.append(parse_box(box_elem))

    frames: Dict[int, FrameAnn] = {}
    for frame_elem in root.findall("frame"):
        frame_num = int(frame_elem.get("num"))
        targets: FrameAnn = {}
        target_list = frame_elem.find("target_list")
        if target_list is not None:
            for target in target_list.findall("target"):
                tid = int(target.get("id"))
                box = parse_box(target.find("box"))
                attr = target.find("attribute")
                vehicle_type = (
                    attr.get("vehicle_type", "unknown") if attr is not None else "unknown"
                )
                targets[tid] = {"box": box, "vehicle_type": vehicle_type}
        frames[frame_num] = targets
    return ignored, frames


def list_frame_paths(seq_dir: Path) -> List[Path]:
    frames = sorted(seq_dir.glob("img*.jpg"))
    if not frames:
        frames = sorted(seq_dir.glob("*.jpg"))
    return frames


def frame_index_from_name(path: Path) -> int:
    # img00001.jpg -> 1
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def draw_ignored_regions(frame: np.ndarray, ignored: Iterable[Box]) -> None:
    overlay = frame.copy()
    for left, top, width, height in ignored:
        x1, y1 = int(left), int(top)
        x2, y2 = int(left + width), int(top + height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def draw_annotations(
    frame: np.ndarray,
    targets: FrameAnn,
    trails: Dict[int, Deque[Tuple[int, int]]],
    show_ignored: bool,
    ignored: List[Box],
) -> None:
    if show_ignored and ignored:
        draw_ignored_regions(frame, ignored)

    # Trajectories first so boxes sit on top.
    for tid, points in trails.items():
        if len(points) < 2:
            continue
        color = color_for_id(tid)
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    for tid, info in targets.items():
        left, top, width, height = info["box"]
        x1, y1 = int(round(left)), int(round(top))
        x2, y2 = int(round(left + width)), int(round(top + height))
        color = color_for_id(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tid} {info['vehicle_type']}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(0, y1 - 4)
        cv2.rectangle(frame, (x1, ty - th - baseline), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


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
    # Drop trails for tracks that left the scene.
    for tid in list(trails):
        if tid not in active:
            del trails[tid]


def render_sequence(
    seq_name: str,
    seq_dir: Path,
    xml_path: Path,
    out_path: Path,
    fps: float,
    trail_length: int,
    show_ignored: bool,
) -> int:
    ignored, annotations = parse_annotation(xml_path)
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
            update_trails(trails, targets, trail_length)
            draw_annotations(frame, targets, trails, show_ignored, ignored)

            hud = f"{seq_name}  frame {frame_idx}/{len(frame_paths)}  tracks {len(targets)}"
            cv2.putText(
                frame,
                hud,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
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
    layout: Dict[str, Path],
    split: str,
    sequences: Optional[List[str]],
) -> List[Tuple[str, str, Path, Path]]:
    """Return list of (split_name, seq_name, seq_dir, xml_path)."""
    splits: List[str]
    if split == "all":
        splits = ["train", "test"]
    else:
        splits = [split]

    wanted = set(sequences) if sequences else None
    jobs: List[Tuple[str, str, Path, Path]] = []
    for split_name in splits:
        xml_dir = layout[split_name]
        for xml_path in sorted(xml_dir.glob("*.xml")):
            seq_name = xml_path.stem
            if wanted is not None and seq_name not in wanted:
                continue
            seq_dir = layout["images"] / seq_name
            if not seq_dir.is_dir():
                print(f"warning: missing images for {seq_name}, skipping")
                continue
            jobs.append((split_name, seq_name, seq_dir, xml_path))
    return jobs


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    layout = discover_layout(dataset_root)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root.parent / "UA-DETRAC_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(layout, args.split, args.sequences)
    if args.max_sequences is not None:
        jobs = jobs[: args.max_sequences]
    if not jobs:
        raise SystemExit("No sequences found to visualize.")

    print(f"Dataset root : {dataset_root}")
    print(f"Images       : {layout['images']}")
    print(f"Output dir   : {output_dir}")
    print(f"Sequences    : {len(jobs)}")

    for i, (split_name, seq_name, seq_dir, xml_path) in enumerate(jobs, start=1):
        out_path = output_dir / split_name / f"{seq_name}.mp4"
        print(f"[{i}/{len(jobs)}] {split_name}/{seq_name} -> {out_path}")
        n = render_sequence(
            seq_name=seq_name,
            seq_dir=seq_dir,
            xml_path=xml_path,
            out_path=out_path,
            fps=args.fps,
            trail_length=args.trail_length,
            show_ignored=args.show_ignored,
        )
        print(f"  wrote {n} frames")

    print("Done.")


if __name__ == "__main__":
    main()
