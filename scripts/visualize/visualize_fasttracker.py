#!/usr/bin/env python3
"""Visualize FastTracker-Benchmark sequences with GT boxes and trajectories.

Dataset layout (Hugging Face ``Hamidreza-Hashemp/FastTracker-Benchmark``)::

    train/
        task_xxx.zip                 # or extracted task_xxx/
            img1/000001.jpg
            gt/gt.txt                # frame,id,left,top,w,h,conf,class,visibility
            gt/labels.txt            # class names (1-indexed by line order)
            video/task_xxx.mp4
            seqinfo.ini

Example:
    .venv/bin/python scripts/visualize/visualize_fasttracker.py /media/7TBSSD/data/tracking/FastTracker-Benchmark
"""

from __future__ import annotations

import argparse
import colorsys
import configparser
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Fallback if a sequence has no gt/labels.txt (line order = class id).
DEFAULT_CLASS_NAMES: Dict[int, str] = {
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

IGNORE_CLASS_NAME = "ignore_region"
Box = Tuple[float, float, float, float]
FrameAnn = Dict[int, dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render FastTracker-Benchmark videos with GT boxes and trajectories."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to FastTracker-Benchmark root (contains train/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_parent>/FastTracker-Benchmark_visualizations",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence names, e.g. task_tunnel task_day_left_turn.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract missing train/*.zip archives into train/<seq>/ before rendering.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. Default: frameRate from seqinfo.ini (typically 30).",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=60,
        help="Number of past centers kept for each trajectory polyline.",
    )
    parser.add_argument(
        "--show-ignore",
        action="store_true",
        help="Also draw ignore_region annotations (class 9) in gray.",
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
        help="Optionally cap frames written per sequence.",
    )
    return parser.parse_args()


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def load_class_names(labels_path: Path) -> Dict[int, str]:
    if not labels_path.is_file():
        return dict(DEFAULT_CLASS_NAMES)
    names: Dict[int, str] = {}
    for i, line in enumerate(labels_path.read_text().splitlines(), start=1):
        name = line.strip()
        if name:
            names[i] = name
    return names or dict(DEFAULT_CLASS_NAMES)


def parse_seqinfo(seqinfo_path: Path) -> Dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    if "Sequence" not in parser:
        return {}
    return dict(parser["Sequence"])


def parse_gt(
    gt_path: Path,
    class_names: Dict[int, str],
    show_ignore: bool,
) -> Dict[int, FrameAnn]:
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
            class_id = int(float(parts[7])) if len(parts) >= 8 else -1
            class_name = class_names.get(class_id, f"class_{class_id}")
            if class_name == IGNORE_CLASS_NAME and not show_ignore:
                continue
            frames[frame_id][track_id] = {
                "box": (left, top, width, height),
                "class_id": class_id,
                "class_name": class_name,
            }
    return dict(frames)


def list_frame_paths(img_dir: Path, ext: str = ".jpg") -> List[Path]:
    if not ext.startswith("."):
        ext = f".{ext}"
    frames = sorted(img_dir.glob(f"*{ext}"))
    if not frames:
        frames = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        frames = sorted(set(frames), key=lambda p: p.name)
    return frames


def frame_index_from_name(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract archive into dest_dir; return the sequence folder path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    seq_name = zip_path.stem
    seq_dir = dest_dir / seq_name
    if seq_dir.is_dir() and (seq_dir / "gt" / "gt.txt").is_file():
        return seq_dir

    print(f"  extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m
            for m in zf.namelist()
            if not m.startswith("__MACOSX/") and "/.__" not in m and not Path(m).name.startswith("._")
        ]
        zf.extractall(dest_dir, members=members)

    if not seq_dir.is_dir():
        # Some archives may nest differently; pick the folder that has gt/gt.txt.
        candidates = [
            p for p in dest_dir.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").is_file()
        ]
        if not candidates:
            raise RuntimeError(f"Extraction of {zip_path} did not produce a sequence folder")
        seq_dir = candidates[0]
    return seq_dir


def discover_sequences(
    dataset_root: Path,
    sequences: Optional[List[str]],
    do_extract: bool,
) -> List[Path]:
    train_dir = dataset_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train/ under {dataset_root}")

    wanted = set(sequences) if sequences else None
    seq_dirs: Dict[str, Path] = {}

    # Already-extracted sequences.
    for path in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        if path.name.startswith(".") or path.name == "__MACOSX":
            continue
        if (path / "gt" / "gt.txt").is_file():
            if wanted is None or path.name in wanted:
                seq_dirs[path.name] = path

    # Zips (extract if requested, or if the extracted folder is missing).
    for zip_path in sorted(train_dir.glob("*.zip")):
        name = zip_path.stem
        if wanted is not None and name not in wanted:
            continue
        if name in seq_dirs:
            continue
        if do_extract:
            seq_dirs[name] = extract_zip(zip_path, train_dir)
        else:
            print(
                f"warning: {zip_path.name} is not extracted; "
                f"pass --extract or unzip it under train/{name}/"
            )

    return [seq_dirs[k] for k in sorted(seq_dirs)]


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


def draw_annotations(
    frame: np.ndarray,
    targets: FrameAnn,
    trails: Dict[int, Deque[Tuple[int, int]]],
) -> None:
    for tid, points in trails.items():
        if len(points) < 2:
            continue
        info = targets.get(tid)
        if info and info["class_name"] == IGNORE_CLASS_NAME:
            color = (120, 120, 120)
        else:
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

        is_ignore = info["class_name"] == IGNORE_CLASS_NAME
        color = (120, 120, 120) if is_ignore else color_for_id(tid)
        thickness = 1 if is_ignore else 2
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, thickness)

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
    seq_dir: Path,
    out_path: Path,
    fps_override: Optional[float],
    trail_length: int,
    show_ignore: bool,
    max_frames: Optional[int],
) -> int:
    seq_name = seq_dir.name
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise RuntimeError(f"Missing GT: {gt_path}")

    class_names = load_class_names(seq_dir / "gt" / "labels.txt")
    annotations = parse_gt(gt_path, class_names, show_ignore)

    meta = parse_seqinfo(seq_dir / "seqinfo.ini") if (seq_dir / "seqinfo.ini").is_file() else {}
    img_dir = seq_dir / meta.get("imDir", "img1")
    ext = meta.get("imExt", ".jpg")
    fps = fps_override
    if fps is None:
        try:
            fps = float(meta.get("frameRate", 30))
        except ValueError:
            fps = 30.0

    frame_paths = list_frame_paths(img_dir, ext)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {img_dir}")

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
            if max_frames is not None and written >= max_frames:
                break
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"  warning: skipping unreadable frame {frame_path}")
                continue

            frame_idx = frame_index_from_name(frame_path)
            targets = annotations.get(frame_idx, {})
            update_trails(trails, targets, trail_length)
            draw_annotations(frame, targets, trails)

            hud = (
                f"{seq_name}  frame {frame_idx}/{len(frame_paths)}  "
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


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root.parent / "FastTracker-Benchmark_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = discover_sequences(dataset_root, args.sequences, args.extract)
    if args.max_sequences is not None:
        seq_dirs = seq_dirs[: args.max_sequences]
    if not seq_dirs:
        raise SystemExit(
            "No sequences found. Wait for download to finish, then pass --extract "
            "or unzip train/*.zip under train/<seq>/."
        )

    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Sequences    : {len(seq_dirs)}")

    for i, seq_dir in enumerate(seq_dirs, start=1):
        out_path = output_dir / f"{seq_dir.name}.mp4"
        print(f"[{i}/{len(seq_dirs)}] {seq_dir.name} -> {out_path}")
        n = render_sequence(
            seq_dir=seq_dir,
            out_path=out_path,
            fps_override=args.fps,
            trail_length=args.trail_length,
            show_ignore=args.show_ignore,
            max_frames=args.max_frames,
        )
        print(f"  wrote {n} frames")

    print("Done.")


if __name__ == "__main__":
    main()
