#!/usr/bin/env python3
"""Run an Ultralytics YOLO model over MOT-style sequences and write det/det.txt.

Legacy standalone detector. Prefer ``.venv/bin/python -m mot_pipeline.run detect``
for shared caches under ``detections/``.

For each sequence folder that contains `img1/` (and usually `seqinfo.ini`), this
runs YOLO on every frame and writes MOTChallenge detections:

    frame, -1, bb_left, bb_top, bb_width, bb_height, conf, class, -1

Example:
    .venv/bin/python scripts/detect/run_yolo_dets.py \
      /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
      --weights /media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt \
      --device cuda:0 --imgsz 1280 --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

# COCO ids that map onto road users (person + vehicles).
DEFAULT_COCO_CLASSES = [0, 1, 2, 3, 5, 7]
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_WEIGHTS = Path("/media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO detections -> MOT det.txt per sequence.")
    p.add_argument("dataset_root", type=Path, help="Folder of sequences (each has img1/).")
    p.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Ultralytics weights (default: {DEFAULT_WEIGHTS}).",
    )
    p.add_argument("--sequences", nargs="+", default=None, help="Subset of sequence names.")
    p.add_argument("--device", default=None, help="cuda:0 / cpu / 0,1. Default: auto.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size (long side).")
    p.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=DEFAULT_COCO_CLASSES,
        help="Class ids to keep (default: traffic COCO classes). Pass -1 to keep all.",
    )
    p.add_argument("--batch", type=int, default=16, help="Batch size for inference.")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only).")
    p.add_argument(
        "--det-name",
        default="det/det.txt",
        help="Relative output path within each sequence (default det/det.txt).",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames per sequence (debug).")
    return p.parse_args()


def frame_index(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def discover_sequences(root: Path, wanted: Optional[List[str]]) -> List[Path]:
    seqs = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == "__MACOSX":
            continue
        if not (p / "img1").is_dir():
            continue
        if wanted is not None and p.name not in wanted:
            continue
        seqs.append(p)
    return seqs


def run_sequence(model, seq_dir: Path, args) -> int:
    frames = sorted((seq_dir / "img1").glob("*.jpg"), key=frame_index)
    if not frames:
        frames = sorted((seq_dir / "img1").glob("*.png"), key=frame_index)
    if not frames:
        return 0
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    classes = None if (args.classes and -1 in args.classes) else args.classes
    out_path = seq_dir / args.det_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predict_kwargs = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        classes=classes,
        verbose=False,
    )
    if args.half:
        predict_kwargs["half"] = True

    n_det = 0
    with out_path.open("w") as f:
        # Explicit fixed-size batches: passing the whole list to predict makes
        # ultralytics run it as ONE batch (OOMs on long sequences).
        for start in range(0, len(frames), args.batch):
            chunk = frames[start : start + args.batch]
            results = model.predict(source=[str(p) for p in chunk], **predict_kwargs)
            for fp, res in zip(chunk, results):
                fidx = frame_index(fp)
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    w, h = x2 - x1, y2 - y1
                    f.write(f"{fidx},-1,{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{c:.4f},{int(k)},-1\n")
                    n_det += 1
    return n_det


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")

    from ultralytics import YOLO  # local import so --help needs no torch

    seqs = discover_sequences(root, args.sequences)
    if not seqs:
        raise SystemExit("No sequences with img1/ found.")

    model = YOLO(args.weights)
    kept = "all" if (args.classes and -1 in args.classes) else \
        ", ".join(COCO_NAMES.get(c, str(c)) for c in args.classes)
    print(f"Weights   : {args.weights}")
    print(f"Device    : {args.device or 'auto'}   imgsz={args.imgsz} conf={args.conf}")
    print(f"Keep class: {kept}")
    print(f"Sequences : {len(seqs)}")

    for i, seq in enumerate(seqs, 1):
        print(f"[{i}/{len(seqs)}] {seq.name} ...", flush=True)
        n = run_sequence(model, seq, args)
        print(f"  wrote {n} detections -> {seq / args.det_name}")
    print("Done.")


if __name__ == "__main__":
    main()
