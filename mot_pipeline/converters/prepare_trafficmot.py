"""Convert TrafficMOT CSV annotations into MOTChallenge folders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

from mot_pipeline.mot_io import write_seqinfo
from mot_pipeline.paths import MOT_ROOT, RAW_DATASETS

SPLITS = ("Fully_annotate", "FirstFrame_annotate")
DEFAULT_FPS = 10.0


def _frame_num_from_name(name: str) -> int:
    """``frame0.csv`` / ``frame0.jpg`` → 0."""
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def list_sequences(dataset_root: Path, split: str) -> List[str]:
    gt_root = dataset_root / "ground_truth" / split
    if not gt_root.is_dir():
        return []
    return sorted(p.name for p in gt_root.iterdir() if p.is_dir())


def convert_sequence(
    dataset_root: Path,
    split: str,
    seq: str,
    out_root: Path,
    *,
    force: bool = False,
) -> Path:
    img_src = dataset_root / "image" / split / seq
    gt_src = dataset_root / "ground_truth" / split / seq
    if not img_src.is_dir():
        raise FileNotFoundError(f"Missing images: {img_src}")
    if not gt_src.is_dir():
        raise FileNotFoundError(f"Missing GT: {gt_src}")

    out_seq = out_root / split / seq
    gt_path = out_seq / "gt" / "gt.txt"
    if gt_path.is_file() and (out_seq / "seqinfo.ini").is_file() and not force:
        return out_seq

    # Contiguous remap: sorted even indices 0,2,...,58 → MOT frames 1..N
    frame_files = sorted(img_src.glob("frame*.jpg"), key=lambda p: _frame_num_from_name(p.name))
    if not frame_files:
        frame_files = sorted(img_src.glob("*.jpg"), key=lambda p: _frame_num_from_name(p.name))
    if not frame_files:
        raise FileNotFoundError(f"No frames in {img_src}")

    native_to_mot: Dict[int, int] = {}
    for mot_idx, fp in enumerate(frame_files, start=1):
        native_to_mot[_frame_num_from_name(fp.name)] = mot_idx

    sample = cv2.imread(str(frame_files[0]))
    if sample is None:
        raise RuntimeError(f"Failed to read {frame_files[0]}")
    h, w = sample.shape[:2]

    out_seq.mkdir(parents=True, exist_ok=True)
    img1 = out_seq / "img1"
    if img1.exists() or img1.is_symlink():
        if img1.is_symlink():
            img1.unlink()
        elif img1.is_dir():
            for child in img1.iterdir():
                child.unlink()
        else:
            img1.unlink()
    img1.mkdir(parents=True, exist_ok=True)

    for mot_idx, fp in enumerate(frame_files, start=1):
        link = img1 / f"{mot_idx:06d}.jpg"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(fp.resolve())

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    with gt_path.open("w") as out_f:
        for csv_path in sorted(gt_src.glob("frame*.csv"), key=lambda p: _frame_num_from_name(p.name)):
            native = _frame_num_from_name(csv_path.name)
            if native not in native_to_mot:
                continue
            mot_frame = native_to_mot[native]
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        cls = int(float(row["classID"]))
                        tid = int(float(row["trackID"]))
                        x = float(row["x"])
                        y = float(row["y"])
                        bw = float(row["w"])
                        bh = float(row["h"])
                    except (KeyError, ValueError):
                        continue
                    out_f.write(
                        f"{mot_frame},{tid},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},1,{cls},1\n"
                    )
                    n_boxes += 1

    write_seqinfo(
        out_seq / "seqinfo.ini",
        name=seq,
        frame_rate=DEFAULT_FPS,
        seq_length=len(frame_files),
        im_width=w,
        im_height=h,
    )
    _ = n_boxes
    return out_seq


def convert_all(
    dataset_root: Optional[Path] = None,
    out_root: Optional[Path] = None,
    *,
    split: str = "Fully_annotate",
    sequences: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Path:
    dataset_root = Path(dataset_root or RAW_DATASETS["trafficmot"]).expanduser().resolve()
    out_root = Path(out_root or MOT_ROOT / "TrafficMOT").expanduser().resolve()
    splits = SPLITS if split == "all" else (split,)
    wanted = set(sequences) if sequences else None
    for sp in splits:
        for seq in list_sequences(dataset_root, sp):
            if wanted is not None and seq not in wanted:
                continue
            convert_sequence(dataset_root, sp, seq, out_root, force=force)
            print(f"  TrafficMOT {sp}/{seq}")
    return out_root


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="TrafficMOT → MOT layout under mot/TrafficMOT/")
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument(
        "--split",
        choices=("Fully_annotate", "FirstFrame_annotate", "all"),
        default="Fully_annotate",
    )
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    out = convert_all(
        args.dataset_root,
        args.out_root,
        split=args.split,
        sequences=args.sequences,
        force=args.force,
    )
    print(f"Done. MOT root: {out}")


if __name__ == "__main__":
    main()
