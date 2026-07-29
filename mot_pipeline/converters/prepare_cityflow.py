"""Convert CityFlow cameras into flat MOTChallenge folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2

from mot_pipeline.mot_io import write_seqinfo
from mot_pipeline.paths import MOT_ROOT, RAW_DATASETS

SPLITS = ("train", "validation")


def discover_cameras(
    dataset_root: Path, split: str
) -> List[Tuple[str, Path]]:
    """Return [(flat_name, camera_dir), ...] for cameras with gt/gt.txt."""
    split_root = dataset_root / split
    if not split_root.is_dir():
        return []
    out: List[Tuple[str, Path]] = []
    for scenario in sorted(p for p in split_root.iterdir() if p.is_dir()):
        for cam in sorted(p for p in scenario.iterdir() if p.is_dir()):
            if not (cam / "gt" / "gt.txt").is_file():
                continue
            flat = f"{scenario.name}_{cam.name}"
            out.append((flat, cam))
    return out


def convert_sequence(
    cam_dir: Path,
    flat_name: str,
    out_split_root: Path,
    *,
    force: bool = False,
    max_frames: Optional[int] = None,
) -> Path:
    out_seq = out_split_root / flat_name
    gt_dst = out_seq / "gt" / "gt.txt"
    img1 = out_seq / "img1"
    vdo = cam_dir / "vdo.avi"
    gt_src = cam_dir / "gt" / "gt.txt"
    if not vdo.is_file():
        raise FileNotFoundError(f"Missing video: {vdo}")
    if not gt_src.is_file():
        raise FileNotFoundError(f"Missing GT: {gt_src}")

    if (
        gt_dst.is_file()
        and (out_seq / "seqinfo.ini").is_file()
        and img1.is_dir()
        and not force
        and max_frames is None
    ):
        n_existing = sum(1 for _ in img1.glob("*.jpg"))
        cap_probe = cv2.VideoCapture(str(vdo))
        n_video = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap_probe.release()
        # Re-extract if a previous partial convert left too few frames.
        if n_existing > 0 and (n_video <= 0 or n_existing >= n_video - 1):
            _link_baseline_dets(cam_dir, out_seq)
            return out_seq

    out_seq.mkdir(parents=True, exist_ok=True)
    (out_seq / "gt").mkdir(parents=True, exist_ok=True)
    if max_frames is None:
        shutil.copy2(gt_src, gt_dst)
    else:
        with gt_src.open() as src, gt_dst.open("w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                frame = int(float(line.replace(",", " ").split()[0]))
                if frame <= max_frames:
                    dst.write(line + "\n")

    if img1.exists():
        if img1.is_symlink():
            img1.unlink()
        elif img1.is_dir():
            shutil.rmtree(img1)
        else:
            img1.unlink()
    img1.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(vdo))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {vdo}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(str(img1 / f"{n:06d}.jpg"), frame)
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    if n == 0:
        raise RuntimeError(f"No frames extracted from {vdo}")
    if width <= 0 or height <= 0:
        sample = cv2.imread(str(img1 / "000001.jpg"))
        height, width = sample.shape[:2]

    write_seqinfo(
        out_seq / "seqinfo.ini",
        name=flat_name,
        frame_rate=fps,
        seq_length=n,
        im_width=width,
        im_height=height,
    )
    _link_baseline_dets(cam_dir, out_seq)
    return out_seq


def _link_baseline_dets(cam_dir: Path, out_seq: Path) -> None:
    """Expose CityFlow baseline dets under MOT ``det/`` for ``--detector existing``."""
    src_dir = cam_dir / "det"
    if not src_dir.is_dir():
        return
    # Prefer YOLO3, then any det_*.txt / det.txt.
    candidates = []
    for name in ("det_yolo3.txt", "det_ssd512.txt", "det_mask_rcnn.txt", "det.txt"):
        p = src_dir / name
        if p.is_file():
            candidates.append(p)
    candidates.extend(
        p for p in sorted(src_dir.glob("det*.txt")) if p not in candidates
    )
    if not candidates:
        return
    dest_dir = out_seq / "det"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in candidates:
        dst = dest_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve())
    # Canonical path used by --detector existing (default det/det.txt).
    canonical = dest_dir / "det.txt"
    if not canonical.exists() and not canonical.is_symlink():
        canonical.symlink_to(candidates[0].resolve())


def convert_all(
    dataset_root: Optional[Path] = None,
    out_root: Optional[Path] = None,
    *,
    split: str = "train",
    sequences: Optional[Sequence[str]] = None,
    force: bool = False,
    max_frames: Optional[int] = None,
) -> Path:
    dataset_root = Path(dataset_root or RAW_DATASETS["cityflow"]).expanduser().resolve()
    out_root = Path(out_root or MOT_ROOT / "CityFlow").expanduser().resolve()
    splits = SPLITS if split == "all" else (split,)
    wanted = set(sequences) if sequences else None
    for sp in splits:
        for flat, cam_dir in discover_cameras(dataset_root, sp):
            if wanted is not None and flat not in wanted and cam_dir.name not in wanted:
                continue
            convert_sequence(
                cam_dir,
                flat,
                out_root / sp,
                force=force,
                max_frames=max_frames,
            )
            print(f"  CityFlow {sp}/{flat}")
    return out_root


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="CityFlow → MOT layout under mot/CityFlow/")
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--split", choices=("train", "validation", "all"), default="train")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args(list(argv) if argv is not None else None)
    out = convert_all(
        args.dataset_root,
        args.out_root,
        split=args.split,
        sequences=args.sequences,
        force=args.force,
        max_frames=args.max_frames,
    )
    print(f"Done. MOT root: {out}")


if __name__ == "__main__":
    main()
