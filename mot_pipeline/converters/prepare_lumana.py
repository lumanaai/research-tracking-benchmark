"""Copy LumanaBenchmark MP4s onto the SSD and extract MOT ``img1/`` frames.

Videos are matched by stem to ``gt_annotations_manually_validated/<seq>/``.
Extras without GT are skipped. Frames land at::

    <dataset_root>/gt_annotations_manually_validated/<seq>/img1/000001.jpg

Optional raw copies (default)::

    <dataset_root>/raw_videos/<seq>.mp4

Because ``mot/LumanaBenchmark/train`` symlinks at that annotations tree, YOLO
and the rest of the pipeline see the frames automatically.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import cv2

from mot_pipeline.mot_io import parse_seqinfo, write_seqinfo
from mot_pipeline.paths import RAW_DATASETS

DEFAULT_VIDEO_DIR = Path(
    "/mnt/nas/data/tracking_dataset/vehicle_tracking/raw_videos_1"
)
ANNOTATIONS = "gt_annotations_manually_validated"


def _seq_dirs(ann_root: Path) -> List[Path]:
    return sorted(
        p
        for p in ann_root.iterdir()
        if p.is_dir() and p.name != "seqmaps" and (p / "gt" / "gt.txt").is_file()
    )


def _max_gt_frame(seq_dir: Path) -> int:
    gt = seq_dir / "gt" / "gt.txt"
    if not gt.is_file():
        return 0
    mx = 0
    for line in gt.read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if parts and parts[0]:
            mx = max(mx, int(float(parts[0])))
    return mx


def _update_seqinfo(seq_dir: Path, n_frames: int, width: int, height: int) -> None:
    """Write seqinfo; never shrink seqLength below annotated GT frame indices.

    OpenCV often yields N-1 frames vs CAP_PROP_FRAME_COUNT / the labeled length.
    TrackEval rejects GT timesteps > seqLength, so keep max(extracted, prior, GT).
    """
    meta = parse_seqinfo(seq_dir)
    frame_rate = float(meta.get("frameRate") or 15.0)
    prior = int(float(meta.get("seqLength") or 0))
    seq_length = max(n_frames, prior, _max_gt_frame(seq_dir))
    write_seqinfo(
        seq_dir / "seqinfo.ini",
        name=seq_dir.name,
        frame_rate=frame_rate,
        seq_length=seq_length,
        im_width=width if width > 0 else int(float(meta.get("imWidth") or 0)),
        im_height=height if height > 0 else int(float(meta.get("imHeight") or 0)),
        im_ext=".jpg",
        im_dir="img1",
    )


def extract_sequence(
    video: Path,
    seq_dir: Path,
    *,
    force: bool = False,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 95,
) -> int:
    img1 = seq_dir / "img1"
    if img1.is_dir() and not force and max_frames is None:
        n_existing = sum(1 for _ in img1.glob("*.jpg"))
        if n_existing > 0:
            return n_existing

    if img1.exists():
        if img1.is_symlink():
            img1.unlink()
        elif img1.is_dir():
            shutil.rmtree(img1)
        else:
            img1.unlink()
    img1.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(str(img1 / f"{n:06d}.jpg"), frame, params)
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    if n == 0:
        raise RuntimeError(f"No frames extracted from {video}")
    if width <= 0 or height <= 0:
        sample = cv2.imread(str(img1 / "000001.jpg"))
        if sample is None:
            raise RuntimeError(f"Cannot read first frame under {img1}")
        height, width = sample.shape[:2]

    _update_seqinfo(seq_dir, n, width, height)
    return n


def prepare(
    dataset_root: Path,
    video_dir: Path,
    *,
    copy_videos: bool = True,
    force: bool = False,
    sequences: Optional[Sequence[str]] = None,
    max_frames: Optional[int] = None,
) -> None:
    ann_root = dataset_root / ANNOTATIONS
    if not ann_root.is_dir():
        raise FileNotFoundError(f"Missing annotations tree: {ann_root}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing video dir: {video_dir}")

    videos = {p.stem: p for p in sorted(video_dir.glob("*.mp4"))}
    seqs = _seq_dirs(ann_root)
    if sequences:
        want = set(sequences)
        seqs = [s for s in seqs if s.name in want]

    raw_out = dataset_root / "raw_videos"
    if copy_videos:
        raw_out.mkdir(parents=True, exist_ok=True)

    matched = 0
    skipped_no_video = 0
    for seq_dir in seqs:
        vid = videos.get(seq_dir.name)
        if vid is None:
            print(f"  SKIP {seq_dir.name}: no matching .mp4 in {video_dir}")
            skipped_no_video += 1
            continue
        if copy_videos:
            dest = raw_out / f"{seq_dir.name}.mp4"
            if force or not dest.is_file():
                shutil.copy2(vid, dest)
            src = dest
        else:
            src = vid
        n = extract_sequence(src, seq_dir, force=force, max_frames=max_frames)
        print(f"  OK {seq_dir.name}: {n} frames → {seq_dir / 'img1'}")
        matched += 1

    extra = sorted(set(videos) - {s.name for s in _seq_dirs(ann_root)})
    print(
        f"Done: extracted {matched} sequences"
        + (f"; skipped {skipped_no_video} without video" if skipped_no_video else "")
        + (f"; ignored {len(extra)} videos without GT" if extra else "")
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=RAW_DATASETS["lumana_benchmark"],
        help="LumanaBenchmark root on the SSD.",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory of <seq>.mp4 files (NAS staging).",
    )
    p.add_argument(
        "--no-copy-videos",
        action="store_true",
        help="Extract from NAS paths in place; do not copy MP4s to dataset_root/raw_videos/.",
    )
    p.add_argument("--force", action="store_true", help="Re-copy / re-extract even if img1/ exists.")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    prepare(
        args.dataset_root,
        args.video_dir,
        copy_videos=not args.no_copy_videos,
        force=args.force,
        sequences=args.sequences,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
