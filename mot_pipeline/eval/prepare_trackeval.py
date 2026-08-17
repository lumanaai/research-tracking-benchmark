"""Stage filtered GT + tracker outputs into a TrackEval MotChallenge layout."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

from mot_pipeline.class_maps import policy_for_benchmark
from mot_pipeline.mot_io import filter_mot_file, parse_seqinfo, write_seqinfo
from mot_pipeline.trackers.base import resolve_tracking_schedule


def prepare_trackeval_layout(
    *,
    seq_dirs: Sequence[Path],
    tracks_dir: Path,
    eval_root: Path,
    tracker_name: str,
    benchmark: str,
    exclude_motorcycles: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """Build ``eval_root/{gt,trackers/<name>/data}/`` for TrackEval.

    GT boxes are vehicle-filtered and rewritten as class ``1`` (TrackEval's
    'pedestrian' slot) for class-agnostic vehicle evaluation.

    When ``extra`` has ``target_fps`` or ``frame_stride``, GT is restricted to
    the same kept frame ids the tracker saw (skipped frames do not exist).
    """
    keep, drop = policy_for_benchmark(benchmark, exclude_motorcycles)
    keep_set: Optional[Set[int]] = set(keep) if keep is not None else None
    drop_set: Set[int] = set(drop)
    extra = extra or {}
    subsample = extra.get("target_fps") is not None or extra.get("frame_stride") is not None

    gt_fol = eval_root / "gt"
    trk_fol = eval_root / "trackers" / tracker_name / "data"
    if eval_root.exists():
        shutil.rmtree(eval_root)
    gt_fol.mkdir(parents=True, exist_ok=True)
    trk_fol.mkdir(parents=True, exist_ok=True)

    seq_info: Dict[str, int] = {}
    for seq_dir in seq_dirs:
        seq = seq_dir.name
        gt_src = seq_dir / "gt" / "gt.txt"
        if not gt_src.is_file():
            raise FileNotFoundError(f"Missing GT: {gt_src}")
        track_src = Path(tracks_dir) / f"{seq}.txt"
        if not track_src.is_file():
            raise FileNotFoundError(f"Missing tracks: {track_src}")

        gt_dst_dir = gt_fol / seq / "gt"
        gt_dst_dir.mkdir(parents=True, exist_ok=True)

        meta = parse_seqinfo(seq_dir)
        seq_len = int(meta.get("seqLength", 0))
        if seq_len <= 0:
            seq_len = _max_frame(gt_src, track_src)

        keep_frames: Optional[Set[int]] = None
        if subsample:
            frame_ids, _, _, _ = resolve_tracking_schedule(meta, seq_len, extra)
            keep_frames = set(frame_ids)

        filter_mot_file(
            gt_src,
            gt_dst_dir / "gt.txt",
            keep=keep_set,
            drop=drop_set,
            keep_frames=keep_frames,
            rewrite_class=1,
            force_conf=1.0,
        )
        write_seqinfo(
            gt_fol / seq / "seqinfo.ini",
            name=seq,
            frame_rate=float(meta.get("frameRate", 30)),
            seq_length=seq_len,
            im_width=int(float(meta.get("imWidth", 1920))),
            im_height=int(float(meta.get("imHeight", 1080))),
            im_ext=meta.get("imExt", ".jpg"),
        )
        shutil.copy2(track_src, trk_fol / f"{seq}.txt")
        seq_info[seq] = seq_len

    return {
        "eval_root": eval_root,
        "gt_folder": gt_fol,
        "trackers_folder": eval_root / "trackers",
        "tracker_name": tracker_name,
        "seq_info": seq_info,
        "keep": sorted(keep_set) if keep_set is not None else None,
        "drop": sorted(drop_set),
        "target_fps": extra.get("target_fps"),
        "frame_stride": extra.get("frame_stride"),
        "gt_subsampled": subsample,
    }


def _max_frame(*paths: Path) -> int:
    mx = 0
    for path in paths:
        if not path.is_file():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                mx = max(mx, int(float(parts[0])))
    return max(mx, 1)
