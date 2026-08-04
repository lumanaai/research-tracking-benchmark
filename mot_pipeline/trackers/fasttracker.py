"""FastTracker adapter — recyclable reference implementation for other trackers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mot_pipeline.mot_io import load_mot_dets, parse_seqinfo
from mot_pipeline.paths import FASTTRACKER_ROOT, PIPELINE_ROOT
from mot_pipeline.protocols import Tracker
from mot_pipeline.trackers.base import (
    load_tracker_config,
    resolve_tracking_schedule,
    write_mot_tracks,
)

DEFAULT_CFG: Dict[str, Any] = {
    "track_thresh": 0.6,
    "track_buffer": 30,
    "match_thresh": 0.9,
    "min_box_area": 100,
    "reset_velocity_offset_occ": 5,
    "reset_pos_offset_occ": 3,
    "enlarge_bbox_occ": 1.2,
    "dampen_motion_occ": 0.85,
    "active_occ_to_lost_thresh": 15,
    "init_iou_suppress": 0.8,
}


def _ensure_fasttracker_on_path() -> None:
    root = str(FASTTRACKER_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _build_det_array(dets, class_aware: bool):
    if not class_aware:
        if not dets:
            return np.empty((0, 5), dtype=np.float32)
        return np.array([[d[0], d[1], d[2], d[3], d[4]] for d in dets], dtype=np.float32)

    import torch

    if not dets:
        return torch.zeros((0, 7), dtype=torch.float32)
    arr = np.array(
        [[d[0], d[1], d[2], d[3], d[4], 1.0, d[5]] for d in dets], dtype=np.float32
    )
    return torch.from_numpy(arr)


class FastTrackerAdapter(Tracker):
    """Wraps ``yolox.tracker.fasttracker.Fasttracker`` with scale=1 dets."""

    name = "fasttracker"

    def __init__(
        self,
        class_aware: bool = False,
        mot20: bool = False,
        conf_default: float = 1.0,
        drop_vertical: bool = False,
        min_box_area: Optional[float] = None,
        frame_rate: Optional[int] = None,
    ) -> None:
        self.class_aware = class_aware
        self.mot20 = mot20
        self.conf_default = conf_default
        self.drop_vertical = drop_vertical
        self.min_box_area_override = min_box_area
        self.frame_rate_override = frame_rate

    def track_sequence(
        self,
        seq_dir: Path,
        det_path: Path,
        out_path: Path,
        config: Dict[str, Any],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ensure_fasttracker_on_path()
        if self.class_aware:
            from yolox.tracker.fasttracker_cls import Fasttracker
        else:
            from yolox.tracker.fasttracker import Fasttracker

        cfg = dict(DEFAULT_CFG)
        cfg.update(config or {})
        extra = extra or {}

        meta = parse_seqinfo(seq_dir)
        img_w = int(meta.get("imWidth", 0)) or None
        img_h = int(meta.get("imHeight", 0)) or None
        seq_len = int(meta.get("seqLength", 0)) or None

        if not det_path.is_file():
            raise FileNotFoundError(f"No detection file: {det_path}")

        keep = set(extra["keep_classes"]) if extra.get("keep_classes") else None
        drop = set(extra["drop_classes"]) if extra.get("drop_classes") else None
        per_frame = load_mot_dets(
            det_path, conf_default=self.conf_default, keep=keep, drop=drop
        )

        if img_h is None or img_w is None:
            max_x = max((d[2] for dl in per_frame.values() for d in dl), default=1.0)
            max_y = max((d[3] for dl in per_frame.values() for d in dl), default=1.0)
            img_w = img_w or int(np.ceil(max_x))
            img_h = img_h or int(np.ceil(max_y))

        frames = sorted(per_frame)
        if seq_len is None:
            seq_len = frames[-1] if frames else 0

        frame_ids, tracker_fps, _, _ = resolve_tracking_schedule(meta, seq_len, extra)
        frame_rate = (
            self.frame_rate_override
            if self.frame_rate_override is not None
            else max(1, int(round(tracker_fps)))
        )

        tracker = Fasttracker(SimpleNamespace(mot20=self.mot20), cfg, frame_rate=frame_rate)
        min_box_area = (
            self.min_box_area_override
            if self.min_box_area_override is not None
            else cfg.get("min_box_area", 100)
        )
        img_info = (img_h, img_w)
        img_size = (img_h, img_w)

        results: List[Tuple[int, list, list, list]] = []
        for frame_id in frame_ids:
            det_array = _build_det_array(per_frame.get(frame_id, []), self.class_aware)
            online_targets = tracker.update(det_array, img_info, img_size)
            tlwhs, ids, scores = [], [], []
            for t in online_targets:
                tlwh = t.tlwh
                if tlwh[2] * tlwh[3] <= min_box_area:
                    continue
                if self.drop_vertical and tlwh[2] / max(tlwh[3], 1e-6) > 1.6:
                    continue
                tlwhs.append(tlwh)
                ids.append(t.track_id)
                scores.append(t.score)
            results.append((frame_id, tlwhs, ids, scores))

        write_mot_tracks(out_path, results)


def default_config_path() -> Path:
    return PIPELINE_ROOT / "configs" / "trackers" / "fasttracker" / "fasttracker_bench.json"


def load_fasttracker_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_tracker_config(path or default_config_path(), DEFAULT_CFG)
