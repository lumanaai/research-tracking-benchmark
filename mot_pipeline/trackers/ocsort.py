"""OC-SORT adapter — motion-only, detection-fed (same contract as FastTracker)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mot_pipeline.mot_io import load_mot_dets, parse_seqinfo
from mot_pipeline.paths import OCSORT_ROOT
from mot_pipeline.protocols import Tracker
from mot_pipeline.trackers.base import (
    dets_to_xyxy_score,
    write_mot_tracks,
    xyxy_ids_to_frame_result,
)

DEFAULT_CFG: Dict[str, Any] = {
    "det_thresh": 0.5,
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "asso_func": "iou",
    "inertia": 0.2,
    "use_byte": True,
}


def _ensure_ocsort_on_path() -> None:
    root = str(OCSORT_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class OCSORTAdapter(Tracker):
    """Wraps ``trackers.ocsort_tracker.ocsort.OCSort`` with scale=1 dets."""

    name = "ocsort"

    def __init__(self, conf_default: float = 1.0) -> None:
        self.conf_default = conf_default

    def track_sequence(
        self,
        seq_dir: Path,
        det_path: Path,
        out_path: Path,
        config: Dict[str, Any],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ensure_ocsort_on_path()
        from trackers.ocsort_tracker.ocsort import OCSort

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

        tracker = OCSort(
            det_thresh=float(cfg["det_thresh"]),
            max_age=int(cfg["max_age"]),
            min_hits=int(cfg["min_hits"]),
            iou_threshold=float(cfg["iou_threshold"]),
            delta_t=int(cfg["delta_t"]),
            asso_func=str(cfg["asso_func"]),
            inertia=float(cfg["inertia"]),
            use_byte=bool(cfg["use_byte"]),
        )

        img_info = (img_h, img_w)
        img_size = (img_h, img_w)
        results: List[Tuple[int, list, list, list]] = []
        for frame_id in range(1, seq_len + 1):
            det_array = dets_to_xyxy_score(per_frame.get(frame_id, []))
            online = tracker.update(det_array, img_info, img_size)
            results.append(xyxy_ids_to_frame_result(frame_id, online))

        write_mot_tracks(out_path, results)
