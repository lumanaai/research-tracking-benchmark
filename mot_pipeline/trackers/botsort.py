"""BoT-SORT adapter — motion-only (no ReID), detection-fed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mot_pipeline.mot_io import frame_index, list_frames, load_mot_dets, parse_seqinfo
from mot_pipeline.paths import PIPELINE_ROOT
from mot_pipeline.protocols import Tracker
from mot_pipeline.trackers.base import (
    dets_to_xyxy_score,
    load_tracker_config,
    resolve_tracking_schedule,
    write_mot_tracks,
)
from mot_pipeline.trackers.botsort_shim import ensure_botsort_importable

# Paper BoT-SORT (no ReID) defaults from tools/track.py; CMC off by default so
# batch sweeps stay image-free like OC-SORT / HybridSORT. Set cmc_method to
# sparseOptFlow (or orb/ecc) to enable camera-motion compensation.
DEFAULT_CFG: Dict[str, Any] = {
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.7,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "cmc_method": "none",
    "mot20": False,
    "min_box_area": 10,
}


def _build_args(cfg: Dict[str, Any], seq_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        track_high_thresh=float(cfg["track_high_thresh"]),
        track_low_thresh=float(cfg["track_low_thresh"]),
        new_track_thresh=float(cfg["new_track_thresh"]),
        track_buffer=int(cfg["track_buffer"]),
        match_thresh=float(cfg["match_thresh"]),
        proximity_thresh=float(cfg["proximity_thresh"]),
        appearance_thresh=float(cfg["appearance_thresh"]),
        with_reid=False,
        cmc_method=str(cfg.get("cmc_method", "none")),
        name=seq_name,
        ablation=False,
        mot20=bool(cfg.get("mot20", False)),
        # Unused when with_reid=False; kept for BoTSORT.__init__ attribute access.
        fast_reid_config="",
        fast_reid_weights="",
        device="cpu",
    )


def _frame_path_map(seq_dir: Path) -> Dict[int, Path]:
    img_dir = seq_dir / "img1"
    if not img_dir.is_dir():
        return {}
    return {frame_index(p): p for p in list_frames(img_dir) if frame_index(p) > 0}


def _load_frame(path: Optional[Path], img_h: int, img_w: int) -> np.ndarray:
    if path is not None and path.is_file():
        img = cv2.imread(str(path))
        if img is not None:
            return img
    return np.zeros((img_h, img_w, 3), dtype=np.uint8)


class BoTSORTAdapter(Tracker):
    """Wraps ``tracker.bot_sort.BoTSORT`` with ReID forced off."""

    name = "botsort"

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
        ensure_botsort_importable()
        from tracker.bot_sort import BoTSORT

        cfg = dict(DEFAULT_CFG)
        cfg.update(config or {})
        extra = extra or {}

        meta = parse_seqinfo(seq_dir)
        img_w = int(meta.get("imWidth", 0)) or None
        img_h = int(meta.get("imHeight", 0)) or None
        seq_len = int(meta.get("seqLength", 0)) or None
        seq_name = meta.get("name") or seq_dir.name

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
        frame_rate = max(1, int(round(tracker_fps)))

        cmc = str(cfg.get("cmc_method", "none")).lower()
        need_images = cmc not in ("none", "")
        path_by_frame = _frame_path_map(seq_dir) if need_images else {}
        if need_images and not path_by_frame:
            raise FileNotFoundError(
                f"BoT-SORT cmc_method={cmc!r} needs frames under {seq_dir / 'img1'}"
            )

        args = _build_args(cfg, seq_name)
        tracker = BoTSORT(args, frame_rate=frame_rate)
        min_box_area = float(cfg.get("min_box_area", 10))

        results: List[Tuple[int, list, list, list]] = []
        for frame_id in frame_ids:
            det_array = dets_to_xyxy_score(per_frame.get(frame_id, []))
            img = (
                _load_frame(path_by_frame.get(frame_id), img_h, img_w)
                if need_images
                else None
            )
            online = tracker.update(det_array, img)
            tlwhs, ids, scores = [], [], []
            for t in online:
                tlwh = t.tlwh
                if tlwh[2] * tlwh[3] <= min_box_area:
                    continue
                tlwhs.append(tlwh.tolist() if hasattr(tlwh, "tolist") else list(tlwh))
                ids.append(int(t.track_id))
                scores.append(float(t.score))
            results.append((frame_id, tlwhs, ids, scores))

        write_mot_tracks(out_path, results)


def default_config_path() -> Path:
    return PIPELINE_ROOT / "configs" / "trackers" / "botsort" / "default.json"


def load_botsort_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_tracker_config(path or default_config_path(), DEFAULT_CFG)
