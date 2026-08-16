"""In-house analytics ByteTrack adapter — imported live from the analytics clone.

Drives ``tracking.byte_track.bytetrack.Bytetrack.run()`` one frame at a time
rather than poking the ``BYTETracker`` engine directly, so the per-class
confidence logic, the untracked-detection passthrough and the per-batch
``cleanup()`` all execute exactly as they do in the analyzer.

Two things that path needs beyond boxes:

* ``(cls, subclass)`` per detection — ``cls`` gates association (tracks only
  match detections of the same coarse object), ``subclass`` indexes the
  per-class confidence thresholds. Detection class ids are translated into that
  numbering via ``class_maps.to_analytics_subclass``.
* Image shapes. On the motion-only path the frame is never read, only its
  ``.shape``, so shape-carrying stand-ins avoid decoding any pixels. A ReID
  variant would need the real frames here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mot_pipeline.class_maps import to_analytics_subclass
from mot_pipeline.mot_io import load_mot_dets, parse_seqinfo
from mot_pipeline.paths import PIPELINE_ROOT
from mot_pipeline.protocols import Tracker
from mot_pipeline.trackers.analytics_shim import (
    AnalyticImage,
    DetectionResults,
    ImagePredictions,
    MiniClassHandler,
    TrackerContext,
    import_tracker_factory,
)
from mot_pipeline.trackers.base import (
    load_tracker_config,
    resolve_tracking_schedule,
    write_mot_tracks,
)

# Verbatim ``l0_track`` block from analyzer_manager/assets/configAnalytic.json,
# minus the byteSReidTrack-only keys. Overridden by --tracker-config.
DEFAULT_CFG: Dict[str, Any] = {
    "name": "bytetrack",
    "track_thresh": 0.25,
    "untrack_objects": [3, 4, 5, 7],
    "first_track_compensation": 0.1,
    "baseline_track_compensation": 0.2,
    "use_per_class_thresh": True,
    "detector_unique": False,
    "n_init": 3,
    "track_buffer_seconds": 6,
    "match_thresh": 0.96,
    "match_thresh_2": 0.84,
    "match_thresh_3": 0.6,
    "min_box_area": 10,
    "mot20": False,
    "use_timestamp": False,
    "night_mode": False,
}

# Output columns of Bytetrack.run(): x1,y1,x2,y2,id,cls,subclass,conf,det_id,age,conflict
COL_TRACK_ID = 4
COL_CONF = 7


class AnalyticsByteTrackAdapter(Tracker):
    """Wraps the analytics ``Bytetrack`` wrapper with a stub analyzer context."""

    name = "analytics_bytetrack"

    def __init__(
        self,
        class_space: str = "expert_eff",
        conf_default: float = 1.0,
        frame_rate: Optional[int] = None,
    ) -> None:
        self.class_space = class_space
        self.conf_default = conf_default
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
        tracker_factory = import_tracker_factory()

        cfg = dict(DEFAULT_CFG)
        cfg.update(config or {})
        extra = extra or {}
        class_space = extra.get("class_space") or self.class_space

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

        frame_ids, tracker_fps, native_fps, _ = resolve_tracking_schedule(
            meta, seq_len, extra
        )
        frame_rate = (
            self.frame_rate_override
            if self.frame_rate_override is not None
            else max(1, int(round(tracker_fps)))
        )

        class_handler = MiniClassHandler()
        context = TrackerContext((img_h, img_w), class_handler)
        tracker = self._create_tracker(tracker_factory, cfg, frame_rate, context)
        if cfg.get("night_mode"):
            # Same event the analyzer fires when a frame comes back monochrome.
            context.set_night_mode(True)

        min_box_area = float(cfg.get("min_box_area") or 0)
        # The tracker itself is agnostic to units; feeding original pixels and a
        # matching detector_resolution keeps its out-of-bounds pruning correct.
        image_stub = SimpleNamespace(shape=(img_h, img_w, 3))

        results: List[Tuple[int, list, list, list]] = []
        for frame_id in frame_ids:
            data = self._build_predictions(
                per_frame.get(frame_id, []), class_space, class_handler
            )
            # Wall-clock timestamps use the native sequence rate + original frame id.
            timestamp = int(round((frame_id - 1) * 1000.0 / native_fps))
            det_res = DetectionResults(
                predictions=[ImagePredictions(data=data, timestamp=timestamp)]
            )
            images = [
                AnalyticImage(
                    frame=image_stub,
                    timestamp=timestamp,
                    frame_number=frame_id,
                    processed=image_stub,
                )
            ]
            tracked = tracker.run(det_res, images)[0]
            results.append(self._to_frame_result(frame_id, tracked, min_box_area))

        write_mot_tracks(out_path, results)

    def _create_tracker(self, tracker_factory, cfg, frame_rate, context):
        return tracker_factory().create(
            str(cfg.get("name", "bytetrack")), cfg, frame_rate, context
        )

    @staticmethod
    def _build_predictions(
        dets: list, class_space: str, class_handler: MiniClassHandler
    ) -> np.ndarray:
        """Build the analyzer's N x 7 ``[x1,y1,x2,y2,conf,cls,subclass]`` array."""
        rows: List[List[float]] = []
        for x1, y1, x2, y2, score, class_id in dets:
            subclass = to_analytics_subclass(class_id, class_space)
            if subclass is None:
                continue
            cls = class_handler.object_of(subclass)
            if cls < 0:  # 'unknown' in 32cls.csv — the analyzer never tracks these
                continue
            rows.append([x1, y1, x2, y2, score, float(cls), float(subclass)])
        if not rows:
            # Shape matters: Bytetrack.run() indexes columns and reads .shape.
            return np.empty((0, 7), dtype=float)
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _to_frame_result(
        frame_id: int, tracked: Any, min_box_area: float
    ) -> Tuple[int, list, list, list]:
        tlwhs: list = []
        ids: list = []
        scores: list = []
        arr = np.asarray(tracked, dtype=float)
        if arr.size == 0:
            return frame_id, tlwhs, ids, scores
        for row in arr:
            track_id = int(row[COL_TRACK_ID])
            if track_id < 0:  # untracked passthrough detection
                continue
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            w, h = x2 - x1, y2 - y1
            if min_box_area and w * h <= min_box_area:
                continue
            tlwhs.append([x1, y1, w, h])
            ids.append(track_id)
            scores.append(float(row[COL_CONF]))
        return frame_id, tlwhs, ids, scores


def default_config_path() -> Path:
    return PIPELINE_ROOT / "configs" / "trackers" / "analytics_bytetrack" / "benchmark.json"


def load_analytics_bytetrack_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_tracker_config(path or default_config_path(), DEFAULT_CFG)
