"""Analytics ByteTrack Plus (Aug19) — independent engine + Aug19 tuning.

Uses vendored ``ByteTrackerPlusAug19/`` (hookified base + CIoU matching + Plus
subclass) while keeping the live analytics ``Bytetrack`` wrapper path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from mot_pipeline.paths import BYTETRACKER_PLUS_AUG19_ROOT, PIPELINE_ROOT
from mot_pipeline.trackers.analytics_bytetrack import (
    DEFAULT_CFG as _AB_DEFAULTS,
    AnalyticsByteTrackAdapter,
)
from mot_pipeline.trackers.analytics_shim import import_tracker_factory
from mot_pipeline.trackers.base import load_tracker_config

# Aug19 image defaults (cost / match / crossover / lifecycle).
DEFAULT_CFG: Dict[str, Any] = {
    **_AB_DEFAULTS,
    "name": "bytetrack_plus_aug19",
    "track_thresh": 0.30,
    "first_track_compensation": 0.05,
    "baseline_track_compensation": 0.05,
    "use_per_class_thresh": True,
    "detector_unique": False,
    "untrack_objects": [],
    "n_init": 3,
    "track_buffer_seconds": 3.0,
    "match_thresh": 0.95,
    "match_thresh_2": 0.85,
    "match_thresh_3": 0.60,
    "use_ciou": True,
    "norm_dist_th": 0.05,
    "norm_speed_th": 0.0004,
    "xover_grace_frames": 8,
    "xover_mahab_th": 9.21,
    "parked_max_lost_multiplier": 2.0,
    "velocity_window_size": 5,
    "velocity_cost_alpha": 0.0,
    "velocity_noise_th": 1.5,
    "fast_exit_th": 4.0,
}


def _load_aug19_engine():
    """Import ``BYTETrackerPlus`` from ByteTrackerPlusAug19 (after analytics shim)."""
    engine_path = BYTETRACKER_PLUS_AUG19_ROOT / "byte_tracker_plus.py"
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"ByteTrackerPlusAug19 engine not found: {engine_path}. "
            "Expected ByteTrackerPlusAug19/ at the project root."
        )
    import_tracker_factory()
    # Project root must be on path so ``ByteTrackerPlusAug19`` imports as a package.
    project_root = str(BYTETRACKER_PLUS_AUG19_ROOT.resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ByteTrackerPlusAug19.byte_tracker_plus import BYTETrackerPlus

    return BYTETrackerPlus


def import_bytetrack_plus_aug19_factory():
    """Factory whose ``create`` builds a ``Bytetrack`` wrapper around the Aug19 engine."""
    import_tracker_factory()
    from tracking.byte_track.bytetrack import Bytetrack

    engine_cls = _load_aug19_engine()

    class BytetrackPlusAug19(Bytetrack):
        name = "bytetrack_plus_aug19"
        engine_type = engine_cls

    class _Factory:
        def create(self, tracker_name: str, msg: dict, fps: int, context):
            tracker = BytetrackPlusAug19(msg, fps, context)
            vehicle_cls = getattr(context.class_handler, "vehicle_value", -1)
            if vehicle_cls is not None and int(vehicle_cls) >= 0:
                tracker.engine.set_vehicle_cls(int(vehicle_cls))
            return tracker

    return _Factory


class AnalyticsByteTrackPlusAug19Adapter(AnalyticsByteTrackAdapter):
    """Same frame loop as analytics ByteTrack; swaps in the Aug19 Plus engine."""

    name = "analytics_bytetrack_plus_aug19"

    def __init__(
        self,
        class_space: str = "expert_eff",
        conf_default: float = 1.0,
        frame_rate: Optional[int] = None,
    ) -> None:
        super().__init__(
            class_space=class_space,
            conf_default=conf_default,
            frame_rate=frame_rate,
        )

    def track_sequence(
        self,
        seq_dir: Path,
        det_path: Path,
        out_path: Path,
        config: Dict[str, Any],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(DEFAULT_CFG)
        merged.update(config or {})
        return super().track_sequence(
            seq_dir, det_path, out_path, merged, extra=extra
        )

    def _create_tracker(self, tracker_factory, cfg, frame_rate, context):
        return import_bytetrack_plus_aug19_factory()().create(
            str(cfg.get("name", "bytetrack_plus_aug19")), cfg, frame_rate, context
        )


def default_config_path() -> Path:
    return (
        PIPELINE_ROOT
        / "configs"
        / "trackers"
        / "analytics_bytetrack_plus_aug19"
        / "aug19.json"
    )


def load_analytics_bytetrack_plus_aug19_config(
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    return load_tracker_config(path or default_config_path(), DEFAULT_CFG)
