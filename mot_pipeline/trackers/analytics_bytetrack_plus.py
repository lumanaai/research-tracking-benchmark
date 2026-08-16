"""Analytics ByteTrack Plus — same wrapper path as ``analytics_bytetrack``, plus engine.

Uses the vendored ``ByteTrackerPlus/byte_tracker_plus.py`` engine (crossover
protection, parked-vehicle hold, fast-exit boundary) while keeping the live
``Bytetrack`` wrapper, class maps, and confidence logic from the analytics clone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from mot_pipeline.paths import BYTETRACKER_PLUS_ROOT, PIPELINE_ROOT
from mot_pipeline.trackers.analytics_bytetrack import (
    DEFAULT_CFG as _AB_DEFAULTS,
    AnalyticsByteTrackAdapter,
)
from mot_pipeline.trackers.analytics_shim import import_tracker_factory
from mot_pipeline.trackers.base import load_tracker_config

# Benchmark defaults = analytics benchmark.json + plus-engine knobs.
DEFAULT_CFG: Dict[str, Any] = {
    **_AB_DEFAULTS,
    "velocity_window_size": 15,
    "velocity_cost_alpha": 0.0,
    "velocity_noise_th": 1.5,
    "fast_exit_th": 4.0,
    "parked_max_lost_multiplier": 10.0,
}


def _load_plus_engine():
    """Import ``BYTETracker`` from ByteTrackerPlus (after analytics is on sys.path)."""
    engine_path = BYTETRACKER_PLUS_ROOT / "byte_tracker_plus.py"
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"ByteTrackerPlus engine not found: {engine_path}. "
            "Expected ByteTrackerPlus/ at the project root."
        )
    # Ensure analytics shim + tracking package are importable first.
    import_tracker_factory()
    mod_name = "mot_pipeline_byte_tracker_plus"
    if mod_name in sys.modules:
        return sys.modules[mod_name].BYTETracker
    spec = importlib.util.spec_from_file_location(mod_name, engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ByteTrackerPlus engine from {engine_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.BYTETracker


def import_bytetrack_plus_factory():
    """Factory whose ``create`` builds a ``Bytetrack`` wrapper around the plus engine."""
    import_tracker_factory()
    from tracking.byte_track.bytetrack import Bytetrack

    engine_cls = _load_plus_engine()

    class BytetrackPlus(Bytetrack):
        name = "bytetrack_plus"
        engine_type = engine_cls

    class _Factory:
        def create(self, tracker_name: str, msg: dict, fps: int, context):
            tracker = BytetrackPlus(msg, fps, context)
            vehicle_cls = getattr(context.class_handler, "vehicle_value", -1)
            if vehicle_cls is not None and int(vehicle_cls) >= 0:
                tracker.engine.set_vehicle_cls(int(vehicle_cls))
            return tracker

    return _Factory


class AnalyticsByteTrackPlusAdapter(AnalyticsByteTrackAdapter):
    """Same frame loop as analytics ByteTrack; swaps in the plus association engine."""

    name = "analytics_bytetrack_plus"

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
        # Parent reads DEFAULT_CFG from the analytics module; merge plus knobs first.
        merged = dict(DEFAULT_CFG)
        merged.update(config or {})
        return super().track_sequence(
            seq_dir, det_path, out_path, merged, extra=extra
        )

    def _create_tracker(self, tracker_factory, cfg, frame_rate, context):
        # Ignore the stock factory; always build the plus wrapper.
        return import_bytetrack_plus_factory()().create(
            str(cfg.get("name", "bytetrack_plus")), cfg, frame_rate, context
        )


def default_config_path() -> Path:
    return (
        PIPELINE_ROOT
        / "configs"
        / "trackers"
        / "analytics_bytetrack_plus"
        / "benchmark.json"
    )


def load_analytics_bytetrack_plus_config(
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    return load_tracker_config(path or default_config_path(), DEFAULT_CFG)
