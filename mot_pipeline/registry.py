"""Name → adapter factories."""

from __future__ import annotations

from mot_pipeline.benchmarks import BENCHMARKS, get_benchmark
from mot_pipeline.detectors import DETECTORS, get_detector
from mot_pipeline.trackers import TRACKERS, get_tracker

__all__ = [
    "BENCHMARKS",
    "DETECTORS",
    "TRACKERS",
    "get_benchmark",
    "get_detector",
    "get_tracker",
]
