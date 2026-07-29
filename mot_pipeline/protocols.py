"""Adapter protocols for benchmarks, detectors, and trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class RunSpec:
    """Frozen description of one experiment run."""

    run_id: str
    benchmark: str
    split: str
    tracker: str
    detector: str
    detector_id: str
    tracker_config_path: Optional[str]
    tracker_config: Dict[str, Any] = field(default_factory=dict)
    sequences: Optional[List[str]] = None
    exclude_motorcycles: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Benchmark(ABC):
    """Normalized MOTChallenge view of a dataset."""

    name: str

    @abstractmethod
    def ensure_mot(
        self,
        split: str,
        force: bool = False,
        sequences: Optional[Sequence[str]] = None,
    ) -> Path:
        """Ensure MOT layout exists; return the sequences root for ``split``."""

    @abstractmethod
    def sequence_dirs(
        self, split: str, sequences: Optional[Sequence[str]] = None
    ) -> List[Path]:
        """List sequence folders under the MOT root for ``split``."""

    def default_split(self) -> str:
        return "train"


class Detector(ABC):
    """Writes MOT ``det.txt`` files under the shared detections cache."""

    name: str

    @abstractmethod
    def detector_id(self, **kwargs: Any) -> str:
        """Stable cache key, e.g. ``yolov8s_imgsz1280_conf0.25``."""

    @abstractmethod
    def run(
        self,
        seq_dirs: Sequence[Path],
        out_root: Path,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Detect on ``seq_dirs``; write ``out_root/<seq>/det.txt``. Return out_root."""


class Tracker(ABC):
    """Association-only tracker: dets in, MOT tracks out."""

    name: str

    @abstractmethod
    def track_sequence(
        self,
        seq_dir: Path,
        det_path: Path,
        out_path: Path,
        config: Dict[str, Any],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track one sequence; write MOT track file to ``out_path``."""
