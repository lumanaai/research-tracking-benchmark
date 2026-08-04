"""Bridge for importing the in-house analytics tracker without touching its source.

``analytics/analyzer_manager/app/tracking/`` is self-contained apart from three
imports of the analyzer's ``general`` package. Importing the real one is not an
option here: ``general.analyzer_general`` loads a deployment JSON at import time
and opens a log directory under ``/usr/src/app/logs/``. So we register
API-compatible stand-ins in ``sys.modules`` first, then import the tracker
package straight out of the clone.
"""

from __future__ import annotations

import csv
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mot_pipeline.paths import ANALYTICS_APP_ROOT, ANALYTICS_CLASSES_CSV

DEFAULT_DETECTOR_IM_SIZE = [384, 640]


class Event:
    """Handler list supporting ``+=`` — mirrors ``general.core.Event``."""

    def __init__(self) -> None:
        self._eventhandlers: List[Any] = []

    def __iadd__(self, handler):
        self._eventhandlers.append(handler)
        return self

    def __isub__(self, handler):
        self._eventhandlers.remove(handler)
        return self

    def __call__(self, *args, **kwargs):
        for handler in self._eventhandlers:
            handler(*args, **kwargs)


class BaseConfig(dict):
    """Attribute-aliased dict — mirrors ``general.core.BaseConfig``.

    Class-level annotations on subclasses act as defaults for keys the incoming
    dict does not carry.
    """

    def __init__(self, args_dict: Optional[dict] = None) -> None:
        super().__init__()
        self.__dict__ = self
        if args_dict is not None:
            for key in args_dict.keys():
                self[key] = args_dict[key]


@dataclass
class AnalyticImage:
    frame: Any
    timestamp: int
    frame_number: int
    processed: Any = None


@dataclass
class ImagePredictions:
    """Stand-in for ``detection.detector.ImagePredictions``."""

    data: Any
    timestamp: Any = 0


@dataclass
class DetectionResults:
    """Stand-in for ``detection.detector.DetectionResults``."""

    predictions: Any
    names: Any = None
    preds: Any = None


class MiniClassHandler:
    """The slice of ``general.core.ClassHandler`` the tracker actually reads.

    Parses the same ``32cls.csv`` the analyzer feeds its detector, so per-class
    thresholds and the subclass -> object id mapping match production exactly.
    A threshold of ``-1`` means "fall back to the tracker's own default".
    """

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        path = Path(csv_path or ANALYTICS_CLASSES_CSV)
        if not path.is_file():
            raise FileNotFoundError(
                f"Analytics class metadata not found: {path}. "
                "Is the analytics repo cloned at the project root?"
            )
        with path.open("rt") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty class metadata file: {path}")

        n_classes = max(int(r["index"]) for r in rows) + 1
        self.n_classes = n_classes
        self.classes: List[str] = [""] * n_classes
        self.object_names: List[str] = [""] * n_classes
        self.subclass_to_object = np.full(n_classes, -1, dtype=int)
        self.confidence_thresholds: Dict[str, np.ndarray] = {
            "day": np.full(n_classes, -1.0, dtype=float),
            "night": np.full(n_classes, -1.0, dtype=float),
        }
        object_ids: Dict[str, int] = {}
        for row in rows:
            idx = int(row["index"])
            self.classes[idx] = row["class"]
            self.object_names[idx] = row["object"]
            self.subclass_to_object[idx] = int(row["object_id"])
            self.confidence_thresholds["day"][idx] = float(row.get("day_threshold", -1))
            self.confidence_thresholds["night"][idx] = float(row.get("night_threshold", -1))
            object_ids.setdefault(row["object"], int(row["object_id"]))

        self.object_ids = object_ids
        self.person_value = object_ids.get("person", -1)
        self.vehicle_value = object_ids.get("vehicle", -1)

    def object_of(self, subclass: int) -> int:
        if 0 <= subclass < self.n_classes:
            return int(self.subclass_to_object[subclass])
        return -1


class TrackerContext:
    """Stand-in for the ``Analyzer`` instance the tracker receives as ``context``.

    ``Bytetrack`` reads exactly three things off it: the detector resolution
    (``[H, W]``, used for out-of-bounds track pruning), the class handler's
    confidence threshold vectors, and a night-mode event it subscribes to.
    """

    def __init__(
        self,
        detector_resolution: Sequence[int],
        class_handler: Optional[MiniClassHandler] = None,
        night_mode: bool = False,
    ) -> None:
        self.detector_resolution = [int(v) for v in detector_resolution]
        self.class_handler = class_handler or MiniClassHandler()
        self.on_night_mode_changed = Event()
        self.is_night_mode = bool(night_mode)

    def set_night_mode(self, night_mode: bool) -> None:
        """Fire the same event the analyzer fires on a monochrome frame."""
        self.is_night_mode = bool(night_mode)
        self.on_night_mode_changed(self.is_night_mode)


def _install_general_shim() -> None:
    general = sys.modules.get("general")
    if general is not None and getattr(general, "__mot_pipeline_shim__", False):
        return

    general = types.ModuleType("general")
    general.__mot_pipeline_shim__ = True
    general.__path__ = []  # mark as package so submodule imports resolve

    core = types.ModuleType("general.core")
    core.BaseConfig = BaseConfig
    core.AnalyticImage = AnalyticImage
    core.Event = Event

    analyzer_general = types.ModuleType("general.analyzer_general")
    analyzer_general.logger = logging.getLogger("analytics_tracker")
    analyzer_general.DEFAULT_DETECTOR_IM_SIZE = list(DEFAULT_DETECTOR_IM_SIZE)

    general.core = core
    general.analyzer_general = analyzer_general
    sys.modules["general"] = general
    sys.modules["general.core"] = core
    sys.modules["general.analyzer_general"] = analyzer_general


def import_tracker_factory():
    """Return ``tracking.tracker.TrackerFactory`` from the analytics clone."""
    _install_general_shim()

    root = Path(ANALYTICS_APP_ROOT).resolve()
    if not (root / "tracking" / "tracker.py").is_file():
        raise FileNotFoundError(
            f"Analytics tracker not found under {root}. "
            "Expected a clone of the analytics repo at the project root."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tracking import tracker as tracker_mod

    # ``tracking`` is a namespace package; make sure it resolved to the clone
    # and not to some other directory that happens to share the name.
    resolved = Path(tracker_mod.__file__).resolve()
    if root not in resolved.parents:
        raise ImportError(
            f"'tracking.tracker' resolved to {resolved}, expected a file under {root}. "
            "Another 'tracking' package is shadowing the analytics one on sys.path."
        )
    return tracker_mod.TrackerFactory
