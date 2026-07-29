"""YOLOX detector stub (same MOT det.txt contract as yolov8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from mot_pipeline.protocols import Detector


class YOLOXDetector(Detector):
    """Placeholder until YOLOX weights / export path is wired."""

    name = "yolox"

    def detector_id(self, **kwargs: Any) -> str:
        return "yolox_unimplemented"

    def run(
        self,
        seq_dirs: Sequence[Path],
        out_root: Path,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError(
            "YOLOX detector is not implemented yet. Provide MOT det.txt via --detector gt "
            "or --detector yolov8, or implement mot_pipeline.detectors.yolox.YOLOXDetector.run "
            "to write the same MOT format: frame,-1,x,y,w,h,conf,class,-1"
        )
