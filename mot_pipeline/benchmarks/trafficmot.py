"""TrafficMOT benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from mot_pipeline.benchmarks.base import MotRootBenchmark
from mot_pipeline.converters.prepare_trafficmot import convert_all


class TrafficMOTBenchmark(MotRootBenchmark):
    name = "trafficmot"
    mot_folder = "TrafficMOT"
    raw_key = "trafficmot"

    def default_split(self) -> str:
        return "Fully_annotate"

    def ensure_mot(
        self,
        split: str = "Fully_annotate",
        force: bool = False,
        sequences: Optional[Sequence[str]] = None,
    ) -> Path:
        convert_all(
            self.raw_root,
            self.mot_root,
            split=split,
            sequences=sequences,
            force=force,
        )
        root = self.sequences_root(split)
        if not root.is_dir():
            raise FileNotFoundError(f"TrafficMOT MOT split missing after convert: {root}")
        return root
