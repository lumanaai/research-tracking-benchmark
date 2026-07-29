"""UA-DETRAC benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from mot_pipeline.benchmarks.base import MotRootBenchmark
from mot_pipeline.converters.prepare_ua_detrac import convert_all


class UADetracBenchmark(MotRootBenchmark):
    name = "ua_detrac"
    mot_folder = "UA-DETRAC"
    raw_key = "ua_detrac"

    def default_split(self) -> str:
        return "train"

    def ensure_mot(
        self,
        split: str = "train",
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
            raise FileNotFoundError(f"UA-DETRAC MOT split missing after convert: {root}")
        return root
