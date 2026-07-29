"""Shared helpers for benchmark adapters."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from mot_pipeline.mot_io import discover_mot_sequences
from mot_pipeline.paths import MOT_ROOT, RAW_DATASETS
from mot_pipeline.protocols import Benchmark


class MotRootBenchmark(Benchmark):
    """Benchmark whose sequences live under ``MOT_ROOT/<folder>/<split>/``."""

    mot_folder: str
    raw_key: str

    def __init__(
        self,
        raw_root: Optional[Path] = None,
        mot_root: Optional[Path] = None,
    ) -> None:
        self.raw_root = Path(raw_root) if raw_root else RAW_DATASETS[self.raw_key]
        self.mot_root = Path(mot_root) if mot_root else MOT_ROOT / self.mot_folder

    def sequences_root(self, split: str) -> Path:
        return self.mot_root / split

    def sequence_dirs(
        self, split: str, sequences: Optional[Sequence[str]] = None
    ) -> List[Path]:
        return discover_mot_sequences(self.sequences_root(split), sequences)
