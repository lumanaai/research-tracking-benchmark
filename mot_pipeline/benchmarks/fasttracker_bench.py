"""FastTracker-Benchmark adapter (already MOT-native under train/)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from mot_pipeline.benchmarks.base import MotRootBenchmark
from mot_pipeline.mot_io import discover_mot_sequences
from mot_pipeline.paths import RAW_DATASETS


class FastTrackerBenchBenchmark(MotRootBenchmark):
    name = "fasttracker_bench"
    mot_folder = "FastTracker-Benchmark"
    raw_key = "fasttracker_bench"

    def default_split(self) -> str:
        return "train"

    def ensure_mot(
        self,
        split: str = "train",
        force: bool = False,
        sequences: Optional[Sequence[str]] = None,
    ) -> Path:
        """Symlink the native ``train/`` tree into ``mot/FastTracker-Benchmark/``."""
        del sequences  # native tree already contains all sequences
        if split != "train":
            raise ValueError("FastTracker-Benchmark only has a 'train' split.")
        raw_train = self.raw_root / "train"
        if not raw_train.is_dir():
            raise FileNotFoundError(f"Missing FastTracker-Benchmark train/: {raw_train}")

        dest = self.sequences_root(split)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            if force:
                dest.unlink()
            elif dest.resolve() == raw_train.resolve():
                return dest
            else:
                dest.unlink()
        elif dest.exists():
            if force:
                if dest.is_dir() and not dest.is_symlink():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            else:
                return dest

        dest.symlink_to(raw_train.resolve())
        return dest

    def sequence_dirs(
        self, split: str, sequences: Optional[Sequence[str]] = None
    ) -> List[Path]:
        root = self.ensure_mot(split, sequences=sequences)
        return discover_mot_sequences(root, sequences)
