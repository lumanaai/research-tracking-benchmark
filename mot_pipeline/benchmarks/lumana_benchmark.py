"""LumanaBenchmark adapter (MOT-native GT under gt_annotations_manually_validated/)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from mot_pipeline.benchmarks.base import MotRootBenchmark
from mot_pipeline.mot_io import discover_mot_sequences


class LumanaBenchmark(MotRootBenchmark):
    name = "lumana_benchmark"
    mot_folder = "LumanaBenchmark"
    raw_key = "lumana_benchmark"

    # Native label tree under the raw dataset root.
    RAW_ANNOTATIONS = "gt_annotations_manually_validated"

    def default_split(self) -> str:
        return "train"

    def ensure_mot(
        self,
        split: str = "train",
        force: bool = False,
        sequences: Optional[Sequence[str]] = None,
    ) -> Path:
        """Symlink annotations into ``mot/LumanaBenchmark/train``."""
        del sequences  # native tree already contains all sequences
        if split != "train":
            raise ValueError("LumanaBenchmark only has a 'train' split.")
        raw_ann = self.raw_root / self.RAW_ANNOTATIONS
        if not raw_ann.is_dir():
            raise FileNotFoundError(
                f"Missing LumanaBenchmark {self.RAW_ANNOTATIONS}/: {raw_ann}"
            )

        dest = self.sequences_root(split)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            if force:
                dest.unlink()
            elif dest.resolve() == raw_ann.resolve():
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

        dest.symlink_to(raw_ann.resolve())
        return dest

    def sequence_dirs(
        self, split: str, sequences: Optional[Sequence[str]] = None
    ) -> List[Path]:
        root = self.ensure_mot(split, sequences=sequences)
        return discover_mot_sequences(root, sequences)
