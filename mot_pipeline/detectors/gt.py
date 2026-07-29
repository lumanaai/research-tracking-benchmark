"""Oracle detector: filtered GT boxes written as MOT det.txt."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Set

from mot_pipeline.class_maps import policy_for_benchmark
from mot_pipeline.mot_io import filter_mot_file, write_json
from mot_pipeline.protocols import Detector


class GTDetector(Detector):
    name = "gt"

    def __init__(self, benchmark: str = "fasttracker_bench", exclude_motorcycles: bool = False):
        self.benchmark = benchmark
        self.exclude_motorcycles = exclude_motorcycles

    def detector_id(self, **kwargs: Any) -> str:
        tag = "nomoto" if self.exclude_motorcycles else "vehicles"
        return f"gt_{tag}"

    def run(
        self,
        seq_dirs: Sequence[Path],
        out_root: Path,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> Path:
        keep, drop = policy_for_benchmark(self.benchmark, self.exclude_motorcycles)
        keep_set: Optional[Set[int]] = set(keep) if keep is not None else None
        drop_set: Set[int] = set(drop)
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        meta = {
            "detector": self.name,
            "detector_id": self.detector_id(),
            "benchmark": self.benchmark,
            "exclude_motorcycles": self.exclude_motorcycles,
            "keep": sorted(keep_set) if keep_set is not None else None,
            "drop": sorted(drop_set),
        }
        write_json(out_root / "meta.json", meta)

        for seq_dir in seq_dirs:
            gt = seq_dir / "gt" / "gt.txt"
            if not gt.is_file():
                raise FileNotFoundError(f"Missing GT for {seq_dir.name}: {gt}")
            dst = out_root / seq_dir.name / "det.txt"
            if dst.is_file() and not force:
                continue
            n = filter_mot_file(
                gt,
                dst,
                keep=keep_set,
                drop=drop_set,
                force_conf=1.0,
            )
            print(f"  [gt] {seq_dir.name}: {n} dets -> {dst}")
        return out_root
