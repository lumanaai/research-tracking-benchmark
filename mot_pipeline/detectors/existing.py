"""Import pre-existing MOT detections from each sequence folder into the shared cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Set

from mot_pipeline.class_maps import policy_for_benchmark, policy_for_coco
from mot_pipeline.mot_io import filter_mot_file, write_json
from mot_pipeline.protocols import Detector


class ExistingDetsDetector(Detector):
    """Reuse detections already sitting under each MOT sequence.

    Looks for ``<seq>/<det_name>`` (default ``det/det.txt``), applies the
    vehicle class filter, and writes into the shared detections cache so trackers
    never mutate the source files.
    """

    name = "existing"

    def __init__(
        self,
        benchmark: str = "fasttracker_bench",
        exclude_motorcycles: bool = False,
        det_name: str = "det/det.txt",
        class_space: str = "auto",
    ) -> None:
        self.benchmark = benchmark
        self.exclude_motorcycles = exclude_motorcycles
        self.det_name = det_name
        # auto: COCO filter when importing YOLO-like files; else benchmark map.
        # Explicit: "coco" | "benchmark" | "none"
        self.class_space = class_space

    def detector_id(self, **kwargs: Any) -> str:
        stem = Path(self.det_name).stem
        moto = "nomoto" if self.exclude_motorcycles else "vehicles"
        return f"existing_{stem}_{moto}"

    def _policies(self):
        if self.class_space == "none":
            return None, set()
        if self.class_space == "coco":
            return policy_for_coco(self.exclude_motorcycles)
        if self.class_space == "benchmark":
            return policy_for_benchmark(self.benchmark, self.exclude_motorcycles)
        # auto: FastTracker-Bench YOLO dumps are COCO; CityFlow baseline has no
        # useful class column; UA-DETRAC / TrafficMOT usually won't have seq dets.
        if self.benchmark == "fasttracker_bench":
            return policy_for_coco(self.exclude_motorcycles)
        if self.benchmark == "cityflow":
            return None, set()
        return policy_for_benchmark(self.benchmark, self.exclude_motorcycles)

    def run(
        self,
        seq_dirs: Sequence[Path],
        out_root: Path,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> Path:
        keep, drop = self._policies()
        keep_set: Optional[Set[int]] = set(keep) if keep is not None else None
        drop_set: Set[int] = set(drop)
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        write_json(
            out_root / "meta.json",
            {
                "detector": self.name,
                "detector_id": self.detector_id(),
                "benchmark": self.benchmark,
                "det_name": self.det_name,
                "class_space": self.class_space,
                "exclude_motorcycles": self.exclude_motorcycles,
                "keep": sorted(keep_set) if keep_set is not None else None,
                "drop": sorted(drop_set),
            },
        )

        missing = []
        for seq_dir in seq_dirs:
            src = seq_dir / self.det_name
            if not src.is_file():
                missing.append(str(src))
                continue
            dst = out_root / seq_dir.name / "det.txt"
            if dst.is_file() and not force:
                print(f"  [existing] skip cached {seq_dir.name}")
                continue
            n = filter_mot_file(
                src,
                dst,
                keep=keep_set,
                drop=drop_set,
            )
            print(f"  [existing] {seq_dir.name}: {n} dets <- {src}")

        if missing:
            raise FileNotFoundError(
                "Missing sequence detection files "
                f"({self.det_name}):\n  " + "\n  ".join(missing) +
                "\nRun `--detector yolov8` (or gt) to produce them, or pass "
                "--det-name for a different relative path."
            )
        return out_root
