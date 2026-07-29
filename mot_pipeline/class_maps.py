"""Vehicle / ignore class maps per benchmark and detector.

Default policy: motorcycles/motorbikes are vehicles. Pass
``exclude_motorcycles=True`` to drop them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class ClassPolicy:
    """Which class ids to keep for vehicle-only tracking / eval."""

    keep: Optional[FrozenSet[int]]
    """If set, only these class ids are kept. ``None`` means keep all
    (used when the dataset has no non-vehicle classes, e.g. CityFlow)."""

    drop: FrozenSet[int] = frozenset()
    """Always dropped (applied before keep)."""

    motorcycle_ids: FrozenSet[int] = frozenset()
    """Ids removed when ``exclude_motorcycles`` is True."""

    def resolved(
        self, exclude_motorcycles: bool = False
    ) -> tuple[Optional[FrozenSet[int]], FrozenSet[int]]:
        drop = set(self.drop)
        keep = set(self.keep) if self.keep is not None else None
        if exclude_motorcycles:
            drop |= set(self.motorcycle_ids)
            if keep is not None:
                keep -= set(self.motorcycle_ids)
        return (
            frozenset(keep) if keep is not None else None,
            frozenset(drop),
        )


# FastTracker-Benchmark labels.txt (1-indexed):
# 1 person, 2-6 buses/trucks/car, 7 bike, 8 motorbike, 9 ignore_region,
# 10 tractor, 11 trailor, 12 wheelchair, 13 heavy_equipment, 14 pm, 15 umbrella
FT_BENCH = ClassPolicy(
    keep=frozenset({2, 3, 4, 5, 6, 8, 10, 11, 13, 14}),
    drop=frozenset({1, 7, 9, 12, 15}),
    motorcycle_ids=frozenset({8}),
)

# UA-DETRAC converter writes car=1, bus=2, van=3, others=4 — all vehicles.
UA_DETRAC = ClassPolicy(
    keep=None,
    drop=frozenset(),
    motorcycle_ids=frozenset(),
)

# TrafficMOT: 1 Motor_Bike … 10 Truck; 5 Bike, 6 Pedestrian are non-vehicle.
TRAFFICMOT = ClassPolicy(
    keep=frozenset({1, 2, 3, 4, 7, 8, 9, 10}),
    drop=frozenset({5, 6}),
    motorcycle_ids=frozenset({1}),
)

# CityFlow GT has no class column (always 1); vehicle-only annotations.
CITYFLOW = ClassPolicy(
    keep=None,
    drop=frozenset(),
    motorcycle_ids=frozenset(),
)

# COCO ids used by stock YOLOv8.
COCO_VEHICLE = ClassPolicy(
    keep=frozenset({2, 3, 5, 7}),  # car, motorcycle, bus, truck
    drop=frozenset({0, 1}),  # person, bicycle
    motorcycle_ids=frozenset({3}),
)

# Team yolov8m-expert_eff taxonomy (NOT COCO numbering for bus/truck):
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 train, 6 truck,
# 19 forklift, …
EXPERT_EFF_VEHICLE = ClassPolicy(
    keep=frozenset({2, 3, 4, 6, 19}),  # car, motorcycle, bus, truck, forklift
    drop=frozenset({0, 1}),  # person, bicycle
    motorcycle_ids=frozenset({3}),
)

BENCHMARK_POLICIES = {
    "fasttracker_bench": FT_BENCH,
    "ua_detrac": UA_DETRAC,
    "trafficmot": TRAFFICMOT,
    "cityflow": CITYFLOW,
}


def policy_for_benchmark(
    name: str, exclude_motorcycles: bool = False
) -> tuple[Optional[FrozenSet[int]], FrozenSet[int]]:
    if name not in BENCHMARK_POLICIES:
        raise KeyError(f"Unknown benchmark class map: {name}")
    return BENCHMARK_POLICIES[name].resolved(exclude_motorcycles)


def policy_for_coco(
    exclude_motorcycles: bool = False,
) -> tuple[Optional[FrozenSet[int]], FrozenSet[int]]:
    return COCO_VEHICLE.resolved(exclude_motorcycles)


def policy_for_yolo_weights(
    weights: Optional[object] = None,
    exclude_motorcycles: bool = False,
) -> tuple[Optional[FrozenSet[int]], FrozenSet[int]]:
    """Pick keep/drop by YOLO weight stem (expert_eff ≠ COCO bus/truck ids)."""
    from pathlib import Path

    stem = Path(str(weights or "")).stem.lower()
    if "expert_eff" in stem:
        return EXPERT_EFF_VEHICLE.resolved(exclude_motorcycles)
    return COCO_VEHICLE.resolved(exclude_motorcycles)
