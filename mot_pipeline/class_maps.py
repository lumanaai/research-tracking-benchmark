"""Vehicle / ignore class maps per benchmark and detector.

Default policy: motorcycles/motorbikes are vehicles. Pass
``exclude_motorcycles=True`` to drop them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional


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

# LumanaBenchmark: single class vehicle=1.
LUMANA = ClassPolicy(
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
# 19 forklift, 23 boat, …
# Keep-set matches in-house product (bicycle + boat included; person dropped).
EXPERT_EFF_VEHICLE = ClassPolicy(
    keep=frozenset({1, 2, 3, 4, 6, 19, 23}),
    drop=frozenset({0}),  # person
    motorcycle_ids=frozenset({3}),
)

BENCHMARK_POLICIES = {
    "fasttracker_bench": FT_BENCH,
    "ua_detrac": UA_DETRAC,
    "trafficmot": TRAFFICMOT,
    "cityflow": CITYFLOW,
    "lumana_benchmark": LUMANA,
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


def class_space_for_yolo_weights(weights: Optional[object] = None) -> str:
    from pathlib import Path

    stem = Path(str(weights or "")).stem.lower()
    return "expert_eff" if "expert_eff" in stem else "coco"


# --- Analytics tracker class space -------------------------------------------
# The in-house analytics tracker consumes ``(cls, subclass)`` per detection,
# where ``subclass`` indexes analyzer_manager/assets/detection/32cls.csv:
#   0 person, 1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 train, 6 truck,
#   19 forklift, 23 boat, ...
# and ``cls`` is that row's coarse ``object_id`` (person=0, vehicle=1, ...),
# which MiniClassHandler derives from the same CSV.
A_PERSON = 0
A_BICYCLE = 1
A_CAR = 2
A_MOTORCYCLE = 3
A_BUS = 4
A_TRUCK = 6
A_FORKLIFT = 19
A_BOAT = 23


@dataclass(frozen=True)
class AnalyticsClassSpace:
    """Maps one detection-class numbering onto 32cls subclass ids."""

    mapping: Mapping[int, int]
    default: Optional[int] = None
    """Subclass for ids absent from ``mapping``. ``None`` drops them."""

    def subclass_of(self, class_id: int) -> Optional[int]:
        sub = self.mapping.get(int(class_id))
        return self.default if sub is None else sub


# FastTracker-Benchmark gt/labels.txt, 1-indexed:
# 1 person, 2 bus_small, 3 bus_big, 4 truck_small, 5 truck_big, 6 car, 7 bike,
# 8 motorbike, 9 ignore_region, 10 tractor, 11 trailor, 12 wheelchair,
# 13 heavy_equipment, 14 pm, 15 umbrella
ANALYTICS_FT_BENCH = AnalyticsClassSpace(
    {
        1: A_PERSON,
        2: A_BUS,
        3: A_BUS,
        4: A_TRUCK,
        5: A_TRUCK,
        6: A_CAR,
        7: A_BICYCLE,
        8: A_MOTORCYCLE,
        10: A_TRUCK,  # tractor — no 32cls equivalent
        11: A_TRUCK,  # trailor
        12: A_PERSON,  # wheelchair
        13: A_FORKLIFT,  # heavy_equipment
        14: A_MOTORCYCLE,  # pm (personal mobility)
        15: A_PERSON,  # umbrella (carried by a person)
    }
)

# UA-DETRAC converter numbering (see converters/prepare_ua_detrac.py).
ANALYTICS_UA_DETRAC = AnalyticsClassSpace(
    {1: A_CAR, 2: A_BUS, 3: A_TRUCK, 4: A_CAR},  # van -> truck, others -> car
    default=A_CAR,
)

# TrafficMOT classes 1-10 (see the dataset ReadMe).
ANALYTICS_TRAFFICMOT = AnalyticsClassSpace(
    {
        1: A_MOTORCYCLE,  # Motor_Bike
        2: A_BUS,  # Bus
        3: A_CAR,  # LMV
        4: A_CAR,  # Auto
        5: A_BICYCLE,  # Bike
        6: A_PERSON,  # Pedestrian
        7: A_TRUCK,  # LCV
        8: A_CAR,  # E-rickshaw
        9: A_TRUCK,  # Tractor
        10: A_TRUCK,  # Truck
    }
)

# CityFlow GT carries no class column (the class field is -1); all annotations
# are vehicles, overwhelmingly cars.
ANALYTICS_CITYFLOW = AnalyticsClassSpace({-1: A_CAR, 1: A_CAR}, default=A_CAR)

# LumanaBenchmark: class_id always 1 (vehicle).
ANALYTICS_LUMANA = AnalyticsClassSpace({1: A_CAR}, default=A_CAR)

ANALYTICS_COCO = AnalyticsClassSpace(
    {
        0: A_PERSON,
        1: A_BICYCLE,
        2: A_CAR,
        3: A_MOTORCYCLE,
        5: A_BUS,
        7: A_TRUCK,
    }
)

# The expert_eff detector already emits 32cls ids.
ANALYTICS_EXPERT_EFF = AnalyticsClassSpace({}, default=None)

ANALYTICS_CLASS_SPACES: dict[str, AnalyticsClassSpace] = {
    "fasttracker_bench": ANALYTICS_FT_BENCH,
    "ua_detrac": ANALYTICS_UA_DETRAC,
    "trafficmot": ANALYTICS_TRAFFICMOT,
    "cityflow": ANALYTICS_CITYFLOW,
    "lumana_benchmark": ANALYTICS_LUMANA,
    "coco": ANALYTICS_COCO,
    "expert_eff": ANALYTICS_EXPERT_EFF,
    "analytics_32cls": ANALYTICS_EXPERT_EFF,
}


def analytics_class_space(space: str) -> AnalyticsClassSpace:
    """Look up a detection-class numbering by name.

    ``expert_eff`` / ``analytics_32cls`` are identity spaces: those detections
    already use 32cls ids, so ids fall through unchanged.
    """
    if space in ("expert_eff", "analytics_32cls"):
        return ANALYTICS_EXPERT_EFF
    if space not in ANALYTICS_CLASS_SPACES:
        raise KeyError(
            f"Unknown analytics class space '{space}'. "
            f"Choose from: {sorted(ANALYTICS_CLASS_SPACES)}"
        )
    return ANALYTICS_CLASS_SPACES[space]


def to_analytics_subclass(class_id: int, space: str) -> Optional[int]:
    """Translate a detection class id into a 32cls subclass id.

    Returns ``None`` when the id has no sensible equivalent, in which case the
    caller should drop the detection.
    """
    cs = analytics_class_space(space)
    if cs is ANALYTICS_EXPERT_EFF:
        return int(class_id) if int(class_id) >= 0 else None
    return cs.subclass_of(class_id)
