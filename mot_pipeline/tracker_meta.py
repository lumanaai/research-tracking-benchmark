"""Static metadata for registered trackers (release year, preferred detector, …).

Used by comparison tables and docs. ``ref_ms_per_frame`` is filled by
``scripts/analysis/bench_tracker_speed.py`` (GT dets, one dense sequence) and
is only a fallback when a findings row has no per-run timing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Paper / lineage year and the detector the method was designed / reported with.
# ``preferred_detector`` is the upstream preference (not necessarily what this
# pipeline sweeps — we usually feed YOLOv8m-expert or GT boxes).
TRACKER_META: Dict[str, Dict[str, Any]] = {
    "fasttracker": {
        "year": 2025,
        "preferred_detector": "YOLOX",
        "ref_ms_per_frame": 5.5908,
        "note": "Occlusion/ROI ByteTrack-style; paper uses YOLOX multi-class.",
    },
    "ocsort": {
        "year": 2023,
        "preferred_detector": "YOLOX",
        "ref_ms_per_frame": 9.2088,
        "note": "Observation-centric KF; motion-only; paper uses ByteTrack YOLOX dets.",
    },
    "hybridsort": {
        "year": 2024,
        "preferred_detector": "YOLOX",
        "ref_ms_per_frame": 16.0810,
        "note": "Weak cues (confidence/height) on OC-SORT; paper uses YOLOX.",
    },
    "analytics_bytetrack": {
        "year": 2022,
        "preferred_detector": "YOLO",
        "ref_ms_per_frame": 8.3883,
        "note": "In-house ByteTrack; production prefers YOLO (keeps low-score boxes).",
    },
    "analytics_bytetrack_plus": {
        "year": 2022,
        "preferred_detector": "YOLO",
        "ref_ms_per_frame": 12.0329,
        "note": "In-house ByteTrack+ (crossover/parked); same YOLO det preference.",
    },
    "analytics_bytetrack_plus_aug19": {
        "year": 2026,
        "preferred_detector": "YOLO",
        "note": "Aug19 ByteTrack+ (CIoU + retuned crossover/lifecycle); YOLO preferred.",
    },
    "botsort": {
        "year": 2022,
        "preferred_detector": "YOLOX",
        "ref_ms_per_frame": 4.9772,
        "note": "Bag-of-tricks on ByteTrack; paper YOLOX (+ FastReID when ReID on).",
    },
    "traffictrack": {
        "year": None,
        "preferred_detector": "—",
        "note": "Stub — not implemented yet.",
    },
}

# Populated by bench_tracker_speed.py; keys match TRACKER_META.
REF_MS_PER_FRAME: Dict[str, float] = {}


def tracker_year(name: str) -> Optional[int]:
    meta = TRACKER_META.get(name) or {}
    year = meta.get("year")
    return int(year) if year is not None else None


def preferred_detector(name: str) -> str:
    meta = TRACKER_META.get(name) or {}
    return str(meta.get("preferred_detector") or "—")


def display_tracker_name(name: str) -> str:
    year = tracker_year(name)
    if year is None:
        return name
    return f"{name} ({year})"


def ref_ms_per_frame(name: str) -> Optional[float]:
    if name in REF_MS_PER_FRAME:
        return float(REF_MS_PER_FRAME[name])
    meta = TRACKER_META.get(name) or {}
    val = meta.get("ref_ms_per_frame")
    return float(val) if val is not None else None


def set_ref_ms_per_frame(values: Dict[str, float]) -> None:
    """Update in-memory + TRACKER_META ref timings (caller may persist to disk)."""
    REF_MS_PER_FRAME.clear()
    REF_MS_PER_FRAME.update({k: float(v) for k, v in values.items()})
    for name, ms in values.items():
        if name in TRACKER_META:
            TRACKER_META[name]["ref_ms_per_frame"] = float(ms)
