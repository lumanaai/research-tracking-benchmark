"""Default SSD / project paths for the MOT pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = Path(__file__).resolve().parent
SSD_ROOT = Path("/media/7TBSSD/data/tracking")

MOT_ROOT = SSD_ROOT / "mot"
DETECTIONS_ROOT = SSD_ROOT / "detections"
EXPERIMENTS_ROOT = SSD_ROOT / "experiments"
FINDINGS_ROOT = EXPERIMENTS_ROOT / "_findings"
WEIGHTS_ROOT = SSD_ROOT / "weights"

RAW_DATASETS = {
    "fasttracker_bench": SSD_ROOT / "FastTracker-Benchmark",
    "ua_detrac": SSD_ROOT / "UA-DETRAC",
    "trafficmot": SSD_ROOT / "TrafficMOT",
    "cityflow": SSD_ROOT / "CityFlow",
}

TRACKEREVAL_ROOT = PROJECT_ROOT / "FastTracker" / "TrackEval"
FASTTRACKER_ROOT = PROJECT_ROOT / "FastTracker"
OCSORT_ROOT = PROJECT_ROOT / "OC_SORT"
HYBRIDSORT_ROOT = PROJECT_ROOT / "HybridSORT"

DEFAULT_YOLO_WEIGHTS = WEIGHTS_ROOT / "yolov8m-expert_eff-1_2.pt"
