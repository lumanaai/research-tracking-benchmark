from mot_pipeline.trackers.analytics_bytetrack import AnalyticsByteTrackAdapter
from mot_pipeline.trackers.analytics_bytetrack_plus import AnalyticsByteTrackPlusAdapter
from mot_pipeline.trackers.analytics_bytetrack_plus_aug19 import (
    AnalyticsByteTrackPlusAug19Adapter,
)
from mot_pipeline.trackers.botsort import BoTSORTAdapter
from mot_pipeline.trackers.fasttracker import FastTrackerAdapter
from mot_pipeline.trackers.hybridsort import HybridSORTAdapter
from mot_pipeline.trackers.ocsort import OCSORTAdapter
from mot_pipeline.trackers.traffictrack import TrafficTrackAdapter

TRACKERS = {
    "fasttracker": FastTrackerAdapter,
    "traffictrack": TrafficTrackAdapter,
    "ocsort": OCSORTAdapter,
    "hybridsort": HybridSORTAdapter,
    "analytics_bytetrack": AnalyticsByteTrackAdapter,
    "analytics_bytetrack_plus": AnalyticsByteTrackPlusAdapter,
    "analytics_bytetrack_plus_aug19": AnalyticsByteTrackPlusAug19Adapter,
    "botsort": BoTSORTAdapter,
}


def get_tracker(name: str, **kwargs):
    if name not in TRACKERS:
        raise KeyError(f"Unknown tracker '{name}'. Choose from: {sorted(TRACKERS)}")
    return TRACKERS[name](**kwargs)
