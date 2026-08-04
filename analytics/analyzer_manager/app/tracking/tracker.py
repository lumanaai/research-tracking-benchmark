from enum import Enum
from typing import List, Dict
from general.analyzer_general import logger
from general.core import AnalyticImage


class TrackerType(str, Enum):
    BASE = "base"
    BYTETRACK = "bytetrack"
    DEEPSORT = "deepsort"
    BYTESREID = "byteSReidTrack"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class BaseTracker:
    name: str = TrackerType.BASE.value
    track_buffer_sec: float

    def __init__(self, msg_dict: dict, fps: int, context):
        self.track_buffer_sec = 3.0
        self.context = context

    def on_night_mode_changed(self, night_mode: bool):
        pass

    def run(self, det, images: List[AnalyticImage]):
        pass

    def merge_ids(self, id_to_keep: int, id_to_merge: int):
        pass

    def get_confidence_th(self) -> float:
        pass

    def on_entities_stats_update(self, stats: Dict):
        pass

    def set_image_size(self, im_size):
        pass

class TrackerFactory:
    _instance = None
    trackers: List[BaseTracker] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrackerFactory, cls).__new__(cls)
        return cls._instance

    def create(self, tracker_name: str, msg: dict, fps: int, context) -> BaseTracker:
        if tracker_name == TrackerType.BYTETRACK:
            logger.info("Initialize bytetrack")
            from .byte_track import bytetrack

            return bytetrack.Bytetrack(msg, fps, context)
        elif tracker_name == TrackerType.BYTESREID:
            from .byte_sreid_track import byteSReidTrack

            return byteSReidTrack.ByteSReidTrack(msg, fps, context)
        elif tracker_name == TrackerType.DEEPSORT:
            raise ValueError(f"deep sort is deprecated")
        else:
            raise ValueError(f"tracker_name must be a supported tracker:{TrackerType.list()}")
