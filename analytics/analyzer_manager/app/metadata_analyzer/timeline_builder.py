from collections import defaultdict, Counter
from dataclasses import dataclass, field
from math import floor, ceil
from typing import Dict, Optional, List, Union
import numpy as np

from general.analyzer_general import logger
from general.cloud_converter import cloud_TrackerTypes_converter, cloud_GlobalTypes_converter
from general.core import BatchDataResolver, EntityClassification
from general.entity import LocationList
from general.entity_db import EntityDB


@dataclass
class EntityTimelineData:
    id_base: int = 0
    object_id: int = -1
    tracker_type: EntityClassification = EntityClassification.UNKNOWN
    global_type: str = "unknown"
    route: LocationList = None


class TimelineBuilder:
    _period: int
    entity_db: EntityDB
    _timeline_db: Dict[int, EntityTimelineData]
    _last_normalized_ts: int

    def __init__(self, entity_db: EntityDB, period_sec: int = 60):
        self.entity_db = entity_db
        self._period = period_sec
        self._timeline_db = {}
        self._last_normalized_ts = -1
        self.entity_db.on_entities_update += self.on_entities_update
        self.max_occupancy: Union[int, np.array] = 0
        self.max_occupancy_object = {}

    def track(self, batch_data: BatchDataResolver, timestamps: List[int]) -> Optional[Dict]:
        report = None
        ts_arr = np.array([t // 1000 for t in timestamps]).astype(int)
        sec = ts_arr % self._period
        normalized_ts = ts_arr - sec
        max_count = np.max(batch_data.object_count, axis=0)
        try:
            self.max_occupancy = np.maximum(max_count, self.max_occupancy)
        except ValueError:
            # might occur when new object classes are added
            self.max_occupancy = max_count

        idxs = np.where(normalized_ts > self._last_normalized_ts)[0]
        if len(idxs) > 0:
            if self._last_normalized_ts >= 0:
                report = self.sync_and_cleanup()
            self._last_normalized_ts = normalized_ts[idxs[0]]
        ent_ids = batch_data.unique(BatchDataResolver.ID).astype(int)
        for ent_id in ent_ids:
            if ent_id > 0 and ent_id not in self._timeline_db:
                self._timeline_db[ent_id] =  self._generate_default_timeline(ent_id)
        return report

    def sync_and_cleanup(self) -> Dict:
        trackers = []
        normalized_ts = int(self._last_normalized_ts * 1000)
        tracker_classes = [obj.tracker_type.name.lower() for obj in self._timeline_db.values()]
        occupancy = dict(Counter(tracker_classes))
        if not np.isscalar(self.max_occupancy):
            occupancy.update(
                {
                    self.entity_db.class_handler.get_object_name(idx): val
                    for idx, val in enumerate(self.max_occupancy)
                    if val > 0
                }
            )
        for ent_id in self._timeline_db:
            ent_timeline = self._timeline_db.get(ent_id)
            report = self._timeline_report(ent_id, ent_timeline)
            trackers.append(report)
        self._timeline_db.clear()
        self.max_occupancy = 0
        return {"timestamp": normalized_ts, "trackers": trackers, "occupancy": occupancy} if trackers else None

    def _generate_default_timeline(self, ent_id: int) -> EntityTimelineData:
        ent_data = self.entity_db.get_entity(ent_id)
        ent_timeline = EntityTimelineData()
        ent_timeline.id_base = ent_data.id_base
        ent_timeline.object_id = self.entity_db.class_handler.get_object_name(ent_data.object_id)
        ent_timeline.tracker_type = ent_data.tracker_type
        ent_timeline.global_type = ent_data.global_type
        ent_timeline.route = ent_data.locations
        return ent_timeline

    def on_entities_update(self, ent_ids: List[int]):
        for ent_id in ent_ids:
            if ent_id in self._timeline_db:
                ent_data = self.entity_db.get_entity(ent_id)
                if ent_data is not None:
                    self._timeline_db[ent_id].tracker_type = ent_data.tracker_type
                    self._timeline_db[ent_id].object_id = self.entity_db.class_handler.get_object_name(
                        ent_data.object_id
                    )
                    self._timeline_db[ent_id].global_type = ent_data.global_type

    def _timeline_report(self, ent_id, ent_timeline: EntityTimelineData) -> Dict:

        first_idx = np.where(ent_timeline.route.timestamps < self._last_normalized_ts * 1000)[0]
        first_idx = first_idx[-1] if len(first_idx) > 0 else 0
        sec = np.copy(ent_timeline.route.timestamps[first_idx:] / 1000 - self._last_normalized_ts)
        locs = ent_timeline.route.locations[first_idx:]
        sec = sec[sec < 60]
        if len(sec) > 1 and sec[1] < 0:
            logger.error(f"Negative timestamp for entity {ent_id}. timestamps: {ent_timeline.route.timestamps}, last ts: {self._last_normalized_ts}, first_idx: {first_idx}")
            sec = sec[1:]
            locs = locs[1:]

        sec = np.append(sec, self._period)
        alt_timeline = defaultdict(lambda: np.uint64(0))
        dwell_timeline = []
        idx = 0
        first_sec = sec[0]
        if first_sec > 0:
            dwell_timeline.append({"index": -1, "dwell": first_sec})
        else:
            first_sec = 0
        last_sec = first_sec
        timeline = np.uint64(0)
        while idx < len(sec) - 1:
            dwell_timeline.append({"index": locs[idx], "dwell": sec[idx + 1] - last_sec})
            bit_range = np.uint64(1 << ceil(sec[idx + 1])) - np.uint64(1 << floor(last_sec))
            alt_timeline[locs[idx]] = np.bitwise_or(alt_timeline[locs[idx]], bit_range)
            if locs[idx] >= 0:
                timeline = np.bitwise_or(timeline, bit_range)
            last_sec = sec[idx + 1]
            idx += 1
        locations = [{"index": int(k), "timeline": str(v)} for k, v in alt_timeline.items() if k >= 0]
        return {
            "idBase": ent_timeline.id_base,
            "idIndex": ent_id,
            "timeline": str(timeline),
            "route": dwell_timeline,
            "type": cloud_TrackerTypes_converter.get(ent_timeline.object_id, 0),
            "globalType": cloud_GlobalTypes_converter[ent_timeline.global_type],
            "trackerClass": ent_timeline.tracker_type.value,
            "positions": locations,
        }

    def calc_objects_motion(self, since_ts: int, max_window:int = 5) -> int:
        max_motion = 0
        for ent_data in self._timeline_db.values():
            if len(ent_data.route.timestamps):
                max_motion = max(max_motion, np.count_nonzero(ent_data.route.timestamps[-max_window:] > since_ts))
        return max_motion
