from copy import copy
from math import floor
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from numpy import ndarray

from .analyzer_general import (
    ROI_SHAPE,
    ROUTE_SHAPE,
    ALPR_TRUCK_TYPES,
    ALPR_CAR_TYPES,
    GLOBAL_TYPE_KEY,
    GLOBAL_BLACKLIST,
    GLOBAL_SUCCESS,
    PARTIAL_MATCH_FILTERS,
    logger,
    GENDER_FILTER,
    AGE_FILTER,
    MALE_VALUE,
    CHILD_VALUE,
)
from .cloud_converter import convert_custom_object
from .core import BatchDataResolver, ClassHandler, AttrProperty, AttrConfidence, EntityClassification


def bbox_to_perimeter_position(bbox):
    position = {"x": round((bbox[0] + bbox[2]) / 2, 3), "y": round((bbox[1] + bbox[3]) / 2, 3)}
    top_left = {"x": round(bbox[0], 3), "y": round(bbox[1], 3)}
    bottom_right = {"x": round(bbox[2], 3), "y": round(bbox[3], 3)}
    return {"position": position, "perimeter": {"topLeft": top_left, "bottomRight": bottom_right}}


class LocationList:
    _pos: int
    _curr_len: int
    _current_index: int
    _last_center: np.array
    TIMESTAMP: int = 0  # position of timestamp in array
    LOCATION: int = 1  # position of location in array
    hysteresis_th: float = 0.6**2

    def __init__(self, initial_pos: ndarray = None, init_len: int = 30):
        self._curr_len = init_len
        self._locations = np.full(init_len, -1, dtype=int)
        self._timestamps = np.full(init_len, -1, dtype=int)
        self._pos = 0
        if initial_pos is not None:
            self._last_center = np.array([(initial_pos[0] + initial_pos[2]), initial_pos[1] + initial_pos[3]]) / 2
        else:
            self._last_center = np.array([-1, -1])

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < self._pos:
            loc = self._locations[self._current_index]
            ts = self._timestamps[self._current_index]
            self._current_index += 1
            return ts, loc
        raise StopIteration

    def __getitem__(self, item):
        if item < self._pos:
            loc = self._locations[item]
            ts = self._timestamps[item]
            return ts, loc
        return None

    def insert(self, location: int, location_coords, timestamp: int) -> bool:
        prev_location = self._locations[self._pos - 1]
        is_moved = False
        if prev_location != location:
            dist2 = np.linalg.norm(location_coords - self._last_center)
            if dist2 > self.hysteresis_th:  # hysteresis
                is_moved = True
                self._timestamps[self._pos] = timestamp
                self._locations[self._pos] = location
                self._last_center = location_coords
                self._extend_if_needed()
                self._pos += 1
        return is_moved

    def calc_max_span(self, accurate: bool = False) -> float:
        locs = self._locations[: self._pos]
        unique_locations = np.unique(locs[locs != -1])
        if len(unique_locations) <= 1:
            return 0
        elif len(unique_locations) > 9:
            if not accurate:
                return 32

        rows = []
        cols = []
        for loc in unique_locations:
            rows.append(loc // ROI_SHAPE[0])
            cols.append(loc % ROI_SHAPE[0])
        r_ptp = np.ptp(rows)
        c_ptp = np.ptp(cols)
        return np.maximum(r_ptp, c_ptp)

    @property
    def last(self):
        if self._pos > 0:
            item = self._pos - 1
            loc = self._locations[item]
            ts = self._timestamps[item]
            return ts, loc
        return None

    @property
    def last_index(self):
        return self._pos

    @property
    def timestamps(self):
        if self._pos > 0:
            return self._timestamps[: self._pos]
        return []

    @property
    def locations(self):
        if self._pos > 0:
            return self._locations[: self._pos]
        return []

    def finalize(self, last_seen: int):
        pos = min(self._pos, self._curr_len - 1)  # precaution - shouldn't be other than self._pos
        self._timestamps[pos] = last_seen
        self._locations[pos] = -1
        self._pos += 1

    def _extend_if_needed(self):
        if self._pos > self._curr_len - 5:
            self._curr_len *= 2
            self._locations.resize(self._curr_len, refcheck=False)
            self._timestamps.resize(self._curr_len, refcheck=False)

    def reactivate(self):
        self._extend_if_needed()


AREA_FACTOR = ROUTE_SHAPE[0] * ROUTE_SHAPE[1]


class ActiveEntityData:
    def __init__(self, track_id: int, first_seen: int, class_handler: ClassHandler, initial_pos: ndarray = None):
        self.first_seen: int = first_seen
        self.track_id: int = track_id
        self.locations: LocationList = LocationList()
        self.attributes: Dict = {}
        self.visual_id: Optional[np.array] = None
        self.last_cloud_sync: int = 0
        self.last_bbox: ndarray = np.zeros(4)
        self.object_id: int = -1
        self.class_id: int = -1
        self.last_seen: int = 0
        self.last_confidence: float = 0
        self.total_samples: int = 0
        self.mean_confidence: float = 0
        self.max_confidence: float = 0
        self.crop_index: int = 0
        self.best_image: Optional[str] = None
        self.zoom_image: Optional[str] = None
        self.flags = set()
        self.appearing_objects: Dict = {}
        self.tracker_type: EntityClassification = EntityClassification.UNKNOWN

        self.encoding: Optional[np.array] = None
        self.max_area: float = 0
        self.class_handler = class_handler
        self._l1_processed = set()
        self.last_bbox = initial_pos
        self.clip: Optional[np.array] = None
        self.custom_object_data = {}

    def bbox_to_location(self, bbox) -> int:
        raise NotImplementedError

    def update(self, obj_data: ndarray, appearance: Dict) -> Tuple[bool, bool, int]:
        locations = obj_data[:, BatchDataResolver.LOCATION].astype(int)
        timestamps = obj_data[:, BatchDataResolver.TIMESTAMP].astype(int)
        confidences = obj_data[:, BatchDataResolver.CONFIDENCE].astype(float)
        bboxes = obj_data[:, BatchDataResolver.POS].astype(float)
        centers = obj_data[:, BatchDataResolver.CENTER].astype(float)
        areas = obj_data[:, BatchDataResolver.AREA].astype(float)

        self.last_seen = timestamps[-1]
        is_moved = False
        enlarged = np.argmax(np.insert(areas, 0, self.max_area)) - 1
        if enlarged >= 0:
            self.max_area = areas[enlarged] * 1.1  # require 10% increase in size

        # update locations
        center_std = np.linalg.norm(np.std(centers, axis=0))
        if center_std > 0.3:
            i = 0
            while i < len(locations):
                is_moved |= self.locations.insert(locations[i], centers[i], timestamps[i])
                # only do it rapidly if there is a change in location
                i += 1
                if not is_moved:  # skip one if no movement
                    i += 1
        else:
            bbox = np.median(np.vstack((self.last_bbox, bboxes)), axis=0)
            center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2 * ROI_SHAPE
            location = floor(min(ROI_SHAPE[1] - 1, floor(bbox[3] * ROI_SHAPE[1])) * ROI_SHAPE[0] + center[0])
            is_moved = self.locations.insert(location, center, timestamps[0])

        self.last_bbox = bboxes[-1]

        # check if object type has changed
        obj_type = np.median(obj_data[:, BatchDataResolver.CLASS]).astype(int)
        obj_subtypes = np.median(obj_data[:, BatchDataResolver.SUBCLASS]).astype(int)
        if len(self.appearing_objects):
            self.appearing_objects = {
                k: max(self.appearing_objects[k], appearance[k])
                for k in self.appearing_objects.keys() & appearance.keys()
            }
        else:
            self.appearing_objects = appearance

        # TODO: Need to consider when to update appearing_objects
        is_type_changed = False
        if obj_type != self.object_id or obj_subtypes != self.class_id:
            if obj_type == -1 or self.mean_confidence < np.median(confidences):
                self.object_id = int(obj_type)
                self.class_id = int(obj_subtypes)
                self._classify_tracker_type()
                is_type_changed = True

        self.last_confidence = confidences[-1]
        self.mean_confidence = self.running_average(self.mean_confidence, confidences)
        self.max_confidence = np.amax([self.max_confidence, np.amax(confidences)])
        self.total_samples += len(timestamps)
        return is_type_changed, is_moved, enlarged

    def update_attr(self, attr: Dict, l1_analyzers: Set):
        if "descriptor" in attr:
            desc = attr.pop("descriptor")
            self.visual_id = desc.value
        if self.attributes:
            for key in attr.keys():
                if key in self.attributes and isinstance(self.attributes[key], list):
                    cur_priority = max([0] + [p.priority for p in self.attributes[key] if isinstance(p, AttrProperty)])
                    new_priority = max([0] + [p.priority for p in attr[key] if isinstance(p, AttrProperty)])
                    if new_priority >= cur_priority:
                        self.attributes[key] = attr[key]
                else:
                    self.attributes[key] = attr[key]
        else:
            self.attributes.update(attr)

        if (
            self.is_blacklisted
            and GLOBAL_TYPE_KEY in self.attributes
            and GLOBAL_SUCCESS == self.attributes[GLOBAL_TYPE_KEY][0].value
        ):
            self.flags.remove(GLOBAL_BLACKLIST)

        self._l1_processed.update(l1_analyzers)
        self._classify_tracker_type()

    def running_average(self, current_value, new_values: ndarray):
        return (current_value * self.total_samples + np.sum(new_values)) / (self.total_samples + len(new_values))

    def deactivate(self):
        self.locations.finalize(self.last_seen)

    def reactivate(self):
        self.locations.reactivate()
        # self.locations = LocationList()

    def route_map(self):
        route_map = np.zeros(ROUTE_SHAPE)
        for i, loc in enumerate(self.locations):
            x, y = np.unravel_index(loc[1], ROI_SHAPE, order="C")
            area_factor = floor((np.sqrt(loc[2]) - 1) / 2)
            x_coord = floor(x * ROUTE_SHAPE[0] / 32)
            y_coord = floor(y * ROUTE_SHAPE[1] / 32)
            if area_factor > 0:
                route_map[
                    x_coord - area_factor : x_coord + area_factor + 1, y_coord - area_factor : y_coord + area_factor + 1
                ] = (i + 1)
            else:
                route_map[x_coord, y_coord] = i + 1
        return np.ravel(route_map)

    def directional_map(self):
        id_map = np.zeros((*ROI_SHAPE, 8), dtype=int)
        last_x = -1
        last_y = -1
        for i, loc in enumerate(self.locations):
            x, y = np.unravel_index(loc[1], ROI_SHAPE, order="C")
            if last_x >= 0:
                dx = x - last_x
                dy = y - last_y
                while dx != 0 or dy != 0:
                    if dx < 0 and dy < 0:
                        id_map[last_x, last_y, 0] = 1
                    elif dx == 0 and dy < 0:
                        id_map[last_x, last_y, 1] = 1
                    elif dx > 0 and dy < 0:
                        id_map[last_x, last_y, 2] = 1
                    elif dx > 0 and dy == 0:
                        id_map[last_x, last_y, 3] = 1
                    elif dx > 0 and dy > 0:
                        id_map[last_x, last_y, 4] = 1
                    elif dx == 0 and dy > 0:
                        id_map[last_x, last_y, 5] = 1
                    elif dx < 0 and dy > 0:
                        id_map[last_x, last_y, 6] = 1
                    elif dx < 0 and dy == 0:
                        id_map[last_x, last_y, 7] = 1
                    dx -= np.sign(dx)
                    dy -= np.sign(dy)
            last_x = x
            last_y = y
        return id_map

    def deploy_max_neighbors(self):
        appearance = {}
        for object_id in self.appearing_objects.keys():
            object_name = self.class_handler.get_object_name(object_id)
            appearance[object_name] = self.appearing_objects[object_id]
        return appearance

    def sync(self, to_ts: int, roi_filter: Optional[Set] = None):
        location_dict = self.generate_location_dict(self.last_cloud_sync, to_ts)
        if roi_filter is not None:
            keys = list(location_dict.keys())
            for key in keys:
                if key not in roi_filter:
                    del location_dict[key]
        if len(location_dict) == 0:
            return None
        object_name = self.class_handler.get_object_name(self.object_id)
        bbox_data = bbox_to_perimeter_position(self.last_bbox)
        last_seen = min(self.last_seen, to_ts)
        info_dict = {
            "idBase": self.first_seen,
            "idIndex": self.track_id,
            "dwell": (last_seen - self.first_seen) / 1000,
            "maxNeighbors": self.deploy_max_neighbors(),
            "markedPositions": list(location_dict.values()),
            "timestamp": last_seen,
            "type": object_name,
            "conf": self.max_confidence,
        }
        info_dict.update(bbox_data)
        self.last_cloud_sync = to_ts
        return info_dict

    def generate_location_dict(self, from_ts: int, to_ts: int) -> Dict:
        ts = self.locations.timestamps
        first_idx = np.argwhere(ts > from_ts - 1)
        location_dict = {}
        if len(first_idx) > 0:
            start_idx = max(0, first_idx[0][0] - 1)
            last_idx = np.argwhere(self.locations.timestamps > to_ts)
            if len(last_idx) > 0:
                end_idx = last_idx[0][0]
            else:
                end_idx = len(ts)

            for idx in range(start_idx, end_idx - 1):
                loc = self.locations[idx]
                if loc[1] not in location_dict:
                    location_dict[loc[1]] = {"index": loc[1], "startTime": loc[0], "endTime": 0}
                location_dict[loc[1]]["endTime"] = self.locations[idx + 1][0]

            loc = self.locations[end_idx - 1]
            if loc[1] not in location_dict:
                location_dict[loc[1]] = {"index": loc[1], "startTime": loc[0], "endTime": 0}
            location_dict[loc[1]]["endTime"] = min(to_ts, self.last_seen)
        else:
            if self.last_seen > from_ts:
                loc = self.locations.last
                location_dict[loc[1]] = {"index": loc[1], "startTime": loc[0], "endTime": min(to_ts, self.last_seen)}
        return location_dict

    def is_valid(self, success_required: bool = False) -> Optional[bool]:
        is_failure = GLOBAL_TYPE_KEY in self.attributes and GLOBAL_SUCCESS != self.attributes[GLOBAL_TYPE_KEY][0].value
        # if success is required - we should return False if the object is not a success
        if success_required and is_failure:
            return False
        # if blacklisted - give it a chance to get out of blacklist
        if self.is_blacklisted:
            return None
        # otherwise return false only if we know its a failure
        return not is_failure

    def match_filters(
        self,
        object_ids: List[int],
        filters_dict: Dict,
        filters_disabled: Dict,
        success_required: bool = False,
        strict_filters: Dict = {},
    ):
        if self.object_id not in object_ids:
            return False

        desc = self.attributes

        if success_required and GLOBAL_TYPE_KEY in desc and GLOBAL_SUCCESS != desc[GLOBAL_TYPE_KEY][0].value:
            return False

        if filters_disabled[self.object_id]:
            return True

        filters = filters_dict[self.object_id]
        if len(desc) == 0:
            desc = self.get_default_attributes()

        if len(strict_filters) and strict_filters[self.object_id]:
            """requires the attributes to be in hgh confidence in order for the filter to match"""

            for key in filters.keys():
                desc_value = desc.get(key, [])

                if isinstance(desc_value, list):
                    desc_value = set([prop.value for prop in desc_value if prop.confidence == AttrConfidence.HIGH])
                else:
                    desc_value = set(desc_value.value) if desc_value.confidence == AttrConfidence.HIGH else set()

                if key in PARTIAL_MATCH_FILTERS:
                    filter_match = any(elem2 in elem1 for elem1 in desc_value for elem2 in filters[key])
                else:
                    filter_match = bool(desc_value & filters[key])

                if not filter_match:
                    return False
            # all attributes matched
            return True
        else:
            for key in filters.keys():
                filter_match = True
                # if we don't have such attribute we enable match on non strict comparison
                if key in desc:
                    if isinstance(desc[key], list):
                        values = set([prop.value for prop in desc[key]])
                        if key in PARTIAL_MATCH_FILTERS:
                            param_match = any(elem2 in elem1 for elem1 in values for elem2 in filters[key])
                        else:
                            param_match = bool(values & filters[key])
                    else:
                        # TODO: check if we need to remove
                        logger.error(f"depricated content {desc[key]}")
                        param_match = desc[key].value in filters[key]
                    filter_match = filter_match and param_match
                if not filter_match:
                    return False
            # all attributes matched
            return True

    @staticmethod
    def to_prop(lst: List[str], confidence=AttrConfidence.LOW) -> List[AttrProperty]:
        return [AttrProperty(l, 0, confidence) for l in lst]

    def get_default_attributes(self):
        class_name, object_name = self.class_handler.get_class_data(self.class_id)
        if class_name in ["car"]:
            class_type = ALPR_CAR_TYPES
        elif class_name in ["truck"]:
            class_type = ALPR_TRUCK_TYPES
        else:
            class_type = [class_name]

        return {
            "type": self.to_prop(class_type, AttrConfidence.HIGH),
            GLOBAL_TYPE_KEY: self.to_prop([GLOBAL_SUCCESS], AttrConfidence.HIGH),
        }

    def get_attribute_report(self) -> Dict:
        class_name, object_name = self.class_handler.get_class_data(self.class_id)
        if len(self.attributes) == 0:
            attr = self.get_default_attributes()
        else:
            attr = copy(self.attributes)
        self._classify_tracker_type()
        if GLOBAL_TYPE_KEY not in self.attributes:
            if self.is_blacklisted:
                attr[GLOBAL_TYPE_KEY] = self.to_prop([GLOBAL_BLACKLIST], AttrConfidence.HIGH)
            else:
                attr[GLOBAL_TYPE_KEY] = self.to_prop([GLOBAL_SUCCESS], AttrConfidence.HIGH)
        att_report = {
            "conf": self.max_confidence,
            "description": attr,
            "idBase": self.first_seen,
            "idIndex": self.track_id,
            "type": object_name,
            "timestamp": self.last_seen,
            "trackerClass": self.tracker_type.value,
        }
        if self.best_image is not None:
            att_report["bestImage"] = self.best_image
        if self.zoom_image is not None:
            att_report["zoomImage"] = self.zoom_image
        if self.clip is not None:
            att_report["clip"] = self.clip
        pos_custom_obj = self.custom_object_data.get("pos", [])
        if pos_custom_obj:
            att_report.update(convert_custom_object(pos_custom_obj))

        return att_report

    @property
    def id_base(self):
        return self.first_seen

    @property
    def is_blacklisted(self):
        return GLOBAL_BLACKLIST in self.flags

    @property
    def global_type(self):
        if GLOBAL_TYPE_KEY in self.attributes:
            return self.attributes[GLOBAL_TYPE_KEY][0].value
        elif self.is_blacklisted:
            return GLOBAL_BLACKLIST
        return GLOBAL_SUCCESS

    def has_l1(self, required_l1):
        return required_l1 in self._l1_processed

    @property
    def dwell(self) -> int:
        return self.last_seen - self.first_seen

    @property
    def has_any_l1(self) -> bool:
        return len(self._l1_processed) > 0

    def _classify_tracker_type(self):
        self.tracker_type = self.class_handler.object_types.get(self.object_id, EntityClassification.UNKNOWN)
        if self.object_id == self.class_handler.person_value:
            if AGE_FILTER in self.attributes and self.attributes[AGE_FILTER][0].value == CHILD_VALUE:
                self.tracker_type = EntityClassification.CHILD
            elif GENDER_FILTER in self.attributes:
                self.tracker_type = (
                    EntityClassification.MALE
                    if self.attributes[GENDER_FILTER][0].value == MALE_VALUE
                    else EntityClassification.FEMALE
                )
            else:
                self.tracker_type = EntityClassification.UNKNOWN_GENDER
        elif self.object_id == self.class_handler.vehicle_value:
            tracker_type = self.class_handler.vehicle_types.get(self.class_id, EntityClassification.UNKNOWN)
            type_from_attr = self.attributes.get("type", None)
            t = ""
            if type_from_attr:
                t = self.attributes["type"][0].value
            self.tracker_type = self.class_handler.vehicle_string_types.get(t, tracker_type)

    def has_movement(self, threshold: int = 1) -> bool:
        locs = self.locations.locations
        unique_locations = np.unique(locs[locs != -1])
        return len(unique_locations) > threshold

    def is_match_custom_object(self, custom_objects_data: Dict, force: bool = False) -> bool:
        if self.custom_object_data:
            required_objects = set(custom_objects_data)
            neg = set(self.custom_object_data.get("neg", []))

            # if all required objects are in neg, then it's not a match
            if len(required_objects.difference(neg)) == 0:
                return False

            pos_ids = set(self.custom_object_data.get("pos", [])).intersection(required_objects)
            is_clip = self.custom_object_data.get("is_clip", False)
            is_reid = self.custom_object_data.get("is_reid", False)
            if is_clip and is_reid:
                return len(pos_ids) > 0

            if len(pos_ids) > 0:
                if force:
                    return True
                else:
                    reid_ok = [not custom_objects_data[p].custom_object.is_reid_enabled or is_reid for p in pos_ids]
                    clip_ok = [not custom_objects_data[p].custom_object.is_clip_enabled or is_clip for p in pos_ids]
                    is_final = any([r and c for r, c in zip(reid_ok, clip_ok)])
                    return is_final or None
        return False if force else None
