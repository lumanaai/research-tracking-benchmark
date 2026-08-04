from typing import Tuple, Optional, Dict, List, Set
from collections import OrderedDict
import numpy as np
import itertools

from numpy import ndarray
from scipy.spatial.distance import cdist

from general import apply_from_dict
from general.core import BatchDataResolver, AnalyticImage, ClassHandler, AlertCategory
from general.entity import ActiveEntityData
from general.img_utils import crop_image
from .base_alerts import ObjectAlert, AlertCandidate, duration_unit_to_sec, duration_unit_to_str
from general.analyzer_general import ROI_SHAPE, SUPPORTED_FILTERS, logger

import enum


class ProximityLevel(enum.Enum):
    far = 0
    close = 1


class ProximityAlert(ObjectAlert):
    type_name = "proximity"
    default_msg = "Proximity alert"

    def __init__(self, alert_dict: Dict, context):
        super(ProximityAlert, self).__init__(alert_dict, context)
        self.min_proximity_th = 0.75  # normalized by offender size, below that objects are considered overlapped
        max_distance_th = 7.5  # in parts of ROI - should be configurable from alert fields
        analytic_config = self.context.get_config()
        if "alertConfig" in analytic_config:
            if "proximity" in analytic_config["alertConfig"]:
                if "min_proximity_th" in analytic_config["alertConfig"]["proximity"]:
                    self.min_proximity_th = analytic_config["alertConfig"]["proximity"]["min_proximity_th"]
                if "distance_th" in analytic_config["alertConfig"]["proximity"]:
                    max_distance_th = analytic_config["alertConfig"]["proximity"]["proximity_radius"]

        self.ids_mapping: OrderedDict[int, Set] = OrderedDict()
        class_handler: ClassHandler = self.context.get_class_handler()
        self.vehicle_value = class_handler.vehicle_value
        self.person_value = class_handler.person_value
        two_wheels = []
        for cl in ["motorcycle", "bicycle"]:
            if cl in class_handler.classes:
                two_wheels.append(class_handler.class_str_to_int(cl))
        self.two_wheels = np.array(two_wheels)

        proximity_level = int(apply_from_dict("proximity", self.formValue, ProximityLevel.close.value))
        duration_unit = apply_from_dict("durationUnit", self.formValue, 0)
        duration = apply_from_dict("duration", self.formValue, 0)
        self.proximity_duration_ms = duration * duration_unit_to_sec[duration_unit] * 1000
        self.max_proximity_th = apply_from_dict("maxProximityRadius", self.formValue, max_distance_th)

        if proximity_level == ProximityLevel.far.value:
            self.check_proximity_criteria = self.check_far_proximity_criteria
        else:
            self.check_proximity_criteria = self.check_close_proximity_criteria

        objects_value = apply_from_dict("objects", self.formValue, [])

        has_vehicle = False
        has_person = False

        # reorder according to object size
        self.objects = OrderedDict()
        size_order = ["vehicle", "person", "pet", "weapon"]
        required_objs = [class_handler.object_int_to_str(obj_data["type"]) for obj_data in objects_value]
        unused_inds = list(range(len(required_objs)))
        ord = []
        for obj in size_order:
            if obj in required_objs:
                idx = required_objs.index(obj)
                ord.append(idx)
                unused_inds.remove(idx)
        for idx in unused_inds:
            ord.append(idx)

        for i in ord:
            object_value = objects_value[i]
            if object_value is not None:
                object_name = self.context.get_class_handler().object_int_to_str(object_value["type"])
            else:
                object_name = "unknown"

            object_filters = apply_from_dict("filters", object_value, {})
            filters = {}
            for key in object_filters.keys():
                if (
                    key in SUPPORTED_FILTERS
                    and object_filters[key] is not None
                    and isinstance(object_filters[key], list)
                    and len(object_filters[key]) > 0
                ):
                    filters[key] = set(
                        [element.lower() if isinstance(element, str) else element for element in object_filters[key]]
                    )

            self.objects[object_value["type"]] = {
                "object": object_value["type"],
                "type": object_name,
                "filters": filters,
                "filtersDisabled": (len(filters) == 0),
                "appearance": False,
                "strict": apply_from_dict("strict", object_value, False),
            }

            self.ids_mapping[object_value["type"]] = set()
            if object_value["type"] == self.vehicle_value:
                has_vehicle = True
            if object_value["type"] == self.person_value:
                has_person = True
        self.post_filter = has_person and has_vehicle
        self.objects_keys = list(self.objects.keys())
        self.pivot = self.objects_keys[0]
        self.extra_field = {"proximity": proximity_level}

        duration_str = duration_unit_to_str[duration_unit]
        if self.category == AlertCategory.CustomizedCapabilities.value:
            self.alert_message = f"Workers are to close to a vehicle for more than {duration} {duration_str}s"
        else:
            objs_str = ", ".join([self.objects[obj]["type"] for obj in self.objects])
            if proximity_level == ProximityLevel.far.value:
                self.alert_message = f"{objs_str} appear together for more than {duration} {duration_str}s".capitalize()
            else:
                self.alert_message = f"{objs_str} are close to each other for more than {duration} {duration_str}s".capitalize()

    def on_entities_update(self, ent_ids: List[int]):
        to_remove = []
        for ent in ent_ids:
            ent_data: ActiveEntityData = self.entity_db.get_entity(ent)
            is_match = ent_data is not None and ent_data.object_id in self.objects
            is_exist = ent in self.active_ids
            if is_exist and not is_match:
                to_remove.append(ent)
            elif is_match and not is_exist:
                self.active_ids.add(ent)
                for key in self.ids_mapping:
                    if key == ent_data.object_id:
                        self.ids_mapping[key].add(ent)
                    else:
                        self.ids_mapping[key].discard(ent)
                        # logger.warning(f"PROXIMITY: entity {ent} was found under object {key} and was moved")
        if len(to_remove):
            self.on_entities_removed(to_remove)

    def on_entities_removed(self, ent_ids: List[int]):
        # remove from trackers list if the entitiy is part of the key
        self.trackers = {k: v for k, v in self.trackers.items() if not any(element in ent_ids for element in k)}
        for ent in ent_ids:
            self.active_ids.discard(ent)
            for s in self.ids_mapping.values():
                s.discard(ent)

    def check_close_proximity_criteria(self, vars_data: ndarray, frame) -> List[Tuple]:
        items = []
        vars_data = vars_data[vars_data[:, BatchDataResolver.FRAME_ID] == frame, :]
        if len(vars_data) < 2:
            return items
        # need to revise how to mark pivot and not pivot ents, since its may be problematic if a pivot object is not in
        # the list, for instance in cases of blacklist
        alert_ids = vars_data[:, BatchDataResolver.ID].astype(int).tolist()
        pivot_ents = list(self.ids_mapping[self.pivot].intersection(alert_ids))
        pivot_idx = np.isin(vars_data[:, BatchDataResolver.ID], pivot_ents)
        num_pivots = np.sum(pivot_idx)
        if num_pivots == 0 or num_pivots == len(vars_data):
            return items
        pivot_dets = vars_data[pivot_idx, :]
        other_dets = vars_data[np.logical_not(pivot_idx), :]
        distances = cdist(pivot_dets[:, BatchDataResolver.CENTER], other_dets[:, BatchDataResolver.CENTER], "euclidean")
        norm_factor = np.linalg.norm(
            pivot_dets[:, BatchDataResolver.CENTER] - pivot_dets[:, BatchDataResolver.POS[0:2]] * ROI_SHAPE, axis=1
        )
        norm_dist = np.array(distances) / norm_factor[:, np.newaxis]
        distance_th = np.minimum(norm_factor * 1.5, self.max_proximity_th)
        violations = np.where(np.logical_and(norm_dist > self.min_proximity_th, distances < distance_th[:, np.newaxis]))
        for i, pivot in enumerate(violations[0]):
            items.append((pivot_dets[pivot, BatchDataResolver.ID], other_dets[violations[1][i], BatchDataResolver.ID]))
        return items

    def check_far_proximity_criteria(self, vars_data: np.array, frame) -> List[Tuple]:
        active_objects = [[]] * len(self.objects)
        alert_ids = (
            np.unique(vars_data[vars_data[:, BatchDataResolver.FRAME_ID] == frame, BatchDataResolver.ID])
            .astype(int)
            .tolist()
        )
        for idx, s in enumerate(self.ids_mapping):
            active_objects[idx] = self.ids_mapping[s].intersection(alert_ids)
        is_eligible = all([len(x) > 0 for x in active_objects])
        if is_eligible:
            return list(itertools.product(*active_objects))
        return []

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:

        alert_candidates = []

        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            return alert_candidates
        for ind, frame in enumerate(images):
            items = self.check_proximity_criteria(vars_data, ind)
            ts = frame.timestamp
            for item in items:
                sort_item = tuple(int(i) for i in sorted(item))
                if sort_item in self.trackers:
                    self.trackers[sort_item]["last_ts"] = ts
                else:
                    self.trackers[sort_item] = {"first_ts": ts, "last_ts": ts, "alertSent": False}

        for items in self.trackers:
            if self.trackers[items]["alertSent"]:
                continue
            last_ts = self.trackers[items]["last_ts"]
            delta_t = last_ts - self.trackers[items]["first_ts"]
            if delta_t >= self.proximity_duration_ms:
                candidate = self.build_alert_candidate(last_ts, list(items), vars_data)
                time_data = vars_data[vars_data[:, BatchDataResolver.TIMESTAMP] == last_ts, :]
                idx = int(time_data[-1, BatchDataResolver.FRAME_ID])
                for item in items:
                    ent_pos = time_data[time_data[:, BatchDataResolver.ID] == item, BatchDataResolver.POS]
                    candidate.crops.append(crop_image(images[idx].frame, ent_pos))
                alert_candidates.append(candidate)
                self.trackers[items]["alertSent"] = True

        return alert_candidates

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)
        alert_info.alertMessage = self.alert_message
        alert_info.extra.update(self.extra_field)
        return alert_info

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        is_checked = super().check_alert_candidate(candidate, force)
        if is_checked:
            validate_obj = dict.fromkeys(self.ids_mapping.keys(), False)
            for ent in candidate.ent_ids:
                ent_data: ActiveEntityData = self.entity_db.get_entity(ent)
                if ent_data is not None:
                    validate_obj[ent_data.object_id] = True
            if all(list(validate_obj.values())):
                logger.info(f"PROXIMITY: {candidate.ent_ids} with validation {validate_obj}")
                return True
            else:
                logger.warning(
                    f"PROXIMITY: {candidate.ent_ids} doesn't seems to be containing the mix of required object id"
                )
                return False
        return is_checked
