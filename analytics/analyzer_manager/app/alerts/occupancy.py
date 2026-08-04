import random
from argparse import Namespace
from collections import Counter
from typing import Optional, Dict, List, Set

import numpy as np

from general import apply_from_dict
from general.analyzer_general import logger, roi_gen
from general.core import BatchDataResolver, AnalyticImage, TrackingType, AlertCategory, AlertInfo, AlertRouting
from .base_alerts import ObjectAlert, AlertCandidate, duration_unit_to_sec


def check_zone(ids_to_check, batch_data, roi_filter, marked_index, batch_size):
    vars_data = batch_data.query(BatchDataResolver.ID, ids_to_check)
    n_vars = len(vars_data)
    if roi_filter:
        locations = vars_data[:, BatchDataResolver.LOCATION].astype(int)
        in_roi = [i for i in range(n_vars) if locations[i] in marked_index]
        vars_data = vars_data[in_roi]

    # bincount = np.bincount(vars_data[:, BatchDataResolver.FRAME_ID].astype(np.uint8), minlength=batch_size)
    bin_ids = [np.array([]) for _ in range(batch_size)]
    if len(vars_data):
        for i in range(batch_size):
            bin_ids[i] = vars_data[vars_data[:, BatchDataResolver.FRAME_ID] == i, BatchDataResolver.ID].astype(int)
    return bin_ids  # bincount, bin_ids


class OccupancyAlert(ObjectAlert):
    type_name = "occupancy"
    duration: int = 0

    def __init__(self, alert_dict: Dict, context):
        super(OccupancyAlert, self).__init__(alert_dict, context)
        self.set_flow_values()

        self.zone_occupancy = {}
        self.globalCount = 0
        self.alert_on_tracked_only = True
        self.objects_str = ", ".join(self.objects)

        if not self.mergedZones and "zones" in alert_dict and len(alert_dict["zones"]) and (self.legacy):
            zones = alert_dict["zones"]
        elif "zones" in self.selectedCamera and len(self.selectedCamera["zones"]) and not (self.legacy):

            zones = self.selectedCamera["zones"]
        else:
            zones = {}

        for key in zones:
            if "markedIdx" in zones[key]:
                zone_id = apply_from_dict("name", zones[key], "")
                if not (len(zone_id)):
                    if not (len(self.zones)) in self.zones:
                        zone_id = len(self.zones)
                    else:
                        zone_id = random.randint(0, 1024)

                # check if id is a string and a number
                if isinstance(zone_id, str) and zone_id.isdigit():
                    zone_id = int(zone_id)

                if zone_id in self.zones:
                    logger.error(f"Cant init zone: {zones[key]}, already exist: {self.zones}")

                else:
                    marked_idx = apply_from_dict("markedIdx", zones[key], None)
                    roiFilter = marked_idx is not None and len(marked_idx) > 0
                    roi = None if not roiFilter else roi_gen(marked_idx)
                    self.zones[zone_id] = Namespace(roiFilter=roiFilter, roi=roi, marked_idx=marked_idx)
        self.alerted_sets: List[Set] = []
        if self.positive_alert:
            self.alert_message = f"More than {self.count} {self.objects_str}s are detected"
        else:
            self.alert_message = f"No {self.objects_str} was detected"
        if len(self.zones):
            self.active_start_time = {k: None for k in self.zones}
        else:
            self.active_start_time = None

    def set_flow_values(self):
        if self.formValue:
            self.positive_alert = True
            self.count = self.apply_from_dict("amount", self.formValue, 1)
            self.mergedZones = False
            duration = self.apply_from_dict("duration", self.formValue, 0)
            units = self.apply_from_dict("durationUnit", self.formValue, 0)
            self.duration = duration * duration_unit_to_sec[units] * 1000  # convert to ms

        # support absense alert
        if (self.category == AlertCategory.Tracking) and (self.flow == TrackingType.Absense.value):
            self.positive_alert = False
            self.count = 0

    def is_criteria_met(self, count):
        return (self.positive_alert and count > self.count) or (not self.positive_alert and count <= self.count)

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:

        alert_candidates = []
        batch_data = self._filter_person_in_car(batch_data)
        alert_ids = batch_data.unique(batch_data.ID, batch_data.CLASS, self.object_ids).astype(int).tolist()
        if self.alert_on_tracked_only and -1 in alert_ids:
            alert_ids.remove(-1)

        if (len(alert_ids) == 0) and self.positive_alert:
            return alert_candidates

        # if we got here we need only to compare type
        batch_size = len(images)
        if len(self.zones):
            zone_ents = {}
            zone_counts_mat = []
            for zone_id in self.zones:
                zone_ents[zone_id] = check_zone(
                    alert_ids, batch_data, self.zones[zone_id].roiFilter, self.zones[zone_id].marked_idx, batch_size
                )
                zone_counts_mat.append([len(ents) for ents in zone_ents[zone_id]])
            zone_counts_mat = np.array(zone_counts_mat)
            sel_z, sel_i = np.where(np.vectorize(self.is_criteria_met)(zone_counts_mat))
            inactive_zones = self.active_start_time.keys() - set(sel_z)
            for k in inactive_zones:
                self.active_start_time[k] = None

            if len(sel_z) > 0:

                sel_count = zone_counts_mat[sel_z, sel_i]

                for idx in range(len(sel_count)):
                    zone_id = sel_z[idx]
                    frame_id = sel_i[idx]
                    if self.active_start_time[zone_id] is None:
                        self.active_start_time[zone_id] = images[frame_id].timestamp
                    duration_met = self.duration <= (images[idx].timestamp - self.active_start_time[zone_id])

                    if (
                        len(set(zone_ents[zone_id][frame_id]).difference(self.alerted_ids)) > 0
                        or not (self.positive_alert)
                    ) and duration_met:
                        zones_occupancy = {}
                        alert_ents = set()
                        for zone_id in zone_ents:
                            zones_occupancy[zone_id] = zone_ents[zone_id][frame_id]
                            alert_ents.update(zone_ents[zone_id][frame_id].tolist())
                        alert_data = zones_occupancy
                        candidate = self.build_alert_candidate(
                            images[frame_id].timestamp, list(alert_ents), extra={"data": alert_data}
                        )
                        alert_candidates.append(candidate)
                        self.alerted_ids.update(alert_ents)
        else:
            frame_ents = check_zone(alert_ids, batch_data, self.roiFilter, self.marked_idx, batch_size)
            active_idx = [idx for idx in range(batch_size) if self.is_criteria_met(len(frame_ents[idx]))]
            if len(active_idx) == 0:
                self.active_start_time = None
            elif self.active_start_time is None:
                self.active_start_time = images[active_idx[0]].timestamp
            for idx in active_idx:
                duration_met = self.duration <= (images[idx].timestamp - self.active_start_time)
                if duration_met and (
                    not self.positive_alert or len(set(frame_ents[idx]).difference(self.alerted_ids)) > 0
                ):
                    # alert_message = f"Detected {ents[active_idx[0]]} {self.objects_str[:-2]}"
                    alert_data = frame_ents[active_idx[0]].tolist()
                    candidate = self.build_alert_candidate(images[active_idx[0]].timestamp, list(alert_data))
                    alert_candidates.append(candidate)
                    self.alerted_ids.update(frame_ents[idx])

        return alert_candidates

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)
        if candidate.extra is None:
            alert_info.alertData = candidate.ent_ids
            total_ents = len(candidate.ent_ids)
        else:
            zones_occupancy = {}
            total_ents = 0
            for zone in candidate.extra["data"]:
                zones_occupancy[zone] = len(candidate.extra["data"][zone])
                total_ents += zones_occupancy[zone]
            alert_info.alertData = zones_occupancy
        alert_info.extra.update({"count": total_ents})
        alert_info.alertMessage = self.alert_message
        self.alerted_sets.append(set(candidate.ent_ids))
        return alert_info

    def on_entities_removed(self, ent_ids: List[int]):
        super().on_entities_removed(ent_ids)
        if len(self.alerted_sets) > 0:
            for ent in ent_ids:
                self.alerted_sets = [s for s in self.alerted_sets if ent not in s]

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        if not self.positive_alert:
            return True
        placeholder_cand = AlertCandidate("", 0)
        filtered_ents = set()
        undefined_ents = set()
        for ent_id in candidate.ent_ids:
            placeholder_cand.ent_ids = [ent_id]
            ent_ok = super().check_alert_candidate(placeholder_cand)
            if ent_ok:
                filtered_ents.add(ent_id)
            elif ent_ok is None:
                undefined_ents.add(ent_id)
        undefined_ents.update(filtered_ents)
        if candidate.extra is not None:
            # zones mode, need to check every zone to see if criteria met
            zones_occupancy = candidate.extra.get("data", {})
            is_checked_val = 0
            possible_values = [False, False if force else None, True]
            for zone in zones_occupancy:
                filtered_zone = filtered_ents.intersection(zones_occupancy[zone])
                if self.is_criteria_met(len(filtered_zone)):
                    zones_occupancy[zone] = filtered_zone
                    is_checked_val = 2
                else:
                    undefined_zone = undefined_ents.intersection(zones_occupancy[zone])
                    if self.is_criteria_met(len(undefined_zone)):
                        is_checked_val = max(is_checked_val, 1)
                    zones_occupancy[zone] = undefined_zone
            if possible_values[is_checked_val]:
                candidate.ent_ids = list(filtered_ents)
                return filtered_ents not in self.alerted_sets
            else:
                candidate.ent_ids = list(undefined_ents)
            return possible_values[is_checked_val]
        else:
            if self.is_criteria_met(len(filtered_ents)):
                candidate.ent_ids = list(filtered_ents)
                return filtered_ents not in self.alerted_sets
            elif not force and self.is_criteria_met(len(undefined_ents)):
                # not certain -> wait for more information
                candidate.ent_ids = list(undefined_ents)
                return None
        return False


class AbsenceAlert(OccupancyAlert):
    def __init__(self, alert_dict: Dict, context):
        super().__init__(alert_dict, context)
        self.irrelevant_ids = {-1}

        # convert to regular zone without roi filter
        if len(self.zones) == 0:
            self.zones["full"] = Namespace(roiFilter=False, roi=None, marked_idx=None)
            self.active_start_time = {"full": None}

        # track which zones got activated
        self.is_alerted = {z: False for z in self.zones}
        self.unknown_ids = set()
        self.last_unknown_check_ts = 0
        self.last_unknown_check_interval = 1000

    def set_flow_values(self):
        super().set_flow_values()
        self.positive_alert = False
        self.count = 0

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []
        batch_data = self._filter_person_in_car(batch_data)
        alert_ids = set(batch_data.unique(batch_data.ID, batch_data.CLASS, self.object_ids).astype(int).tolist())
        alert_ids = list(alert_ids - self.irrelevant_ids)
        batch_size = len(images)
        if (
            len(alert_ids) > 0
            and self.unknown_ids
            and self.last_unknown_check_ts + self.last_unknown_check_interval < images[-1].timestamp
        ):
            self.on_entities_update(list(self.unknown_ids))
            self.last_unknown_check_ts = images[-1].timestamp
            alert_ids = list(set(alert_ids) - self.irrelevant_ids)
        for zone_id in self.zones:
            if len(alert_ids) == 0:
                active_idx = list(range(batch_size))
                zone_ents = []
            else:
                zone_ents = check_zone(
                    alert_ids, batch_data, self.zones[zone_id].roiFilter, self.zones[zone_id].marked_idx, batch_size
                )
                active_idx = [idx for idx in range(batch_size) if len(set(zone_ents[idx]) - self.unknown_ids) == 0]
            if len(active_idx) <= batch_size // 2:
                self.active_start_time[zone_id] = None
                self.is_alerted[zone_id] = False
                continue
            else:
                if self.active_start_time[zone_id] is None:
                    self.active_start_time[zone_id] = images[active_idx[-1]].timestamp
                # no need to raise another alert if nothing occurred in the zone and its already alerted
                elif not self.is_alerted[zone_id]:
                    duration_met = self.duration <= (images[active_idx[-1]].timestamp - self.active_start_time[zone_id])
                    if duration_met:
                        alert_ids = set([int(item) for sublist in zone_ents for item in sublist])
                        candidate = self.build_alert_candidate(images[active_idx[-1]].timestamp, list(alert_ids))
                        alert_candidates.append(candidate)
                        self.is_alerted[zone_id] = True
        return alert_candidates

    def on_entities_update(self, ent_ids: List[int]):
        for ent_id in ent_ids:
            placeholder = AlertCandidate("", 0, [ent_id])
            ent_data = self.entity_db.get_entity(ent_id)
            if ent_data is None or ent_data.object_id not in self.object_ids:
                is_checked = False
            else:
                is_checked = super(OccupancyAlert, self).check_alert_candidate(placeholder)
            if is_checked is None:
                if ent_id not in self.unknown_ids:
                    self.unknown_ids.add(ent_id)
                    self.l1_requested_ents.add(ent_id)
            elif not is_checked:
                self.irrelevant_ids.add(ent_id)
                self.unknown_ids.discard(ent_id)
            else:  # meet the criteria so its relevant
                self.irrelevant_ids.discard(ent_id)
                self.unknown_ids.discard(ent_id)

    def on_entities_removed(self, ent_ids: List[int]):
        for ent_id in ent_ids:
            self.irrelevant_ids.discard(ent_id)
            self.unknown_ids.discard(ent_id)

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        # unknown ents are forwarded in the alert candidates to be checked here
        unknown_ents = candidate.ent_ids
        if unknown_ents:
            placeholder_cand = AlertCandidate("", 0)
            for eid in unknown_ents:
                placeholder_cand.ent_ids = [eid]
                is_checked = super(OccupancyAlert, self).check_alert_candidate(placeholder_cand, force)
                # if checked it means that the ent is valid and fit the criteria, so it should be ignored
                if is_checked:
                    self.irrelevant_ids.discard(eid)
                    return False
                elif force:
                    is_checked = False

                if is_checked is not None and not is_checked:
                    candidate.ent_ids.remove(eid)
        if candidate.ent_ids:
            return None  # still unknown ents exists
        return True


class RegionCountingAlert(OccupancyAlert):

    min_obj_for_vcc = 10
    sample_period_ms = 5 * 60 * 1000  # 5 minutes
    increased_detector_max_det = 300
    positive_alert = True
    count = np.inf
    mergedZones = False
    type_name = "regionCounting"
    polygons: List

    def __init__(self, alert_dict: Dict, context):
        super(RegionCountingAlert, self).__init__(alert_dict, context)
        alert_config = self.context.get_config().get("alertConfig", {}).get(self.type_name, {})
        if alert_config:
            self.min_obj_for_vcc = alert_config.get("min_obj_for_vcc", self.min_obj_for_vcc)
            self.increased_detector_max_det = alert_config.get(
                "increased_detector_max_det", self.increased_detector_max_det
            )
        self.sample_period_ms = (
            self.context.get_config()["countingPolicy"].get("regionUpdateRate", 0) * 1000 or self.sample_period_ms
        )

        super().__init__(alert_dict, context)
        self.last_sample_period = -1
        if self.min_obj_for_vcc >= 0:
            self.routing = AlertRouting.MANUAL

        # unified the zones to one zone
        self.zones = {}

        # increase the detector capacity
        if self.increased_detector_max_det > 0:
            self.context.context.detector.args.max_det = self.increased_detector_max_det

        if self.object_ids is None or len(self.object_ids) == 0:
            self.object_ids = self.context.get_class_handler().object_ids
            for obj_id in self.object_ids:
                self.l1_required_type[obj_id] = None

    def set_flow_values(self):
        self.polygons = self.formValue.get("polygons", [])

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []
        period = images[-1].timestamp // self.sample_period_ms
        if period > self.last_sample_period:
            count_per_obj = {}
            curr_ents = set()
            curr_size = 0
            for obj_id in self.object_ids:
                alert_ids = batch_data.unique(batch_data.ID, batch_data.CLASS, obj_id).astype(int).tolist()
                if self.alert_on_tracked_only and -1 in alert_ids:
                    alert_ids.remove(-1)
                if alert_ids:
                    batch_size = len(images)
                    frame_ents = check_zone(alert_ids, batch_data, self.roiFilter, self.marked_idx, batch_size)
                    ent_counts = Counter(int(eid) for frame in frame_ents for eid in frame)
                    threshold = batch_size / 2
                    consistent_ents = [eid for eid, cnt in ent_counts.items() if cnt >= threshold]
                else:
                    consistent_ents = []

                # avoid counting the same object
                curr_ents.update(consistent_ents)
                count_per_obj[obj_id] = len(curr_ents) - curr_size
                curr_size = len(curr_ents)
            ts = period * self.sample_period_ms
            candidate = self.build_alert_candidate(ts, list(curr_ents), extra={"regionCount": count_per_obj})
            if count_per_obj.get(self.person_value, 0) >= self.min_obj_for_vcc:
                candidate.validation_images.append(images[-1].frame)
                candidate.extra["polygons"] = self.polygons
            alert_candidates.append(candidate)
            self.last_sample_period = period
        return alert_candidates

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super(OccupancyAlert, self).build_alert_info(candidate)
        alert_info.extra = candidate.extra
        if candidate.extra["regionCount"].get(self.person_value, 0) < self.min_obj_for_vcc:
            alert_info.routing = AlertRouting.NO_ROUTING.value
        return alert_info

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        return True
