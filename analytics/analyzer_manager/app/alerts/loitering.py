from typing import Dict, List
import numpy as np

from general.analyzer_general import logger
from general.core import (
    BatchDataResolver,
    AnalyticImage,
    AlertInfo,
    AlertCategory,
    ProtectedGearWearOptions,
    AlertRouting,
    MotionData,
    ProtectedGearType,
)
from .base_alerts import ObjectAlert, AlertCandidate, duration_unit_to_sec, duration_unit_to_str


class LoiteringAlert(ObjectAlert):
    type_name = "loitering"
    motion_filter = True
    history_filter = True
    def_time_window_factor = 10
    min_size_ambient_motion = 3
    loitering_ambient_motion_filter = True

    def __init__(self, alert_dict: Dict, context):
        super(LoiteringAlert, self).__init__(alert_dict, context)
        self.set_flow_values()
        self.threshold *= 1000  # sec to ms
        self.trackers = {}
        self.max_hold = 30000  # leave time to overcome blacklist\filter issues

        # get the alert configuration if exists
        cfg = self.alerts_config.get(self.type_name, {})
        if self.ambient_motion_filter: # override motion filter since not relevant to loitering
            self.loitering_ambient_motion_filter = True
            self.ambient_motion_filter = False
        time_window_factor = cfg.get("time_window_factor", self.def_time_window_factor)
        self.active_timespan = self._calculate_active_timespan(time_window_factor)

    def _calculate_active_timespan(self, time_window_factor):
        # at least the required time but no more than extra 1 min
        return max(self.threshold, min(self.threshold * time_window_factor, self.threshold + 60000))

    def set_flow_values(self):
        count, units_str = self._parse_duration()
        self.alert_message = f"is loitering for more than {count} {units_str}"

    def _parse_duration(self):
        count = self.threshold
        units = 0
        if self.formValue:
            self.positive_alert = True
            count = self.apply_from_dict("duration", self.formValue, 0)
            units = self.apply_from_dict("durationUnit", self.formValue, 0)
            self.threshold = count * duration_unit_to_sec[units]
        units_str = duration_unit_to_str[units]
        return count, units_str

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        object_name = self.context.get_class_handler().object_int_to_str(alert_info.object_id)
        alert_info.alertMessage = f"{object_name} {self.alert_message}".capitalize()

    def calc_dwell(self, idx):
        return self.trackers[idx]["last_ts"] - self.trackers[idx]["first_ts"]

    def on_entities_removed(self, ent_ids: List[int]):
        super(LoiteringAlert, self).on_entities_removed(ent_ids)
        for ent_id in ent_ids:
            if ent_id in self.trackers:
                del self.trackers[ent_id]

    def reset_history(self, active_obj):
        if active_obj in self.trackers and not self.trackers[active_obj]["alertSent"]:
            del self.trackers[active_obj]

    def is_active_batch(self, images: List[AnalyticImage], motion_data: MotionData, batch_data) -> List[AlertCandidate]:
        alert_info = []

        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            for active_obj in self.active_ids:
                self.reset_history(active_obj)
            return alert_info

        alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()
        trackers = list(self.trackers.keys())
        new_ids = set(alert_ids).difference(trackers)
        active_ids = new_ids.union(trackers)
        for active_obj in active_ids:

            var_data = vars_data[vars_data[:, BatchDataResolver.ID] == active_obj, :]
            if len(var_data) == 0:
                self.reset_history(active_obj)
                continue
            ts = var_data[:, BatchDataResolver.TIMESTAMP]

            if active_obj in new_ids:
                self.trackers[active_obj] = {
                    "first_ts": ts[0],
                    "last_ts": ts[-1],
                    "alertSent": False,
                    "max_motion": 100 * int(not self.loitering_ambient_motion_filter),
                    "max_area": 0,
                }
            else:
                self.trackers[active_obj]["last_ts"] = ts[-1]
                self.trackers[active_obj]["max_area"] = max(self.trackers[active_obj]["max_area"], np.max(var_data[:, BatchDataResolver.AREA]))
            if self.trackers[active_obj]["alertSent"]:
                continue
            self.motion_update(active_obj, np.clip(var_data, 0, 1), motion_data)
            dwell = self.calc_dwell(active_obj)
            if dwell > self.threshold:
                if ( # ambient motion only if object is large enough
                    self.trackers[active_obj]["max_motion"] >= self.ambient_motion_th
                    or self.trackers[active_obj]["max_area"] < self.min_size_ambient_motion
                ):
                    self.trackers[active_obj]["alertSent"] = True
                    extra = {"extra_fields": {"duration": dwell}}
                    candidate = self.build_alert_candidate(
                        self.trackers[active_obj]["last_ts"], active_obj, var_data, extra=extra, images=images
                    )
                    alert_info.append(candidate)
                elif dwell > self.active_timespan:  # it's been enough time without any motion, indicating FP
                    self.trackers[active_obj]["alertSent"] = True
                    logger.info(
                        f"{self.type_name} alert for {active_obj} was filtered due to lack of motion in active timespan"
                    )
        return alert_info

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super().build_alert_info(candidate)
        loitering_period = round(candidate.extra["extra_fields"]["duration"] / 1000, 2)
        alert_info.alertData = float(loitering_period)
        return alert_info

    def motion_update(self, active_obj: int, var_data: np.array, motion_data: MotionData):
        if self.loitering_ambient_motion_filter:
            curr_motion = self.trackers[active_obj]["max_motion"]
            if curr_motion < self.ambient_motion_th < motion_data.max:
                tl = np.floor(np.min(var_data[:, BatchDataResolver.POS[:2]], axis=0) * 32).astype(int)
                br = np.ceil(np.max(var_data[:, BatchDataResolver.POS[2:]], axis=0) * 32).astype(int)
                motion_index = np.max(motion_data.unified[tl[1] : br[1], tl[0] : br[0]])
                if motion_index > curr_motion:
                    self.trackers[active_obj]["max_motion"] = motion_index

# DEPRECATED - use PpeAlert from protective_equipment instead
# class PpeAlert(LoiteringAlert):
#     default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE
#     motion_filter = False
#     history_filter = False
#     ambient_motion_filter = False

#     def set_flow_values(self):
#         count, units_str = self._parse_duration()
#         gear_type = self.formValue.get("gear", ProtectedGearType.hard_hat.value)
#         attr_name = "helmet" if gear_type == ProtectedGearType.hard_hat.value else "vest"
#         if gear_type != ProtectedGearType.hard_hat.value:
#             self.routing = AlertRouting.NO_ROUTING

#         if (
#             self.formValue.get("wear", ProtectedGearWearOptions.not_wearing.value)
#             == ProtectedGearWearOptions.not_wearing.value
#         ):
#             self.alert_message = f"Workers Don’t wear a safety {attr_name} for more than {count} {units_str}"
#             self.special_filter = "not wearing"
#         else:
#             self.alert_message = f"Workers wear a safety {attr_name} for more than {count} {units_str}"
#             self.special_filter = "wearing"

#     def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
#         alert_info.alertMessage = self.alert_message


class ZoneProtectionAlert(LoiteringAlert):
    type_name = "zoneProtection"
    motion_filter = True
    history_filter = True
    ambient_motion_filter = True
    max_ambient_motion_th = 50
    def_ambient_motion_th = 10
    def_time_window_factor = 3
    max_time_window_factor = 10
    def_motion_filter_th = 2
    max_motion_filter_th = 5

    def __init__(self, alert_dict: Dict, context):
        super(ZoneProtectionAlert, self).__init__(alert_dict, context)

        cfg = self.alerts_config.get(self.type_name, {})
        sensitivity = cfg.get("sensitivity", 50)

        self.set_sensitivity(sensitivity)

    def set_sensitivity(self, sensitivity):
        reference_sensitivity = 50
        factor = (sensitivity - reference_sensitivity) / reference_sensitivity
        time_window_factor = self.def_time_window_factor
        if factor < 0:  # less sensitive
            factor = -factor
            self.ambient_motion_th += (self.max_ambient_motion_th - self.def_ambient_motion_th) * factor
            time_window_factor = (self.def_time_window_factor - 1) * factor + 1
            self.motion_filter_th += (self.max_motion_filter_th - self.def_motion_filter_th) * factor
            self.motion_filter = True
        else:
            self.ambient_motion_th = self.def_ambient_motion_th * (1 - factor)
            time_window_factor += (self.max_time_window_factor - self.def_time_window_factor) * -factor
            if sensitivity > 75:
                self.motion_filter = False
            else:
                self.motion_filter = True
                self.motion_filter_th = self.def_motion_filter_th if reference_sensitivity == 50 else 1
        self.active_timespan = self._calculate_active_timespan(time_window_factor)

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        if self.category == AlertCategory.CustomizedCapabilities.value:
            alert_info.alertMessage = self.alert_message
        else:
            object_name = self.context.get_class_handler().object_int_to_str(alert_info.object_id)
            alert_info.alertMessage = f"{object_name} {self.alert_message}".capitalize()

    def set_flow_values(self):
        count, units_str = self._parse_duration()
        self.alert_message = f"is loitering for more than {count} {units_str} in a restricted area"
