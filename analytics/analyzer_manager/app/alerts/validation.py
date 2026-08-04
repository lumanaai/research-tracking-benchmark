## THIS ALERT IS DEPRECATED AND WILL BE REMOVED IN FUTURE RELEASES. PLEASE USE THE NEW ALERTS FRAMEWORK INSTEAD @integrated.py.

from .base_alerts import ObjectAlert


class ValidationAlert(ObjectAlert):
    """Deprecated: routed to DHOIntegratedAlert/DFOIntegratedAlert via alerts.py generate_alert()."""
    type_name = "validationAlert"


# from enum import Enum
# from typing import Dict, List

# import numpy as np

# from general.core import BatchDataResolver, AnalyticImage, AlertInfo
# from .base_alerts import ObjectAlert, AlertCandidate, duration_unit_to_sec


# class AlarmType(int, Enum):
#     DoorHeldOpen = 0
#     DoorForcedOpen = 1
#     UNKNOWN = 99


# class ValidationAlert(ObjectAlert):
#     type_name = "validationAlert"
#     hysteresis_threshold_ms = 100000
#     motion_filter = True
#     history_filter = True

#     def __init__(self, alert_dict: Dict, context):
#         # Use proper class for super() and initialize
#         super(ValidationAlert, self).__init__(alert_dict, context)
#         self.alert_on_tracked_only = True
#         self.has_local_actions = True
#         self.alerted_ids = {}
#         self.activation_data = []
#         self.hold_in_sec = 5
#         self.alert_activated = False
#         class_handler = self.context.get_class_handler()

#         # for now we look for people
#         self.object_ids = [
#             class_handler.person_value,
#             # class_handler.vehicle_value,
#         ]
#         self.objects = [
#             class_handler.object_int_to_str(class_handler.person_value),
#             # class_handler.object_int_to_str(class_handler.vehicle_value),
#         ]

#         self.filters = {
#             class_handler.person_value: {},
#             # class_handler.vehicle_value: {},
#         }
#         self.filtersDisabled = {
#             class_handler.person_value: True,
#             # class_handler.vehicle_value: False,
#         }
#         self.strict_filters = {
#             class_handler.person_value: True,
#             # class_handler.vehicle_value: True,
#         }
#         self.recognition_filters = {
#             class_handler.person_value: {"enabled": False},
#             # class_handler.vehicle_value: {"enabled": False},
#         }
#         self.l1_required_type = {
#             class_handler.person_value: None,
#             # class_handler.vehicle_value: None,
#         }
#         self.l1_required = False
#         analytic_config = self.context.get_config()
#         if "alertConfig" in analytic_config:
#             self.hysteresis_threshold_ms = (
#                 analytic_config["alertConfig"]
#                 .get("objectAppearance", {})
#                 .get("hysteresis", self.hysteresis_threshold_ms)
#             )
#         self.max_hold = 30000  # leave time to overcome blacklist\filter issues
#         self.alarm_type = int(self.formValue.get("alarmType", AlarmType.UNKNOWN.value))
#         self._parse_duration()

#     def activateAlert(self, data):
#         self.activation_data.append(data)
#         self.alert_activated = True

#     def _parse_duration(self):
#         count = self.hold_in_sec
#         units = 0
#         if self.formValue:
#             self.positive_alert = True
#             count = self.apply_from_dict("duration", self.formValue, 0)
#             units = self.apply_from_dict("durationUnit", self.formValue, 0)
#             self.hold_in_sec = count * duration_unit_to_sec[units]

#     def clearAlertData(self, extra_fields: dict, person_detected: bool):
#         if person_detected:
#             if self.alarm_type == AlarmType.DoorHeldOpen:
#                 extra_fields["clearEvent"] = True
#                 extra_fields["eventNote"] = f"Alert cleared, person is holding the door"
#                 extra_fields["clearDuration"] = self.hold_in_sec
#             elif self.alarm_type == AlarmType.DoorForcedOpen:
#                 extra_fields["clearEvent"] = False
#                 extra_fields["eventNote"] = f"Alert is active, person detected entering the door"
#                 extra_fields["clearDuration"] = self.hold_in_sec
#             else:
#                 extra_fields["eventNote"] = f"Unsupported alarm type, not clearing alert"
#                 extra_fields["clearEvent"] = False
#                 extra_fields["clearDuration"] = self.hold_in_sec
#         else:
#             if self.alarm_type == AlarmType.DoorHeldOpen:
#                 extra_fields["clearEvent"] = False
#                 extra_fields["eventNote"] = f"Alert is active, door is being held open"
#                 extra_fields["clearDuration"] = self.hold_in_sec
#             elif self.alarm_type == AlarmType.DoorForcedOpen:
#                 extra_fields["clearEvent"] = True
#                 extra_fields["eventNote"] = f"Alert is cleared, no person detected entering the door"
#                 extra_fields["clearDuration"] = self.hold_in_sec
#             else:
#                 extra_fields["eventNote"] = f"Unsupported alarm type, not clearing alert"
#                 extra_fields["clearEvent"] = False
#                 extra_fields["clearDuration"] = self.hold_in_sec
#         return extra_fields

#     def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
#         alert_candidates = []

#         if self.alert_activated:
#             timestamps = np.array([img.timestamp for img in images])
#             for activation_data in self.activation_data:

#                 vars_data = self.get_batch_candidates_data(batch_data)
#                 ids_to_report = []
#                 extra_fields = {"extra_fields": activation_data}

#                 if len(vars_data) > 0:
#                     alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()

#                     for id_ in alert_ids:
#                         batch_ts = vars_data[vars_data[:, BatchDataResolver.ID] == id_, BatchDataResolver.TIMESTAMP]
#                         last_seen = self.alerted_ids.get(id_, 0)
#                         # If configured to fire on presence, allow alerts when either seen first time
#                         # or hysteresis period has elapsed since last alert. Otherwise only alert
#                         # when object is seen for the first time (last_seen == 0).
#                         try:
#                             if last_seen == 0 or batch_ts[-1] >= last_seen + self.hysteresis_threshold_ms:
#                                 ids_to_report.append(id_)
#                         except Exception:
#                             # defensive: if timestamp arithmetic fails, fallback to first-seen behaviour
#                             if last_seen == 0:
#                                 ids_to_report.append(id_)
#                         else:
#                             if last_seen == 0:
#                                 ids_to_report.append(id_)
#                         self.alerted_ids[id_] = batch_ts[-1].astype(int)

#                 trigger_ts = activation_data.get("timestamp", images[-1].timestamp)
#                 ts_idx = np.argmin(np.abs(timestamps - trigger_ts))
#                 ts = timestamps[ts_idx]

#                 # update alert info in case of active
#                 if len(ids_to_report) > 0:
#                     objects_active = ids_to_report[0]
#                     idx = np.where(vars_data[:, BatchDataResolver.ID] == objects_active)[0][0]
#                     extra_fields = self.clearAlertData(extra_fields, person_detected=True)

#                     alert_candidates.append(
#                         self.build_alert_candidate(
#                             ts, objects_active, vars_data[idx, :], images=images, extra={"extra_fields": extra_fields}
#                         )
#                     )

#                 else:
#                     extra_fields = self.clearAlertData(extra_fields, person_detected=False)
#                     alert_candidates.append(self.build_alert_candidate(ts, extra={"extra_fields": extra_fields}))
#             self.alert_activated = False
#             self.activation_data = []

#         return alert_candidates

#     def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
#         alert_info = super().build_alert_info(candidate)
#         extra_fields = candidate.extra.get("extra_fields", {}) if candidate.extra else {}
#         alert_info.alertMessage = extra_fields.get("eventNote", None)
#         alert_info.clearEvent = extra_fields.get("clearEvent", None)
#         return alert_info

#     def on_entities_removed(self, ent_ids: List[int]):
#         for ent in ent_ids:
#             self.active_ids.discard(ent)
#             if ent in self.alerted_ids:
#                 del self.alerted_ids[ent]
