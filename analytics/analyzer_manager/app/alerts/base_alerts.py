import json
import re as re
from collections import deque
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple

import cv2
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious  # noqa

from general import apply_from_dict
from general.analyzer_general import roi_gen, SUPPORTED_FILTERS, logger, ROI_SHAPE, encode_debug_data
from general.cloud_converter import convert_to_alert_data
from general.core import (
    AlertType,
    AlertStatus,
    AlertsAction,
    AlertsConfidence,
    BatchDataResolver,
    AlertCategory,
    SafetyType,
    CustomizedCapabilitiesType,
    ProtectedGearWearOptions,
    AnalyticImage,
    alert_mapping,
    AlertInfo,
    AlertRouting,
    MotionData,
    ProtectedGearType,
    HttpMethods,
)
from general.db_handler import CustomObjectData
from general.entity import ActiveEntityData
from general.img_utils import crop_image
from .schedule import Schedule, Policy, ZonePolicy


def find_value_by_prefix(dictionary: dict, prefix: str):
    for key in dictionary.keys():
        if key.startswith(prefix):
            return dictionary[key]
    return None  # Return None if no key matches the prefix


def find_recognition_lists(dictionary: dict, prefix: str):
    """Find subject list and groups list for a given prefix (include/exclude).

    Separates keys like 'includePeople' (subjects) from 'includePersonGroups' (groups)
    by checking whether 'Group' appears in the key name.

    Returns:
        (subjects, groups) — either may be None if not present.
    """
    subjects = None
    groups = None
    for key, value in dictionary.items():
        if key.startswith(prefix) and value:
            if "Group" in key or "group" in key:
                groups = value
            else:
                subjects = value
    return subjects, groups


@dataclass
class AlertCandidate:
    alert_id: str
    timestamp: int
    # object_id: int = -1
    ent_ids: List[int] = field(default_factory=list)
    var_data: Optional[np.array] = None
    extra: Optional[dict] = None
    crops: List[np.array] = field(default_factory=list)
    validation_images: List[np.array] = field(default_factory=list)
    last_valid_check: int = None

    def __post_init__(self):
        self.last_valid_check = self.timestamp


class BaseAlert(object):
    type_name: str = "base"
    alert_message = ""
    debug_data = ""
    object_based_alert = False
    is_store_snapshot = False
    is_store_thumb_for_local_action = False
    default_routing = AlertRouting.NO_ROUTING
    special_filter: str = None
    routing_crop_size = 512
    max_hold: int = 5000  # default time to hold candidate until we have definitive L1 results
    max_hold_if_strict: int = 10000  # default time to hold candidate until we have definitive L1 results
    activate_ddata = False
    is_db_valid = True

    def __init__(self, alert_dict: Dict, context):
        self.context = context
        self.enabled: bool = apply_from_dict("enabled", alert_dict, True)
        self.event_id: str = apply_from_dict("_id", alert_dict, "")
        self.variableAlert: bool = apply_from_dict("variableAlert", alert_dict, False)
        self.active_ids = set()
        analytic_config = self.context.get_config()
        self.alerts_config = apply_from_dict("alertConfig", analytic_config, {})
        class_handler = self.context.get_class_handler()
        self.vehicle_value = class_handler.vehicle_value
        self.person_value = class_handler.person_value
        self.object_ids = []
        self.objects = []
        self.custom_objects = {}

        # local action related - if needed, store additional thumbnail for local action
        self.has_local_actions = apply_from_dict("hasLocalActions", alert_dict, False)
        self.accountId = apply_from_dict("accountId", alert_dict, None)
        self.partitionId = apply_from_dict("partitionId", alert_dict, None)
        self.policyCode = apply_from_dict("policyCode", alert_dict, None)
        self.priorityNumber = apply_from_dict("priorityNumber", alert_dict, None)

        actions = apply_from_dict("actions", alert_dict, [])
        if self.has_local_actions:
            for action in actions:
                form_value = action.get("formValue", {})
                if form_value.get("core") and (form_value.get("method") == HttpMethods.POST.value):
                    self.is_store_thumb_for_local_action = True

        version = apply_from_dict("version", alert_dict, "2.0.0")
        self.legacy = not (version == "2.0.0")

        # selectedCamera
        self.selectedCamera = apply_from_dict("selectedCamera", alert_dict, {})
        self.timezone = apply_from_dict("timezone", alert_dict, apply_from_dict("timezone", self.selectedCamera, None))
        self.locationId = apply_from_dict("locationId", self.selectedCamera, "")
        self.cameraId = apply_from_dict("cameraId", self.selectedCamera, "")

        # markedIdx
        if self.legacy:
            marked_idx = apply_from_dict("markedIdx", alert_dict, None)
            self.mergedZones = apply_from_dict("measureCrossZones", alert_dict, True)
        else:
            marked_idx = apply_from_dict("markedIdx", self.selectedCamera, None)
            self.mergedZones = False
        self.roiFilter: bool = marked_idx is not None and len(marked_idx) > 0
        self.roi: Any = None if not self.roiFilter else roi_gen(marked_idx)
        self.marked_idx = set() if not self.roiFilter else set(marked_idx)

        # settings
        setting_dict = alert_dict.get("settings", {})
        self.settings = setting_dict

        if self.legacy:
            blockout_seconds = apply_from_dict("blockNotificationPeriod", setting_dict, 0)
        else:
            blockout_seconds = apply_from_dict("reactivationTh", setting_dict, 0)
        self.blockout: float = blockout_seconds * 1000 if blockout_seconds is not None else 0
        self.priority = 0
        if "priority" in setting_dict:
            self.pushAlert = not (setting_dict["priority"] == AlertsConfidence.low.value)
            self.alertThumbnail = not (setting_dict["priority"] == AlertsConfidence.low.value)
            self.alertZoomThumbnail = setting_dict["priority"] == AlertsConfidence.high.value
            self.priority = int(setting_dict["priority"])
        else:
            self.pushAlert: bool = apply_from_dict("pushAlert", setting_dict, True)
            self.alertThumbnail: bool = apply_from_dict("alertThumbnail", setting_dict, False)
            self.alertZoomThumbnail: bool = apply_from_dict("alertZoomThumbnail", setting_dict, False)
            self.alertThumbnail = self.alertThumbnail and self.pushAlert
            self.alertZoomThumbnail = self.alertZoomThumbnail and self.pushAlert
            self.priority = 1 if self.pushAlert else 0

        self.showOnAlertPage = 1
        self.showOnWalls = 1
        if "show" in setting_dict:
            disabled = setting_dict["show"].get("disabled", False)
            if disabled:
                self.showOnAlertPage, self.showOnWalls = 0, 0
            else:
                show_on = setting_dict["show"].get("on", [0, 1])
                self.showOnAlertPage = 1 if 0 in show_on else 0
                self.showOnWalls = 1 if 1 in show_on else 0

        self.internal_alert: bool = apply_from_dict("internal", setting_dict, False)
        routing_offset = apply_from_dict("routing", setting_dict, 0)
        routing = self.default_routing.value

        prompt = setting_dict.get("prompt", None)
        if prompt:
            self.special_filter = str(prompt)
            if routing == AlertRouting.NO_ROUTING:
                routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE

        if routing_offset in [AlertRouting.MANUAL.value, AlertRouting.MANUAL_OLD.value]:
            routing += AlertRouting.MANUAL.value
        self.routing: AlertRouting = AlertRouting(routing)
        self.activate_ddata = (
            self.context.get_config().get("alertConfig", {}).get("internal_debug_enable", True) and self.internal_alert
        )

        self.is_store_snapshot = self.is_store_snapshot or self.routing != AlertRouting.NO_ROUTING
        # actions
        self.actions = {"messages": [], "gpioActions": [], "speakerActions": []}
        # self.parse_actions(apply_from_dict("actions", alert_dict, []))

        # basic configuration. Here this code will be removed once converting
        # I avoided if else to minimize mistakes

        if "configuration" in alert_dict and alert_dict["configuration"] is not None:
            configuration_dict = alert_dict["configuration"]
        else:
            configuration_dict = {}
        if self.object_based_alert:
            object_value = apply_from_dict("object", configuration_dict, None)
            objects_value = apply_from_dict("objects", configuration_dict, None)
            if objects_value is not None and len(objects_value):
                values = [list(obj.values())[0] for obj in objects_value]
                self.objects = [class_handler.object_int_to_str(obj) for obj in values]
                self.object_ids = values
            elif object_value is not None:
                # legacy support
                self.objects = [class_handler.object_int_to_str(object_value)]
                self.object_ids = [object_value]
            else:
                self.objects = ["unknown"]
        else:
            self.objects = [None]

        filters: Dict = apply_from_dict("filters", configuration_dict, {})
        strict_filters = self.alerts_config.get("strict_filters", False)
        self.alert_confidence = apply_from_dict("confidence", setting_dict, AlertsConfidence.low.value)
        self.success_required_filter: bool = self.alerts_config.get(
            "success_filter", (self.alert_confidence == AlertsConfidence.high.value)
        )
        unique_detector = apply_from_dict("detector_unique", apply_from_dict("l0_track", analytic_config, {}), False)
        success_required_default: bool = self.alerts_config.get("success_required_default", True)

        if success_required_default and not unique_detector:
            self.success_required_filter = True

        self.filters = {}
        self.filtersDisabled = {}
        self.strict_filters = {}
        self.recognition_filters = {}
        self.recognition_enabled = False
        for object_id in self.object_ids:
            self.filters[object_id] = {}
            for key in filters.keys():
                if (
                    key in SUPPORTED_FILTERS
                    and filters[key] is not None
                    and isinstance(filters[key], list)
                    and len(filters[key]) > 0
                ):
                    self.filters[object_id][key] = set(
                        [element.lower() if isinstance(element, str) else element for element in filters[key]]
                    )

            # check if filters are set
            self.filtersDisabled[object_id] = (
                (len(self.filters[object_id]) == 0) or (self.objects is None) or (len(self.objects) > 1)
            )
            self.strict_filters[object_id] = strict_filters

        if (
            "detectionAdditionalAttributes" in configuration_dict
            and configuration_dict["detectionAdditionalAttributes"] is not None
        ):
            additional_att_dict = configuration_dict["detectionAdditionalAttributes"]
        else:
            additional_att_dict = {}

        self.count: Optional[float] = apply_from_dict("count", additional_att_dict, None)
        direction = apply_from_dict("direction", additional_att_dict, "above")
        self.positive_alert: bool = direction != "below"
        self.sensitivity: Optional[float] = apply_from_dict("sensitivity", additional_att_dict, None)

        self._last_active_alert = -100000000

        flow_dict = apply_from_dict("selectedFlow", alert_dict, {})
        # schedule
        self.schedule = Schedule()
        self.formValue = apply_from_dict("formValue", flow_dict, {})
        if "schedule" in self.formValue and self.formValue["schedule"] is not None:
            self.schedule = Schedule(self.formValue["schedule"])
            if self.timezone is None and not self.schedule.all_week:
                raise ValueError(f"Alert {self.event_id} has no timezone set, but has schedule")
        else:
            self.schedule = Schedule()

        self.category: Optional[int] = apply_from_dict("category", flow_dict, -1)
        self.flow: Optional[int] = apply_from_dict("flowType", flow_dict, -1)
        self.detection: Optional[int] = apply_from_dict("detection", configuration_dict, None)
        if self.category >= 0:
            # new alert format
            alert_type_value = self.get_detection_type()
        else:
            if self.detection is not None:
                alert_type_value = self.detection
            else:
                alert_type_value = apply_from_dict("alertType", alert_dict, AlertType.unknown.value)

        # set global parameters from formValue
        if self.formValue and self.object_based_alert:
            self.objects = []
            objects_value = apply_from_dict("objects", self.formValue, None)
            self.filters = {}
            self.filtersDisabled = {}
            self.strict_filters = {}

            if objects_value is not None:
                for object_value in objects_value:
                    object_id = object_value["type"]
                    self.objects.append(class_handler.object_int_to_str(object_id))
                    self.object_ids.append(object_id)
                    self.filters[object_id] = {}

                    filters = apply_from_dict("filters", object_value, {})
                    if isinstance(filters, dict):
                        for key in filters.keys():
                            if (
                                key in SUPPORTED_FILTERS
                                and filters[key] is not None
                                and isinstance(filters[key], list)
                                and len(filters[key]) > 0
                            ):
                                self.filters[object_id][key] = set(
                                    [
                                        element.lower() if isinstance(element, str) else element
                                        for element in filters[key]
                                    ]
                                )
                    elif isinstance(filters, list):
                        for filt in filters:
                            filter_name = filt.get("name", "none")
                            if filt.get("enabled", False) and filter_name in SUPPORTED_FILTERS:
                                values = filt.get("value", [])
                                if values:
                                    self.filters[object_id][filter_name] = set(
                                        [elem.lower() if isinstance(elem, str) else elem for elem in values]
                                    )
                                colors = filt.get("colors", [])
                                colors_name = filter_name[:-4] + "Colors"
                                if colors and colors_name in SUPPORTED_FILTERS:
                                    self.filters[object_id][colors_name] = set(colors)

                    # check if filters are set
                    self.filtersDisabled[object_id] = len(self.filters[object_id]) == 0
                    self.strict_filters[object_id] = apply_from_dict("strict", object_value, strict_filters)

                    # handle recognition filters
                    self.recognition_filters[object_id] = {"enabled": False}
                    for t in ["include", "exclude"]:
                        rec_list, rec_groups = find_recognition_lists(object_value, t)
                        if rec_list or rec_groups:
                            self.recognition_filters[object_id] = {
                                "metadata": [],
                                "identifiers": np.array([]),
                                "enabled": True,
                                "exclude": t == "exclude",
                                "l1_analyzer": context.get_l1_manager().get_default_recognition_analyzer(object_id),
                                "subjects": rec_list,
                                "groups": rec_groups or [],
                                "object_id": object_id,
                            }
                            self.recognition_enabled = True
                            break
            custom_objects = [int(o) for o in self.formValue.get("customObjects", [])]
            if custom_objects:
                custom_object_db = self.context.get_analytics_db().get("custom_objects", [])
                for obj in custom_objects:
                    obj_data: Optional[CustomObjectData] = None
                    for db_obj in custom_object_db:
                        if obj == db_obj.custom_object.customObjectId:
                            obj_data = db_obj
                            break
                    if obj_data is None:
                        raise ValueError(f"Custom object id {obj} not found in DB")
                    obj_id = obj_data.object_type
                    if obj_id not in self.object_ids:
                        self.object_ids.append(obj_id)
                        self.objects.append(class_handler.object_int_to_str(obj_id))
                        self.filters[obj_id] = {}
                    reid_th = obj_data.custom_object.reidThr
                    if reid_th is not None and reid_th > 0:
                        self.filtersDisabled[obj_id] = False
                    else:
                        self.filtersDisabled[obj_id] = self.filtersDisabled.get(obj_id, True)
                    if obj_id not in self.custom_objects:
                        self.custom_objects[obj_id] = {}
                    self.custom_objects[obj_id][obj_data.custom_object.customObjectId] = copy(obj_data)

            if len(self.objects) == 0:
                self.objects = ["unknown"]

            if custom_objects or self.recognition_enabled:
                self.on_db_update()

                # invalidate the alert if data isnt valid
                if not self.is_db_valid:
                    raise RuntimeError("This alert requires data from the synced DB and the data isn't valid")

        self.l1_required_type = {}
        self.l1_requested_ents = set()
        l1_required = False
        for object_id in self.filters:
            self.l1_required_type[object_id] = None
            if not self.filtersDisabled[object_id]:
                self.l1_required_type[object_id] = context.get_l1_manager().get_default_filter_analyzer(
                    object_id, self.filters[object_id]
                )
            l1_required |= self.l1_required_type[object_id] is not None
        self.l1_required = l1_required

        # handle l1 that is requested in the messages
        pattern = re.compile(r"\{\{(.*?)\}\}")  # noqa
        default_l1 = context.get_l1_manager().default_desc_analyzer()
        # Find all matches of the pattern
        matches = set(re.findall(pattern, str(alert_dict)))
        matches.discard("person_name")  # remove person_name since it's not really l1 and only available in face alert
        self.l1_for_msg = {}

        if matches:
            match_object = set([match.split("_")[0] for match in matches])
            for obj in self.objects:
                if obj in match_object:
                    obj_id = class_handler.object_str_to_int(obj)
                    self.l1_for_msg[obj_id] = [default_l1[obj_id]]
                    if (
                        obj_id == class_handler.vehicle_value
                        and any(["license_plate" in m for m in matches])
                        and class_handler.plate_subclass_value > -1
                    ):
                        self.l1_for_msg[obj_id].append(default_l1[class_handler.plate_subclass_value])

        self.alert_type: AlertType = AlertType(alert_type_value)
        self.threshold = self.count  # use count as default threshold
        self.zones = {}
        self.trackers = {}
        self.policies = {}
        self.zones_policy = None
        self.apply_from_dict = apply_from_dict
        if any(self.strict_filters.values()):
            self.max_hold = max(self.max_hold_if_strict, self.max_hold)

    def is_blockedout(self, alert_ts):
        return self.blockout > 0 and alert_ts - self.blockout < self._last_active_alert

    def set_flow_values(self):
        pass

    def on_db_update(self):
        if self.custom_objects:
            custom_object_db = self.context.get_analytics_db().get("custom_objects", [])
            for obj in self.custom_objects:
                for obj_id in self.custom_objects[obj].keys():
                    obj_data: Optional[CustomObjectData] = None
                    for db_obj in custom_object_db:
                        if obj_id == db_obj.custom_object.customObjectId:
                            obj_data = db_obj
                            break
                    if obj_data is None:
                        logger.warning(f"Custom object id {obj_id} not found in DB during alert {self.event_id} update")
                        self.custom_objects[obj].pop(obj_id, None)
                    else:
                        self.custom_objects[obj][obj_id] = copy(obj_data)
        if self.recognition_enabled:
            self._update_recognition_groups()

    def _update_recognition_groups(self):
        """Resolve group references in recognition filters from the DB."""
        analytics_db = self.context.get_analytics_db()
        prev_valid = self.is_db_valid
        all_valid = True

        for object_id, rec_filter in self.recognition_filters.items():
            if not rec_filter.get("enabled"):
                continue

            # Parse the static subject list (always valid)
            base_metadata, base_identifiers = self._parse_recognition_list(object_id, rec_filter.get("subjects") or [])

            # Resolve groups from DB
            group_metadata = []
            group_identifiers = np.array([])
            if rec_filter.get("groups"):
                groups = rec_filter["groups"]
                result = self._resolve_recognition_groups(object_id, groups, analytics_db)
                if result[0] is None:
                    # DB issue — alert already invalidated via validate_alert
                    all_valid = False
                    continue
                group_metadata, group_identifiers = result

            # Combine subjects + group members
            combined_metadata = base_metadata + group_metadata
            if len(group_identifiers) > 0:
                if len(base_identifiers) > 0:
                    if object_id == self.vehicle_value:
                        combined_identifiers = np.concatenate([base_identifiers, group_identifiers])
                    else:
                        combined_identifiers = np.concatenate(
                            [np.atleast_2d(base_identifiers), np.atleast_2d(group_identifiers)]
                        )
                else:
                    combined_identifiers = group_identifiers
            else:
                combined_identifiers = base_identifiers

            rec_filter["metadata"] = combined_metadata
            rec_filter["identifiers"] = combined_identifiers

        self.is_db_valid = all_valid
        if all_valid and not prev_valid:
            # DB was invalid and is now resolved — re-validate the alert
            self.context.validate_alert(self.event_id, True)

    def _resolve_recognition_groups(self, object_id, groups, analytics_db):
        """Resolve group references into metadata + identifiers arrays.

        Returns (metadata, identifiers) or (None, None) on error.
        Calls validate_alert(False) to invalidate the alert when data is missing.
        """
        metadata = []
        identifiers = []

        if object_id == self.person_value:
            person_db = analytics_db.get("persons", {})
            group_db = analytics_db.get("persons_groups", [])
            existing = set()
            for g in groups:
                gid = g.get("groupId", g.get("id"))
                group_data = [gd for gd in group_db if gd.id == gid]
                if len(group_data) != 1:
                    self.context.validate_alert(
                        self.event_id, False, f"Group ID {gid} not found in persons_groups database"
                    )
                    return None, None
                group_persons = group_data[0].personIds
                if not group_persons:
                    self.context.validate_alert(self.event_id, False, f"Group ID {gid} has no persons")
                    return None, None
                for pid in group_persons:
                    if pid in existing:
                        continue
                    if pid not in person_db:
                        self.context.validate_alert(
                            self.event_id, False, f"Person ID {pid} not found in persons database"
                        )
                        return None, None
                    person_data = person_db[pid]
                    reps = [r for r in person_data.pos_reps if r.faceIdVersion == 2]
                    if len(reps) == 0:
                        self.context.validate_alert(self.event_id, False, f"Person ID {pid} has no V2 representatives")
                        return None, None
                    for rep in reps:
                        metadata.append({"name": person_data.person.name, "personId": pid})
                        identifiers.append(np.frombuffer(rep.faceIdBin, dtype=np.float32))
                    existing.add(pid)
            identifiers = np.array(identifiers) if identifiers else np.array([])

        elif object_id == self.vehicle_value:
            vehicle_db = analytics_db.get("vehicles", {})
            group_db = analytics_db.get("vehicle_groups", [])
            for g in groups:
                gid = g.get("groupId", g.get("id"))
                group_data = [gd for gd in group_db if gd.id == gid]
                if len(group_data) != 1:
                    self.context.validate_alert(
                        self.event_id, False, f"Group ID {gid} not found in vehicle_groups database"
                    )
                    return None, None
                group_vehicles = group_data[0].vehicleIds
                if not group_vehicles:
                    self.context.validate_alert(self.event_id, False, f"Group ID {gid} has no vehicles")
                    return None, None
                for vid in group_vehicles:
                    if vid not in vehicle_db:
                        self.context.validate_alert(
                            self.event_id, False, f"Vehicle ID {vid} not found in vehicles database"
                        )
                        return None, None
                    metadata.append({"vehicle_id": vid})
                    identifiers.append(vehicle_db[vid].plate.lower())
            identifiers = np.array(identifiers).T if identifiers else np.array([])
        else:
            return None, None

        return metadata, identifiers

    def set_last_active(self, timestamp: int):
        self._last_active_alert = timestamp

    def get_detection_type(self):
        if self.category == AlertCategory.Safety.value and self.flow == SafetyType.Weapon.value:
            self.objects = ["weapon"]
            self.object_ids = [self.context.get_class_handler().object_str_to_int("weapon")]
        elif self.category == AlertCategory.CustomizedCapabilities.value:
            self.adjust_custom_filters()
        return alert_mapping.get(self.category, {}).get(self.flow, AlertType.unknown)

    def adjust_custom_filters(self):

        if self.category == AlertCategory.CustomizedCapabilities.value:
            if self.flow == CustomizedCapabilitiesType.ProtectiveGear.value:
                gear_type = self.formValue.get("gear", ProtectedGearType.hard_hat.value)
                attr_name = "hard_hat" if gear_type == ProtectedGearType.hard_hat.value else "safety_vest"

                if self.formValue["wear"] == ProtectedGearWearOptions.not_wearing.value:
                    filt_value = ["no_" + attr_name]
                else:  # self.formValue["wear"] == ProtectedGearOptions.hard_hat.value:
                    filt_value = [attr_name]

                self.formValue["objects"] = [
                    {
                        "type": self.context.get_class_handler().object_str_to_int("person"),
                        "filters": {"protectiveGearType": filt_value},
                        "strict": True,
                    }
                ]

            elif self.flow == CustomizedCapabilitiesType.PersonalSafety.value:
                self.formValue["objects"] = [
                    {
                        "type": self.context.get_class_handler().object_str_to_int("person"),
                        "filters": {},
                        "strict": True,
                    },
                    {
                        "type": self.context.get_class_handler().object_str_to_int("vehicle"),
                        "filters": {},
                        "strict": True,
                    },
                ]

    def check_alert_batch(self, images, motion_data, batch_data):
        status = AlertStatus.DISABLED
        alert_info = {}
        if self.enabled:
            # schedule was checked in the alert manager level
            alert_info = self.is_active_batch(images, motion_data, batch_data)

            # blockout is checked when validating the candidate
            status = AlertStatus.ACTIVE if alert_info else AlertStatus.IDLE
        return status, alert_info

    def activateAlert(self, data):
        logger.error("activateAlert not implemented on this alert type")
        pass

    def out_of_schedule_maintenance(self, images, motion_data, batch_data):
        pass

    def is_in_schedule(self, now, day) -> bool:

        # if the alert holds schedule - it win
        if self.schedule and not self.schedule.all_week:
            return self.schedule.check_schedule(now, day)

        # first we are checking if we have controlZone
        if self.partitionId and len(self.partitionId) > 0:
            if self.zones_policy:
                return self.zones_policy.check_zone(now, day)
            else:
                # This is place holder - cant be removed later on - its to avoid cloud bugs
                logger.error(f"Alert {self.event_id} has partitionId set but no zones policy")
                return True

        # the alert isnt in controlZone, operate by Policy
        work_by_policy = False
        for policy in self.policies.values():
            work_by_policy = True
            if policy.check_policy(now, day):
                # the logic is OR between policies
                return True
        if work_by_policy:
            # event exist in policy but didnt find a match for current time
            return False

        # if we dont have schedule, dont have control zone and we dont have policies return True
        return True

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BatchDataResolver
    ) -> List[AlertCandidate]:
        return []

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        return True

    def create_alert_policy(self, policy_msg):
        if "schedule" in policy_msg and policy_msg["schedule"] is not None:
            self.schedule = Schedule(self.formValue["schedule"])

    def handle_alert_policy(self, policy_msg, init=False):
        action = apply_from_dict("action", policy_msg, "unknown")
        policy_id = apply_from_dict("_id", policy_msg, "")
        # in case of init alert is armed by default
        if init:
            policy_msg["arm"] = True

        if policy_id:
            if action == AlertsAction.add.value:
                self.policies[policy_id] = Policy(policy_msg)
            elif action == AlertsAction.remove.value:
                self.policies.pop(policy_id, None)
            elif action == AlertsAction.update.value:
                self.policies.pop(policy_id, None)
                self.policies[policy_id] = Policy(policy_msg)

    def handle_zone_policy(self, zone_msg, init=False):
        action = apply_from_dict("action", zone_msg, "unknown")
        policy_id = apply_from_dict("_id", zone_msg, "")

        if policy_id:
            if action == AlertsAction.add.value:
                self.zones_policy = ZonePolicy(zone_msg)
            elif action == AlertsAction.remove.value:
                self.zones_policy = None
            elif action == AlertsAction.update.value:
                if self.zones_policy:
                    logger.info("Updating zone policy")
                    self.zones_policy.update(zone_msg)
                elif init:
                    self.zones_policy = ZonePolicy(zone_msg)
                else:
                    logger.error("Zone policy wasnt configured, creating")
                    self.zones_policy = ZonePolicy(zone_msg)

    def get_alert_policies(self):
        return self.policies

    def set_alert_policies(self, policies):
        self.policies = policies

    def build_alert_info_dc(self, candidate: AlertCandidate):
        alert_info = AlertInfo(
            eventId=self.event_id,
            timestamp=int(candidate.timestamp),
            detectionType=self.alert_type.value,
            alertMessage=self.alert_message,
            debugData=self.debug_data,
            priority=self.priority,
            flowType=self.flow,
            category=self.category,
            internal=int(self.internal_alert),
            routing=self.routing.value,
            hasLocalActions=self.has_local_actions,
            accountId=self.accountId,
            partitionId=self.partitionId,
            policyCode=self.policyCode,
            priorityNumber=self.priorityNumber,
            storeLocalThumbnail=self.is_store_thumb_for_local_action,
            showOnAlertPage=self.showOnAlertPage,
            showOnWalls=self.showOnWalls,
            _actions=self.actions,
            _variableAlert=self.variableAlert,
            clearEvent=None,
        )
        return alert_info

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = self.build_alert_info_dc(candidate)
        alert_info.crops += candidate.crops
        alert_info.validation_images += candidate.validation_images
        extra_fields = candidate.extra.get("extra_fields", {}) if candidate.extra else {}
        alert_info.extra.update(extra_fields)
        if self.special_filter is not None:
            alert_info.specialFilter = self.special_filter
        alert_info.alertMessage = self.alert_message
        alert_info.debugData = candidate.extra.get("debug_data", None) if candidate.extra else None
        return alert_info

    def build_alert_candidate(
        self,
        timestamp,
        ent_ids: Union[int, List[int]] = None,
        var_data: np.array = None,
        extra: dict = None,
        crops: List[np.array] = None,
        images: List[AnalyticImage] = None,
        validation_images: List[np.array] = None,
    ) -> AlertCandidate:
        candidate = AlertCandidate(self.event_id, timestamp)
        if ent_ids is not None:
            candidate.ent_ids = ent_ids if isinstance(ent_ids, list) else [ent_ids]
        if var_data is not None:
            candidate.var_data = np.atleast_2d(var_data)
        if extra is not None:
            candidate.extra = extra
        if crops is not None:
            candidate.crops += crops
        elif images is not None and candidate.var_data is not None:
            idx = int(candidate.var_data[-1, BatchDataResolver.FRAME_ID])
            candidate.crops.append(crop_image(images[idx].frame, candidate.var_data[-1, BatchDataResolver.POS]))
        if validation_images is not None:
            candidate.validation_images += validation_images
        elif self.routing != AlertRouting.NO_ROUTING and images is not None and candidate.var_data is not None:
            idx = int(candidate.var_data[-1, BatchDataResolver.FRAME_ID])
            bbox = candidate.var_data[-1, BatchDataResolver.POS]
            validation_crop = self._generate_validation_crop(bbox, images[idx], margins=[0.2, 0.2])
            candidate.validation_images += [validation_crop]

        return candidate

    def on_entities_update(self, ent_ids: List[int]):
        pass

    def on_entities_removed(self, ent_ids: List[int]):
        pass

    def on_entities_purged(self, ent_ids: List[int]):
        pass

    def on_night_mode_changed(self, night_mode: bool):
        pass

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        alert_info.alertMessage = self.alert_message

    def _parse_recognition_list(self, obj_id: int, rec_list: List[Dict]) -> Tuple[List, List]:
        metadata = []
        identifiers = []
        if obj_id == self.person_value:
            for person in rec_list:
                name = person.get("name", "")
                person_id = person.get("personId", 0)
                for rep in person["representatives"]:
                    metadata.append({"name": name, "personId": person_id})
                    identifiers.append(np.frombuffer(bytes.fromhex(rep), dtype=np.float32))
            identifiers = np.array(identifiers)
        if obj_id == self.vehicle_value:
            for vehicle in rec_list:
                metadata.append({"vehicle_id": vehicle["id"]})
                identifiers.append(vehicle["plate"].lower())
            identifiers = np.array(identifiers)
        return metadata, identifiers

    def register_alert(self, alert_info):
        pass

    def apply_roi_filter(self, vars_data: np.ndarray, position=BatchDataResolver.LOCATION) -> np.ndarray:
        n_vars = len(vars_data)
        if self.roiFilter and n_vars > 0:
            locations = vars_data[:, position].astype(int)
            in_roi = [i for i in range(n_vars) if locations[i] in self.marked_idx]
            vars_data = vars_data[in_roi]
        return vars_data

    def _generate_validation_crop(self, bbox: np.ndarray, image: AnalyticImage, margins: List[float] = None):
        return crop_image(image.frame, bbox, margins=margins, bgr_map=True)

    def _generate_validation_crop_const_size(self, bbox: np.ndarray, image: AnalyticImage, margins: List[float] = None):
        # first make the crop 1:1 aspect ratio
        if margins is not None:
            y_gb = (bbox[3] - bbox[1]) * margins[1]
            x_gb = (bbox[2] - bbox[0]) * margins[0]
            bbox = np.clip(bbox + np.array([-x_gb, -y_gb, x_gb, y_gb]), 0, 1)
        image_sz = image.frame.shape[1::-1]
        bbox = bbox * np.array([*image_sz, *image_sz])
        bbox_sizes = bbox[2:] - bbox[:2]
        centers = (bbox[:2] + bbox[2:]) / 2
        mx_ind = np.argmax(bbox_sizes)
        min_ind = 1 - mx_ind

        def increase_dim_by_factor(dim, factor):
            opposite_dim = 1 - dim
            min_bbox_tl = centers[dim] - bbox_sizes[dim] * factor / 2
            max_bbox_br = centers[dim] + bbox_sizes[dim] * factor / 2
            if min_bbox_tl < 0:
                bbox[dim] = 0
                bbox[2 + dim] = min(bbox_sizes[min_ind] * factor, image.frame.shape[opposite_dim])
            elif max_bbox_br > image.frame.shape[opposite_dim]:
                bbox[2 + dim] = image.frame.shape[opposite_dim]
                bbox[dim] = max(0, image.frame.shape[opposite_dim] - bbox_sizes[dim] * factor)
            else:
                bbox[dim] = min_bbox_tl
                bbox[2 + dim] = max_bbox_br

        if bbox_sizes[mx_ind] < self.routing_crop_size:
            factor = self.routing_crop_size / bbox_sizes[mx_ind]
            increase_dim_by_factor(mx_ind, factor)
            bbox_sizes = bbox[2:] - bbox[:2]

        factor = bbox_sizes[mx_ind] / bbox_sizes[min_ind]
        increase_dim_by_factor(min_ind, factor)

        bbox = bbox.astype(int)
        crop = image.frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        crop_sz = crop.shape[1::-1]
        if abs(crop_sz[0] - crop_sz[1]) > 3:  # allow small dim difference
            mx_ind = np.argmax(bbox_sizes)
            min_ind = 1 - mx_ind
            delta = crop_sz[mx_ind] - crop_sz[min_ind]
            # if mx index is 1 then we need to pad top o.w left
            bottom = delta * min_ind
            right = delta * mx_ind
            crop = cv2.copyMakeBorder(crop, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(crop, (self.routing_crop_size, self.routing_crop_size), interpolation=cv2.INTER_LINEAR)

    def update_motion(self, motion_data: MotionData):
        pass

    def serialize_debug_data(self, data: Dict[str, Any]) -> str:
        """Serialize and encode as base64 string"""
        return encode_debug_data(data)


class ObjectAlert(BaseAlert):
    alert_on_tracked_only = True
    object_based_alert = True
    filter_driver = True
    driver_occlusion_th = 0.5
    motion_filter = False
    motion_filter_th = 1
    history_filter = False
    max_history_time = 5 * 60 * 60 * 1000  # 5 hours
    max_history_len = 5
    min_history_span = 1
    min_history_iou = 0.95
    min_history_sim = 0.9
    ambient_motion_filter = False
    ambient_motion_th = 10
    ambient_motion_min_area = 3

    def __init__(self, alert_dict: Dict, context):
        super(ObjectAlert, self).__init__(alert_dict, context)
        self.entity_db = context.get_entity_db()
        self.alerted_ids = set()
        candidate_ids = self.entity_db.query(self.objects, filters=self.filters)
        self.active_id = set(candidate_ids)

        # get parameters from config if exists
        cfg = self.alerts_config.get(self.type_name, {})
        self.filter_driver = cfg.get("filter_driver", self.filter_driver)
        self.driver_occlusion_th = cfg.get("driver_occlusion_th", self.driver_occlusion_th)
        self.history_filter = cfg.get("history_filter", self.history_filter)
        self.motion_filter = cfg.get("motion_filter", self.motion_filter)
        self.motion_filter_th = cfg.get("motion_filter", self.motion_filter_th)
        self.max_history_len = cfg.get("max_history_len", self.max_history_len)
        self.min_history_span = cfg.get("min_history_span", self.min_history_span)
        self.min_history_iou = cfg.get("min_history_iou", self.min_history_iou)
        self.min_history_sim = cfg.get("min_history_sim", self.min_history_sim)
        self.ambient_motion_filter = cfg.get("ambient_motion_filter", self.ambient_motion_filter)
        self.ambient_motion_th = cfg.get("ambient_motion_th", self.ambient_motion_th)

        # conditions are met to filter person in car
        # vehicles and person defined and the object should be person to allow the filtering
        self.filter_driver &= (
            self.vehicle_value >= 0 and self.person_value >= 0 and self.person_value in self.object_ids
        )

        self.history = []
        self.motion_buffer = deque(maxlen=3)

    def update_motion(self, motion_data: MotionData):
        if self.ambient_motion_filter:
            self.motion_buffer.append(motion_data.unified)

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        checked_out = True
        none_or_false = False if force else None
        for ent_id in candidate.ent_ids:
            ent_data: ActiveEntityData = self.entity_db.get_entity(ent_id)
            if ent_data is not None:
                if self.motion_filter and not ent_data.has_movement(self.motion_filter_th):
                    return none_or_false  # until entity is purged
                if self.ambient_motion_filter:
                    pos = (
                        candidate.var_data[-1, BatchDataResolver.POS]
                        if candidate.var_data is not None
                        else ent_data.last_bbox
                    )
                    area = (pos[2] - pos[0]) * (pos[3] - pos[1]) * ROI_SHAPE[0] * ROI_SHAPE[1]
                    if area > self.ambient_motion_min_area:
                        tl = np.maximum((pos[:2] * ROI_SHAPE) - 1, 0).astype(int)
                        br = np.minimum((pos[2:] * ROI_SHAPE) + 1, ROI_SHAPE).astype(int)
                        unified = np.max(self.motion_buffer, axis=0)
                        motion_index = np.max(unified[tl[1] : br[1], tl[0] : br[0]])
                        if motion_index < self.ambient_motion_th:
                            return False
                is_custom = None
                is_custom_obj = self.custom_objects.get(ent_data.object_id, None) is not None
                if is_custom_obj:
                    is_custom = ent_data.is_match_custom_object(self.custom_objects[ent_data.object_id], force)

                # if custom object matches - return True
                if is_custom:
                    return True
                # if its verified non match and there are other no other filters - return False
                elif is_custom is False and not self.filters.get(ent_data.object_id, None):
                    return False

                # otherwise continue with normal filtering
                if not is_custom:
                    if self.l1_required:
                        required_l1 = self.l1_required_type[ent_data.object_id]
                        if required_l1:
                            has_l1 = ent_data.has_l1(required_l1)
                            if (has_l1 or force) and self.filters.get(ent_data.object_id, None):
                                is_match = ent_data.match_filters(
                                    self.object_ids,
                                    self.filters,
                                    self.filtersDisabled,
                                    self.success_required_filter,
                                    self.strict_filters,
                                )
                                if not is_match:
                                    return False
                            else:
                                return none_or_false
                    elif self.filters.get(ent_data.object_id, None):
                        is_match = ent_data.match_filters(
                            self.object_ids,
                            self.filters,
                            self.filtersDisabled,
                            self.success_required_filter,
                            self.strict_filters,
                        )
                        if not is_match:
                            return False
                    elif is_custom_obj:
                        return none_or_false
                    is_valid = ent_data.is_valid(self.success_required_filter)
                else:
                    is_valid = True
                if self.history_filter and is_valid is not None:
                    is_valid = not self._compare_to_history(ent_data)
            if is_valid is not None or force:
                checked_out &= bool(is_valid)
            elif is_valid is None:
                return none_or_false
        return checked_out

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super().build_alert_info(candidate)

        if len(candidate.ent_ids) == 1:
            known_dets = candidate.var_data
            if known_dets is not None:
                if len(known_dets) > 0:
                    known_dets = np.atleast_2d(known_dets)
                else:
                    known_dets = None
            if candidate.ent_ids[0] > 0:
                alert_info.idIndex = candidate.ent_ids[0]
                ent_data = self.entity_db.get_entity(candidate.ent_ids[0])
                if ent_data is not None:
                    alert_info.object_id = ent_data.object_id
                    alert_info.idBase = ent_data.id_base
                    alert_info.trackerClass = int(ent_data.tracker_type)
                    alert_info.extra["entity_data"] = convert_to_alert_data(ent_data.get_attribute_report())
                    if candidate.extra is not None:
                        alert_info.extra["entity_data"].update(candidate.extra.get("entity_data", {}))
            elif known_dets is not None:
                object_id = candidate.var_data[BatchDataResolver.CLASS]
                alert_info.object_id = object_id
            custom_objects = alert_info.extra.get("entity_data", {}).get("custom_objects_ids", [])
            if custom_objects:
                prompt_data = {
                    "custom_object_prompts": [],
                    "blipThr": [],
                }
                alert_info.extra["entity_data"]["custom_objects_names"] = []
                for _id in custom_objects:
                    obj_data = self.custom_objects.get(alert_info.object_id, {}).get(_id, None)
                    if obj_data is not None:
                        name = obj_data.custom_object.name
                        alert_info.extra["entity_data"]["custom_objects_names"].append(name)
                        blip_thr = obj_data.custom_object.blipThr
                        if blip_thr is not None and blip_thr > 0:
                            prompt_data["blipThr"].append(blip_thr)
                            prompt_data["custom_object_prompts"].append(obj_data.custom_object.prompt)
                    else:
                        # specif custom object is not part of the alert data
                        # (the object is matching custom object unrelated to the alert filters
                        logger.debug(
                            f"Custom object id {_id} not found in alert data for object type {alert_info.object_id}"
                        )
                if (
                    alert_info.routing == AlertRouting.NO_ROUTING.value
                    and prompt_data["custom_object_prompts"]
                    and self.alerts_config.get("enable_routing_for_custom_objects", False)
                ):
                    alert_info.routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE.value
                    alert_info.specialFilter = json.dumps(prompt_data)
                pass
            self.generate_alert_message(candidate, alert_info)
        return alert_info

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        obj_id = alert_info.object_id
        if obj_id >= 0:
            object_name = self.context.get_class_handler().object_int_to_str(obj_id)
            alert_info.alertMessage = f"{object_name} {self.type_name}".capitalize()

    def on_entities_update(self, ent_ids: List[int]):
        for ent in ent_ids:
            ent_data: ActiveEntityData = self.entity_db.get_entity(ent)
            is_match = ent_data in self.object_ids
            # is_match = ent_data.match_filters(
            #     self.object_ids, self.filters, self.filtersDisabled, self.success_required_filter, self.strict_filters
            # )
            is_exist = ent in self.active_ids
            if is_exist and not is_match:
                self.active_ids.discard(ent)
            elif is_match and not is_exist:
                self.active_ids.add(ent)

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
            self.alerted_ids.discard(ent)

    def get_batch_candidates_data(self, batch_data: BatchDataResolver) -> np.array:
        batch_data = self._filter_person_in_car(batch_data)
        candidates = batch_data.unique(batch_data.ID, batch_data.CLASS, self.object_ids).astype(int).tolist()
        if self.alert_on_tracked_only and -1 in candidates:
            candidates.remove(-1)

        # candidates = list(self.get_batch_candidates(batch_data))
        vars_data = np.asarray([])
        if len(candidates) > 0:
            vars_data = batch_data.query(BatchDataResolver.ID, candidates)
            vars_data = self.apply_roi_filter(vars_data)
        return vars_data

    def _filter_person_in_car(self, batch_data: BatchDataResolver) -> BatchDataResolver:
        if not self.filter_driver or len(batch_data) < 2:
            return batch_data

        # get relevant entities
        persons = batch_data.query(BatchDataResolver.CLASS, self.person_value)
        vehicles = batch_data.query(BatchDataResolver.CLASS, self.vehicle_value)

        # filter 2 wheels
        # vehicles = vehicles[np.isin(vehicles[:, BatchDataResolver.SUBCLASS], self.two_wheels, invert=True), :]
        if len(persons) == 0 or len(vehicles) == 0:
            return batch_data
        frames_p = np.unique(persons[:, BatchDataResolver.FRAME_ID])
        frames_v = np.unique(persons[:, BatchDataResolver.FRAME_ID])
        frames = frames_p if len(frames_p) < len(frames_v) else frames_v
        idx_to_remove = set()
        for f in frames:
            frame_vehicles = vehicles[vehicles[:, BatchDataResolver.FRAME_ID] == f, BatchDataResolver.INDEX].astype(int)
            frame_persons = persons[persons[:, BatchDataResolver.FRAME_ID] == f, BatchDataResolver.INDEX].astype(int)
            overlaps = np.any(
                batch_data.all_overlaps[np.ix_(frame_persons, frame_vehicles)] > self.driver_occlusion_th, axis=1
            )
            idx_to_remove.update(frame_persons[overlaps].tolist())
        if idx_to_remove:
            new_data = deepcopy(batch_data)
            new_data.data = np.array([v for v in batch_data.data if v[BatchDataResolver.INDEX] not in idx_to_remove])
            return new_data
        return batch_data

    def register_alert(self, alert_info: AlertInfo):
        if self.history_filter:
            ent_data: ActiveEntityData = self.entity_db.get_entity(alert_info.idIndex)
            if ent_data is not None:
                if ent_data.locations.calc_max_span() <= self.min_history_span:
                    if len(self.history) == self.max_history_len:
                        self.history.pop()
                    self.history.append(
                        {
                            "bbox": ent_data.last_bbox,
                            "descriptor": copy(ent_data.visual_id),
                            "timestamp": ent_data.last_seen,
                        }
                    )
            records_to_remove = []
            for rid, record in enumerate(self.history):
                if record["timestamp"] < alert_info.timestamp - self.max_history_time:
                    records_to_remove.append(rid)

            for rid in reversed(records_to_remove):
                self.history.pop(rid)

    def _compare_to_history(self, ent_data) -> bool:
        match_idx = -1
        for idx, hist in enumerate(self.history):
            iou = bbox_ious(ent_data.last_bbox[np.newaxis, :], hist["bbox"][np.newaxis, :])[0][0]
            if ent_data.visual_id is not None and hist["descriptor"] is not None:
                sim = np.dot(hist["descriptor"], ent_data.visual_id)
                # can reduce iou threshold because we have visual similarity
                if sim > self.min_history_sim and iou > self.min_history_iou * 0.8:
                    match_idx = idx
                    logger.warning(
                        f"Alert {self.event_id} candidate Id {ent_data.track_id} is filtered by history base on iou ({iou}) and visual similarity ({sim})"
                    )
                    break
            else:
                if iou > self.min_history_iou:
                    match_idx = idx
                    logger.warning(
                        f"Alert {self.event_id} candidate Id {ent_data.track_id} is filtered by history base on iou ({iou})"
                    )
                    if ent_data.visual_id is not None:
                        hist["descriptor"] = copy(ent_data.visual_id)  # update descriptor since there is none
                    break

        if match_idx > -1:
            # if there is a match, put  first in the list so next time it will be checked first and deleted last
            priority_hist = self.history.pop(match_idx)
            priority_hist["bbox"] = ent_data.last_bbox  # update bbox
            priority_hist["timestamp"] = ent_data.last_seen
            self.history.insert(0, priority_hist)
        return match_idx > -1


duration_unit_to_sec = {0: 1, 1: 60, 2: 3600, 3: 86400}
duration_unit_to_str = {0: "second", 1: "minute", 2: "hour", 3: "day"}
