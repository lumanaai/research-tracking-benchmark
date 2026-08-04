import datetime
import json
import random
from collections import deque
from copy import deepcopy
from typing import Dict, Set, List, Optional

import numpy as np
import pytz

from general.analyzer_general import logger, log_exception
from general.core import (
    AlertType,
    AlertsAction,
    AlertStatus,
    ClassHandler,
    BatchDataResolver,
    CommProtocol,
    alert_mapping,
    AlertInfo,
    AlertCategory,
    SafetyType,
    AdvanceAnalyzerType,
    TrackingType,
    AlertsConfidence,
    AlertRouting,
    RetailAlertType,
    ZoneRegion,
)
from general.entity_db import EntityDB
from level1.face.face_utils import conf_order_map, FaceConfidence, conf2cosine_th
from .activity import FallAlert
from .appearance import AppearanceAlert
from .base_alerts import BaseAlert, AlertCandidate
from .classification import ClassificationModelAlert
from .clip_trigger import ClipAlert
from .disappeare import DisappeareAlert
from .doors import DoorAlert
from .face_appearance import FaceAppearanceAlert
from .gloves import GlovesAlert, HandsAlert
from .integrated import DFOIntegratedAlert, DHOIntegratedAlert, TriggerType
from .lane import LaneAlert
from .license_plate import LicensePlateAlert
from .linecrossing import LineCrossingAlert, CountingAlert
from .loitering import LoiteringAlert, ZoneProtectionAlert
from .missing_obj import MissingObjectAlert
from .motion import MotionAlert
from .occupancy import OccupancyAlert, AbsenceAlert, RegionCountingAlert
from .periodic import PeriodicAlert, SnapshotAlert
from .phone import PhoneAlert
from .protective_equipment import PpeAlert
from .proximity import ProximityAlert
from .rare_appearance import FireAlert, BrandishedWeaponAlert
from .rare_appearance import WeaponAppearanceAlert
from .recognition import LPRAlert, FaceAlert, ContainerIdAlert
from .schedule import DayName
from .state_object_alert import ObjectStateDurationAlert, ObjectStateChangedAlert, ObjectStateTransitionAlert
from .state_shelves_alert import EmptyShelfAlert, EmptyShelfCounterAlert, EmptyShelfDropAlert
from .tailgating import TailgatingAlert
from .tampering import TamperAlert
from .trafficcontrol import TrafficControlAlert
from .validation import ValidationAlert
from .violence import FightingAlert
from .zones import ZoneAlert

INTERNAL_WEAPON_ALERT_ID = "000000000000000000000001"
INTERNAL_FIRE_ALERT_ID = "000000000000000000000002"
INTERNAL_VIOLENCE_ALERT_ID = "000000000000000000000003"
INTERNAL_FALLING_ALERT_ID = "000000000000000000000004"
INTERNAL_TAMPERING_ALERT_ID = "000000000000000000000005"


INTERNAL_ALERTS: Dict[str, Dict] = {
    "weapon": {
        "type": WeaponAppearanceAlert,
        "probability": 0.3,
        "id": INTERNAL_WEAPON_ALERT_ID,
        "category": AlertCategory.Safety.value,
        "flowType": SafetyType.Weapon.value,
        "formValue": {},
    },
    "fire": {
        "type": FireAlert,
        "probability": 0.1,
        "id": INTERNAL_FIRE_ALERT_ID,
        "category": AlertCategory.Safety.value,
        "flowType": SafetyType.Fire.value,
        "formValue": {},
    },
    "falling": {
        "type": FallAlert,
        "probability": 0.3,
        "id": INTERNAL_FALLING_ALERT_ID,
        "category": AlertCategory.Safety.value,
        "flowType": SafetyType.Fall.value,
        "formValue": {},
    },
    "violence": {
        "type": FightingAlert,
        "probability": 0.3,
        "id": INTERNAL_VIOLENCE_ALERT_ID,
        "category": AlertCategory.Safety.value,
        "flowType": SafetyType.Fighting.value,
        "formValue": {},
    },
}
TAMPERING_INTERNAL_ALERT = {
    "type": TamperAlert,
    "probability": 0,
    "id": INTERNAL_TAMPERING_ALERT_ID,
    "category": AlertCategory.Safety.value,
    "flowType": SafetyType.Tampering.value,
    "formValue": {
        "location": 0,  # indoor for increased sensitivity
        "duration": 1,  # 1 minute
        "durationUnit": 1,  # minutes
    },
}


class AlertManager:
    _alerts: Dict[str, BaseAlert]
    _required_classes: Dict[str, Set[str]]
    _analytic_config: Dict
    _class_handler: ClassHandler

    ALERT_TYPE_TO_CLASS = {
        AlertType.videoTampering: TamperAlert,
        AlertType.motionScore: MotionAlert,
        AlertType.tailgating: TailgatingAlert,
        AlertType.lineCrossing: LineCrossingAlert,
        AlertType.loitering: LoiteringAlert,
        AlertType.disappeare: DisappeareAlert,
        AlertType.appearance: AppearanceAlert,
        AlertType.zoneTrespassing: ZoneAlert,
        AlertType.proximity: ProximityAlert,
        AlertType.trafficControl: TrafficControlAlert,
        AlertType.lpr: LPRAlert,
        AlertType.lpc: LicensePlateAlert,
        AlertType.occupancy: OccupancyAlert,
        AlertType.weapon: WeaponAppearanceAlert,
        AlertType.faceDetection: FaceAlert,
        AlertType.doors: DoorAlert,
        AlertType.fire: FireAlert,
        AlertType.fall: FallAlert,
        AlertType.periodicText: PeriodicAlert,
        AlertType.ppe: PpeAlert,
        AlertType.clip: ClipAlert,
        AlertType.fighting: FightingAlert,
        AlertType.gloves: GlovesAlert,
        AlertType.hands: HandsAlert,
        AlertType.phone: PhoneAlert,
        AlertType.missingObject: MissingObjectAlert,
        AlertType.containerId: ContainerIdAlert,
        AlertType.absence: AbsenceAlert,
        AlertType.counting: CountingAlert,
        AlertType.zoneProtection: ZoneProtectionAlert,
        AlertType.faceAppearance: FaceAppearanceAlert,
        AlertType.snapshot: SnapshotAlert,
        AlertType.emptyShelf: EmptyShelfAlert,
        AlertType.emptyShelfCounter: EmptyShelfCounterAlert,
        AlertType.emptyShelfDrop: EmptyShelfDropAlert,
        AlertType.validationAlert: ValidationAlert,
        AlertType.brandishingWeapon: BrandishedWeaponAlert,
        AlertType.stateObjectChange: ObjectStateChangedAlert,
        AlertType.stateObjectTransition: ObjectStateTransitionAlert,
        AlertType.stateObjectDuration: ObjectStateDurationAlert,
        AlertType.distOnLane: LaneAlert,
        AlertType.regionCount: RegionCountingAlert,
    }

    def __init__(self, init_dict, policies_dict, zone_dict, config, entity_db: EntityDB, l1_manager, context):
        self.stagnation_ts_threshold = 3000
        self.curr_shelf_counter = None
        self.curr_shelves = []
        self.internal_alert_limit = 100
        self.alert_status = True
        self.max_alert_hold = 1000
        self.max_alert_hold_for_rec = 10000
        self._advance_fall: bool = False
        self._alerts = {}
        self._required_classes = {}
        self._analytic_config = config
        self.context = context
        self._l1_manager = l1_manager
        self._entity_db = entity_db
        self._entity_db.on_entities_update += self.on_entities_update
        self._entity_db.on_entities_removed += self.on_entities_removed
        self._entity_db.on_entities_purged += self.on_entities_purged
        self.context.on_db_update += self.on_db_update
        self.active_alerts: List[AlertCandidate] = []
        self.active_recognition_alerts: List[AlertCandidate] = []
        self._last_raised_alerts: Dict[str, deque] = {}
        self.bad_alerts: Dict[str, str] = {}
        self.context.on_night_mode_changed += self.on_night_mode_changed
        self.min_face_conf: FaceConfidence = FaceConfidence.MEDIUM
        self.alert_mapping = deepcopy(alert_mapping)
        self.is_location_center = context.is_location_center
        self.camera_type = context.camera_type
        if self._analytic_config.get("alertConfig", {}).get("linecross_as_count", False):
            self.alert_mapping[AlertCategory.Tracking.value][TrackingType.LineCrossing.value] = AlertType.counting.value
        if "alerts" in init_dict:
            for alert_dict in init_dict["alerts"]:
                self._add_alert(alert_dict, check_existing=False)
        if "policies" in policies_dict:
            for policy in policies_dict["policies"]:
                self.parse_policy_msg(policy, init=True)
        if "controlZone" in zone_dict:
            self.parse_zone_msg(zone_dict["controlZone"], init=True)

        try:
            self._internal_alert_counter = 0
            self._add_internal_alerts()
        except Exception as e:
            log_exception(logger, "error adding internal alerts", e)

    #        self._handle_shelves()

    def on_entities_removed(self, ent_ids: List[int]):
        for alert in self._alerts:
            self._alerts[alert].on_entities_removed(ent_ids)

    def on_entities_update(self, ent_ids: List[int]):
        # called every time entity metadata is changed (new or update)
        for alert in self._alerts:
            self._alerts[alert].on_entities_update(ent_ids)

    def on_entities_purged(self, ent_ids: List[int]):
        # called every time entity is deleted
        for alert in self._alerts:
            self._alerts[alert].on_entities_purged(ent_ids)

    def get_class_handler(self):
        return self.context.class_handler

    def get_entity_db(self):
        return self._entity_db

    def get_l1_manager(self):
        return self._l1_manager

    def get_config(self):
        return self._analytic_config

    def get_app_config(self):
        return self.context.app_config

    def get_clip_dispatcher(self):
        return self.context.clip_dispatcher

    def get_camera_id(self):
        return self.context.camera_id

    def _add_l1_requests(self, candidate: AlertCandidate, alert: BaseAlert, active_ents: Dict[int, Set[str]]):
        for ent in candidate.ent_ids:
            ent_data = self._entity_db.get_entity(ent)
            if ent_data is not None:
                req_l1 = active_ents.get(ent, set())
                req_l1.add(alert.l1_required_type[ent_data.object_id])
                rec_filter = alert.recognition_filters.get(ent_data.object_id, {})
                if rec_filter.get("enabled", False):
                    req_l1.add(rec_filter["l1_analyzer"])
                if ent_data.object_id in alert.l1_for_msg:
                    req_l1.update(alert.l1_for_msg[ent_data.object_id])
                active_ents[ent] = req_l1
            else:
                pass

    def check_alerts(self, images, batch_data: BatchDataResolver, motion_data) -> Dict[int, Set[str]]:
        now = datetime.datetime.fromtimestamp(images[0].timestamp / 1000).replace(tzinfo=pytz.utc)
        active_ents = {}
        for alert_id in self._alerts:
            alert: BaseAlert = self._alerts[alert_id]
            if alert.timezone is not None:
                now = now.astimezone(pytz.timezone(self._alerts[alert_id].timezone))
            day = DayName(now.weekday()).name
            alert.update_motion(motion_data)
            if alert.is_in_schedule(now, day):
                alert_status, alert_candidates = alert.check_alert_batch(images, motion_data, batch_data)
                if alert_status is AlertStatus.ACTIVE:
                    for candidate in alert_candidates:
                        candidate: AlertCandidate
                        self.active_alerts.append(candidate)
                        self._add_l1_requests(candidate, alert, active_ents)
                elif alert.l1_requested_ents:
                    for ent in alert.l1_requested_ents:
                        ent_data = self._entity_db.get_entity(ent)
                        if ent_data is not None:
                            req_l1 = active_ents.get(ent, set())
                            req_l1.add(alert.l1_required_type[ent_data.object_id])
                            active_ents[ent] = req_l1
                    alert.l1_requested_ents.clear()
            else:
                alert.out_of_schedule_maintenance(images, motion_data, batch_data)
        # stagnation avoidance
        for alert_candidate in self.active_alerts:
            alert = self._alerts.get(alert_candidate.alert_id, None)
            if (
                alert is not None
                and alert_candidate.last_valid_check < images[-1].timestamp - self.stagnation_ts_threshold
            ):
                self._add_l1_requests(alert_candidate, alert, active_ents)
                alert_candidate.last_valid_check = images[-1].timestamp
        return active_ents

    def _check_filter_active_alerts(self, curr_time: int) -> List[AlertInfo]:
        candidates_to_keep = []
        alerts_results = []
        for alert_candidate in sorted(self.active_alerts, key=lambda candidate: candidate.timestamp):
            alert = self._alerts.get(alert_candidate.alert_id, None)
            if alert is None:  # the alert was removed - no need to keep the candidate
                logger.debug(f"alert {alert_candidate.alert_id} was removed and candidate flushed")
                continue
            # if blockout is enabled, supress other alerts
            elif alert.is_blockedout(alert_candidate.timestamp):
                logger.debug(f"alert {alert_candidate.alert_id} is blocked out")
                continue
            is_held = alert_candidate.timestamp < curr_time - alert.max_hold
            is_checked_out = alert.check_alert_candidate(alert_candidate, force=is_held)
            if is_checked_out is None:
                candidates_to_keep.append(alert_candidate)

            elif is_checked_out:
                self.active_recognition_alerts.append(alert_candidate)
            else:
                logger.debug(f"alert {alert_candidate.alert_id} did not checked out as valid")

        self.active_alerts = candidates_to_keep
        return alerts_results

    def _check_recognition_active_alerts(self, curr_time: int) -> List[AlertInfo]:
        candidates_to_keep = []
        alerts_results = []
        for alert_candidate in sorted(self.active_recognition_alerts, key=lambda candidate: candidate.timestamp):
            alert = self._alerts.get(alert_candidate.alert_id, None)
            if alert is None:  # the alert was removed - no need to keep the candidate
                logger.debug(f"alert {alert_candidate.alert_id} was removed and candidate flushed")
                continue
                # if blockout is enabled, supress other alerts
            elif alert.is_blockedout(alert_candidate.timestamp):
                logger.debug(f"alert {alert_candidate.alert_id} is blocked out")
                continue
            is_filtered_out = False
            if alert.recognition_enabled:
                is_held = alert_candidate.timestamp < curr_time - self.max_alert_hold_for_rec
                is_filtered_out = self._apply_recognition_filter(alert.recognition_filters, alert_candidate, is_held)
                if is_filtered_out is None:
                    candidates_to_keep.append(alert_candidate)
                    is_filtered_out = True  # not to raise alert

            if not is_filtered_out:
                alert_info = alert.build_alert_info(alert_candidate)
                alerts_results.append(alert_info)
                alert.set_last_active(alert_candidate.timestamp)
        self.active_recognition_alerts = candidates_to_keep
        return alerts_results

    def get_active_alerts(self, curr_time: int) -> List[AlertInfo]:
        active_alerts = []
        if self.active_alerts:
            self._check_filter_active_alerts(curr_time)
        if self.active_recognition_alerts:
            active_alerts = self._check_recognition_active_alerts(curr_time)
            self.register_alerts(active_alerts)
        return active_alerts

    def _is_repeating_alerts(self, alert_candidate: AlertCandidate, alert: BaseAlert):
        is_repeated = False
        if (
            alert.object_based_alert  # support just object alerts
            and len(self._last_raised_alerts[alert.event_id]) > 0  # must have history
            and len(alert_candidate.ent_ids) == 1  # must be single entity alert
        ):
            ent_data = self._entity_db.get_entity(alert_candidate.ent_ids)
            if ent_data is not None:
                pass
                # TBD if we need to implement this
                # obj = ent_data.object_id
                # descriptor = ent_data.encoding
        return is_repeated

    def generate_alert(self, alert_msg: dict) -> BaseAlert:
        alert_type = self.get_alert_type_from_dict(alert_msg)
        if alert_type == AlertType.developer:
            # arrange the alert msg to the inlined alert
            actual_flow = json.loads(alert_msg.get("selectedFlow", {}).get("formValue", {}).get("json", ""))
            if "selectedFlow" in actual_flow and len(actual_flow) == 1:  # actual flow is just the selectedFlow def
                alert_msg["selectedFlow"] = actual_flow["selectedFlow"]
            else:  # actual flow is the full alert definition
                actual_flow.pop("_id", None)
                alert_msg.update(actual_flow)
            alert_type = self.get_alert_type_from_dict(alert_msg)

        if alert_type == AlertType.classification:
            try:
                # first try to treat it as a model classification alert
                alert = ClassificationModelAlert(alert_msg, context=self)
                return alert
            except ValueError:
                # if failed, treat it as a clip alert
                alert = ClipAlert(alert_msg, context=self)
                return alert
        # omulla - will enable once we will put the line crossing
        elif alert_type == AlertType.validationAlert:
            alarm_type = alert_msg.get("selectedFlow", {}).get("formValue", {}).get("alarmType", -1)
            if alarm_type == TriggerType.DHO.value:
                return DHOIntegratedAlert(alert_msg, context=self)
            elif alarm_type == TriggerType.DFO.value:
                return DFOIntegratedAlert(alert_msg, context=self)
            else:
                raise ValueError(f"Invalid integrated alert type, must be 0 (DHO) or 1 (DFO), got {alarm_type}")

        alert_class = self.ALERT_TYPE_TO_CLASS.get(alert_type)
        if alert_class is None:
            logger.error("alert type unrecognized")

        if alert_type == AlertType.appearance:
            alert_obj = alert_msg.get("configuration", {}).get("object", "unknown")
            if isinstance(alert_obj, int):
                alert_obj = self.context.class_handler.object_int_to_str(alert_obj)
            if alert_obj == "weapon":
                alert_class = WeaponAppearanceAlert
            elif alert_obj == "fire":
                alert_class = FireAlert

        return None if alert_class is None else alert_class(alert_msg, context=self)

    def get_alert_type_from_dict(self, alert_dict: Dict) -> AlertType:
        if "configuration" in alert_dict and "detection" in alert_dict["configuration"]:
            return AlertType(alert_dict["configuration"]["detection"])
        elif (
            "selectedFlow" in alert_dict
            and "category" in alert_dict["selectedFlow"]
            and "flowType" in alert_dict["selectedFlow"]
        ):
            return AlertType(
                self.alert_mapping.get(alert_dict["selectedFlow"]["category"], {}).get(
                    alert_dict["selectedFlow"]["flowType"], AlertType.unknown.value
                )
            )
        else:
            return AlertType.unknown

    @staticmethod
    def get_alert_id_from_dict(alert_dict: Dict) -> str:
        if "_id" in alert_dict:
            return alert_dict["_id"]
        else:
            return ""

    def get_required_classes(self):
        return list(self._required_classes.keys())

    def _update_alert(self, alert_id, update_dict):
        pass

    def _remove_alert(self, alert_dict):
        alert_id = AlertManager.get_alert_id_from_dict(alert_dict)

        # remove from bad alerts
        self.bad_alerts.pop(alert_id, None)
        self.alert_status = len(self.bad_alerts) == 0
        if alert_id in self._alerts.keys():
            alert = self._alerts.pop(alert_id)
            if alert.objects is not None:
                self._remove_from_class_tracker(alert.objects, alert.event_id)
        else:
            logger.error(f"Didnt find alert {alert_id} to remove")

    def _add_alert(self, alert_dict: Dict, check_existing=True):
        reason = "unknown"
        try:
            new_alert = self.generate_alert(alert_dict)
        except Exception as e:
            new_alert = None
            log_exception(logger, f"error occurred when generating alert with from msg: {alert_dict}", e)
            reason = str(e)
        if new_alert is None:
            alert_cat = alert_dict.get("selectedFlow", {}).get("category", -1)
            if alert_cat in [AlertCategory.Status.value, AlertCategory.Integrations.value]:
                logger.warning(f"Cloud based alert received with message: {alert_dict}, ignoring it")
            else:
                self.bad_alerts[alert_dict.get("_id", "unknown")] = reason
                self.alert_status = False
        else:
            self._alerts[new_alert.event_id] = new_alert
            self._last_raised_alerts[new_alert.event_id] = deque(maxlen=5)
            if new_alert.objects is not None:
                self._add_to_class_tracker(new_alert.objects, new_alert.event_id)
            # if the new alert is a known internal alert, remove all internal alerts
            if check_existing and type(new_alert) in [alert["type"] for alert in INTERNAL_ALERTS.values()]:
                ids_to_remove = [alert["id"] for alert in INTERNAL_ALERTS.values()]
                for def_id in ids_to_remove:
                    self._alerts.pop(def_id, None)
            elif check_existing and isinstance(new_alert, TamperAlert):
                self._alerts.pop(INTERNAL_TAMPERING_ALERT_ID, None)

    def validate_alert(self, alert_id, is_valid: bool, reason: str = "unknown"):
        if is_valid:
            self.bad_alerts.pop(alert_id, None)
            self.alert_status = True if not self.bad_alerts else False
            logger.info(f"Alert validated: {alert_id}")
        else:
            self.bad_alerts[alert_id] = reason
            self.alert_status = False
            logger.info(f"Alert invalidated: {alert_id}, Reason: {reason}")

    def _add_to_class_tracker(self, required_classes, event_id):
        for required_class in required_classes:
            if required_class not in self._required_classes:
                self._required_classes[required_class] = {event_id}
            else:
                self._required_classes[required_class].add(event_id)

    def _remove_from_class_tracker(self, required_classes, event_id):
        for required_class in required_classes:
            self._required_classes[required_class].remove(event_id)
            if len(self._required_classes[required_class]) == 0:
                del self._required_classes[required_class]

    def traffic_control_alert_info(self, alert_dict):
        traffic_alerts = {}
        if "actions" in alert_dict and alert_dict["actions"] is not None:
            if "directActions" in alert_dict["actions"] and alert_dict["actions"]["directActions"] is not None:
                for direct_msg in alert_dict["actions"]["directActions"]:
                    if "trafficController" in direct_msg and (direct_msg["trafficController"]):
                        traffic_alerts = {"address": direct_msg["address"], "port": direct_msg["port"]}
        return traffic_alerts

    def get_traffic_alert_controllers(self):
        controllers = []
        for alert in self._alerts.values():
            for action in alert.actions["messages"]:
                if action.protocol == CommProtocol.UDP.value and action.trafficController:
                    controllers.append({"action": action, "zones": alert.zones.keys()})
        return controllers

    def parse_alerts_msg(self, alert_msg):
        action = self.get_msg_action(alert_msg)
        if action is not None:
            alert_dict = alert_msg["alert"]
            if action is AlertsAction.add:
                self._add_alert(alert_dict)
            elif action is AlertsAction.remove:
                self._remove_alert(alert_dict)
            elif action is AlertsAction.update:
                # Storing policies before deleting alert
                alert_id = AlertManager.get_alert_id_from_dict(alert_dict)
                alert_policies = None
                if alert_id in self._alerts:
                    alert_policies = self._alerts[alert_id].get_alert_policies()

                self._remove_alert(alert_dict)
                self._add_alert(alert_dict)

                # Saving policies after reading the alert
                if alert_policies is not None:
                    self._alerts[alert_id].set_alert_policies(alert_policies)

    def validate_external_event(self, alert_msg):
        alert_id = alert_msg.get("eventId", "")
        data = alert_msg.get("data", {})
        if alert_id in self._alerts:
            self._alerts[alert_id].activateAlert(data)
        else:
            logger.error(f"Didnt find alert {alert_id} to activate from external event")

    def parse_policy_msg(self, policy_msg, init=False):
        alert_ids = self.get_alerts_id_policy(policy_msg)
        for alert_id in alert_ids:
            self._alerts[alert_id].handle_alert_policy(policy_msg, init)

    def parse_zone_msg(self, zone_msg, init=False):
        if zone_msg and "cameraIds" in zone_msg and self.context.camera_id in zone_msg["cameraIds"]:
            for alert in self._alerts.values():
                alert.handle_zone_policy(zone_msg, init)

    def get_alerts_id_policy(self, policy_msg):
        if "alertIds" in policy_msg:
            policy_alerts_ids = policy_msg["alertIds"]
            return list(set(policy_alerts_ids).intersection(self._alerts.keys()))
        return []

    @staticmethod
    def get_msg_action(alert_msg):
        if "alert" in alert_msg:
            alert_dict = alert_msg["alert"]
            action = AlertsAction(alert_dict["action"])
            return action
        return None

    def update_model(self, key, config):
        if key == "doors":
            for alert in self._alerts.values():
                if isinstance(alert, DoorAlert):
                    alert.update_model(config)
        elif key == "wc":
            for alert in self._alerts.values():
                if isinstance(alert, WeaponAppearanceAlert):
                    alert.update_model(config)

    def on_night_mode_changed(self, is_night):
        for alert in self._alerts.values():
            alert.on_night_mode_changed(is_night)

    def _add_internal_alerts(self):
        # check if configuration allows internal alerts (defaults on):
        alert_config = self._analytic_config.get("alertConfig", {})
        internal_alert_en = alert_config.get("internal_alerts", True)
        prob = alert_config.get("internal_alert_ratio", 0.75)

        if internal_alert_en:
            has_alert = False
            has_tampering_alert = not alert_config.get("internal_tampering_alert_enabled", True)

            def gen_internal_alert_from_config(alert_data):
                # generate alert structure
                internal_alert_conf = {
                    "selectedFlow": {
                        "category": alert_data["category"],
                        "flowType": alert_data["flowType"],
                        "formValue": alert_data["formValue"],
                    },
                    "settings": {
                        "priority": AlertsConfidence.low.value,
                        "confidence": AlertsConfidence.high.value,
                        "routing": AlertRouting.ROUTE_VCC_INTERNAL.value,
                        "internal": True,
                    },
                    "enabled": True,
                    "_id": alert_data["id"],
                }
                internal_alert = self.generate_alert(internal_alert_conf)
                self._alerts[internal_alert.event_id] = internal_alert
                self._last_raised_alerts[internal_alert.event_id] = deque(maxlen=5)

            # check if alert is already defined
            alert_types = {alert_def["type"] for alert_def in INTERNAL_ALERTS.values()}
            has_alert = any(type(a) in alert_types for a in self._alerts.values())
            has_tampering_alert = has_tampering_alert or any(isinstance(a, TamperAlert) for a in self._alerts.values())
            # randomly generate internal alerts if enabled
            if random.random() < prob:
                type_probability = alert_config.get("internal_alert_type_prob", {})
                self.internal_alert_limit = alert_config.get("internal_alert_limit", self.internal_alert_limit)
                if type_probability:
                    for k, v in type_probability.items():
                        if k in INTERNAL_ALERTS:
                            INTERNAL_ALERTS[k]["probability"] = v

                # no alert like internal alert, generate one
                if not has_alert:
                    # first decide which alert to generate:
                    alerts = list(INTERNAL_ALERTS.keys())
                    probabilities = [INTERNAL_ALERTS[alert]["probability"] for alert in alerts]
                    chosen_alert = random.choices(alerts, weights=probabilities, k=1)[0]
                    alert_data_ = INTERNAL_ALERTS[chosen_alert]
                    gen_internal_alert_from_config(alert_data_)

            if not has_tampering_alert:
                gen_internal_alert_from_config(TAMPERING_INTERNAL_ALERT)

    def get_alert(self, alert_id) -> Optional[BaseAlert]:
        return self._alerts.get(alert_id, None)

    def _apply_recognition_filter(
        self, recognition_filters: Dict, candidate: AlertCandidate, force: bool
    ) -> Optional[bool]:
        to_filter = True
        exclude_filter = True  # default is to exclude
        for ent_id in candidate.ent_ids:
            ent_filtered = None
            ent_data = self._entity_db.get_entity(ent_id)
            if ent_data is not None:
                rec_filter = recognition_filters[ent_data.object_id]
                if rec_filter["enabled"]:
                    exclude_filter = rec_filter["exclude"]
                    if ent_data.has_l1(rec_filter["l1_analyzer"]):
                        description = ent_data.attributes
                        is_match, extra = self.match_recognition(description, rec_filter)
                        if is_match is not None:
                            ent_filtered = is_match == rec_filter["exclude"]
                            if extra is not None:
                                if candidate.extra is None:
                                    candidate.extra = {}
                                candidate.extra.update(extra)
                else:
                    ent_filtered = False
            else:
                return True
            to_filter = None if ent_filtered is None else to_filter and ent_filtered
        if to_filter is None and force:
            return not exclude_filter
        return to_filter

    def match_recognition(self, description: Dict, recognition_filters: Dict) -> (Optional[bool], Optional[Dict]):
        identifiers = recognition_filters.get("identifiers")
        if identifiers is None or len(identifiers) == 0:
            return None, None
        if recognition_filters["l1_analyzer"] == AdvanceAnalyzerType.FACE.value:
            if "faceIdHexV2" in description:
                face_id = description["faceIdHexV2"].data
                face_conf = description["faceConfidence"]
                conf_ok = conf_order_map[face_conf] >= conf_order_map[self.min_face_conf]
                if conf_ok:
                    match_scores = face_id @ recognition_filters["identifiers"].T
                    mx_id = np.argmax(match_scores)
                    if match_scores[mx_id] > conf2cosine_th[face_conf]:
                        name = recognition_filters["metadata"][mx_id].get("name", "")
                        return True, {"entity_data": {"person_name": [name]}}
                    return False, None
        elif recognition_filters["l1_analyzer"] in [AdvanceAnalyzerType.ALPR.value, AdvanceAnalyzerType.LPREC.value]:
            if "plate" in description:
                plate = description["plate"][0].value.lower()
                return plate in recognition_filters["identifiers"], None
        return None, None

    @property
    def suspended_alerts(self):
        return self.active_alerts + self.active_recognition_alerts

    def register_alerts(self, alerts: List[AlertInfo]):
        for alert_info in alerts:
            alert = self._alerts.get(alert_info.eventId, None)
            if alert is not None:
                alert.register_alert(alert_info)
                if alert_info.internal:
                    self._internal_alert_counter += 1
                    if self._internal_alert_counter > self.internal_alert_limit:
                        # after several internal alerts, remove it
                        alert.enabled = False

        pass

    @property
    def batch_size(self):
        return self.context.batch_size

    def is_required_advance_tracking(self) -> bool:
        is_required = False
        types_required = []  # [CountingAlert]
        for alert in self._alerts.values():
            if type(alert) in types_required:
                is_required = True
                break
        return is_required

    def get_analytics_db(self) -> Dict:
        return self.context.analytic_db

    def on_db_update(self):
        # self._handle_shelves() # no longer required

        for alert in self._alerts.values():
            alert.on_db_update()

    def _handle_shelves(self):
        new_shelves = self.context.analytic_db.get("shelves", [])
        need_update = True
        if len(self.curr_shelves) == 0:
            need_update = len(new_shelves) > 0

        # Update current shelves
        self.curr_shelves = new_shelves

        if need_update:
            shelf_alert_id = "shelf_counter_internal_alert"
            if self.curr_shelf_counter is not None:
                self.bad_alerts.pop(self.curr_shelf_counter, None)
                self._alerts.pop(self.curr_shelf_counter, None)
            if len(self.curr_shelves) > 0:
                try:
                    shelf_counter_conf = {
                        "selectedFlow": {
                            "category": AlertCategory.Retail.value,
                            "flowType": RetailAlertType.EmptyShelfCounter.value,
                            "formValue": {"shelves": [shelf.id for shelf in self.curr_shelves]},
                        },
                        "settings": {
                            "priority": 0,
                            "confidence": AlertsConfidence.high.value,
                            "routing": AlertRouting.NO_ROUTING.value,
                            "internal": False,
                        },
                        "enabled": True,
                        "_id": shelf_alert_id,
                    }
                    shelf_counter_alert = self.generate_alert(shelf_counter_conf)
                    self.curr_shelf_counter = shelf_counter_alert.event_id
                    self._alerts[shelf_counter_alert.event_id] = shelf_counter_alert

                except Exception as e:
                    log_exception(logger, "error occurred when generating shelf counter alert", e)
                    self.bad_alerts[shelf_alert_id] = str(e)

    def handle_zone_regions(self, zone_regions: Dict[str, ZoneRegion]):
        for _id, zone_region in zone_regions.items():
            alert_id = _id
            if alert_id in self._alerts:
                logger.info(f"Removing existing zone counting alert for zone {_id}")
                self._remove_alert({"_id": alert_id})

            zone_counter_conf = {
                "selectedFlow": {
                    "category": AlertCategory.Tracking.value,
                    "flowType": TrackingType.RegionCounting.value,
                    "formValue": {
                        "objects": [
                            {"type": 0, "filters": {}, "strict": True},
                            {"type": 1, "filters": {}, "strict": True},
                        ],
                        "polygons": zone_region.polygons,
                    },
                },
                "selectedCamera": {
                    "edgeId": self.context.edge_id,
                    "cameraId": self.context.camera_id,
                    "markedIdx": zone_region.roi,
                },
                "settings": {
                    "priority": 0,
                    "confidence": AlertsConfidence.high.value,
                    "routing": AlertRouting.NO_ROUTING.value,
                    "internal": True,
                },
                "enabled": True,
                "_id": alert_id,
            }
            region_counting_alert = self.generate_alert(zone_counter_conf)
            self._alerts[region_counting_alert.event_id] = region_counting_alert

    def set_advance_fall(self, is_advance_bool: bool = False):
        if self._advance_fall != is_advance_bool:
            # update fall alert to trigger the change
            for alert in self._alerts.values():
                if isinstance(alert, FallAlert):
                    alert.set_advance_fall(is_advance_bool)
        self._advance_fall = is_advance_bool

    @property
    def advance_fall(self) -> bool:
        return self._advance_fall
