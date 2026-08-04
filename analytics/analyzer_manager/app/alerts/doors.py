from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np

from general.analyzer_general import logger, ROI_SHAPE
from general.core import AnalyticImage
from general.img_utils import crop_image
from level1.doors.doors_classifier import DoorsClassifier
from .alerts import BaseAlert
from .base_alerts import AlertCandidate, duration_unit_to_sec, duration_unit_to_str


class DoorState(int, Enum):
    CLOSE = -1
    UNKNOWN = 0
    OPEN = 1


class DoorAlertType(int, Enum):
    CLOSE = 0
    OPEN = 1
    CHANGED = 2


MINIMAL_DOOR_DURATION = 2000


@dataclass
class DoorInfo:
    name: str
    coordinates: np.ndarray
    alert_type: DoorAlertType
    duration: int = 0
    state: DoorState = DoorState.UNKNOWN
    last_known_state: DoorState = DoorState.UNKNOWN
    last_state_changed: int = 0
    next_check_ts: int = 0
    last_checked_ts: int = 0
    slice: np.ndarray = None
    last_scores: deque = None
    valid_state_checks = 0
    valid_state_checks_th = 0
    message_postfix: str = ""

    def __post_init__(self):
        # coordinates to roi for motion based activation
        self.roi = np.zeros(ROI_SHAPE).astype(int)
        coords = self.coordinates * [*ROI_SHAPE, *ROI_SHAPE]
        coords[:2] = np.floor(coords[:2])
        coords[2:] = np.ceil(coords[2:])
        coords = np.clip(coords, 0, ROI_SHAPE[0]).astype(int)
        self.slice = coords
        self.valid_state_checks_th = self.duration / 2000


def check_criteria(door: DoorInfo) -> bool:
    last_state = door.last_known_state
    is_active = False
    if door.duration <= MINIMAL_DOOR_DURATION:
        if (
            door.alert_type is DoorAlertType.CHANGED
            and door.state is not last_state
            and door.state is not DoorState.UNKNOWN
        ):
            is_active = True
        elif door.alert_type is DoorAlertType.OPEN and door.state is DoorState.OPEN:
            is_active = True
        elif door.alert_type is DoorAlertType.CLOSE and door.state is DoorState.CLOSE:
            is_active = True
    return is_active


def check_criteria_duration(door: DoorInfo, timestamp) -> bool:
    is_active = False
    if door.last_state_changed > 0 and timestamp - door.last_state_changed > door.duration:
        is_active = (door.alert_type == DoorAlertType.CLOSE and door.state == DoorState.CLOSE) or (
            door.alert_type == DoorAlertType.OPEN and door.state == DoorState.OPEN
        )
        # at least half of the duration we want to have a known state
        is_active &= door.valid_state_checks > door.valid_state_checks_th
        door.last_state_changed = 0
    return is_active


class DoorAlert(BaseAlert):
    type_name = "doors"
    doors: Dict[int, DoorInfo]
    motion_sensitivity = 10
    default_sensitivity = 0.25
    period = 1000
    history_size = 2

    def __init__(self, alert_dict: Dict, context):
        self.doors = {}
        super(DoorAlert, self).__init__(alert_dict, context)

        # local history
        analytic_config = self.context.get_config()
        doors_config = {}
        if "l1_models" in analytic_config and "attributes" in analytic_config["l1_models"]:
            doors_config = analytic_config["l1_models"]["attributes"].get("doors", {})
        doors_config["enable"] = True
        self.door_config = doors_config
        self.doors_classifier = DoorsClassifier(doors_config)
        self.margins = [self.doors_classifier.required_margins] * 2
        if self.sensitivity is None:
            self.sensitivity = self.default_sensitivity
        self.l1_required = False
        self.set_flow_values()

    def update_model(self, config):
        self.door_config.update(config)
        self.doors_classifier = DoorsClassifier(self.door_config)

    def set_flow_values(self):
        elements = self.formValue.get("doors", [])
        event_type = DoorAlertType(self.formValue.get("eventType", DoorAlertType.CHANGED.value))
        duration = int(self.formValue.get("duration", 0))  # to ms
        units = self.apply_from_dict("durationUnit", self.formValue, 0)
        duration *= duration_unit_to_sec[units]

        # no logic in duration for changed event
        if event_type is DoorAlertType.CHANGED:
            duration = 0

        for item in elements:
            if item["cameraId"] == self.cameraId:
                did = item["id"]
                d = DoorInfo(
                    item["name"],
                    np.asarray(item["position"]),
                    event_type,
                    duration * 1000,
                )
                d.last_scores = deque(np.repeat(0, self.history_size), maxlen=self.history_size)
                if event_type is not DoorAlertType.CHANGED:
                    d.message_postfix = f"for more than {duration} {duration_unit_to_str[units]}"
                self.doors[did] = d

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        active_alerts = []
        for door_id in self.doors:
            door = self.doors[door_id]
            next_check_ts = 0

            if (
                door.valid_state_checks < door.valid_state_checks_th
                or np.std(np.asarray(door.last_scores)) > self.sensitivity
                or np.any(
                    motion_data.unified[door.slice[1] : door.slice[3], door.slice[0] : door.slice[2]]
                    >= self.motion_sensitivity
                )
            ):
                next_check_ts = self.calc_next_check(door)
            else:
                # as long as there is no motion, we can run the analytics in loger periods
                if door.last_checked_ts + self.period * 5 > images[-1].timestamp:
                    continue

            for image in images:
                if image.timestamp >= next_check_ts:
                    is_active = False
                    crop = crop_image(image.frame, door.coordinates, margins=self.margins, bgr_map=False)
                    state, score = self.doors_classifier.forward_on_crop_list([crop])
                    score = score[0]
                    if state[0] == 0:
                        score = score * -1
                    new_state = self._door_state_from_score(score)
                    last_state = door.last_known_state
                    door.valid_state_checks += 0 if new_state == DoorState.UNKNOWN else 1
                    log_line = (
                        f"time: {image.timestamp}, detected state: {new_state.name},"
                        + f"score: {score}, valid count: {door.valid_state_checks}"
                    )
                    logger.debug(log_line)
                    # print(log_line)

                    door.last_scores.append(score)
                    if new_state != DoorState.UNKNOWN and new_state != last_state:
                        door.state = self._door_state_from_score(np.mean(np.asarray(door.last_scores)))
                        if door.last_known_state is DoorState.UNKNOWN:
                            door.last_known_state = new_state
                        else:
                            is_active = check_criteria(door)
                            if door.state != DoorState.UNKNOWN:
                                door.last_known_state = door.state
                                door.last_state_changed = image.timestamp * (door.duration > MINIMAL_DOOR_DURATION)
                                door.valid_state_checks = 0
                    elif new_state != DoorState.UNKNOWN and door.last_state_changed > 0:
                        is_active = check_criteria_duration(door, image.timestamp)

                    # update next time of sample
                    door.last_checked_ts = door.next_check_ts = image.timestamp
                    next_check_ts = self.calc_next_check(door, new_state)
                    if is_active:
                        candidate = self.build_alert_candidate(
                            image.timestamp, ent_ids=door_id, extra={"state": door.state}, crops=[crop]
                        )
                        active_alerts.append(candidate)
                        logger.info(f"Door alert: validity checks {door.valid_state_checks}")
        return active_alerts

    def calc_next_check(self, door: DoorInfo, new_state: DoorState = None) -> int:
        if new_state is not None and new_state is not DoorState.UNKNOWN and new_state != door.state:
            return door.last_checked_ts + int(self.period / 4)  # increase pace if state changes
        elif door.state is DoorState.UNKNOWN:
            return door.last_checked_ts + int(self.period / 2)  # increase pace if we cant tell the door state
        else:
            return door.last_checked_ts + self.period  # default pace, if no motion the pace is reduced

    def _door_state_from_score(self, score) -> DoorState:
        if np.abs(score) > self.sensitivity:
            return DoorState(np.sign(score))
        return DoorState.UNKNOWN

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)
        door_info = self.doors[candidate.ent_ids[0]]
        alert_info._crop = candidate.crops
        new_state = "closed" if candidate.extra["state"] < 0 else "opened"
        alert_info.alertMessage = f"Door {door_info.name} is {new_state} {door_info.message_postfix}"
        return alert_info
