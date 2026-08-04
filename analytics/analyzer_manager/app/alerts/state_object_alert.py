from copy import copy
from typing import Dict, List, Set

from alerts.base_alerts import BaseAlert, AlertCandidate, duration_unit_to_sec, duration_unit_to_str
from general.core import AnalyticImage, MotionData, BatchDataResolver, AlertInfo
from general.img_utils import crop_image


class ObjectStateAlert(BaseAlert):
    min_duration = 2000
    change_object_latency = True

    def __init__(self, alert_dict: Dict, context):
        super(ObjectStateAlert, self).__init__(alert_dict, context)
        entity_db = context.get_entity_db()
        self.state_object_manager = entity_db.state_object_manager
        self.object_id = -1
        self.set_flow_values()
        if not self.state_object_manager.enabled or self.object_id not in self.state_object_manager.objects:
            self.enable = False
            raise ValueError("State object manager is not enabled or object_id is invalid")
        analytic_config = self.context.get_config()

        if "state_objects" in analytic_config:
            self.change_object_latency = analytic_config["state_objects"].get(
                "alert_changes_object_latency", self.change_object_latency
            )
        self.object_handler = self.state_object_manager.objects[self.object_id]

        # state parameters
        self.is_alerted = False

    def set_flow_values(self):
        self.object_id = self.formValue.get("objectState", {}).get("id", -1)

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super().build_alert_info(candidate)
        alert_info.alertMessage = self.format_message(candidate)
        return alert_info

    def format_message(self, candidate: AlertCandidate) -> str:
        return self.alert_message


class ObjectStateTransitionAlert(ObjectStateAlert):
    from_state: Set[int]
    to_state: Set[int]
    min_duration: int = 1000

    def __init__(self, alert_dict: Dict, context):
        super(ObjectStateTransitionAlert, self).__init__(alert_dict, context)
        self.last_change = None
        self.alert_message = "%s changed state from %s to %s"
        if self.change_object_latency:
            self.object_handler.set_latency(1)

    def set_flow_values(self):
        super(ObjectStateTransitionAlert, self).set_flow_values()
        self.from_state = set(self.formValue.get("fromStateIds", []))
        self.to_state = set(self.formValue.get("toStateIds", []))
        if len(self.from_state) * len(self.to_state) == 0:
            raise ValueError("Both fromState and toState must be specified")

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BatchDataResolver
    ) -> List[AlertCandidate]:
        candidates: List[AlertCandidate] = []

        changes = self.object_handler.batch_state_changes
        for change in changes:
            from_match = change.stateIdFrom in self.from_state and change.stateIdFrom >= 0
            to_match = change.stateIdTo in self.to_state and change.stateIdTo >= 0
            if from_match and to_match:
                self.is_alerted = True
                self.last_change = copy(change)
            else:
                self.is_alerted = False
                self.last_change = None

        if self.is_alerted:
            #  now check duration condition
            if images[-1].timestamp - self.object_handler.last_state_change > self.min_duration:
                extra_dict = {
                    "extra_fields": {
                        "customObjectStateId": self.object_id,
                        "objectName": self.object_handler.name,
                        "stateIdFrom": self.last_change.stateIdFrom,
                        "stateNameFrom": self.object_handler.states_data[self.last_change.stateIdFrom].description,
                        "stateIdTo": self.last_change.stateIdTo,
                        "stateNameTo": self.object_handler.states_data[self.last_change.stateIdTo].description,
                    }
                }
                crop = crop_image(images[0].frame, self.object_handler.coordinates, margins=[0, 0], bgr_map=True)
                candidate = self.build_alert_candidate(images[-1].timestamp, extra=extra_dict, crops=[crop])
                candidates.append(candidate)
                self.is_alerted = False

        return candidates

    def format_message(self, candidate: AlertCandidate) -> str:
        state_from = candidate.extra.get("extra_fields", {}).get("stateNameFrom", "Unknown")
        state_to = candidate.extra.get("extra_fields", {}).get("stateNameTo", "Unknown")
        return self.alert_message % (self.object_handler.name, state_from, state_to)


class ObjectStateChangedAlert(ObjectStateTransitionAlert):

    def __init__(self, alert_dict: Dict, context):
        super(ObjectStateChangedAlert, self).__init__(alert_dict, context)

        # any change
        states = [s for s in self.object_handler.state_ids if s >= 0]
        self.from_state = set(states)
        self.to_state = set(states)

    def set_flow_values(self):
        super(ObjectStateTransitionAlert, self).set_flow_values()


class ObjectStateDurationAlert(ObjectStateAlert):
    duration: int = 0  # in milliseconds
    target_states: Set[int]
    last_alerted_state: int = -1

    def __init__(self, alert_dict: Dict, context):
        super(ObjectStateDurationAlert, self).__init__(alert_dict, context)
        if self.change_object_latency:
            new_latency = 0
            if self.duration >= 30 * 1000:
                new_latency = 9
            elif self.duration >= 15 * 1000:
                new_latency = 7
            elif self.duration >= 10 * 1000:
                new_latency = 5
            self.object_handler.set_latency(new_latency)

    def set_flow_values(self):
        super().set_flow_values()
        self.target_states = set(self.formValue.get("stateIds", []))
        if not self.target_states:
            raise ValueError("target states must be specified")

        count = self.apply_from_dict("duration", self.formValue, 0)
        units = self.apply_from_dict("durationUnit", self.formValue, 0)
        self.duration = max(count * duration_unit_to_sec[units] * 1000, self.min_duration)  # convert to milliseconds

        self.alert_message = f"%s in state %s for more than {count} {duration_unit_to_str[units]}"

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BatchDataResolver
    ) -> List[AlertCandidate]:
        candidates: List[AlertCandidate] = []
        if self.is_alerted:
            if self.object_handler.current_state != self.last_alerted_state:
                self.is_alerted = False
        elif self.object_handler.current_state in self.target_states:
            state_duration = images[-1].timestamp - self.object_handler.last_state_change
            if state_duration >= self.duration:
                self.is_alerted = True
                extra_dict = {
                    "extra_fields": {
                        "customObjectStateId": self.object_id,
                        "objectName": self.object_handler.name,
                        "stateId": self.object_handler.current_state,
                        "stateName": self.object_handler.states_data[self.object_handler.current_state].description,
                        "duration": state_duration,
                    }
                }
                crop = crop_image(images[0].frame, self.object_handler.coordinates, margins=[0, 0], bgr_map=True)
                candidate = self.build_alert_candidate(images[-1].timestamp, extra=extra_dict, crops=[crop])
                candidates.append(candidate)
                self.last_alerted_state = self.object_handler.current_state

        return candidates

    def format_message(self, candidate: AlertCandidate) -> str:
        state_name = candidate.extra.get("extra_fields", {}).get("stateName", "Unknown")
        return self.alert_message % (self.object_handler.name, state_name)
