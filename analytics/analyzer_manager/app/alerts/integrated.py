from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

from general.core import AnalyticImage, AlertInfo
from .base_alerts import AlertCandidate, ObjectAlert, duration_unit_to_sec
from .linecrossing import LineCrossingAlert
from .loitering import LoiteringAlert


# from general.analyzer_general import logger, log_exception

class TriggerType(IntEnum):
    DHO = 0  # Door Held Open - alert if NO presence
    DFO = 1  # Door Forced Open - alert if presence detected
    UNKNOWN = 99


@dataclass
class TriggerFlag:
    is_triggered: bool = False
    timestamp: int = -1
    trigger_type: TriggerType = TriggerType.DHO

    def reset(self):
        self.is_triggered = False
        self.timestamp = -1
        self.trigger_type = TriggerType.DHO


def generate_clear_alert_note(is_clear_event: bool, trigger_type: TriggerType) -> str:

    if is_clear_event:
        return (
            "Alert is cleared, person is holding the door"
            if trigger_type == TriggerType.DHO
            else "Alert is cleared, no person detected entering the door"
        )
    else:
        return (
            "Alert is active, door is being held open"
            if trigger_type == TriggerType.DHO
            else "Alert is active, person detected entering the door"
        )
    # no need to log error- if we got here the trigger must be of a known type


class IntegratedAlertMixin:
    """
    Mixin class that provides common functionality for integrated alerts (DHO/DFO).
    This mixin should be used with LineCrossingAlert or LoiteringAlert as the parent class.
    """

    form_value: str = "val_type"
    hysteresis_threshold_ms = 100000
    motion_filter = True
    history_filter = True
    pre_trigger_ms = 3 * 1000
    post_trigger_ms = 3 * 1000
    trigger_type: TriggerType = TriggerType.DHO  # override in subclass

    def _init_integrated_alert(self, alert_dict: Dict, context):
        """Initialize common integrated alert functionality. Call from __init__."""
        self.alert_on_tracked_only = True
        self.has_local_actions = True
        self.internal_alert = False  # not internal — these are real user-facing alerts
        self.alerted_ids = {}
        self._trigger_alerted_ids = {}  # separate from parent's alerted_ids (stores int timestamps)
        self.activation_data = []
        self.hold_in_sec = 5
        self.alert_activated = False
        self.positive_alert = True

        class_handler = self.context.get_class_handler()

        # look for people or vehicles
        self.object_ids = [
            class_handler.person_value,
            # class_handler.vehicle_value,
        ]
        self.objects = [
            class_handler.object_int_to_str(class_handler.person_value),
            # class_handler.object_int_to_str(class_handler.vehicle_value),
        ]

        self.filters = {
            class_handler.person_value: {},
            # class_handler.vehicle_value: {},
        }
        self.filtersDisabled = {
            class_handler.person_value: True,
            # class_handler.vehicle_value: False,
        }
        self.strict_filters = {
            class_handler.person_value: True,
            # class_handler.vehicle_value: True,
        }
        self.recognition_filters = {
            class_handler.person_value: {"enabled": False},
            # class_handler.vehicle_value: {"enabled": False},
        }

        analytic_config = self.context.get_config()
        if "alertConfig" in analytic_config:
            self.hysteresis_threshold_ms = (
                analytic_config["alertConfig"]
                .get("objectAppearance", {})
                .get("hysteresis", self.hysteresis_threshold_ms)
            )
        
        for object_id in self.filters:
            self.l1_required_type[object_id] = None
        self.l1_required = False

        self.max_hold = 30000  # leave time to overcome blacklist\filter issues
        self.cands_buffer = []
        self.trigger_flag = TriggerFlag()

    def set_flow_values(self):
        super().set_flow_values()
        # LoiteringAlert.set_flow_values writes threshold in seconds; convert to ms
        # after LoiteringAlert.__init__ has created trackers.
        if isinstance(self, LoiteringAlert) and hasattr(self, "trackers"):
            self.threshold *= 1000
        self.val_type = self.formValue.get(self.form_value, None)
        self.pre_trigger_ms = self.formValue.get("val_pre_trigger_ms", self.pre_trigger_ms)
        self.post_trigger_ms = self.formValue.get("val_post_trigger_ms", self.post_trigger_ms)
        self.alarm_type = int(self.formValue.get("alarmType", TriggerType.UNKNOWN.value))
        self._parse_hold_duration()

    def _parse_hold_duration(self):
        """Parse hold duration from formValue (user-configured door hold time)."""
        if self.formValue:
            count = self.apply_from_dict("duration", self.formValue, 0)
            units = self.apply_from_dict("durationUnit", self.formValue, 0)
            self.hold_in_sec = count * duration_unit_to_sec[units]

    def activateAlert(self, data):
        self.activation_data.append(data)
        self.alert_activated = True

    def _cleanup_expired_candidates(self, buffer: List[AlertCandidate], current_ts: int) -> List[AlertCandidate]:
        """Remove candidates older than the specified time window."""
        return [
            cand for cand in buffer if current_ts - cand.timestamp <= self.pre_trigger_ms
        ]  # and cand.ent_ids[0] not in self.active_ids]

    def _get_active_presence(self, batch_ts: int) -> Optional[AlertCandidate]:
        """
        Check if there are currently tracked entities in the zone (DHO only).
        LoiteringAlert's one-shot candidates expire from the buffer, but trackers
        remain active as long as the entity is present. Use this to detect presence
        when the buffer is empty but entities are still in the zone.
        """
        if not isinstance(self, LoiteringAlert) or not hasattr(self, "trackers"):
            return None
        for ent_id, tracker in self.trackers.items():
            # Entity must be recently seen (within last batch window) and have sufficient dwell
            if batch_ts - tracker["last_ts"] <= self.pre_trigger_ms:
                dwell = tracker["last_ts"] - tracker["first_ts"]
                if dwell > self.threshold:
                    cand = self.build_alert_candidate(tracker["last_ts"], ent_ids=ent_id)
                    return cand
        return None

    def _validate_trigger(self, batch_ts) -> List[AlertCandidate]:
        """
        Validate trigger based on accumulated candidates.
        Processes each activation independently.
        DFO: alerts when presence detected (clearEvent=False when cands found)
        DHO: alerts when NO presence detected (clearEvent=True when cands found)
        """
        results = []
        num_cands = len(self.cands_buffer)
        source_cand = self.cands_buffer[-1] if num_cands > 0 else None

        # For DHO: if buffer is empty, check if entities are currently in the zone
        if num_cands == 0 and self.trigger_type == TriggerType.DHO:
            active_cand = self._get_active_presence(batch_ts)
            if active_cand is not None:
                source_cand = active_cand
                num_cands = 1

        for activation_data in self.activation_data:
            extra_fields = {"extra_fields": activation_data}
            ts = activation_data.get("timestamp", self.trigger_flag.timestamp)

            if num_cands > 0:
                # Per-entity hysteresis: suppress if same entity alerted recently
                ent_ids_to_report = []
                for ent_id in source_cand.ent_ids:
                    last_seen = self._trigger_alerted_ids.get(ent_id, 0)
                    if last_seen == 0 or ts >= last_seen + self.hysteresis_threshold_ms:
                        ent_ids_to_report.append(ent_id)
                    self._trigger_alerted_ids[ent_id] = ts

                if len(ent_ids_to_report) > 0:
                    # DFO: presence detected -> real alert (clearEvent=False)
                    # DHO: presence detected -> clear alert (clearEvent=True)
                    is_cleared = self.trigger_type == TriggerType.DHO
                    extra_fields["clearEvent"] = is_cleared
                    extra_fields["eventNote"] = generate_clear_alert_note(is_cleared, self.trigger_type)
                    extra_fields["clearDuration"] = self.hold_in_sec
                    cand = self.build_alert_candidate(
                        ts,
                        ent_ids=source_cand.ent_ids,
                        var_data=source_cand.var_data,
                        extra={"extra_fields": extra_fields},
                        crops=source_cand.crops,
                        validation_images=source_cand.validation_images,
                    )
                    cand.internal_alert = 0  # mark as non-internal alert
                    results.append(cand)
                else:
                    # All entities suppressed by hysteresis — fall through to no-presence path
                    is_cleared = self.trigger_type == TriggerType.DFO
                    extra_fields["clearEvent"] = is_cleared
                    extra_fields["eventNote"] = generate_clear_alert_note(is_cleared, self.trigger_type)
                    extra_fields["clearDuration"] = self.hold_in_sec
                    cand = self.build_alert_candidate(ts, extra={"extra_fields": extra_fields})
                    cand.internal_alert = 0
                    results.append(cand)
            elif batch_ts - self.trigger_flag.timestamp > self.post_trigger_ms:
                # DFO: no presence -> clear alert (clearEvent=True)
                # DHO: no presence -> real alert (clearEvent=False)
                is_cleared = self.trigger_type == TriggerType.DFO
                extra_fields["clearEvent"] = is_cleared
                extra_fields["eventNote"] = generate_clear_alert_note(is_cleared, self.trigger_type)
                extra_fields["clearDuration"] = self.hold_in_sec
                cand = self.build_alert_candidate(ts, extra={"extra_fields": extra_fields})
                cand.internal_alert = 0  # mark as non-internal alert
                results.append(cand)

        if results:
            self.alert_activated = False
            self.activation_data = []
            self.cands_buffer = []
            self.trigger_flag.reset()

        return results

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        # cleanup expired candidates
        self.cands_buffer = self._cleanup_expired_candidates(self.cands_buffer, images[-1].timestamp)

        # check for new events from parent class
        self.cands_buffer += super().is_active_batch(images, motion_data, batch_data)

        # check for a new trigger
        if self.alert_activated and not self.trigger_flag.is_triggered:
            # Snap trigger timestamp to nearest image frame
            timestamps = np.array([img.timestamp for img in images])
            raw_ts = (
                self.activation_data[0].get("timestamp", images[-1].timestamp)
                if self.activation_data
                else images[-1].timestamp
            )
            ts_idx = np.argmin(np.abs(timestamps - raw_ts))
            first_ts = int(timestamps[ts_idx])
            self.trigger_flag = TriggerFlag(True, first_ts, self.trigger_type)

        # determine trigger validity against accumulated candidates
        if self.trigger_flag.is_triggered:
            return self._validate_trigger(images[-1].timestamp)
        return []
    
    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = ObjectAlert.build_alert_info(self, candidate)
        extra_fields = candidate.extra.get("extra_fields", {}) if candidate.extra else {}
        alert_info.alertMessage = extra_fields.get("eventNote", None)
        alert_info.clearEvent = extra_fields.get("clearEvent", None)
        return alert_info


class DFOIntegratedAlert(IntegratedAlertMixin, LineCrossingAlert):
    """Door Forced Open Alert - alerts when presence IS detected after trigger."""

    type_name = "dfoAlert"
    trigger_type = TriggerType.DFO

    def __init__(self, alert_dict: Dict, context):
        super(DFOIntegratedAlert, self).__init__(alert_dict, context)
        self._init_integrated_alert(alert_dict, context)
        self.set_flow_values()

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
            if ent in self.alerted_ids:
                del self.alerted_ids[ent]
            if ent in self.id_location:
                del self.id_location[ent]
            self._trigger_alerted_ids.pop(ent, None)


class DHOIntegratedAlert(IntegratedAlertMixin, LoiteringAlert):
    """Door Held Open Alert - alerts when NO presence is detected after trigger."""

    type_name = "dhoAlert"
    trigger_type = TriggerType.DHO

    def __init__(self, alert_dict: Dict, context):
        super(DHOIntegratedAlert, self).__init__(alert_dict, context)
        self._init_integrated_alert(alert_dict, context)
        self.set_flow_values()

    # def build_alert_info(self, candidate: AlertCandidate):
    #     from .base_alerts import BaseAlert
    #     return BaseAlert.build_alert_info(self, candidate)

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
            if ent in self.alerted_ids:
                del self.alerted_ids[ent]
            self._trigger_alerted_ids.pop(ent, None)
