from typing import Dict, List, Optional

import cv2
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious  # noqa

from alerts.base_alerts import ObjectAlert, AlertCandidate, duration_unit_to_sec, duration_unit_to_str
from general.analyzer_general import logger
from general.core import (
    AlertRouting,
    AnalyticImage,
    BatchDataResolver,
    AlertInfo,
    MotionData,
    ProtectedGearWearOptions,
    ProtectedGearType,
)
from general.img_utils import crop_image
from level1.pa_recognition.person_attributes import AttributesModel as PaModel
from level1.pa_recognition.ppe import PpeModel


class PersonData:
    def __init__(self, first_seen: int, last_seen: int):
        self.is_alerted: bool = False  # has this person already triggered an alert
        self.first_seen: int = first_seen
        self.last_seen: int = last_seen
        self.best_crop: Optional[np.ndarray] = (
            None  # best crop image for classification (meets size and occlusion requirements)
        )
        self.best_bbox: Optional[np.ndarray] = None  # bbox of best crop for IoU comparison
        self.best_occlusion: float = 1.0  # occlusion of the best crop (lower is better)
        self.classified_bbox: Optional[np.ndarray] = None  # bbox of last classified crop for IoU change detection
        self.classifier_runs: int = 0  # number of classifier invocations for this entity
        self.last_classified_ts: int = 0  # timestamp of last classifier run
        self.positive_count: int = 0  # total positive classifications
        self.total_classifications: int = 0  # total valid classifications (True/False, excluding None)


class PpeAlert(ObjectAlert):
    type_name = "ppe"
    default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE
    motion_filter = False
    history_filter = False
    ambient_motion_filter = False
    no_activity_timeout = 10000  # ms
    occlusion_threshold = 0.4
    crop_iou_change_th = 0.8  # require 20% IoU change to replace crop
    occlusion_improvement_th = 0.075  # allow up to 7.5% worse occlusion (absolute, 0-1 scale)
    max_classifier_trials = 10
    cooldown_infer_interval = 2000  # ms, minimum time between classifier runs per entity
    min_vote_samples = 2  # minimum classifications needed for majority decision
    loiter_duration = 2000  # ms, default time to trigger alert if condition is met
    crop_margin = 0.05  # default margin for crop size estimation

    def __init__(self, alert_dict: Dict, context):
        super().__init__(alert_dict, context)
        self.set_flow_values()

        # Classifier setup - use dedicated PPE model for hard_hat, PA model for vest
        analytic_config = context.get_config()
        l1_attrs = analytic_config.get("l1_models", {}).get("attributes", {}).get("human_parsing", {})

        if self.gear_type == ProtectedGearType.hard_hat.value:
            classifier_config = l1_attrs.get("ppe", {})
            classifier_config["enable"] = True
            self.classifier = PpeModel(classifier_config)
            self.score_idx = self._find_index("hard_hat")
        else:
            classifier_config = l1_attrs.get("pa", {})
            classifier_config["enable"] = True
            self.classifier = PaModel(classifier_config)
            self.score_idx = self._find_index("safety_vest")
            self.undef_idx = self.classifier.interpreter._mapping[self.score_idx].undefined_index

        self.min_crop_size = self.classifier.args.minimal_crop_size

        # Disable L1 pipeline — classification is done directly in is_active_batch
        self.l1_required = False
        for obj_id in self.l1_required_type:
            self.l1_required_type[obj_id] = None

        self.tracking_data: Dict[int, PersonData] = {}
        self.excluded_ents: set = set()

    def set_flow_values(self):
        count, units_str = self._parse_duration()
        self.gear_type = (
            self.formValue.get("gear", ProtectedGearType.hard_hat.value)
            if self.formValue
            else ProtectedGearType.hard_hat.value
        )
        attr_name = "helmet" if self.gear_type == ProtectedGearType.hard_hat.value else "vest"
        if self.gear_type != ProtectedGearType.hard_hat.value:
            self.routing = AlertRouting.NO_ROUTING

        wear_value = (
            self.formValue.get("wear", ProtectedGearWearOptions.not_wearing.value)
            if self.formValue
            else ProtectedGearWearOptions.not_wearing.value
        )
        self.min_vote_samples = (
            self.formValue.get("min_vote_samples", self.min_vote_samples) if self.formValue else self.min_vote_samples
        )

        # Adaptive cooldown: if loiter_duration is too short to collect min_vote_samples
        # at the normal cooldown rate, reduce it until enough samples are gathered
        if self.loiter_duration <= self.cooldown_infer_interval * self.min_vote_samples:
            self.initial_cooldown = max(0, self.loiter_duration // (self.min_vote_samples + 1))
        else:
            self.initial_cooldown = self.cooldown_infer_interval

        if wear_value == ProtectedGearWearOptions.not_wearing.value:
            self.alert_message = f"Workers Don't wear a safety {attr_name} for more than {count} {units_str}"
            self.special_filter = "not wearing"
            self.wear_expected = False
        else:
            self.alert_message = f"Workers wear a safety {attr_name} for more than {count} {units_str}"
            self.special_filter = "wearing"
            self.wear_expected = True

    def _parse_duration(self):
        count = 0
        units = 0
        if self.formValue:
            self.positive_alert = True
            count = self.apply_from_dict("duration", self.formValue, self.loiter_duration // 1000)
            units = self.apply_from_dict("durationUnit", self.formValue, 0)
        self.loiter_duration = int(count * duration_unit_to_sec[units] * 1000)
        units_str = duration_unit_to_str[units]
        return count, units_str

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        alert_info.alertMessage = self.alert_message

    def check_crop_quality(
        self, bbox: np.ndarray, person: PersonData, frame_shape: tuple, occlusion: float = 0.0
    ) -> bool:
        """Check if bbox meets minimum size, is not occluded, occlusion isn't worse, and has changed enough."""
        if occlusion > self.occlusion_threshold:
            return False
        # Estimate crop pixel dimensions from normalized bbox + frame size (with default 5% margins)
        image_h, image_w = frame_shape[:2]
        est_w = (bbox[2] - bbox[0]) * image_w
        est_h = (bbox[3] - bbox[1]) * image_h
        est_w += 2 * int(est_w * self.crop_margin)
        est_h += 2 * int(est_h * self.crop_margin)
        if not (est_h > self.min_crop_size or est_w > self.min_crop_size):
            return False
        if occlusion - person.best_occlusion > self.occlusion_improvement_th:
            return False
        if person.best_bbox is None:
            return True
        iou = bbox_ious(
            np.atleast_2d(bbox).astype(float),
            np.atleast_2d(person.best_bbox).astype(float),
        )[0, 0]
        return iou < self.crop_iou_change_th

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False):
        """Skip L1/filter checks — classification is done directly in is_active_batch."""
        return True

    def _find_index(self, label: str) -> int:
        for idx, attr in self.classifier.interpreter._mapping.items():
            if attr.label == label:
                return idx
        raise ValueError(f"Attribute '{label}' not found in classifier mapping")

    def classify(self, person: PersonData) -> Optional[bool]:
        """Run PPE classifier on person's best crop.
        Returns True if alert condition is met, False if not, None if unreliable."""
        results = self.classifier.forward_on_crop_list([person.best_crop])
        scores = results[0]["scores"]
        confidence = self.classifier.interpreter.calc_confidence_score(scores)
        if confidence < 0:
            return None  # unreliable crop, retry later
        if self.gear_type != ProtectedGearType.hard_hat.value:
            undef_score = scores[self.undef_idx]
            undef_th = self.classifier.interpreter._mapping[self.undef_idx].high_conf
            if undef_score > undef_th:
                return None  # undefined/unreliable for vest
        th = self.classifier.interpreter._mapping[self.score_idx].high_conf
        is_wearing = scores[self.score_idx] >= th
        return bool(is_wearing) == self.wear_expected

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BatchDataResolver
    ) -> List[AlertCandidate]:
        alert_candidates = []
        vars_data = self.get_batch_candidates_data(batch_data)
        # Determine current timestamp
        current_ts = 0
        if len(vars_data) > 0:
            # Filter out already-excluded entities early
            if self.excluded_ents:
                mask = ~np.isin(vars_data[:, BatchDataResolver.ID].astype(int), list(self.excluded_ents))
                vars_data = vars_data[mask]
        if len(vars_data) > 0:
            current_ts = np.max(vars_data[:, BatchDataResolver.TIMESTAMP])
        elif images:
            current_ts = images[-1].timestamp
        # Phase 1: Update tracking from detections
        if len(vars_data) > 0:
            overlaps = batch_data.overlaps
            alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()
            for ent_id in alert_ids:
                var_data = vars_data[vars_data[:, BatchDataResolver.ID] == ent_id]
                last_det = var_data[-1]  # use the last detection row for this entity
                ts = last_det[BatchDataResolver.TIMESTAMP]
                bbox = last_det[BatchDataResolver.POS]
                frame_idx = int(last_det[BatchDataResolver.FRAME_ID])
                crop_idx = int(last_det[BatchDataResolver.INDEX])
                occlusion = overlaps[crop_idx] if crop_idx < len(overlaps) else 0.0

                if ent_id in self.tracking_data:
                    person = self.tracking_data[ent_id]
                    person.last_seen = ts
                    if self.check_crop_quality(bbox, person, images[frame_idx].frame.shape, occlusion):
                        person.best_crop = crop_image(
                            images[frame_idx].frame, bbox, [self.crop_margin, self.crop_margin], bgr_map=False
                        )
                        person.best_bbox = bbox
                        person.best_occlusion = occlusion
                else:
                    person = PersonData(first_seen=ts, last_seen=ts)
                    if self.check_crop_quality(bbox, person, images[frame_idx].frame.shape, occlusion):
                        person.best_crop = crop_image(
                            images[frame_idx].frame, bbox, [self.crop_margin, self.crop_margin], bgr_map=False
                        )
                        person.best_bbox = bbox
                        person.best_occlusion = occlusion
                    self.tracking_data[ent_id] = person

        # Phase 2: Classify and check majority vote when dwell is met
        for ent_id, person in self.tracking_data.items():
            if person.is_alerted or person.best_crop is None or ent_id in self.excluded_ents:
                continue
            if person.classifier_runs >= self.max_classifier_trials:
                self.excluded_ents.add(ent_id)
                continue
            effective_cooldown = (
                self.initial_cooldown
                if person.total_classifications < self.min_vote_samples
                else self.cooldown_infer_interval
            )
            if person.last_classified_ts > 0 and (current_ts - person.last_classified_ts) < effective_cooldown:
                continue
            if person.classified_bbox is not None and person.best_bbox is not None:
                iou = bbox_ious(
                    np.atleast_2d(person.best_bbox).astype(float),
                    np.atleast_2d(person.classified_bbox).astype(float),
                )[0, 0]
                if iou >= self.crop_iou_change_th:
                    continue  # crop hasn't changed enough since last classification, skip
            result = self.classify(person)
            person.classified_bbox = person.best_bbox
            person.classifier_runs += 1
            person.last_classified_ts = current_ts
            if result is True:
                person.positive_count += 1
                person.total_classifications += 1
            elif result is False:
                person.total_classifications += 1
            # None — uncertain, don't count

            # Check majority vote once dwell threshold is met and enough samples collected
            dwell = current_ts - person.first_seen
            if dwell >= self.loiter_duration and person.total_classifications >= self.min_vote_samples:
                if person.positive_count > person.total_classifications / 2:
                    candidate = self._create_alert(ent_id, person, dwell, vars_data, images)
                    alert_candidates.append(candidate)

        # Phase 3: Remove expired entities
        if current_ts > 0:
            expired = [
                eid for eid, p in self.tracking_data.items() if current_ts - p.last_seen > self.no_activity_timeout
            ]
            self._remove_tracked(expired)

        return alert_candidates

    def _create_alert(
        self, ent_id: int, person: PersonData, dwell: int, vars_data, images: List[AnalyticImage]
    ) -> AlertCandidate:
        self.excluded_ents.add(ent_id)
        person.is_alerted = True
        ent_var_data = None
        if len(vars_data) > 0:
            matching = vars_data[vars_data[:, BatchDataResolver.ID] == ent_id]
            if len(matching) > 0:
                ent_var_data = matching
        extra = {"extra_fields": {"duration": dwell}}
        candidate = self.build_alert_candidate(
            person.last_seen,
            ent_id,
            ent_var_data,
            extra=extra,
            images=images,
            crops=[cv2.cvtColor(person.best_crop, cv2.COLOR_RGB2BGR)] if person.best_crop is not None else None,
        )
        logger.info(
            f"Generated PPE alert candidate for entity {ent_id} with duration {dwell} ms after {person.classifier_runs} classification attempts"
        )
        return candidate

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super().build_alert_info(candidate)
        duration = round(candidate.extra["extra_fields"]["duration"] / 1000, 2)
        alert_info.alertData = float(duration)
        return alert_info

    def _remove_tracked(self, ent_ids: List[int]):
        for ent_id in ent_ids:
            self.tracking_data.pop(ent_id, None)
            self.excluded_ents.discard(ent_id)

    def on_entities_removed(self, ent_ids: List[int]):
        super().on_entities_removed(ent_ids)
        self._remove_tracked(ent_ids)
