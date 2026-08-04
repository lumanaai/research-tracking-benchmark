from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from alerts.base_alerts import ObjectAlert, AlertCandidate, duration_unit_to_sec
from general.core import (
    AlertRouting,
    AnalyticImage,
    BatchDataResolver,
    calculate_overlap_matrix,
    BDR,
    AlertInfo,
    MemoryHandler,
)
from general.img_utils import crop_image
from level1.pa_recognition.hands import HandsModel


class GlovesWearOption(Enum):
    not_wearing = 0
    wearing = 1


class GlovesColorOptions(Enum):
    transparent = "transparent"
    colored = "colored"


HAND_CLASS_NAME = "hand"


@dataclass
class HandsData:
    last_center: np.ndarray  # = field(default_factory=lambda: np.full(2, np.inf))
    crops: List[np.array] = field(default_factory=lambda: [])
    classifications: List[Optional[bool]] = field(default_factory=lambda: [])
    timestamps: List[int] = field(default_factory=lambda: [])


class PersonData:
    hands: List[HandsData]  # just 2
    buffer_idx: int = None
    is_alerted: bool = False
    repeats: int = 0
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None

    def __init__(self, hands_data: np.ndarray, buffer_idx: int, first_seen: int = None, last_seen: int = None):
        self.hands = [HandsData(hands_data[0, BDR.CENTER])]
        if len(hands_data) > 1:
            second_hand = HandsData(hands_data[1, BDR.CENTER])
        else:
            second_hand = HandsData(np.full(2, np.inf))
        self.hands.append(second_hand)
        self.buffer_idx = buffer_idx
        self.first_seen = first_seen
        self.last_seen = last_seen

    def add_hands(self, hands_data: np.ndarray, crops: List[np.array]):
        last_centers = np.stack([h.last_center for h in self.hands])
        distances = np.linalg.norm(hands_data[:, BDR.CENTER, np.newaxis] - last_centers, axis=2)
        if len(hands_data) == 1:
            closest_index = np.argmin(distances[0])
            self.add_single_hand(closest_index, hands_data[0], crops[0])
        else:  # 2 hands
            row, col = np.unravel_index(np.argmin(distances), distances.shape)
            self.add_single_hand(col, hands_data[row], crops[row])
            self.add_single_hand(1 - col, hands_data[1 - row], crops[1 - row])

    def add_single_hand(self, idx, hand_data, crop):
        self.hands[idx].last_center = hand_data[BDR.CENTER]
        self.hands[idx].crops.append(crop)
        self.hands[idx].classifications.append(None)
        self.hands[idx].timestamps.append(hand_data[BDR.TIMESTAMP])

    def last_seen_hand(self) -> int:
        timestamps = [h.timestamps[-1] for h in self.hands if len(h.timestamps) > 0]
        if timestamps:
            return max(timestamps)
        return self.last_seen if self.last_seen is not None else 0

    def get_unclassified(self) -> Tuple[List[np.ndarray], List[int]]:
        len_first = len(self.hands[0].crops)
        indices = [idx for idx, classific in enumerate(self.hands[0].classifications) if classific is None]
        crops = [self.hands[0].crops[i] for i in indices]
        if len(self.hands[1].crops) == 0:
            return crops, indices
        indices2 = [idx for idx, classific in enumerate(self.hands[1].classifications) if classific is None]
        crops2 = [self.hands[1].crops[i] for i in indices2]
        return crops + crops2, indices + [i + len_first for i in indices2]

    def update_classification(self, index, classification):
        len_first = len(self.hands[0].crops)
        if index >= len_first:
            index -= len_first
            self.hands[1].classifications[index] = classification
        else:
            self.hands[0].classifications[index] = classification
        pass


class GlovesAlert(ObjectAlert):
    object_type = "bodypart"
    classifier: HandsModel = None
    confidence_th = 0.2
    alert_message = "Gloves violation detected"
    latency = 5
    default_routing = AlertRouting.NO_ROUTING
    wear_requirement: bool = True
    color_requirement: Optional[bool] = None
    max_supported_person = 10
    no_activity_timeout = 10 * 1000  # 10 sec
    min_hand_overlap = 0.8
    is_solve_conflicts = True
    loiter_duration = 0
    classifier_timeout = None

    def __init__(self, alert_dict: Dict, context):
        super(GlovesAlert, self).__init__(alert_dict, context)
        self.required_l1 = False
        self.class_filter = self.context.get_class_handler().class_str_to_int(HAND_CLASS_NAME)
        self.person_obj = self.context.get_class_handler().class_str_to_int("person")
        self.l1_required_type[self.person_obj] = None

        if self.class_filter < 0 or self.person_obj < 0:
            raise ValueError(f"Class {HAND_CLASS_NAME} not found in class handler")
        self.set_flow_values()

        self.classifier_config = {}
        analytic_config = self.context.get_config()

        if "l1_models" in analytic_config and "attributes" in analytic_config["l1_models"]:
            self.classifier_config = (
                analytic_config["l1_models"]["attributes"].get("human_parsing", {}).get("hands", {})
            )
        self.classifier = self._create_classifier()
        self.required_classifications = self._compile_classification_requirements()
        self.crop_margins = [self.classifier.required_margins] * 2
        self.crops_per_hand = max(self.latency, context.batch_size)
        self.memory_handler = MemoryHandler(
            num_buffers=self.max_supported_person,
            num_crops=self.crops_per_hand * 2,
            crop_size=list(self.classifier.image_size) + [3],
            dtype=np.uint8,
        )
        self.tracking_data: Dict[int, PersonData] = {}
        confidence = self.settings.get("confidence", 2)  # default is high confidence
        if confidence < 2:
            self.is_solve_conflicts = False

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []
        vars_data = batch_data.query(BatchDataResolver.SUBCLASS, self.class_filter)
        vars_data = vars_data[vars_data[:, BatchDataResolver.CONFIDENCE] > self.confidence_th]
        len_th = self.classifier.minimal_crop_size / np.array(batch_data.full_resolution)
        is_large_enough_w = vars_data[:, BDR.POS[2]] - vars_data[:, BDR.POS[0]] >= len_th[0]
        is_large_enough_h = vars_data[:, BDR.POS[3]] - vars_data[:, BDR.POS[1]] >= len_th[1]
        vars_data = vars_data[np.logical_and(is_large_enough_w, is_large_enough_h)]

        person_data = batch_data.query(BatchDataResolver.CLASS, self.person_obj)
        person_data = person_data[person_data[:, BDR.ID] >= 0]
        vars_data = self.apply_roi_filter(vars_data, BatchDataResolver.CENTER_LOCATION)

        # gather gloves data from detections
        if len(vars_data) > 0 and len(person_data) > 0:
            batch_tracking = defaultdict(list)

            for image in images:
                hands = vars_data[vars_data[:, BatchDataResolver.TIMESTAMP] == image.timestamp]
                persons = person_data[person_data[:, BatchDataResolver.TIMESTAMP] == image.timestamp]

                if len(hands) > 0 and len(persons) > 0:
                    image_assignment = self.assign_hands(persons, hands)
                    for pid, assigned in image_assignment.items():
                        if len(assigned) > 0:
                            batch_tracking[pid].append(assigned)

            # update tracking data
            for ent_id, hands_list in batch_tracking.items():
                if ent_id not in self.tracking_data:
                    buffer_idx = self.memory_handler.acquire()
                    if buffer_idx < 0:
                        continue
                    self.tracking_data[ent_id] = PersonData(
                        hands_list[0], buffer_idx, last_seen=int(hands_list[0][0, BDR.TIMESTAMP])
                    )

                # to be ignored since fully suppressed after debounce
                if self.tracking_data[ent_id].is_alerted:
                    continue
                buffer_idx = self.tracking_data[ent_id].buffer_idx
                for hands in hands_list:
                    image = images[int(hands[0, BDR.FRAME_ID])].frame
                    crops = []
                    for hand in hands:
                        crop = self.classifier.apply_resize(
                            crop_image(image, hand[BDR.POS], margins=self.crop_margins, bgr_map=False),
                            self.memory_handler.next(buffer_idx),
                        )
                        crops.append(crop)
                    self.tracking_data[ent_id].add_hands(hands, crops)

        # do classifications if required
        all_candidates = []
        ents_idx = []
        hands_idx = []
        for ent_id, person_data in self.tracking_data.items():
            if not person_data.is_alerted:
                candidates, hand_indices = person_data.get_unclassified()
                all_candidates += candidates
                ents_idx += [ent_id] * len(candidates)
                hands_idx += hand_indices
        if len(all_candidates) > 0:
            all_class, all_conf = [], []
            for req in self.required_classifications:
                classifications, confidences = self.classifier.classify(all_candidates, req)
                all_class.append(classifications)
                # all_conf.append(confidences)
            classifications = np.logical_or.reduce(np.vstack(all_class), axis=0)
            # confidences = np.max(np.vstack(all_conf), axis=0)
            for idx, classification in enumerate(classifications):
                self.tracking_data[ents_idx[idx]].update_classification(hands_idx[idx], classification)

        # check for alerts and maintenance
        ids_to_remove = []
        for ent_id, person_data in self.tracking_data.items():
            if not person_data.is_alerted:
                if images[-1].timestamp - person_data.last_seen_hand() > self.no_activity_timeout:
                    ids_to_remove.append(ent_id)
                    continue
                # handle expired classifications
                # this mechanism is what keeps the loitering time, since only when vote passed we update last seen,
                # so it requires the person to continuously get phone classifications
                if self.classifier_timeout is not None:
                    ref_time = max(person_data.last_seen or 0, person_data.last_seen_hand())
                    if images[-1].timestamp - ref_time > self.classifier_timeout:
                        ids_to_remove.append(ent_id)
                        continue
                vote_passed = False
                vote_attempted = False
                for hand in person_data.hands:
                    classifications = np.array([int(c) for c in hand.classifications if c is not None])
                    if len(classifications) >= self.latency:
                        vote_attempted = True
                        if np.mean(classifications[-self.latency :]) > 0.5:
                            vote_passed = True
                            break

                if vote_passed:
                    person_data.repeats += 1
                    person_data.last_seen = images[-1].timestamp
                    person_data.first_seen = (
                        images[-1].timestamp if person_data.first_seen is None else person_data.first_seen
                    )
                    if person_data.last_seen - person_data.first_seen >= self.loiter_duration:
                        ent_data = batch_data.query(BDR.ID, ent_id)
                        alert_ts = int(ent_data[-1, BDR.TIMESTAMP]) if len(ent_data) > 0 else images[-1].timestamp
                        candidate = self.build_alert_candidate(
                            alert_ts,
                            ent_id,
                            ent_data,
                            images=images,
                            crops=[
                                cv2.cvtColor(hand.crops[-1], cv2.COLOR_RGB2BGR)
                                for hand in person_data.hands
                                if len(hand.crops) > 0
                            ],
                        )
                        alert_candidates.append(candidate)
                        person_data.is_alerted = True
                        self.memory_handler.release(person_data.buffer_idx)
                        person_data.buffer_idx = -1
                elif vote_attempted:
                    # cancellation: vote failed, person likely stopped the violation
                    person_data.repeats = 0
                    person_data.first_seen = None
                    person_data.last_seen = images[-1].timestamp
                    # only clear hands that actually reached the vote threshold
                    for h in person_data.hands:
                        hand_cls = np.array([int(c) for c in h.classifications if c is not None])
                        if len(hand_cls) >= self.latency:
                            h.classifications.clear()
                            h.crops.clear()
                            h.timestamps.clear()

        self.on_entities_removed(ids_to_remove)
        return alert_candidates

    def assign_hands(self, persons, hands):
        # first based on overlap
        overlaps = calculate_overlap_matrix(hands[:, BDR.POS], persons[:, BDR.POS])
        matches = overlaps > self.min_hand_overlap
        person_per_hand = np.sum(matches, axis=1)
        hand_per_person = np.sum(matches, axis=0)
        conflict_hands = np.where(person_per_hand > 1)[0]

        if self.is_solve_conflicts:
            for conflict in conflict_hands:
                # reallocate based on distance
                distances = np.linalg.norm(hands[conflict, BDR.CENTER] - persons[:, BDR.CENTER], axis=1)
                valid_matches = np.logical_and(matches[conflict], hand_per_person <= 2)
                distances[~valid_matches] = np.inf
                closest_person = np.argmin(distances)

                # update
                matches[conflict, :] = False
                matches[conflict, closest_person] = True
                hand_per_person = np.sum(matches, axis=0)

            # if for some reason we still have persons with more than 2 hands, we need to invalidate
            conflict_persons = np.where(hand_per_person > 2)[0]
            for conflict in conflict_persons:
                # invalidate based on distance to the closest hand
                distances = np.linalg.norm(persons[conflict, BDR.CENTER] - hands[:, BDR.CENTER], axis=1)
                distances[~matches[:, conflict]] = np.inf
                sorted_indices = np.argsort(distances)
                matches[sorted_indices[2:], conflict] = False
        else:
            matches[conflict_hands, :] = False
            hand_per_person = np.sum(matches, axis=0)
            conflict_persons = np.where(hand_per_person > 2)[0]
            matches[:, conflict_persons] = False

        assignments = {}
        # now there is no conflict and each hand is assigned to exactly on person, and each person has at most 2 hands
        for i, person in enumerate(persons):
            assignments[int(person[BDR.ID])] = hands[matches[:, i]]
        return assignments

    def set_flow_values(self):
        self.wear_requirement = (
            self.formValue.get("wear", GlovesWearOption.wearing.value) == GlovesWearOption.wearing.value
        )
        color_requirement = self.formValue.get("gloves", None)
        if color_requirement and color_requirement in GlovesColorOptions.__members__:
            self.color_requirement = color_requirement == GlovesColorOptions.colored.value
        duration = self.formValue.get("duration", 0)
        units = self.apply_from_dict("durationUnit", self.formValue, 0)
        self.loiter_duration = int(duration * duration_unit_to_sec[units] * 1000)  # convert to milliseconds

    def _create_classifier(self):
        return HandsModel(self.classifier_config)

    def _compile_classification_requirements(self):
        return self.classifier.compile_classification_requirements(self.wear_requirement, self.color_requirement)

    def update_model(self, config):
        self.classifier_config.update(config)
        self.classifier = self._create_classifier()

    def on_entities_removed(self, ent_ids: List[int]):
        for ent_id in ent_ids:
            if ent_id in self.tracking_data:
                self.memory_handler.release(self.tracking_data[ent_id].buffer_idx)
                del self.tracking_data[ent_id]

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        alert_info.alertMessage = self.alert_message


class HandsAlert(GlovesAlert):
    """
    HandsAlert is a specialized alert for detecting hands violations.
    It extends GlovesAlert to handle specific requirements for hand detection.
    """

    alert_message = "Hands violation detected"
    latency = 10
    wear_requirement: Optional[bool] = None
    color_requirement: Optional[bool] = None

    def set_flow_values(self):
        # HandsAlert does not require gloves, so we set wear_requirement to None
        pass
