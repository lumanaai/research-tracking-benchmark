from collections import deque
from dataclasses import dataclass
from typing import List, Dict

import cv2
import numpy as np

from alerts.base_alerts import BaseAlert, AlertCandidate
from alerts.face_db_utils import FaceEntity, parse_face_list_items, load_face_groups_from_db, build_face_db_result
from general.core import BatchDataResolver, AlertRouting, AnalyticImage, MotionData, BDR, AdvanceAnalyzerType, AlertInfo
from general.img_utils import crop_image_by_bbox_and_ar


@dataclass
class DetRecord:
    timestamp: int
    location: np.ndarray


def _match_to_db(recognizer_crops, recognizer, face_matrix, sim_threshold, faces_data, score_factor=0.0):
    candidates = {}
    if faces_data:
        face_ids = np.squeeze(recognizer.forward_on_crop_list(recognizer_crops))

        # match the faces with the database
        cos_sim = face_matrix @ np.atleast_2d(face_ids).T
        m_idx, n_idx = np.where(cos_sim > sim_threshold)
        for i, m in enumerate(m_idx):
            person_id = faces_data[m].id
            if person_id not in candidates:
                candidates[person_id] = []
            candidates[person_id].append((n_idx[i], cos_sim[m, n_idx[i]] + score_factor))
    return candidates


class FaceAppearanceAlert(BaseAlert):
    type_name = "suspect"
    default_routing = AlertRouting.ROUTE_VCC_DEFAULT_FALSE
    faces_data_v1: List[FaceEntity]
    faces_data_v2: List[FaceEntity]
    face_matrix_v1: np.ndarray
    face_matrix_v2: np.ndarray
    sim_threshold: float = 0.45
    sim_threshold_v2: float = 0.3
    high_conf_th: float = 0.5
    expiration_timeout_ms: int = 60 * 1000
    lm_expiration_timeout: int = 5 * 1000
    history_cleanup_timeout: int = 60 * 60 * 1000
    location_based_filtering_threshold: float = 2.0
    lm_variability_threshold: float = 0  # 15
    detection_threshold: float = 0.15
    id_to_name: Dict[int, str]
    id_to_best_image: Dict[int, str]
    min_box_width: int = 65
    max_hold = 5000
    conf_to_th_conversion: Dict[int, float] = {
        0: 0.45,  # low
        1: 0.5,  # medium
        2: 0.55,  # high
    }

    conf_to_th_conversion_v2: Dict[int, float] = {
        0: 0.27,  # low
        1: 0.3,  # medium
        2: 0.35,  # high
    }

    conf_to_width_conversion: Dict[int, int] = {
        0: 55,  # low
        1: 65,  # medium
        2: 80,  # high
    }
    v2_score_factor = 0.3

    def __init__(self, alert_dict: dict, context):
        super(FaceAppearanceAlert, self).__init__(alert_dict, context)
        self.is_db_valid = True
        self.list_subjects = []
        self.groups = []
        self.set_flow_values()
        self.on_db_update()
        if not self.is_db_valid:
            raise RuntimeError(f"FaceAppearance alert {self.event_id} is not valid due to DB issue")
        self.face_subclass = self.context.get_class_handler().face_subclass_value
        l1_manager = self.context.get_l1_manager()
        self.face_analyzer = l1_manager.get_analyzer(AdvanceAnalyzerType.FACE)

        if self.face_subclass < 0 or self.face_analyzer is None or not self.face_analyzer.enabled:
            raise ValueError("Face not configured for this camera")
        self.margins = self.face_analyzer.args.margins_for_face_only
        self.detector_ar = self.face_analyzer.detector_ar
        self.detector = self.face_analyzer.detector
        self.recognizer = self.face_analyzer.recognizer
        self.recognizer_v2 = self.face_analyzer.recognizer_v2

        # history filtering mechanism to avoid sending alerts for the same person
        self.history: Dict[int, DetRecord] = {}

        # held mechanism to wait for more detections
        self.held_candidates: Dict[int, AlertCandidate] = {}
        self.held_lm = deque(maxlen=20)
        self.last_lm_update = -1000000

        # change sensitivity based on confidence
        confidence = self.settings.get("confidence", 1)  # default is medium confidence
        self.sim_threshold = self.conf_to_th_conversion.get(confidence, self.sim_threshold)
        self.sim_threshold_v2 = self.conf_to_th_conversion_v2.get(confidence, self.sim_threshold_v2)
        self.min_box_width = self.conf_to_width_conversion.get(confidence, self.min_box_width)

    def set_flow_values(self):
        super().set_flow_values()
        # read suspect list from form
        people_config = self.formValue.get("people", {})
        elements = people_config.get("list", [])
        self.list_subjects = parse_face_list_items(elements)
        self.groups = self.formValue.get("personGroups", [])

    def on_db_update(self):
        prev_valid_db = self.is_db_valid
        entities = list(self.list_subjects)

        # load groups from DB if configured
        if self.groups:
            self.is_db_valid = False
            person_db = self.context.get_analytics_db().get("persons", {})
            group_db = self.context.get_analytics_db().get("persons_groups", [])

            def on_error(msg):
                self.context.validate_alert(self.event_id, False, msg)

            group_entities = load_face_groups_from_db(self.groups, person_db, group_db, on_error)
            if group_entities is None:
                return
            entities.extend(group_entities)

        self.is_db_valid = True

        if len(entities) == 0:
            self.is_db_valid = False
            self.context.validate_alert(self.event_id, False, "No persons found in the suspect list")
            return

        result = build_face_db_result(entities)
        self.faces_data_v1 = result.entities_v1
        self.faces_data_v2 = result.entities_v2
        self.face_matrix_v1 = result.matrix_v1
        self.face_matrix_v2 = result.matrix_v2
        self.id_to_name = result.id_to_name
        self.id_to_best_image = result.id_to_best_image

        if not prev_valid_db:  # alert wasn't valid due to DB issue and now it is
            self.context.validate_alert(self.event_id, True)

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BDR
    ) -> List[AlertCandidate]:
        alert_candidates = []
        face_data = batch_data.query(BDR.SUBCLASS, self.face_subclass)
        face_data = self.filter_detections(face_data, batch_data.full_resolution)
        if len(face_data) > 0 and self.face_analyzer.enabled:
            # first crop all faces and get their locations
            im_size = images[-1].frame.shape[1::-1]
            norm_factor = np.array([*im_size, *im_size])
            crops = []
            crop_poses = []
            for face in face_data:
                bb = face[BDR.POS] * norm_factor
                crop, crop_pos = crop_image_by_bbox_and_ar(
                    images[int(face[BDR.FRAME_ID])].frame,
                    bb,
                    self.detector_ar,
                    self.margins,
                    bgr_map=False,
                    return_pos=True,
                )
                crops.append(crop)
                crop_poses.append(crop_pos)
            # run detection to verify the faces and get landmarks
            bboxes, landmarks, scores = self.detector.forward_on_crop_list(crops)
            recognizer_crops = []
            good_idxs = []
            for i, bbox in enumerate(bboxes):
                if bbox is not None:
                    landmarks_image_coords = landmarks[i] + crop_poses[i][:2]
                    if self.held_lm:
                        lm_dist = np.min(
                            [np.linalg.norm(landmarks_image_coords - np.array(lm[1])) for lm in self.held_lm]
                        )
                    else:
                        lm_dist = 100000
                    if lm_dist > self.lm_variability_threshold:
                        # run recognition on the face
                        recognizer_crops.append(self.recognizer_v2.canonize_face(crops[i], landmarks[i]))
                        good_idxs.append(i)
                        self.held_lm.append((face_data[i, BDR.TIMESTAMP], landmarks_image_coords))
                    else:
                        pass

            # run recognition on the faces
            if len(recognizer_crops) > 0:
                candidates = _match_to_db(
                    recognizer_crops,
                    self.recognizer_v2,
                    self.face_matrix_v2,
                    self.sim_threshold_v2,
                    self.faces_data_v2,
                    self.v2_score_factor,
                )
                candidates_v1 = _match_to_db(
                    recognizer_crops, self.recognizer, self.face_matrix_v1, self.sim_threshold, self.faces_data_v1
                )
                if len(candidates_v1) > 0:
                    for person_id, match_list in candidates_v1.items():
                        if person_id not in candidates:
                            candidates[person_id] = []
                        candidates[person_id].extend(match_list)

                for person_id, match_list in candidates.items():
                    match_list = sorted(match_list, key=lambda x: x[1], reverse=True)
                    det_idxs = [good_idxs[m[0]] for m in match_list]
                    person_face_data = np.array([face_data[i] for i in det_idxs])
                    is_new = self.check_candidate_history(person_id, person_face_data)
                    if person_id in self.held_candidates:
                        cand_extra = self.held_candidates[person_id].extra
                        cand_extra["num_detections"] += len(match_list)
                        cand_extra["max_score"] = max(cand_extra["max_score"], match_list[0][1])
                        cand_extra["high_conf"] = (
                            cand_extra["num_detections"] > 3 or match_list[0][1] > self.high_conf_th
                        )
                        self.held_candidates[person_id].validation_images += [crops[d] for d in det_idxs]
                    elif is_new:
                        is_high_conf = len(match_list) > 1 or match_list[0][1] > self.high_conf_th
                        person_crops = [cv2.cvtColor(crops[d], cv2.COLOR_RGB2BGR) for d in det_idxs]
                        alert_ts = int(np.min(person_face_data[:, BDR.TIMESTAMP]))
                        extra = {
                            "person_name": self.id_to_name[person_id],
                            "person_id": person_id,
                            "person_best_image": self.id_to_best_image[person_id],
                            "num_detections": len(match_list),
                            "max_score": match_list[0][1],
                            "high_conf": is_high_conf,
                        }
                        cand = self.build_alert_candidate(
                            alert_ts,
                            None,
                            person_face_data,
                            extra=extra,
                            crops=person_crops[:1],
                            validation_images=person_crops,
                        )
                        self.held_candidates[person_id] = cand
        self.history_cleanup(images[-1].timestamp)

        # check if any of the candidate are being held enough
        person_ids = list(self.held_candidates.keys())
        for p in person_ids:
            candidate = self.held_candidates[p]
            if candidate.timestamp - images[-1].timestamp > self.max_hold or len(candidate.validation_images) > 5:
                # candidate is expired
                alert_candidates.append(candidate)
                del self.held_candidates[p]
        return alert_candidates

    def check_candidate_history(self, person_id: int, detections: np.ndarray) -> bool:
        last_seen = np.max(detections[:, BDR.TIMESTAMP])
        location = np.mean(detections[:, BDR.CENTER], axis=0)
        is_new = True

        if person_id in self.history:
            last_ts = self.history[person_id].timestamp
            last_location = self.history[person_id].location
            if (
                last_seen - last_ts < self.expiration_timeout_ms
                or np.linalg.norm(location - last_location) < self.location_based_filtering_threshold
            ):
                is_new = False

        self.history[person_id] = DetRecord(last_seen, location)
        return is_new

    def history_cleanup(self, current_time: int):
        # Remove old entries from the history
        for person_id in list(self.history.keys()):
            if current_time - self.history[person_id].timestamp > self.history_cleanup_timeout:
                del self.history[person_id]

        # if we haven't updated the stored locations in along while
        while len(self.held_lm) > 0 and self.held_lm[0][0] < current_time - self.lm_expiration_timeout:
            self.held_lm.popleft()

    def filter_detections(self, face_data: np.ndarray, full_res: np.ndarray) -> np.ndarray:
        face_data = self.apply_roi_filter(face_data, BatchDataResolver.CENTER_LOCATION)
        if len(face_data) == 0:
            return face_data
        conf_ok = face_data[:, BDR.CONFIDENCE] > self.detection_threshold

        # filter out small boxes
        widths = face_data[:, BDR.POS[2]] - face_data[:, BDR.POS[0]]
        width_ok = widths > self.min_box_width / full_res[0]

        return face_data[width_ok & conf_ok, :]

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super().build_alert_info(candidate)
        person_data = alert_info.extra.get("entity_data", {})
        person_name = candidate.extra.get("person_name")
        alert_info.alertMessage = f"Suspect {person_name} appeared"
        if candidate.extra.get("high_conf", True) and len(candidate.validation_images) > 1:
            alert_info.routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE.value
        alert_info.specialFilter = candidate.extra.get("person_best_image")
        alert_info.extra = candidate.extra
        person_data["person_name"] = [person_name]
        alert_info.extra["entity_data"] = person_data
        if len(alert_info.validation_images) > 5:
            alert_info.validation_images = [v for v in alert_info.validation_images if v.size > 0][:5]
        return alert_info
