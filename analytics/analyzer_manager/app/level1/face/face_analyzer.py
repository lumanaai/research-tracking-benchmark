from copy import deepcopy
from typing import List, Any, Dict, Optional

import numpy as np

from general.analyzer_general import GLOBAL_TYPE_KEY, GLOBAL_FAILURE, GLOBAL_SUCCESS, logger, max_per_id_mask
from general.core import (
    BatchDataResolver,
    AdvanceAnalyzerType,
    DescriptorVector,
    WorkItem,
    AttrProperty,
    AdvanceAnalyticResults,
)
from general.img_utils import crop_image_by_bbox, crop_image_by_bbox_and_ar, is_image_monochrome
from level1.advanced import BaseAdvanceAnalyzer, AttrData, BaseAdvanceAnalyzerConfig
from .face_detection import FaceDetectorFactory
from .face_recognition import FaceRecognizerFactory
from .face_utils import (
    FaceConfidence,
    conf_order_map,
    conf2score,
    face_quality_estimator,
    face_conf_to_attr_conf,
    get_min_confidence,
)


class FaceConfig(BaseAdvanceAnalyzerConfig):

    detector: str = "retinaface"
    detector_config: Dict = {}
    recognizer: str = "sface"
    recognizer_config: Dict = {}
    recognizer_v2: str = "webface"
    recognizer_v2_config: Dict = {}
    classifier: str = "ediffiqa" # "magface"
    classifier_config: Dict = {}
    device: str = "cuda"
    min_crop_width: int = 300
    max_crop_width: int = 600
    eye_dist_levels: Dict = {"high": 44, "medium": 36, "low": 28}
    face_angle_levels: Dict = {"high": 28, "medium": 45, "low": 65}
    sharpness_levels: Dict = {"high": 250, "medium": 175, "low": 100}
    contrast_levels: Dict = {"high": 24, "medium": 17, "low": 14}
    ang_diff_th: float = 12.5
    face_nose_angle_th: float = 135.0
    conf_threshold = {"high": 0.92, "medium": 0.90, "low": 0.8}
    margins = [0.3, 0.3]
    dilation_factor = 0.3
    min_conf = "low"
    preferred_conf = "medium"
    crop_to_aspect_ratio = True
    pre_detector: str = "yunet"
    pre_detector_config: Dict = {}
    use_faces_from_detector: bool = True  # if False, use faces from people detector
    min_face_person_overlap: float = 0.90
    face_only_min_crop_width: int = 35
    face_only_max_crop_width: int = 130
    skew_th: float = 0.5
    margins_for_face_only = [0.5, 0.5]
    sensitivity_factor: float = 1.0
    face_consistency_threshold: float = 0.4

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        if self.sensitivity_factor != 1.0:
            factor = max(min(2.0 - self.sensitivity_factor, 1.5), 0.5)  # 0.5 < factor < 1.5
            self.face_only_min_crop_width = int(self.face_only_min_crop_width * factor)
            self.min_crop_width = int(self.min_crop_width * factor)
            self.min_face_person_overlap *= factor
            self.eye_dist_levels["low"] *= factor
            self.sharpness_levels["low"] *= factor
            self.contrast_levels["low"] *= factor
            self.conf_threshold["low"] *= factor
            self.face_angle_levels["low"] *= 2 - factor
            self.face_nose_angle_th *= 2 - factor
            self.skew_th *= 2 - factor


class FaceAnalyzer(BaseAdvanceAnalyzer):
    crop_on_success_only = True
    name = "FaceAnalyzer"
    _type = AdvanceAnalyzerType.FACE
    use_low_prio_set = True
    max_retries = 10
    low_prio_max_time = 5
    analyzer_batch_limit = 4
    add_face_metrics = True
    min_require_results: AdvanceAnalyticResults = AdvanceAnalyticResults.SUCCESS
    _config_type = FaceConfig
    args: FaceConfig
    pre_detector_ar = -1
    do_merge_averaging = False

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)
        self.prepare_for_detector = None
        self.face_subclass = self.class_handler.face_subclass_value
        self.use_faces_from_detector = self.args.use_faces_from_detector and self.face_subclass >= 0

        if self.use_faces_from_detector:
            self.args.pre_detector = None
            self.args.min_crop_width = self.args.face_only_min_crop_width
            self.args.max_crop_width = self.args.face_only_max_crop_width
            # self.args.crop_to_aspect_ratio = False
            self.crop_margins = self.args.margins_for_face_only
        elif self.args.pre_detector:
            self.pre_detector = FaceDetectorFactory.create(self.args.pre_detector, self.args.pre_detector_config)
            self.pre_detector.args.conf_threshold /= 2
            self.prepare_for_detector = self._pre_detect_face
            self.pre_detector_ar = self.pre_detector.input_size[1] / self.pre_detector.input_size[0]
            self.pre_detector_ar = self.pre_detector.input_size[1] / self.pre_detector.input_size[0]

        self.supported_classes = [self.class_handler.person_value]
        self.supported_subclasses = self.class_handler.get_object_classes(self.class_handler.person_value)
        if self.prepare_for_detector is None:
            if self.args.crop_to_aspect_ratio:  # default
                self.prepare_for_detector = self._crop_to_ar
            else:  # do nothing
                self.prepare_for_detector = lambda x: x

        self.detector = FaceDetectorFactory.create(self.args.detector, self.args.detector_config)
        self.recognizer = FaceRecognizerFactory.create(self.args.recognizer, self.args.recognizer_config)
        self.recognizer_v2 = FaceRecognizerFactory.create(self.args.recognizer_v2, self.args.recognizer_v2_config)
        self.classifier = FaceRecognizerFactory.create(self.args.classifier, self.args.classifier_config)
        self.detector_ar = self.detector.input_size[1] / self.detector.input_size[0]
        self.min_conf = conf_order_map[FaceConfidence(self.args.min_conf)]
        self.preferred_conf = conf_order_map[FaceConfidence(self.args.preferred_conf)]
        self.score_map = {"high": 1, "medium": 0.75, "low": 0.5, "none": 0}
        self.post_score_success_th = self.score_map[self.args.preferred_conf] * 0.9 * 0.9
        self.post_score_failure_th = self.score_map[self.args.min_conf] * 0.9 * 0.9
        self.keys_to_filter = ["detection_score", GLOBAL_TYPE_KEY]
        self.face_statistics = {
            "confidence": 0,
            "distance": 0,
            "angles": 0,
            "blur": 0,
            "contrast": 0,
            "sharpness": 0,
            "other": 0,
            "count": 0,
        }
        self.zoom_ar = context.analytic_config.get("trainThumbPolicy", {}).get("crops_ar", {}).get("face", None)
        self.zoom_crop_margins = self.args.margins

    def prepare_batch_data(self, batch_data: BatchDataResolver) -> BatchDataResolver:
        # this function will replace the person position with the faces position for all faces that are inside
        # a single person. other persons and objects will be invalidated
        if self.enabled and self.use_faces_from_detector:

            new_batch_data = deepcopy(batch_data)
            new_batch_data.data[:, BatchDataResolver.ID] = -1
            detections = batch_data.query(batch_data.SUBCLASS, self.face_subclass)
            if len(detections) == 0:
                return new_batch_data
            # first match face with person
            face_idxs = detections[:, BatchDataResolver.INDEX].astype(int)
            person_ids = batch_data.data[:, BatchDataResolver.ID].astype(int)
            person_mask = (batch_data.data[:, batch_data.SUBCLASS] == self.class_handler.person_value).astype(int)
            faces_overlaps = batch_data.all_overlaps[face_idxs]
            face_m, person_m = np.where(faces_overlaps * person_mask > self.args.min_face_person_overlap)
            unique, counts = np.unique(face_m, return_counts=True)
            mapping = dict(zip(unique, counts))
            matches = {
                face_idxs[face_m[i]]: person_m[i]
                for i in range(len(face_m))
                if mapping.get(face_m[i], 0) == 1 and person_ids[person_m[i]] > -1
            }
            for face, pidx in matches.items():
                new_batch_data.data[pidx, BatchDataResolver.POS] = batch_data.data[face, batch_data.POS]
                new_batch_data.data[pidx, BatchDataResolver.ID] = batch_data.data[pidx, BatchDataResolver.ID]
            return new_batch_data
        return batch_data

    def calculate_pre_score(self, batch_data: BatchDataResolver, image_batch=None) -> Dict[int, float]:

        detections = batch_data.query(batch_data.SUBCLASS, self.supported_subclasses)
        if self.filter_untracked:
            detections = detections[detections[:, BatchDataResolver.ID] >= 0]
        scores = {}

        # scores based on crop width: should be higher than th and lower than width (orientation)
        if len(detections) > 0:
            width, height = batch_data.full_resolution
            pos = detections[:, BatchDataResolver.POS]
            crop_widths = (pos[:, 2] - pos[:, 0]) * width
            crop_heights = (pos[:, 3] - pos[:, 1]) * height
            norm_widths = np.minimum((crop_widths - self.args.min_crop_width) / self.args.max_crop_width, 1)
            if self.use_faces_from_detector:
                crop_ar_skew = np.abs(crop_widths - crop_heights) / np.maximum(crop_widths, crop_heights)
                crop_scores = np.where(
                    (crop_ar_skew > self.args.skew_th) | (norm_widths < 0),
                    -1,
                    norm_widths * 0.9 + (1 - crop_ar_skew) * 0.1,
                )
            else:
                crop_scores = np.where((crop_widths > crop_heights) | (norm_widths < 0), -1, norm_widths)
            scores.update({int(detections[i, BatchDataResolver.INDEX]): scr for i, scr in enumerate(crop_scores)})
        return scores

    def calculate_post_score(self, obj_id: int, attr_data: AttrData) -> float:
        att = attr_data.attributes
        if GLOBAL_FAILURE in att[GLOBAL_TYPE_KEY]:
            return -1
        return att["detection_score"] * self.score_map[att["faceConfidence"]] * (0.8 + attr_data.pre_score / 5)

    def _crop_to_ar(self, crops, is_predetector_ar=False):
        crops_out = []
        for crop in crops:
            h, w = crop.shape[0:2]
            if is_predetector_ar:
                crops_out.append(crop[: int(w * self.pre_detector_ar)])
            else:
                crops_out.append(crop[: int(w * self.detector_ar)])
        return crops_out

    def _pre_detect_face(self, crops, pre_crop_ar=True):
        if pre_crop_ar:
            crops = self._crop_to_ar(crops, True)
        bboxes_pre, landmarks, scores = self.pre_detector.forward_on_crop_list(crops)
        crops_out = []
        for idx, bbox in enumerate(bboxes_pre):
            if bbox is None:
                crops_out.append(None)
            elif bbox[2] - bbox[0] < self.args.eye_dist_levels["low"] * 2:
                crops_out.append(None)
            else:
                if self.args.crop_to_aspect_ratio:
                    face_closeup = crop_image_by_bbox_and_ar(
                        crops[idx], bbox, self.detector_ar, self.args.margins, True
                    )
                else:
                    face_closeup = crop_image_by_bbox(np.array(crops[idx]), bbox, self.args.margins)
                crops_out.append(face_closeup)
        return crops_out

    def _run_model(self, batch: List[WorkItem]) -> List[Any]:
        proc_fail = {GLOBAL_TYPE_KEY: [GLOBAL_FAILURE], "faceConfidence": FaceConfidence.NONE.value, "faceId": []}
        results = [proc_fail] * len(batch)
        crops = self.prepare_for_detector([item.image for item in batch])
        good_indices = [i for i in range(len(crops)) if not (crops[i] is None or crops[i].size == 0)]
        valid_crops = [crops[i] for i in good_indices]
        if valid_crops:
            bboxes, landmarks, scores = self.detector.forward_on_crop_list(valid_crops)

        face_confs = []
        det_scores = []
        det_bbox = []
        recognizer_crops = []
        recognizer_inds = []
        face_metric_lst = []
        recognizer_valid_inds = []

        for i, orig_ind in enumerate(good_indices):
            bbox, landmark, det_conf = bboxes[i], landmarks[i], scores[i]
            if bbox is None:
                continue
            rec_crop = self.recognizer.canonize_face(valid_crops[i], landmark)
            face_conf, face_metrics = self.classify_face(rec_crop, landmark, det_conf)
            # if conf_order_map[face_conf] >= self.min_conf:
            recognizer_crops.append(rec_crop)
            recognizer_inds.append(orig_ind)
            recognizer_valid_inds.append(i)
            face_confs.append(face_conf)
            det_scores.append(det_conf)
            det_bbox.append(bbox)
            face_metric_lst.append(face_metrics)

        if len(recognizer_crops) > 0:
            face_ids = self.recognizer.forward_on_crop_list(recognizer_crops)
            face_ids_v2 = self.recognizer_v2.forward_on_crop_list(recognizer_crops)
            # classifications, class_scores = self.classifier.forward_on_crop_list(recognizer_crops)    # used for magface
            classifier_inputs = [valid_crops[idx] for idx in recognizer_valid_inds]
            classifications, class_scores = self.classifier.forward_on_crop_list(classifier_inputs)
            for i, orig_ind in enumerate(recognizer_inds):
                face_metrics = {"location": batch[orig_ind].extra.location}
                face_pre_score = face_metric_lst[i].pop("score", 0)
                if self.add_face_metrics:
                    face_metrics.update(face_metric_lst[i])
                results[orig_ind] = {
                    GLOBAL_TYPE_KEY: [GLOBAL_SUCCESS],
                    "faceIdHex": DescriptorVector(np.squeeze(face_ids[i])),
                    "faceIdHexV2": DescriptorVector(np.squeeze(face_ids_v2[i])),
                    "faceConfidence": self.calc_conf(classifications[i], face_confs[i], face_metrics).value,
                    "faceScore": conf2score[face_confs[i]],
                    "zoom_image": self.crop_zoom_function(crops[orig_ind], det_bbox[i]),
                    "facePreConfidence": face_confs[i].value,
                    "facePreConfidenceScore": face_pre_score,
                    "faceClassification": classifications[i].value,
                    "faceClassificationScore": class_scores[i],
                    "detection_score": det_scores[i],
                    "faceMetrics": face_metrics,
                }
        # reorder results based on maximal face confidence to remove zoom images for lower confidence faces
        ids = [wi.id for wi in batch]
        total_conf = [res.get("faceScore", 0) * 1000 + res.get("faceClassificationScore", 0) for res in results]

        mask = max_per_id_mask(np.array(ids), np.array(total_conf)).tolist()
        for i, m in enumerate(mask):
            if not m:
                results[i]["zoom_image"] = None

        if self.face_statistics["count"] > 100:
            self._post_reset_statistics()
        return results

    def calc_conf(self, classificaion_conf: FaceConfidence, pre_conf: FaceConfidence, face_metrics) -> FaceConfidence:
        """
        Calculate the final confidence based on classification and pre-confidence.
        """
        min_conf = get_min_confidence(classificaion_conf, pre_conf)
        if (
            min_conf == FaceConfidence.NONE
            and face_metrics.get("eye_dist", 0) >= self.args.eye_dist_levels.get("low")
            and classificaion_conf
            in [
                FaceConfidence.HIGH,
                FaceConfidence.MEDIUM,
            ]
        ):
            return FaceConfidence.LOW
        return min_conf

    def to_properties(self, attr_data: Dict) -> Dict:
        face_metrics = []
        for k in ["faceClassification", "facePreConfidence"]:
            v, s = attr_data.pop(k), attr_data.pop(k + "Score")
            att_conf = face_conf_to_attr_conf[v]
            face_metrics.append(AttrProperty(k, s, att_conf))

        input_metrics = attr_data.pop("faceMetrics", {})
        for k, v in input_metrics.items():
            face_metrics.append(AttrProperty(k, v, att_conf))
        attr_data["faceMetrics"] = face_metrics
        return attr_data  # should be regular dict

    def merge_with_hist(self, attr: Dict, obj_id: int) -> (Optional[Dict], float):
        # if the current face confidence is as good or better than previous results,
        # average all face ids with the same confidence, otherwise don't return value

        if conf_order_map[attr["faceConfidence"]] < self.min_conf:
            return None, -1

        # if there is only one result, return it
        if len(self.processed[obj_id]) == 1:
            if GLOBAL_FAILURE in self.processed[obj_id][0].attributes[GLOBAL_TYPE_KEY]:
                return None, -1
            return attr, self.processed[obj_id][0].post_score

        # otherwise, return best result and average accordingly
        confs = [FaceConfidence(res.attributes["faceConfidence"]) for res in self.processed[obj_id]]
        max_conf = max(confs, key=lambda c: conf_order_map[c], default=None)
        scores = [res.attributes.get("faceClassificationScore", 0) for res in self.processed[obj_id]]
        if max_conf == confs[-1]:
            good_idx = [i for i in range(len(confs)) if confs[i] == max_conf]
            face_ids = np.array([self.processed[obj_id][idx].attributes["faceIdHex"].data for idx in good_idx])
            if len(good_idx) > 1:
                matches = face_ids @ face_ids.T
                if np.any(matches[-1, :] < self.args.face_consistency_threshold):
                    good_idx = good_idx[-1:]
                    face_ids = face_ids[-1:]
            if self.do_merge_averaging:
                new_id = np.mean(np.asarray(face_ids), axis=0)
                attr["faceIdHex"].data = new_id / np.linalg.norm(new_id)
                merge_score = np.mean([self.processed[obj_id][idx].post_score for idx in good_idx])
            else:
                max_score_idx = np.argmax([scores[i] for i in good_idx])
                attr["faceIdHex"].data = face_ids[max_score_idx]
                merge_score = self.processed[obj_id][good_idx[max_score_idx]].post_score
            return attr, merge_score

        # in case the current result is worse than before, don't return anything
        return None, -1

    def add_to_low_prio_set(self, obj_id, metadata):
        can_add = True
        if obj_id in self.processed:
            confs = [FaceConfidence(res.attributes["faceConfidence"]) for res in self.processed[obj_id]]
            max_conf = max(confs, key=lambda c: conf_order_map[c], default=None)
            if conf_order_map[max_conf] >= self.preferred_conf:
                can_add = False
        if can_add:
            self.low_prio_set.add(obj_id)

    def get_metadata(self, crop_info):
        return {}

    def _post_reset_statistics(self):
        logger.http(self.face_statistics)

        for key in self.face_statistics:
            self.face_statistics[key] = 0

    def _log_face_reason(self, failure_category: Optional[str] = None):
        if failure_category:
            self.face_statistics[failure_category] += 1
        self.face_statistics["count"] += 1

    @staticmethod
    def calculate_p2p_angle(eye1, eye2):
        dy = eye2[1] - eye1[1]
        dx = eye2[0] - eye1[0]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle

    @staticmethod
    def angle_between_eyes_nose(eyes_center, nose):
        dy = nose[0] - eyes_center[0]
        dx = nose[1] - eyes_center[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle

    @staticmethod
    def calc_triag_angle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    @staticmethod
    def get_face_angles(landmarks):
        """Calculate angles based on facial landmarks."""

        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

        left_eye_angle = FaceAnalyzer.calc_triag_angle(right_eye, left_eye, nose)
        right_eye_angle = FaceAnalyzer.calc_triag_angle(left_eye, right_eye, nose)
        left_mouth_angle = FaceAnalyzer.calc_triag_angle(nose, left_mouth, right_mouth)
        right_mouth_angle = FaceAnalyzer.calc_triag_angle(left_mouth, right_mouth, nose)
        top_nose_angle = FaceAnalyzer.calc_triag_angle(left_eye, nose, right_eye)
        bottom_nose_angle = FaceAnalyzer.calc_triag_angle(left_mouth, nose, right_mouth)
        # left_nose_angle = FaceAnalyzer.calc_triag_angle(left_eye, nose, left_mouth)
        # right_nose_angle = FaceAnalyzer.calc_triag_angle(right_eye, nose, right_mouth)

        return {
            "left_eye": left_eye_angle,
            "right_eye": right_eye_angle,
            "top_nose": top_nose_angle,
            "bottom_nose": bottom_nose_angle,
            "left_mouth": left_mouth_angle,
            "right_mouth": right_mouth_angle,
            # "left_nose": left_nose_angle,
            # "right_nose": right_nose_angle,
        }

    def classify_face(self, image, landmarks, det_conf) -> (FaceConfidence, Dict):
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[0:5]
        eye_dist = np.linalg.norm(np.array(left_eye - right_eye), 2)
        sharpness, contrast = face_quality_estimator(image)
        angles = FaceAnalyzer.get_face_angles(landmarks)

        score = 0
        face_metrics = {k + "_angle": v for k, v in angles.items()}
        face_metrics["eye_dist"] = eye_dist
        face_metrics["sharpness"] = sharpness
        face_metrics["contrast"] = contrast
        face_metrics["score"] = score

        if eye_dist > self.args.eye_dist_levels["high"]:
            score += 1
        elif eye_dist <= self.args.eye_dist_levels["low"]:
            self._log_face_reason("distance")
            return FaceConfidence.NONE, face_metrics

        if det_conf < self.args.conf_threshold["low"]:
            self._log_face_reason("confidence")
            return FaceConfidence.NONE, face_metrics
        elif det_conf < self.args.conf_threshold["medium"]:
            score -= 3
        elif det_conf < self.args.conf_threshold["high"]:
            score -= 1

        if is_image_monochrome(image):
            score -= 3

        if contrast < self.args.contrast_levels["low"]:
            self._log_face_reason("contrast")
            return FaceConfidence.NONE, face_metrics

        if sharpness < self.args.sharpness_levels["low"]:
            self._log_face_reason("sharpness")
            return FaceConfidence.NONE, face_metrics

        if sharpness < self.args.sharpness_levels["medium"]:
            score -= 2
        elif sharpness < self.args.sharpness_levels["high"]:
            score -= 1

        elif contrast < self.args.contrast_levels["medium"]:
            score -= 2
        elif contrast < self.args.contrast_levels["high"]:
            score -= 1

        # Check eye and mouth angles
        total_angles = 0
        lr_diff = np.array([angles["left_eye"] - angles["right_eye"], angles["left_mouth"] - angles["right_mouth"]])
        lr_diff = np.repeat(np.abs(lr_diff), 2)

        for a_idx, angle in enumerate(["left_eye", "right_eye", "left_mouth", "right_mouth"]):
            if (
                self.args.face_angle_levels["low"] > angles[angle] > self.args.face_angle_levels["high"]
                or lr_diff[a_idx] < self.args.ang_diff_th
            ):
                total_angles += 1

        # Check nose angles
        for angle in ["top_nose", "bottom_nose"]:
            if angles[angle] < self.args.face_nose_angle_th:
                total_angles += 1
        score += total_angles

        if total_angles < 3:
            self._log_face_reason("angles")

        face_metrics["score"] = score

        # Determine grade based on score
        if score < 3:
            self._log_face_reason("other")
            return FaceConfidence.NONE, face_metrics
        else:
            self._log_face_reason(None)
            res = FaceConfidence.HIGH
            if score < 6:
                res = FaceConfidence.LOW
            elif score < 7:
                res = FaceConfidence.MEDIUM
            return res, face_metrics
