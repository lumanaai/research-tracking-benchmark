from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious  # noqa
from cython_bbox import bbox_overlaps as bbox_ious  # noqa

from detection.yolov5.yolov8_detector import YoloV8ExpertArgs
from general.analyzer_general import logger, ROI_SHAPE
from general.common_models import CalibratedModelWrapper
from general.core import (
    BatchDataResolver,
    AnalyticImage,
    AlertInfo,
    AlertRouting,
    calc_max_overlap,
    BDR,
    bbox_to_location,
)
from general.img_utils import crop_image
from general.offline_analytics import check_expert_availability, OfflineAnalyticsClient, OfflineAnalyticsClientFactory
from level1.weapons_classifier.weapon_analyzer import WCResnet18Classifier
from preprocessing.snapshot_resizer import SnapshotResizerFactory
from .alerts import BaseAlert
from .base_alerts import AlertCandidate


def joint_bbox_np(bbox1: np.ndarray, bbox2: np.ndarray):

    # Compute top-left and bottom-right coordinates for the joint bbox
    joint_tl = np.minimum(bbox1[:2], bbox2[:2])
    joint_br = np.maximum(bbox1[2:], bbox2[2:])

    # Combine into a single array
    joint_bbox = np.concatenate((joint_tl, joint_br))

    return joint_bbox


@dataclass
class HazardTrackingData:
    timestamp: int
    center: np.ndarray
    crop: np.ndarray
    confidence: float
    ent_id: int
    bbox: np.ndarray
    person_id: int
    classification: Optional[bool] = None
    classification_score: Optional[float] = None


def get_hazard_list_stats(hazards: List[HazardTrackingData]):
    conf = np.median([h.confidence for h in hazards])
    if len(hazards) > 1:
        centers = np.array([h.center for h in hazards])
        avg_point = np.mean(centers, axis=0)
        location_std = np.std(np.linalg.norm(centers - avg_point, axis=1))
    else:
        avg_point = hazards[-1].center
        location_std = 0
    return conf, location_std, avg_point


class RareAppearanceAlert(BaseAlert):
    type_name = "appearance"
    distance_th = 5
    latency = 5
    time_delta = 3000
    active_ratio = 0.5
    object_based_alert = True
    require_classifier = False
    object_type = ""
    class_filters = []
    classifier: Optional[CalibratedModelWrapper] = None
    person_filter = True
    confidence_th = 0.0
    blacklist_enabled = False
    blacklist: List[Dict]
    blacklist_conf = 0.25
    blacklist_distance_std = 5
    blacklist_distance_th = 3
    blacklist_expiration_ms = 24 * 60 * 60 * 1000  # 22 hours
    is_store_snapshot = True
    default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE
    validation_crop_margins = [0.1, 0.1]
    use_expert: bool = False
    expert_check_interval = 5000
    last_expert_check = -1000000
    offline_client: OfflineAnalyticsClient = None
    expert_detection_log = deque(maxlen=3)
    request_timeout = 10000
    expert_suppression_window = 60 * 1000  # 1 minute
    suppression_iou_th = 0.3

    def __init__(self, alert_dict: Dict, context):
        super(RareAppearanceAlert, self).__init__(alert_dict, context)

        if self.require_classifier:
            self.classifier, self.classifier_config = self._init_classifier()

        self.hazard_obj = self.context.get_class_handler().object_str_to_int(self.object_type)
        if self.class_filters:
            self.class_filters = [self.context.get_class_handler().class_str_to_int(s) for s in self.class_filters]
            self.query_batch = lambda batch_data: batch_data.query(BatchDataResolver.SUBCLASS, self.class_filters)
        else:
            self.query_batch = lambda batch_data: batch_data.query(BatchDataResolver.CLASS, self.hazard_obj)
        self.person_obj = self.context.get_class_handler().person_value
        self.active_ratio = (1 - self.sensitivity) if self.sensitivity else self.active_ratio

        self.hazard_tracking: List[dict] = []
        self.margins = [self.classifier.required_margins] * 2 if self.require_classifier else [0, 0]
        self.l1_required = False
        self.l1_required_type[self.hazard_obj] = None
        self.blacklist = []

        self.expert_retries = 0
        if self.use_expert:  # if this part failed the alert should be invalidated, so it may raise an exception
            offline_analytics_settings = (
                self.context.get_app_config().get("analytics", {}).get("offlineAnalyticsUri", {})
            )
            expert_url = check_expert_availability(offline_analytics_settings)
            self.offline_client = OfflineAnalyticsClientFactory.get_client(expert_url, self.context.get_camera_id())
        self.flexibility = self.expert_check_interval // 2
        self.next_expert_detection_timestamp = self.last_expert_check
        self._skip_debug_data = False
        # multi-responses expert handling
        self.expert_requests_queue = []
        self.expert_suppressed_alerts = {}

        # snapshot resizer
        expert_args = YoloV8ExpertArgs({})
        expert_img_size = tuple(expert_args.im_size)
        self.resizer = SnapshotResizerFactory.get_resizer(self.context.get_camera_id(), expert_img_size)
        self.resizer.antialias = expert_args.antialias
        self.max_timestamp = -1

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []
        vars_data = self.query_batch(batch_data)
        vars_data = vars_data[vars_data[:, BDR.CONFIDENCE] > self.confidence_th]
        person_data = batch_data.query(BDR.CLASS, self.person_obj)

        n_vars = len(vars_data)
        if self.roiFilter and len(vars_data) > 0:
            locations = vars_data[:, BatchDataResolver.LOCATION].astype(int)
            in_roi = [i for i in range(n_vars) if locations[i] in self.marked_idx]
            vars_data = vars_data[in_roi]

        if len(vars_data) > 0:
            # gather hazard data
            hazard_data = []
            for image in images:
                detections = vars_data[vars_data[:, BatchDataResolver.TIMESTAMP] == image.timestamp]
                persons = person_data[person_data[:, BatchDataResolver.TIMESTAMP] == image.timestamp]
                # filter based on overlaps
                if len(detections) > 1:
                    confs = detections[:, BatchDataResolver.CONFIDENCE]
                    idxs = np.argsort(confs)[::-1]
                    detections = detections[idxs]
                    overlaps = calc_max_overlap(detections[:, BatchDataResolver.POS])
                    bad_idx = set()
                    for i in range(len(detections)):
                        det_overlaps = np.where(overlaps[i] > 0.001)[0]
                        bad_idx.update(det_overlaps)
                        overlaps[i, :] = 0
                        overlaps[:, i] = 0
                    detections = np.delete(detections, list(bad_idx), axis=0)

                for detection in detections:
                    center = detection[BatchDataResolver.CENTER]
                    no_person_near = self.person_filter
                    min_person = -1
                    bbox = detection[BatchDataResolver.POS]
                    if self.person_filter and len(persons) > 0:
                        person_dists = np.linalg.norm(center - persons[:, BatchDataResolver.CENTER], axis=1)
                        min_person = np.argmin(person_dists)
                        no_person_near = person_dists[min_person] > self.distance_th
                        if no_person_near:
                            joint_bbox = np.vstack((detections, persons))
                            no_person_near = np.all(calc_max_overlap(joint_bbox[:, BatchDataResolver.POS]) < 0.01)
                        bbox = joint_bbox_np(bbox, persons[min_person][BatchDataResolver.POS])
                        min_person = int(persons[min_person, BatchDataResolver.ID])

                    if not no_person_near:
                        crop = crop_image(
                            image.frame,
                            # cv2.resize(image.frame, [1280, 704]),
                            detection[BatchDataResolver.POS],
                            margins=self.margins,
                            bgr_map=False,
                        )
                        conf = detection[BatchDataResolver.CONFIDENCE]
                        id_ = detection[BatchDataResolver.ID]
                        hazard_data.append(
                            HazardTrackingData(image.timestamp, center, crop, conf, int(id_), bbox, min_person)
                        )

            self._update_tracking(hazard_data)
            batch_ts = [image.timestamp for image in images]
            for obj in self.hazard_tracking:
                if not obj["active"]:
                    obj["hazards"] = obj["hazards"][-1:]
                elif len(obj["hazards"]) >= self.latency:
                    if self.is_blacklisted(obj):
                        obj["active"] = False
                        continue
                    if self.require_classifier:
                        for_classification = [w for w in obj["hazards"] if w.classification is None]
                        if len(for_classification) == 0:
                            continue
                        self.perform_classification(for_classification)
                        sample_size = max(self.latency, len(for_classification))
                        positives = [w for w in obj["hazards"][-sample_size:] if w.classification]
                        if len(positives) / len(for_classification) >= self.active_ratio:
                            batch_positives = [w for w in positives if w.timestamp in batch_ts]
                            selected_obj = obj["hazards"][-1]
                            if len(batch_positives):
                                max_conf_ind = np.argmax([w.classification_score for w in batch_positives])
                                selected_obj = batch_positives[max_conf_ind]
                            img_idx = batch_ts.index(selected_obj.timestamp)
                            person_ids = [p.person_id for p in positives if p.person_id >= 0]
                            if selected_obj.person_id < 0 and person_ids:
                                selected_obj.person_id = person_ids[-1]
                            alert_candidate = self._gen_alert_candidate(obj, selected_obj, images[img_idx])
                            if self.activate_ddata and self.internal_alert and not self._skip_debug_data:
                                alert_candidate = self.add_debug_data(alert_candidate)
                            alert_candidates.append(alert_candidate)
                        else:  # not classified as hazards - remain tracking
                            pass
                    else:
                        batch_ids = [i for i in range(len(obj["hazards"])) if obj["hazards"][i].timestamp in batch_ts]
                        idx = np.argmax([obj["hazards"][i].confidence for i in batch_ids])
                        selected_idx = batch_ids[idx]
                        img_idx = batch_ts.index(obj["hazards"][selected_idx].timestamp)
                        alert_candidate = self._gen_alert_candidate(obj, obj["hazards"][selected_idx], images[img_idx])
                        if self.activate_ddata and self.internal_alert and not self._skip_debug_data:
                            alert_candidate = self.add_debug_data(alert_candidate)
                        alert_candidates.append(alert_candidate)

        # clean-ups
        ts_th = images[-1].timestamp - self.time_delta
        to_delete = []
        for track_idx, obj in enumerate(self.hazard_tracking):
            valid_idxs = [i for i in range(len(obj["hazards"])) if obj["hazards"][i].timestamp > ts_th]
            if len(valid_idxs) == 0:
                to_delete.append(track_idx)
            else:
                # obj["hazards"] = [obj["hazards"][i] for i in valid_idxs[-self.latency :]]
                # this should keep the last relevant example since its already sorted by timestamp
                obj["hazards"] = obj["hazards"][-self.latency :]
        to_delete.reverse()
        for idx in to_delete:
            self.apply_to_blacklist(self.hazard_tracking[idx])
            del self.hazard_tracking[idx]
        if self.blacklist_enabled and to_delete:
            ts_th = images[-1].timestamp - self.blacklist_expiration_ms
            self.blacklist = [bl for bl in self.blacklist if bl["timestamp"] > ts_th]
        if self.use_expert:
            self.detect_with_expert(batch_data, alert_candidates, images)
        return alert_candidates

    def add_debug_data(self, candidate: AlertCandidate) -> AlertCandidate:
        candidate.extra = candidate.extra or {}

        # helper function to extract bbox from different sources
        def extract_bbox_and_center(candidate):
            center = None
            if candidate.extra.get("from_expert", False):
                if "expert_data" in candidate.extra:
                    expert_data = candidate.extra["expert_data"]
                    # try different bbox field names from expert data
                    bbox = expert_data.get("weapon_location") or expert_data.get("perimeter", [])
                else:
                    # basic expert candidate (RareAppearanceAlert)
                    bbox = candidate.extra.get("bbox", [])
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            else:
                # regular candidate with hazard_data
                if "hazard_data" in candidate.extra:
                    bbox = candidate.extra["hazard_data"].bbox
                    center = candidate.extra["hazard_data"].center
                else:
                    bbox = []
            # calculate center from bbox if available
            return bbox, center

        # helper function to extract confidence/score
        def extract_confidence_score(candidate):
            if candidate.extra.get("from_expert", False):
                if "expert_data" in candidate.extra:
                    expert_data = candidate.extra["expert_data"]
                    return (expert_data.get("confidence"), expert_data.get("score") or expert_data.get("weapon_score"))
                return None, None
            else:
                if "hazard_data" in candidate.extra:
                    hazard_data = candidate.extra["hazard_data"]
                    return hazard_data.confidence, hazard_data.classification_score
                return None, None

        # helper function to extract classification
        def extract_classification(candidate):
            if candidate.extra.get("from_expert", False):
                # Expert data might not have classification in the same format
                return None
            else:
                if "hazard_data" in candidate.extra:
                    return candidate.extra["hazard_data"].classification
                return None

        # helper function to extract crop
        def extract_crop(candidate):
            if candidate.extra.get("from_expert", False):
                # For expert candidates, crop might be in validation_images or crops
                if hasattr(candidate, "crops") and candidate.crops:
                    return candidate.crops[0]
                return None
            else:
                if "hazard_data" in candidate.extra:
                    return candidate.extra["hazard_data"].crop
                return None

        # extract all fields using helper functions
        bbox, center = extract_bbox_and_center(candidate)
        confidence, classification_score = extract_confidence_score(candidate)
        classification = extract_classification(candidate)
        crop = extract_crop(candidate)
        # build debug data
        debug_data = {
            "alert_id": candidate.alert_id,
            "ent_ids": candidate.ent_ids,
            "timestamp": candidate.timestamp,
            "validation_images": candidate.validation_images,
            "bbox": bbox,
            "center": center,
            "classification": classification,
            "classification_score": classification_score,
            "confidence": confidence,
            "crop": crop,
            "snapshot": candidate.extra.get("snapshot", None),
            "from_expert": candidate.extra.get("from_expert", False),
        }
        # add expert-specific fields if available
        if candidate.extra.get("from_expert", False) and "expert_data" in candidate.extra:
            expert_data = candidate.extra["expert_data"]
            debug_data.update(
                {
                    "person_id": expert_data.get("person_id"),
                    "camera_id": expert_data.get("camera_id"),
                    "weapon_location": expert_data.get("weapon_location"),
                    "perimeter": expert_data.get("perimeter"),
                }
            )
        candidate.extra["debug_data"] = self.serialize_debug_data(debug_data)
        return candidate

    def detect_with_expert(
        self, batch_data: BatchDataResolver, candidates: List[AlertCandidate], images: List[AnalyticImage]
    ):
        # remove outdated expert requests from queue and re-enable requests if all timed out
        expert_requests_cpy = self.expert_requests_queue.copy()
        for req_ts in expert_requests_cpy:
            if images[-1].timestamp - req_ts > self.request_timeout:
                self.expert_requests_queue.remove(req_ts)

        check_th = self.last_expert_check + self.expert_check_interval
        timestamps = np.array([img.timestamp for img in images])
        idxs = np.argwhere(timestamps >= check_th)
        if idxs.size > 0:
            idx = idxs[0][0]
            success = self.offline_client.send_expert_request(images[idx], flexibility=self.flexibility)
            if success:
                self.last_expert_check = images[idx].timestamp
                self.expert_requests_queue.append(images[idx].timestamp)

        # process any pending expert responses
        expert_requests_cpy = self.expert_requests_queue.copy()
        prev_max_timestamp = self.max_timestamp
        for req_ts in expert_requests_cpy:
            detections = self.offline_client.get_expert_detections(req_ts, self.flexibility)
            if detections is not None:
                detections = [d for d in detections if d is not None]
                if len(detections) > 0:
                    self.expert_requests_queue.remove(req_ts)
                    self.max_timestamp = max(self.max_timestamp, req_ts)
                ts_to_remove = [
                    k
                    for k in self.expert_suppressed_alerts.keys()
                    if k - timestamps[0] > self.expert_suppression_window
                ]
                for k in ts_to_remove:
                    self.expert_suppressed_alerts.pop(k)
                suppressed_rois = list(self.expert_suppressed_alerts.values())
                if suppressed_rois:
                    suppressed_rois = np.vstack(suppressed_rois)
                else:
                    suppressed_rois = np.empty((0, 4))
                for det in sorted(detections, key=lambda x: x["timestamp"]):
                    if det["timestamp"] > self.next_expert_detection_timestamp:
                        self.next_expert_detection_timestamp = self.last_expert_check + self.expert_check_interval
                        # find rare appearance candidate
                        img_det = np.atleast_2d(det["detections"])
                        if img_det.size > 0:
                            rare_det = img_det[np.isin(img_det[:, 5], self.class_filters), :]
                            rare_det = rare_det[rare_det[:, 4] > self.confidence_th, :]
                            if rare_det.size and self.roiFilter:
                                locations = bbox_to_location(rare_det[:, :4])
                                in_roi = [i for i in range(len(locations)) if locations[i] in self.marked_idx]
                                rare_det = rare_det[in_roi, :]
                            if rare_det.size and suppressed_rois.size:
                                ious = np.max(bbox_ious(rare_det[:, :4], suppressed_rois), axis=1)
                                rare_det = rare_det[ious < self.suppression_iou_th, :]
                        else:
                            rare_det = np.empty((0, 6))
                        self.expert_detection_log.append((det["timestamp"], rare_det))
                        if len(self.expert_detection_log) == self.expert_detection_log.maxlen:
                            has_alert = (
                                np.mean([int(d[1].size > 0) for d in self.expert_detection_log]) >= self.active_ratio
                            )
                            if has_alert:
                                # generate alert candidate
                                all_detections = np.vstack([d[1] for d in self.expert_detection_log if d[1].size > 0])
                                bbox = np.hstack(
                                    (np.min(all_detections[:, :2], axis=0), np.max(all_detections[:, 2:4], axis=0))
                                )
                                ts = self.expert_detection_log.popleft()[0]
                                self.expert_detection_log.clear()
                                self.next_expert_detection_timestamp = self.last_expert_check + self.blockout
                                crop = crop_image(images[-1].frame, bbox, margins=self.margins, bgr_map=False)
                                candidate = self.build_alert_candidate(
                                    ts, ent_ids=-1, extra={"from_expert": True, "bbox": bbox}, crops=[crop]
                                )
                                candidates.append(candidate)
                                self.expert_suppressed_alerts[ts] = all_detections
        # clean up old requests from queue in case newer requests were processed
        if self.max_timestamp > prev_max_timestamp:
            expert_requests_cpy = self.expert_requests_queue.copy()
            for remain_ts in expert_requests_cpy:
                if remain_ts < self.max_timestamp:
                    self.expert_requests_queue.remove(remain_ts)

    def _gen_alert_candidate(self, obj: dict, data: HazardTrackingData, image: AnalyticImage) -> AlertCandidate:
        alert_candidate = self.build_alert_candidate(
            data.timestamp,
            ent_ids=-1,
            extra={"hazard_data": data, "snapshot": self.resizer.resize(image, is_bgr=False)},
            validation_images=[self._generate_validation_crop(data.bbox, image, margins=self.validation_crop_margins)],
        )
        self.apply_to_blacklist(obj)
        obj["active"] = False
        obj["hazards"] = obj["hazards"][-1:]
        return alert_candidate

    def _update_tracking(self, hazard_data):
        for hazard in hazard_data:
            added = False
            dist = [np.linalg.norm(hazard.center - obj["center"]) for obj in self.hazard_tracking]
            obj_sorted = np.argsort(dist)
            for obj in obj_sorted:
                if (
                    dist[obj] < self.distance_th
                    and self.hazard_tracking[obj]["hazards"][-1].timestamp < hazard.timestamp
                ):
                    self.hazard_tracking[obj]["hazards"].append(hazard)
                    self.hazard_tracking[obj]["center"] = hazard.center  # np.mean([w.center for w in obj["weapons"]])
                    added = True
                    break

            if not added:
                self.hazard_tracking.append({"center": hazard.center, "hazards": [hazard], "active": True})

    def build_alert_info(self, candidate: AlertCandidate):
        # score = results_scores[selected_idx]
        # conf = data.confidence
        # crop = data.crop
        alert_info = super().build_alert_info(candidate)
        alert_info.object_id = self.hazard_obj
        if candidate.extra.get("from_expert", False):
            alert_info.perimeter = candidate.extra.get("bbox", None)
            alert_info.extra = {
                "score": None,
                "classification": None,
                "from_expert": True,
            }
        else:
            data: HazardTrackingData = candidate.extra["hazard_data"]
            alert_info.idIndex = -1  # data.ent_id
            alert_info.idBase = -1  # data.timestamp
            alert_info.extra = {
                "score": data.classification_score,
                "classification": data.classification,
                "from_expert": False,
            }
            alert_info.perimeter = data.bbox
            alert_info.crops.append(cv2.cvtColor(data.crop, cv2.COLOR_RGB2BGR))
            alert_info.timestamp = data.timestamp

        return alert_info

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        return True

    def _init_classifier(self):
        return None, {}

    def apply_to_blacklist(self, hazard_tracking: Dict):
        if self.blacklist_enabled and hazard_tracking.get("active", True):
            hazards: List[HazardTrackingData] = hazard_tracking.get("hazards", [])
            if len(hazards) > 0:
                conf, location_std, avg_point = get_hazard_list_stats(hazards)
                conf = np.mean([h.confidence for h in hazards])
                if conf < self.blacklist_conf:
                    if len(hazards) == 1 or location_std < self.blacklist_distance_std:
                        # check if already exists in blacklist
                        for bl in self.blacklist:
                            if np.linalg.norm(bl["center"] - avg_point) < self.blacklist_distance_th:
                                bl["center"] = avg_point
                                bl["timestamp"] = hazards[-1].timestamp
                                return
                        self.blacklist.append(
                            {
                                "center": avg_point,
                                "timestamp": hazards[-1].timestamp,
                            }
                        )

    def is_blacklisted(self, hazard_tracking):
        if self.blacklist_enabled and len(self.blacklist) > 0:
            hazards: List[HazardTrackingData] = hazard_tracking.get("hazards", [])
            conf, location_std, avg_point = get_hazard_list_stats(hazards)
            if conf < self.blacklist_conf and location_std < self.blacklist_distance_std:
                for bl in self.blacklist:
                    if np.linalg.norm(bl["center"] - avg_point) < self.blacklist_distance_th:
                        bl["timestamp"] = hazards[-1].timestamp
                        return True
        return False

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        BaseAlert.generate_alert_message(self, candidate, alert_info)

    def perform_classification(self, for_classification: List[HazardTrackingData]):
        pass


class WeaponAppearanceAlert(RareAppearanceAlert):
    require_classifier = True
    object_type = "weapon"
    # class_filters = ["gun"]
    classifier: WCResnet18Classifier = None
    time_delta = 2000
    confidence_th = 0.2
    distance_th = 8
    active_ratio = 0.51
    alert_message = "Threat is detected"
    person_filter = True
    latency = 2
    last_expert_check = -1000000
    expert_url: str = None
    candidate_buffer: dict
    candidate_hold = 5000
    candidate_max_count = 4

    use_expert: bool = True
    expert_check_interval = 1500

    def __init__(self, alert_dict: Dict, context):
        super(WeaponAppearanceAlert, self).__init__(alert_dict, context)

        analytic_config = self.context.get_config()

        if "alertConfig" in analytic_config:
            if "weaponAppearance" in analytic_config["alertConfig"]:
                cfg = analytic_config["alertConfig"]["weaponAppearance"]
                if "minBlockout" in cfg:
                    self.blockout = max(self.blockout, cfg["minBlockout"])
                self.candidate_hold = cfg.get("candidateHold", self.candidate_hold)
                self.candidate_max_count = cfg.get("candidateMaxCount", self.candidate_max_count)
                self.expert_check_interval = cfg.get("expertCheckInterval", self.expert_check_interval)
                self.distance_th = cfg.get("distanceTh", self.distance_th)
                self.latency = cfg.get("latency", self.latency)
                self.use_expert = cfg.get("useExpert", self.use_expert)
                self.confidence_th = cfg.get("confidenceTh", self.confidence_th)
        self.alerted_persons = deque(maxlen=10)
        self.candidate_buffer = {}

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        self._skip_debug_data = True  # avoid adding debug data on first run
        candidates = super().is_active_batch(images, motion_data, batch_data)
        self._skip_debug_data = False  # enable debug data for progressive candidates

        ready_candidates = []
        if candidates:
            for cand in candidates:
                person_id = WeaponAppearanceAlert._person_id_from_candidate(cand)
                if person_id >= 0:
                    if person_id not in self.candidate_buffer:
                        self.candidate_buffer[person_id] = {"timestamp": cand.timestamp, "candidates": [cand]}
                    else:
                        self.candidate_buffer[person_id]["candidates"].append(cand)
                        prev_ts = self.candidate_buffer[person_id]["timestamp"]
                        self.candidate_buffer[person_id]["timestamp"] = min(cand.timestamp, prev_ts)
                elif cand.extra.get("in_roi", True):
                    # if the candidate is not from expert and is in roi, we can use it directly
                    if self.activate_ddata and self.internal_alert:
                        cand = self.add_debug_data(cand)
                    ready_candidates.append(cand)
        if self.candidate_buffer:
            person_ids = list(self.candidate_buffer.keys())
            last_ts = images[-1].timestamp
            for pid in person_ids:
                data = self.candidate_buffer[pid]
                in_roi = any([c.extra.get("in_roi", True) for c in data["candidates"]])
                if not in_roi:
                    # check if the person is in roi
                    person_locs = set(batch_data.query(BDR.ID, pid, BDR.LOCATION).astype(int))
                    in_roi = len(person_locs.intersection(self.marked_idx)) > 0
                    data["candidates"][-1].extra["in_roi"] = in_roi
                if in_roi:
                    if (
                        last_ts - data["timestamp"] > self.candidate_hold
                        or len(data["candidates"]) >= self.candidate_max_count
                    ):
                        # select the best candidate:
                        if len(data["candidates"]) == 1:
                            best_cand = data["candidates"][0]
                        else:
                            scores = [WeaponAppearanceAlert._score_from_candidate(c) for c in data["candidates"]]
                            max_idx = np.argmax(scores)
                            best_cand = data["candidates"][max_idx]
                        if self.activate_ddata and self.internal_alert:
                            best_cand = self.add_debug_data(best_cand)
                        ready_candidates.append(best_cand)
                        del self.candidate_buffer[pid]
                elif last_ts - data["timestamp"] > self.candidate_hold * 2:
                    # remove the candidate if it is not in roi and hold time is exceeded
                    del self.candidate_buffer[pid]

        return ready_candidates

    def detect_with_expert(
        self, batch_data: BatchDataResolver, candidates: List[AlertCandidate], images: List[AnalyticImage]
    ):
        active_alerts = [obj for obj in self.hazard_tracking if obj["active"]] + candidates
        if not candidates and not active_alerts:
            # use expert detector
            check_th = self.last_expert_check + self.expert_check_interval

            # only check with expert if there are person detections
            person_detections = batch_data.query(BatchDataResolver.CLASS, self.person_obj)
            images_with_persons = set(person_detections[:, BDR.FRAME_ID].astype(int).tolist())
            for idx, img in enumerate(images):
                if img.timestamp > check_th and idx in images_with_persons:
                    img_data = batch_data.query(BatchDataResolver.FRAME_ID, idx)
                    success = self.offline_client.send_weapon_request(
                        img,
                        img_data,
                        flexibility=self.expert_check_interval // 2,
                        confidence=self.alert_confidence,
                    )
                    if success:
                        self.last_expert_check = img.timestamp
                        self.expert_retries = 0
                    else:
                        self.expert_retries += 1
                        if self.expert_retries > 3:
                            self.use_expert = False
                            logger.error("Expert detector is disabled due to multiple failures")
                    break
        alerts = self.offline_client.get_active_weapon_alerts()
        if alerts is None:
            self.expert_retries += 1
            if self.expert_retries > 3:
                self.use_expert = False
                logger.error("Expert detector is disabled due to multiple failures")
        if alerts:
            for alert in alerts:
                in_roi = not self.roiFilter
                if self.roiFilter:
                    location = alert.get("weapon_location")
                    lower_y = np.minimum(ROI_SHAPE[1] - 1, int((location[3] * ROI_SHAPE[1])))
                    center_x = (location[0] + location[2]) * ROI_SHAPE[0] / 2
                    loc_idx = np.ravel_multi_index((lower_y, int(center_x)), ROI_SHAPE, order="C")
                    in_roi = loc_idx in self.marked_idx
                if in_roi:
                    alert_candidate = self.build_alert_candidate(
                        alert["timestamp"],
                        ent_ids=-1,
                        extra={
                            "expert_data": alert,
                            "from_expert": True,
                            "in_roi": in_roi,
                            "snapshot": alert.get("snapshot"),
                            "is_brandished": alert.get("is_brandished"),
                        },
                        validation_images=[alert.pop("validation_crop")],
                        crops=[alert.pop("zoom_crop")],
                    )
                    candidates.append(alert_candidate)

    def build_alert_info(self, candidate: AlertCandidate):

        if candidate.extra.get("from_expert", False):
            data = candidate.extra["expert_data"]
            alert_info = super(RareAppearanceAlert, self).build_alert_info(candidate)
            alert_info.object_id = self.hazard_obj
            alert_info.idIndex = -1  # data.ent_id
            alert_info.idBase = -1  # data.timestamp
            alert_info.perimeter = data.get("perimeter", [])
            alert_info.extra.update({"from_expert": True, "in_roi": candidate.extra.get("in_roi", True)})
        else:
            alert_info = super().build_alert_info(candidate)
        return alert_info

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        person_id = WeaponAppearanceAlert._person_id_from_candidate(candidate)
        if person_id >= 0:
            if person_id in self.alerted_persons:
                return False
            self.alerted_persons.append(person_id)
        return True

    def update_model(self, config):
        self.classifier_config.update(config)
        self.classifier = WCResnet18Classifier(self.classifier_config)

    def _init_classifier(self):
        wc_config = {}
        analytic_config = self.context.get_config()

        if "l1_models" in analytic_config and "attributes" in analytic_config["l1_models"]:
            wc_config = analytic_config["l1_models"]["attributes"].get("weapons_classification", {})
            wc_config = {}
        return WCResnet18Classifier(wc_config), wc_config

    def perform_classification(self, for_classification: List[HazardTrackingData]):
        crop_list = [w.crop for w in for_classification]
        results_cls, results_scores = self.classifier.forward_on_crop_list(crop_list)
        for wid, w in enumerate(for_classification):
            w.classification = results_cls[wid] == 0
            w.classification_score = results_scores[wid]
            # in order to save all the classified crops, uncomment the following lines
            # w.classification = False
            # cv2.imwrite(
            #     f"/mnt/d/weapons/out/weapon_{w.timestamp}_{results_cls[wid]}.png",
            #     cv2.cvtColor(w.crop, cv2.COLOR_RGB2BGR),
            # )

    @staticmethod
    def _person_id_from_candidate(candidate: AlertCandidate):
        if candidate.extra.get("from_expert", False):
            data = candidate.extra["expert_data"]
            person_id = data.get("person_id", -1)
        else:
            data = candidate.extra["hazard_data"]
            person_id = data.person_id
        return person_id

    @staticmethod
    def _score_from_candidate(candidate: AlertCandidate) -> float:
        if candidate.extra.get("from_expert", False):
            data = candidate.extra["expert_data"].get("perimeter", [])
        else:
            data = candidate.extra["hazard_data"].bbox
        score = (data[2] - data[0]) * (data[3] - data[1]) if len(data) else 0
        return score


class BrandishedWeaponAlert(WeaponAppearanceAlert):
    alert_message = "Brandished weapon is detected"
    hand_iou_th = 0.05
    confidence_th = 0.1

    def __init__(self, alert_dict: Dict, context):
        super(BrandishedWeaponAlert, self).__init__(alert_dict, context)
        self.query_batch = self._query_batch_brandished
        self.hand_cls = self.context.get_class_handler().class_str_to_int("hand")

    def _query_batch_brandished(self, batch_data: BatchDataResolver):
        hands_data = batch_data.query(BatchDataResolver.SUBCLASS, self.hand_cls, BDR.POS)
        weapon_data = batch_data.query(BatchDataResolver.CLASS, self.hazard_obj)
        ious = np.max(bbox_ious(weapon_data, hands_data), axis=1) if len(hands_data) > 0 else np.zeros(len(weapon_data))
        return weapon_data[ious > self.hand_iou_th]

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        if candidate.extra.get("is_brandished") is False:
            return False
        return super().check_alert_candidate(candidate, force)


class FireAlert(RareAppearanceAlert):
    require_classifier = False
    object_type = "hazard"
    class_filters = ["fire"]  # "smoke", smoke is too unreliable - using just fire for now
    person_filter = False
    time_delta = 10000  # 10 seconds
    latency = 10  # require 10 detections in 10 seconds to trigger alert
    confidence_th = 0.15
    blacklist_distance_var = 5
    blacklist_enabled = True
    alert_message = "Fire is detected"
    validation_crop_margins = [0.25, 0.25]
    smoke_enabled = False
    latency_per_setting = {0: 3, 1: 5, 2: 10}  # lower confidence requires more latency
    confidence_per_setting = {0: 0.1, 1: 0.12, 2: 0.15}  # low, medium, high

    use_expert: bool = True
    expert_check_interval = 5000
    expert_suppression_window = 15 * 60 * 1000  # 15 minute

    def __init__(self, alert_dict: Dict, context):
        super(FireAlert, self).__init__(alert_dict, context)
        self.blockout = 60 * 1000  # 1 minute
        self.set_flow_values()
        self.confidence_setting = self.settings.get("confidence", 2)  # default is high confidence
        self.confidence_th = self.confidence_per_setting[self.confidence_setting]
        self.latency = self.latency_per_setting[self.confidence_setting]
        self.on_night_mode_changed(False)  # initialize with night mode off
        if self.confidence_setting == 0 and self.smoke_enabled:
            self.special_filter = "smoke"

    def on_night_mode_changed(self, night_mode: bool):
        class_filters = ["fire"]
        if (self.confidence_setting == 0 or not night_mode) and self.smoke_enabled:
            class_filters.append("smoke")
        self.class_filters = [self.context.get_class_handler().class_str_to_int(s) for s in class_filters]

    def _update_tracking(self, hazard_data):
        if len(self.hazard_tracking) > 0:
            self.hazard_tracking[-1]["hazards"] += hazard_data
            self.hazard_tracking[-1]["timestamp"] = hazard_data[-1].timestamp
        else:
            self.hazard_tracking = [{"timestamp": hazard_data[-1].timestamp, "hazards": hazard_data, "active": True}]

    def set_flow_values(self):
        if self.formValue:
            self.smoke_enabled = self.formValue.get("type", "fire") == "smoke"
