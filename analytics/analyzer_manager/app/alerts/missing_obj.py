from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from general.core import AnalyticImage, BDR, calculate_overlap_matrix, AlertInfo, medoid, MemoryHandler
from general.image_encoder import ImageEncoder
from general.img_utils import crop_image, measure_contrast
from .base_alerts import AlertCandidate, BaseAlert


@dataclass
class ObjectInfo:
    name: str
    descriptor: np.ndarray
    position: np.ndarray
    crops: List[np.ndarray] = field(default_factory=lambda: [])
    crops_ts: List[int] = field(default_factory=lambda: [])
    descriptors: List[Optional[np.array]] = field(default_factory=lambda: [])
    is_alerted: bool = False  # if set to true it will not raise alert on startup
    last_motion_active: int = -1000000
    last_sampled_time = -1000000
    is_motion_active: bool = False
    is_vehicle: bool = False
    last_observed_descriptor: np.ndarray = None
    alert_renewal_time: int = -1000000
    is_suspected: bool = False
    suspected_ts: int = -1000000
    alerted_descriptors: List[Optional[np.array]] = field(default_factory=lambda: [])


class MissingObjectAlert(BaseAlert):
    type_name = "missingObjectAlerts"
    object_based_alert = False
    crops_per_obj = 5
    tracked_objects: List[ObjectInfo]
    sample_time_ms: int = 5000
    motion_inactive_time_ms = 1000
    alert_timeout: int = 60000
    match_threshold: float = 0.85
    orig_desc_match_threshold: float = 0.8
    obj_overlap_threshold: float = 0.1
    n_obj: int
    position_matrix: np.ndarray
    min_acceptable_contrast = 10  # measured in std of gray level
    max_hold = 10000
    max_obj_alert_history = 10

    def __init__(self, init_dict: Dict, context):
        super(MissingObjectAlert, self).__init__(init_dict, context)
        self.encoder = ImageEncoder({})
        self.set_flow_values()
        self.memory_handler = MemoryHandler(
            num_buffers=len(self.tracked_objects),
            num_crops=self.crops_per_obj,
            crop_size=list(self.encoder.image_size) + [3],
            dtype=np.uint8,
        )
        # get the alert configuration if exists
        cfg = self.alerts_config.get(self.type_name, {})
        self.max_hold = cfg.get("max_hold", self.max_hold)

    def set_flow_values(self):
        if self.formValue:
            sensitivity = self.apply_from_dict("sensitivity", self.formValue, 0.5)
            if sensitivity <= 0.5:
                self.match_threshold *= sensitivity / 0.5
                self.orig_desc_match_threshold *= sensitivity / 0.5
            else:
                self.match_threshold += (1 - self.match_threshold) * (sensitivity - 0.5) / 0.5
                self.orig_desc_match_threshold += (1 - self.match_threshold) * (sensitivity - 0.5) / 0.5
        if self.selectedCamera:
            self.tracked_objects = []
            objects = self.apply_from_dict("objectsData", self.selectedCamera, [])
            for idx, obj in enumerate(objects):
                desc = np.frombuffer(bytes.fromhex(obj["descriptor"]), dtype=np.float32)
                pos = np.array(obj["position"])
                self.tracked_objects.append(ObjectInfo(name=idx, descriptor=desc, position=pos))
            self.n_obj = len(self.tracked_objects)
            assert self.n_obj > 0, "No objects to track in the missing object alert"
            self.position_matrix = np.array([obj.position for obj in self.tracked_objects])

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data: BDR) -> List[AlertCandidate]:
        alert_candidates = []
        iou = np.zeros([self.n_obj, len(images)])
        timestamps = np.array([im.timestamp for im in images])
        batch_size = batch_data.batch_size

        for fid in range(batch_size):
            detections = batch_data.query(BDR.FRAME_ID, fid, BDR.POS)
            if len(detections):
                overlaps = calculate_overlap_matrix(self.position_matrix, detections)
                iou[:, fid] = np.max(overlaps, axis=1)

        for obj_idx, obj in enumerate(self.tracked_objects):
            if obj.is_suspected:
                if timestamps[0] > obj.suspected_ts + self.max_hold:
                    to_alert = True
                    cropped = crop_image(images[-1].frame, obj.position, margins=[0, 0], bgr_map=False)
                    contrast = measure_contrast(cropped)
                    if contrast > self.min_acceptable_contrast:
                        crop = self.encoder.apply_resize(cropped, self.memory_handler.next(obj_idx))
                        obj.crops.append(crop)
                        query_descriptor = self.encoder.forward_on_crop_list([crop])
                        base_score = query_descriptor @ obj.descriptor
                        if base_score > self.match_threshold:
                            to_alert = False
                            obj.last_observed_descriptor = query_descriptor
                            obj.descriptors.clear()
                            obj.crops.clear()
                            obj.crops_ts.clear()
                    if to_alert:
                        candidate = self._gen_candidate_for_obj(images[-1], obj)
                        if candidate:
                            alert_candidates.append(candidate)
                        continue
                    else:
                        obj.is_suspected = False
                else:
                    continue

            motion_active = np.where(iou[obj_idx] > self.obj_overlap_threshold)[0]
            if obj.is_alerted:
                # auto block for a minute and then test to see if the object appeared again
                if timestamps[0] > obj.alert_renewal_time:
                    if motion_active.size == 0:
                        image = images[0].frame
                        crop = self.encoder.apply_resize(crop_image(image, obj.position, margins=[0, 0], bgr_map=False))
                        query_descriptor = self.encoder.forward_on_crop_list([crop])
                        base_score = query_descriptor @ obj.descriptor
                        if base_score > self.match_threshold:
                            obj.is_alerted = False
                        else:
                            obj.alert_renewal_time = timestamps[0] + self.alert_timeout
                continue

            if motion_active.size > 0:  # wait for inactivity
                obj.last_motion_active = timestamps[motion_active[-1]]
                obj.is_motion_active = True
                continue

            if obj.is_motion_active:
                sample_images = np.where(timestamps > obj.last_motion_active + self.motion_inactive_time_ms)[0]
                if sample_images.size > 0:
                    s_idx = sample_images[0]
                    image = images[s_idx].frame
                    crop = self.encoder.apply_resize(crop_image(image, obj.position, margins=[0, 0], bgr_map=False))
                    idx_to_analyze = [idx for idx, d in enumerate(obj.descriptors) if d is None]
                    crops = [crop]
                    if len(idx_to_analyze) > 0:
                        crops.extend([obj.crops[idx] for idx in idx_to_analyze])
                        desc = self.encoder.forward_on_crop_list(crops)
                        query_descriptor = desc[0]
                        for idx, d in zip(idx_to_analyze, desc[1:]):
                            obj.descriptors[idx] = d
                    else:
                        query_descriptor = self.encoder.forward_on_crop_list([crop])
                    base_score = query_descriptor @ obj.descriptor  # original from alert
                    curr_score = 0
                    descriptors = np.array([d for d in obj.descriptors if d is not None])
                    if descriptors.size > 0:  # from steady state descriptors
                        curr_score = query_descriptor @ np.mean(descriptors, axis=0)

                    if base_score < self.orig_desc_match_threshold or 0 < curr_score < self.match_threshold:
                        self.suspect_obj_missing(obj, images[-1])
                    else:
                        obj.is_motion_active = False

            else:  # regular sampling of crops to generate baseline based on lighting condition and object position
                sample_images = np.where(timestamps > obj.last_sampled_time + self.sample_time_ms)[0]
                if sample_images.size > 0:
                    s_idx = sample_images[0]
                    image = images[s_idx].frame
                    cropped = crop_image(image, obj.position, margins=[0, 0], bgr_map=False)
                    contrast = measure_contrast(cropped)
                    if contrast > self.min_acceptable_contrast:
                        crop = self.encoder.apply_resize(cropped, self.memory_handler.next(obj_idx))
                        obj.crops.append(crop)
                        ts = images[s_idx].timestamp
                        obj.crops_ts.append(ts)
                        obj.descriptors.append(None)
                        obj.next_sampled_time = ts + self.sample_time_ms
                        obj.crops_ts = obj.crops_ts[-self.crops_per_obj :]
                        obj.crops = obj.crops[-self.crops_per_obj :]
                        obj.descriptors = obj.descriptors[-self.crops_per_obj :]
                idx_to_analyze = [idx for idx, d in enumerate(obj.descriptors) if d is None]

                if len(obj.crops) == self.crops_per_obj and len(idx_to_analyze) == self.crops_per_obj:
                    crops = [obj.crops[idx] for idx in idx_to_analyze]
                    desc = self.encoder.forward_on_crop_list(crops)
                    med_desc = desc[medoid(desc)]
                    if obj.last_observed_descriptor is not None:
                        match_scores = np.median(obj.last_observed_descriptor @ desc.T)
                        if (
                            match_scores < self.match_threshold
                            or obj.descriptor @ med_desc < self.orig_desc_match_threshold
                        ):
                            self.suspect_obj_missing(obj, images[-1])
                    obj.last_observed_descriptor = med_desc

        return alert_candidates

    def suspect_obj_missing(self, obj: ObjectInfo, image: AnalyticImage):
        obj.is_suspected = True
        obj.suspected_ts = image.timestamp

    def _gen_candidate_for_obj(self, image: AnalyticImage, obj: ObjectInfo) -> Optional[AlertCandidate]:

        obj.is_suspected = False
        obj.is_alerted = True
        obj.alert_renewal_time = image.timestamp + self.alert_timeout

        if obj.alerted_descriptors:
            max_score = np.max(obj.last_observed_descriptor @ np.array(obj.alerted_descriptors).T)

            if max_score > self.match_threshold:
                return None

        obj.alerted_descriptors.append(obj.last_observed_descriptor)
        if len(obj.alerted_descriptors) > self.max_obj_alert_history:
            obj.alerted_descriptors.pop(0)
        crop = crop_image(image.frame, obj.position, margins=[0, 0], bgr_map=True)
        candidate = self.build_alert_candidate(obj.suspected_ts, crops=[crop], extra={"object": obj.name})
        return candidate

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super(MissingObjectAlert, self).build_alert_info(candidate)
        alert_info.alertMessage = f"{candidate.extra['object']} object is missing or moved"
        return alert_info
