from collections import deque
from copy import copy
from typing import List, Tuple, Optional

import cv2
import numpy as np

from alerts.base_alerts import AlertCandidate, BaseAlert
from general.core import AlertRouting, BDR
from general.img_utils import crop_image_by_bbox, fix_bbox_to_ar
from level1.violence.violence_models import ViolenceDetector, ViolenceClassifier

VALIDATION_CROP_SIZE = 384
VALIDATION_MOSAIC_DIM = 3
DETECTION_QUEUE_SIZE = 15


class FightingAlert(BaseAlert):
    """
    Violence detection alert using YOLO-based trigger + MobilenetLSTM V2 classifier.

    Pipeline:
        1. Early exit guards (cooldown, person count < 2, same entities still present)
        2. YOLO violence detector on grayscale triplet frames -> bounding boxes
        3. When enough detections accumulate in a sliding window ->
           crop frames to union bbox of detections -> MobilenetLSTM classifier
        4. Three-tier decision: hit / marginal-retry / miss
    """

    default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE
    violence_detection_ts = -10000000
    violence_timeout = 60000  # 1 minute
    conf_to_latency_conversion = {0: 1, 1: 1, 2: 1}
    min_detections_to_trigger = 2  # minimum frames with detector hits in the window
    sampling_time = 1000  # time in ms to sample frames for detector trigger (e.g., every 1 second)
    classifier_crop_margin = [0, 0]  # pixels to expand around union bbox for classifier input
    detector_bz = 2
    alert_message = "Violence detected"

    def __init__(self, alert_dict: dict, context):
        super(FightingAlert, self).__init__(alert_dict, context)
        self.in_triggered_state = False
        self.violent_ent_ids = set()
        self.last_violent_ents = set()

        analytic_config = self.context.get_config()
        vd_config = analytic_config.get("l1_models", {}).get("attributes", {}).get("violence", {})

        # Override class defaults from analyticConfig
        self.min_detections_to_trigger = vd_config.get("min_detections_to_trigger", self.min_detections_to_trigger)
        self.sampling_time = vd_config.get("sampling_time", self.sampling_time)
        self.violence_timeout = vd_config.get("violence_timeout", self.violence_timeout)
        self.classifier_crop_margin = vd_config.get("classifier_crop_margin", self.classifier_crop_margin)

        self.last_triplet_ts = -self.sampling_time  # timestamp of the last frame of the most recent triplet
        self.collecting_triplet = False  # whether we are currently accumulating frames for a triplet
        self.triplet_fill_idx = 0
        self.triplet_count = 0  # how many triplets collected in current detector batch
        self.positive_detections = 0
        self.crop_bbox = None

        # Stage 1: violence detector trigger
        yolo_config = vd_config.get("yolo", {})
        self.yolo_detector = ViolenceDetector(yolo_config)
        self.detector_buf = np.zeros(
            (self.detector_bz, 3, *self.yolo_detector.args.im_size), dtype=self.yolo_detector.data_type
        )

        # Stage 2: MobilenetLSTM classifier (crop-based, ImageNet-normalized)
        classifier_config = vd_config.get("classifier", {})
        self.classifier = ViolenceClassifier(classifier_config)
        self.sequence_len = self.classifier.sequence_length
        buffer_size = self.classifier.inference_buffer_size
        self.channels = buffer_size[1]
        self.mem = np.zeros(buffer_size, dtype=self.classifier.data_type)
        self.frame_idx = 0

        self.detector_triggeres_imgs = np.zeros(
            (self.min_detections_to_trigger, 3, *self.yolo_detector.args.im_size), dtype=self.yolo_detector.data_type
        )  # for debug data, store the frames that triggered the detector
        self.detection_queue = deque(maxlen=DETECTION_QUEUE_SIZE)

        # Validation mosaic setup: 3x3 grid of 384x384 crops
        self.validation_crop_size = [VALIDATION_CROP_SIZE, VALIDATION_CROP_SIZE]
        mosaic_sz = [VALIDATION_MOSAIC_DIM, VALIDATION_MOSAIC_DIM]
        validation_num_samples = VALIDATION_MOSAIC_DIM * VALIDATION_MOSAIC_DIM
        full_sz = VALIDATION_CROP_SIZE * VALIDATION_MOSAIC_DIM
        self.validation_mem = np.zeros((full_sz, full_sz, 3), dtype=np.uint8)
        self.validation_mosaic = mosaic_sz

        start_frame = self.sequence_len // validation_num_samples
        end_frame = self.sequence_len * (validation_num_samples - 1) // validation_num_samples
        self.validation_idxs = np.linspace(start_frame, end_frame, num=validation_num_samples, dtype=int).tolist()

    def _person_count_per_idx(self, batch_data: BDR):
        person_count_per_frame = batch_data.query(BDR.CLASS, self.person_value, BDR.FRAME_ID).astype(int)
        return np.bincount(person_count_per_frame, minlength=batch_data.batch_size)

    def _run_detector_batch(self, orig_h: int, orig_w: int):
        """Run detector on the full detector buffer and process detections."""
        batch = [self.detector_buf[i] for i in range(self.triplet_count)]
        raw = self.yolo_detector.infer(batch)
        batch_dets = self.yolo_detector.post_infer(raw, None)
        model_h, model_w = self.yolo_detector.args.im_size
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        for i, dets in enumerate(batch_dets):
            if len(dets) > 0:
                self.positive_detections += 1
                dets[:, [0, 2]] *= scale_x
                dets[:, [1, 3]] *= scale_y
                self.detection_queue.append(dets)
                if self.positive_detections <= self.min_detections_to_trigger:
                    self.detector_triggeres_imgs[self.positive_detections - 1] = self.detector_buf[i]
        self.triplet_count = 0

    @staticmethod
    def _get_union_bbox(detection_queue: deque) -> Optional[Tuple[float, float, float, float]]:
        """Compute union bbox across all detections in the queue."""
        all_bboxes = []
        for dets in detection_queue:
            if len(dets) > 0:
                all_bboxes.append(dets[:, :4])
        if not all_bboxes:
            return None
        stacked = np.concatenate(all_bboxes, axis=0)
        return (
            float(stacked[:, 0].min()),
            float(stacked[:, 1].min()),
            float(stacked[:, 2].max()),
            float(stacked[:, 3].max()),
        )

    def _compute_crop_bbox(self, orig_h, orig_w) -> Optional[Tuple[int, int, int, int]]:
        """Get the expanded union bbox for cropping classifier input frames."""
        union = self._get_union_bbox(self.detection_queue)
        if union is None:
            return None
        target_h, target_w = self.classifier.args.im_size[-2:]
        ar_wh = target_w / target_h
        result = fix_bbox_to_ar(union, ar_wh, (orig_w, orig_h), margins=[0, 0], min_size=(target_w, target_h))
        return tuple(result.tolist())

    def _get_entities_in_bbox(self, batch_data: BDR, crop_bbox, orig_h: int, orig_w: int) -> set:
        """Return person entity IDs whose bbox overlaps with the crop region."""
        if crop_bbox is None:
            return set(batch_data.unique(BDR.ID, BDR.CLASS, self.person_value).astype(int).tolist()) - {-1}
        norm_crop = np.array(
            [
                crop_bbox[0] / orig_w,
                crop_bbox[1] / orig_h,
                crop_bbox[2] / orig_w,
                crop_bbox[3] / orig_h,
            ]
        )
        ids = set()
        person_data = batch_data.query(BDR.CLASS, self.person_value, retrieve=BDR.POS + [BDR.ID])
        for row in person_data:
            ent_bbox, ent_id = row[:4], int(row[4])
            if ent_id == -1:
                continue
            if (
                ent_bbox[0] < norm_crop[2]
                and ent_bbox[2] > norm_crop[0]
                and ent_bbox[1] < norm_crop[3]
                and ent_bbox[3] > norm_crop[1]
            ):
                ids.add(ent_id)
        return ids

    def is_active_batch(self, images, motion_data, batch_data) -> List[AlertCandidate]:
        last_ts = images[-1].timestamp
        alert_candidates = []

        # clean detection queue if not enough detections left to trigger
        left_slots = self.sequence_len - self.frame_idx
        if left_slots < self.min_detections_to_trigger - self.positive_detections and not self.in_triggered_state:
            self.reset_trigger()

        if not self.in_triggered_state:
            # early exit guards (checked once per batch, before iterating frames)
            if not self.collecting_triplet:
                early_exit = False
                if last_ts - self.violence_detection_ts < self.violence_timeout:
                    early_exit = True
                elif np.all(self._person_count_per_idx(batch_data) < 2):
                    early_exit = True
                else:
                    batch_ids = set(batch_data.unique(BDR.ID, BDR.CLASS, self.person_value).astype(int).tolist())
                    if len(self.last_violent_ents.intersection(batch_ids)) > 0:
                        early_exit = True
                if early_exit:
                    self.reset_trigger()
                    return alert_candidates

            # collect triplets spaced by sampling_time, batch-infer when detector_bz are ready
            for image in images:
                if not self.collecting_triplet:
                    if image.timestamp - self.last_triplet_ts >= self.sampling_time:
                        self.collecting_triplet = True
                        self.triplet_fill_idx = 0
                    else:
                        continue
                self.detector_buf[self.triplet_count, self.triplet_fill_idx] = (
                    self.yolo_detector.transform_single_image(image.frame)
                )
                self.triplet_fill_idx += 1
                if self.triplet_fill_idx == 3:
                    self.last_triplet_ts = image.timestamp
                    self.collecting_triplet = False
                    self.triplet_count += 1
                    if self.triplet_count == self.detector_bz:
                        self._run_detector_batch(*images[0].frame.shape[:2])
                        break

        # if minimal detections condition met, start accumulating frames for classifier input
        if self.positive_detections >= self.min_detections_to_trigger:
            # extract union bboxes once
            if not self.in_triggered_state:
                self.crop_bbox = (
                    list(self._compute_crop_bbox(images[0].frame.shape[0], images[0].frame.shape[1]))
                    if self.crop_bbox is None
                    else self.crop_bbox
                )

                self.in_triggered_state = True
                orig_h, orig_w = images[0].frame.shape[:2]
                self.violent_ent_ids.update(self._get_entities_in_bbox(batch_data, self.crop_bbox, orig_h, orig_w))
            # build sequence from all available frames
            for image in images:
                t, c = divmod(self.frame_idx, self.channels)
                frame_cropped = (
                    crop_image_by_bbox(image.frame, self.crop_bbox, margins=self.classifier_crop_margin)
                    if self.crop_bbox is not None
                    else image.frame
                )
                self.classifier.transform_single_image(frame_cropped, self.mem[t, c], channel_idx=c)
                self._add_to_mosaic(frame_cropped, self.frame_idx)
                self.frame_idx += 1
                if self.frame_idx == self.sequence_len:
                    # enough frames collected for classifier
                    score = self.classifier.infer([self.mem])
                    violence = self.classifier.post_infer(score, None)
                    if violence:
                        val_image = self.validation_mem.copy()
                        cand = self.build_alert_candidate(
                            images[0].timestamp, crops=[images[0].frame], validation_images=[val_image]
                        )
                        if self.activate_ddata and self.internal_alert:
                            cand = self._add_debug_data(cand, score[0])
                        alert_candidates.append(cand)
                        self.violence_detection_ts = image.timestamp
                        self.last_violent_ents = copy(self.violent_ent_ids)
                    self.reset_trigger()
                    break

        return alert_candidates

    def _add_debug_data(self, candidate: AlertCandidate, score: float) -> AlertCandidate:
        candidate.extra = candidate.extra or {}
        debug_data = {
            "alert_id": candidate.alert_id,
            "timestamp": candidate.timestamp,
            "detector_triggered_images": self.detector_triggeres_imgs.copy(),
            "detector_detections": list(self.detection_queue),
            "crop_bbox": self.crop_bbox,
            "classifier_input_seq": self.mem.copy(),
            "violence_score": score[0],
            "validation_images": candidate.validation_images,
        }
        candidate.extra["debug_data"] = self.serialize_debug_data(debug_data)
        return candidate

    def reset_trigger(self):
        self.in_triggered_state = False
        self.frame_idx = 0
        self.detection_queue.clear()
        self.positive_detections = 0
        self.triplet_count = 0
        self.crop_bbox = None
        self.violent_ent_ids.clear()

    def _add_to_mosaic(self, img, frame_idx):
        if frame_idx in self.validation_idxs:
            t, c = divmod(self.validation_idxs.index(frame_idx), self.validation_mosaic[1])
            h, w = self.validation_crop_size
            y0 = t * h
            y1 = (t + 1) * h
            x0 = c * w
            x1 = (c + 1) * w
            self.validation_mem[y0:y1, x0:x1] = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

