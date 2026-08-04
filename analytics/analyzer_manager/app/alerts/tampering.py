import enum
from collections import deque
from typing import Any, List, Tuple, Dict

import cv2
import numpy as np
import scipy

from general.analyzer_general import logger
from general.core import AnalyticImage
from general.img_utils import brightness_score_batch
from .base_alerts import BaseAlert, AlertCandidate, duration_unit_to_sec, duration_unit_to_str


class Location(enum.Enum):
    indoor = 0
    outdoor = 1


class TamperAlert(BaseAlert):
    type_name = "videoTampering"
    alert_message = "Camera tampering"
    object_based_alert = False
    min_history = 72  # minimal size of embedding history
    num_sigmas = 4.0  # number of std deviations for thresholding
    first_stage_cooldown_ms = 3000  # minimal dt between consecutive first-stage detections to run embedding validation

    def __init__(self, init_dict: Dict, context):
        super(TamperAlert, self).__init__(init_dict, context)
        self.set_flow_values()
        outdoor_sensetivity = 0.25
        indoor_sensetivity = 0.5
        if self.location == Location.indoor.value:
            self.sensitivity = indoor_sensetivity
        else:
            self.sensitivity = outdoor_sensetivity
        self.fallback_sensitivity = self.sensitivity
        short_term_timespan_sec = 20
        short_term_queue_size = 10
        long_term_timespan_sec = 30
        long_term_queue_size = 10

        if self.count is not None:
            short_term_timespan_sec = self.count * 2
        self.tamper_detector = TamperingDetection(
            short_term_timespan_sec=short_term_timespan_sec,
            long_term_timespan_sec=long_term_timespan_sec,
            short_term_queue_size=short_term_queue_size,
            long_term_queue_size=long_term_queue_size,
        )
        if self.sensitivity is not None:
            self.tamper_detector.set_sensitivity(self.sensitivity)

        self.alertZoomThumbnail = False  # bypass zoom alerts for now
        self.extra_fields = {"extra_fields": {"sensitivity": self.sensitivity, "embedding_validation": "disabled"}}
        # embedding-based validation is used with an active image quality monitor
        self.embeddings_enabled = False  # a one time switch
        self.trigger_delay_ms = 4000  # maximal delay between actual tampering and embedding-based validation
        self.last_emb_valid = 0  # timestamp of the last embedding validation
        if hasattr(self.context.context, "iq_monitor") and self.context.context.iq_monitor.enabled:
            self.iq_m_ptr = self.context.context.iq_monitor
        else:
            self.iq_m_ptr = None

    def set_flow_values(self):
        units = 0
        self.location = Location.indoor.value  # default
        count = self.count
        if self.formValue:
            self.location = int(self.apply_from_dict("location", self.formValue, Location.indoor.value))
            count = int(self.apply_from_dict("duration", self.formValue, 10))
            units = self.apply_from_dict("durationUnit", self.formValue, 0)
            self.count = count * duration_unit_to_sec[units]
        self.alert_message = f"Tampering for more than {count} {duration_unit_to_str[units]}s"

    def _check_embedding_distance(
        self, frame: np.ndarray, detection_ts: int, mode: str = "cross_dists"
    ) -> Tuple[bool, bool]:
        # run current image through the model to get embedding
        _, curr_emb = self.iq_m_ptr.net_metrics.forward_on_crop_list([frame])
        curr_norm = (curr_emb / np.linalg.norm(curr_emb)).ravel()  # normalize once upfront, flatten to 1-D
        # determine daypart from brightness using the monitor's last known threshold
        day_part = 1  # default to day
        if self.iq_m_ptr.mode_th_history and self.iq_m_ptr.mode_th_history[-1] is not None:
            bright = brightness_score_batch([frame])[0]
            day_part = int(bright > self.iq_m_ptr.mode_th_history[-1])
        bucket = self.iq_m_ptr.embedding_history_day if day_part == 1 else self.iq_m_ptr.embedding_history_night
        if len(bucket) < self.min_history:
            return False, False
        # unpack timestamps and embeddings into numpy arrays in one pass
        all_ts, all_embs = zip(*bucket)
        hist_embs = np.array(all_embs)
        # filter to pre-tampering embeddings using the actual detection timestamp (ms)
        mask = np.array(all_ts) < detection_ts - self.trigger_delay_ms
        hist_embs = hist_embs[mask]
        if len(hist_embs) < self.min_history:
            return False, False
        if mode == "cross_dists":
            dists = scipy.spatial.distance.pdist(hist_embs, metric="cosine")
        elif mode == "consec_dists":
            dists = 1 - np.sum(hist_embs[:-1] * hist_embs[1:], axis=1)
        # distance from current embedding to recent pre-tampering embeddings
        # check -1, -2 and -3 offsets (3 last samples): if even one exceeds the threshold we flag it
        threshold = np.median(dists) + self.num_sigmas * np.std(dists)
        refer_dist = max((1 - hist_embs[-3:] @ curr_norm).tolist())
        is_anomalous = refer_dist > threshold
        if is_anomalous:
            # cross-daypart validation: check if the current embedding is close to the opposite daypart's history
            # if it is, this is likely a lighting/mode transition rather than true tampering
            other_bucket = (
                self.iq_m_ptr.embedding_history_night if day_part == 1 else self.iq_m_ptr.embedding_history_day
            )
            if len(other_bucket) >= self.min_history:
                # this block is meant to suppress rapid but legit shift from bright scene to dark scene (i.e. infra red transition)
                first_normed = np.asarray(other_bucket[0][1], dtype=np.float32).ravel()
                last_normed = np.asarray(other_bucket[-1][1], dtype=np.float32).ravel()
                # compute cosine distance to the earliest and latest embeddings in the opposite daypart
                cross_refer_dist = min(
                    float(1 - np.dot(curr_norm, first_normed)),
                    float(1 - np.dot(curr_norm, last_normed)),
                )
                # use the same threshold — if the current frame looks like the other daypart's baseline, suppress
                if cross_refer_dist < threshold:
                    other_label = "night" if day_part == 1 else "day"
                    logger.info(
                        f"Cross-daypart check suppressed anomaly: cosine distance to {other_label} history "
                        f"{cross_refer_dist:.4f} < threshold {threshold:.4f}. Likely a lighting transition."
                    )
                    is_anomalous = False
        if is_anomalous:
            # add anomalous embedding to the history to avoid repeated alerts for the same tampering event
            (
                self.iq_m_ptr.embedding_history_day.append((detection_ts, curr_norm))
                if day_part == 1
                else self.iq_m_ptr.embedding_history_night.append((detection_ts, curr_norm))
            )
            logger.info(
                f"Embedding anomaly detected (daypart={'day' if day_part else 'night'}): "
                f"cosine distance {refer_dist:.4f} exceeds threshold {threshold:.4f}) "
                "Anomalous embedding added to history to prevent alert fatigue for the same tampering event."
            )
        return True, is_anomalous

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        # enable embedding-based validation after monitor's first accumulation period has passed
        if not self.embeddings_enabled:
            if self.iq_m_ptr and self.iq_m_ptr.current_len > self.iq_m_ptr.accumulation_period:
                self.embeddings_enabled = True
                self.sensitivity = 1.0
                self.tamper_detector.set_sensitivity(self.sensitivity)
                self.extra_fields = {
                    "extra_fields": {"sensitivity": self.sensitivity, "embedding_validation": "enabled"}
                }

        next_ts = self.tamper_detector.next_image_timestamp()
        alert_info = []
        for image in images:
            if next_ts < image.timestamp:
                frame = image.frame
                timestamp = image.timestamp
                is_tampered, tamper_cost = self.tamper_detector.detect_tampering(frame, timestamp)
                if is_tampered:
                    # print(f"First stage tampering detection at {timestamp // 60000}:{timestamp % 60000 // 1000}, timestamp: {timestamp}")  # for debug
                    alert_extra = self.extra_fields
                    if self.embeddings_enabled:
                        if timestamp - self.last_emb_valid < self.first_stage_cooldown_ms:
                            is_tampered = False  # suppress: too soon since last embedding validation
                            continue
                        valid, is_tampered = self._check_embedding_distance(frame, detection_ts=timestamp)
                        if not valid:
                            # temporary fallback: re-evaluate this frame with fallback (less sensitive) threshold, embeddings stay enabled for next check
                            fallback_factor = 0.5 / self.fallback_sensitivity if self.fallback_sensitivity > 0 else 100
                            fallback_th = TamperingDetection.get_thresholds() * fallback_factor
                            is_tampered = any(tamper_cost > fallback_th)
                            alert_extra = {"extra_fields": {"sensitivity": self.fallback_sensitivity}}
                            logger.info(
                                f"Embedding validation unavailable for this check, using fallback sensitivity ({self.fallback_sensitivity}). Tampered: {is_tampered}"
                            )
                            self.tamper_detector.reset()
                        self.last_emb_valid = timestamp
                    if is_tampered:
                        logger.info(f"Tampering alert, cost: {tamper_cost}")
                        alert_info.append(self.build_alert_candidate(timestamp, extra=alert_extra))
                        self.tamper_detector.reset()
                next_ts = self.tamper_detector.next_image_timestamp()
        return alert_info


class TamperingDetection:
    # constant configuration
    use_image_diff: bool = False
    use_chroma_diff: bool = False
    use_filt_grad_dir_histogram: bool = True
    use_image_stat: bool = False
    default_tampering_threshold: List[float] = [4, 4, 0.95, 1.2, 5]
    resize_dim: Tuple[int, int] = (60, 80)
    enabled: bool
    direction_hist_n_bins = 48
    chroma_hist_n_bins = 32

    def __init__(
        self,
        short_term_timespan_sec: int = 20,
        long_term_timespan_sec: int = 120,
        short_term_queue_size: int = 10,
        long_term_queue_size: int = 10,
        tampering_thresholds: List[float] = None,
    ):
        self._short_term_timespan: int = short_term_timespan_sec
        self._long_term_timespan: int = long_term_timespan_sec
        self._short_term_queue_size: int = short_term_queue_size
        self._long_term_queue_size: int = long_term_queue_size

        self._short_term: deque = deque(maxlen=self._short_term_queue_size)
        self._long_term: deque = deque(maxlen=self._long_term_queue_size)
        self.short_delta_t = self._short_term_timespan / self._short_term_queue_size
        self.long_delta_t = self._long_term_timespan / self._long_term_queue_size
        if tampering_thresholds is None:
            tampering_thresholds = TamperingDetection.default_tampering_threshold
        self.tampering_th = TamperingDetection.get_thresholds(tampering_thresholds)

        self.last_timestamp_short = 0
        self.last_timestamp_long = self._short_term_timespan
        self.enabled = False
        self.last_long_similarities: float = 0

    def reset(self):
        self.last_timestamp_short = 0
        self.last_timestamp_long = 0
        self.last_long_similarities = 0
        self._short_term.clear()
        self._long_term.clear()

    def set_sensitivity(self, sensitivity_level: float):
        factor = 0.5 / sensitivity_level if sensitivity_level > 0 else 100
        self.tampering_th = self.tampering_th * factor

    def next_image_timestamp(self):
        return self.last_timestamp_short + self.short_delta_t * 1000

    def detect_tampering(self, image, timestamp) -> Tuple[bool, Any]:
        if self.short_delta_t * 1000 < (timestamp - self.last_timestamp_short):
            resized = TamperingDetection._prepare_image(image)
            descriptor = self.process_image(resized)
            inserted_to_long = self.insert_to_queues(descriptor, timestamp)
            if len(self._long_term) == self._long_term.maxlen:
                desc_long = [elem[0] for elem in self._long_term]
                desc_short = [elem[0] for elem in self._short_term]
                if inserted_to_long:
                    self.last_long_similarities = self.compute_similarities(desc_long)
                sim_between = self.compute_similarities(desc_long, desc_short)
                cost = np.log(sim_between / self.last_long_similarities)
                return any(cost > self.tampering_th), cost  # noqa
        return False, None

    def insert_to_queues(self, descriptor, timestamp) -> bool:
        element = None
        is_inserted_to_long = False
        if len(self._short_term) == self._short_term_queue_size:
            element = self._short_term.popleft()
        self._short_term.append((descriptor, timestamp))
        self.last_timestamp_short = timestamp
        if element is not None:
            if self.long_delta_t * 1000 < (element[1] - self.last_timestamp_long):
                self._long_term.append(element)
                self.last_timestamp_long = element[1]
                is_inserted_to_long = True
        return is_inserted_to_long

    @staticmethod
    def get_thresholds(tampering_thresholds=default_tampering_threshold):
        active_indexes = np.ones(len(tampering_thresholds), dtype=bool)
        if not TamperingDetection.use_image_diff:
            active_indexes[0:2] = 0
        if not TamperingDetection.use_chroma_diff:
            active_indexes[4] = 0

        return np.asarray(tampering_thresholds)[active_indexes]

    @staticmethod
    def _prepare_image(image):
        return cv2.resize(image, TamperingDetection.resize_dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def process_image(image, grad_th: float = 20):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)

        if TamperingDetection.use_image_stat:
            is_ir = np.all(image[:10, :10, 0] == image[:10, :10, 2])
            image_med = np.median(image_gray)
            descriptor = [is_ir, image_med]
        else:
            descriptor = [[], []]

        # image
        if TamperingDetection.use_image_diff:
            image_float = image.astype(float)
            descriptor.append(image_float)
        else:
            descriptor.append([])

        # direction histogram
        n_bins = TamperingDetection.direction_hist_n_bins

        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, dx=1, dy=0)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, dx=0, dy=1)
        grad_dir = np.mod((np.arctan2(sobel_y, sobel_x) + np.pi), np.pi) / np.pi
        hist = np.histogram(grad_dir, n_bins, range=(0, 1))
        descriptor.append(hist[0])

        if TamperingDetection.use_filt_grad_dir_histogram:
            grad_amp = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
            grad_dir[grad_amp < grad_th] = 0
            hist_filt = np.histogram(grad_dir, n_bins, range=(0, 1))
            descriptor.append(hist_filt[0])
        else:
            descriptor.append([])

        # chroma histogram
        if TamperingDetection.use_chroma_diff:
            n_bins = TamperingDetection.chroma_hist_n_bins
            norm_rgb = np.linalg.norm(image_float, ord=1, axis=2)
            norm_red = image[:, :, 2] / norm_rgb
            norm_green = image[:, :, 1] / norm_rgb
            chroma_hist = np.histogram2d(
                norm_red.flatten(),
                norm_green.flatten(),
                bins=n_bins,
                range=np.array([[0, 1], [0, 1]]),
            )
            descriptor.append(chroma_hist[0])
        else:
            descriptor.append([])

        return descriptor

    @staticmethod
    def get_descriptor_criteria() -> List[str]:
        criteria = []
        if TamperingDetection.use_image_diff:
            criteria = criteria + ["L1 Image Diff", "L2 Image Diff"]
        criteria.append("Grad dir L1")
        criteria.append("Grad dir EMD")
        if TamperingDetection.use_filt_grad_dir_histogram:
            criteria.append("Filtered Grad dir L1")
            criteria.append("Filtered Grad dir EMD")
        if TamperingDetection.use_chroma_diff:
            criteria.append("Chroma L1")
        return criteria

    @staticmethod
    def compare_descriptors(desc1, desc2):
        cost = []
        if TamperingDetection.use_image_stat:
            # currently not used but may be required in the future
            is_ir_1, is_ir_2 = desc1[0], desc2[0]  # noqa
            im_med_1, im_med_2 = desc1[1], desc2[1]  # noqa
        feature_idx = 2
        if TamperingDetection.use_image_diff:
            # image diff L1, L2
            diffs_image = np.linalg.norm(desc1[feature_idx] - desc2[feature_idx], ord=1, axis=2)
            cost.append(np.linalg.norm(diffs_image, ord=1))
            cost.append(np.linalg.norm(diffs_image, ord=2))

        feature_idx = 3
        # gradient direction histogram L1 diff
        grad_dir_l1 = np.linalg.norm(desc1[feature_idx] - desc2[feature_idx], ord=1)
        # earth moving distance
        grad_dir_emv = scipy.stats.wasserstein_distance(desc1[feature_idx], desc2[feature_idx])

        feature_idx = 4
        if TamperingDetection.use_filt_grad_dir_histogram:
            # gradient direction histogram L1 diff
            grad_dir_l1_filt = np.linalg.norm(desc1[feature_idx] - desc2[feature_idx], ord=1)
            grad_dir_l1 = min(grad_dir_l1, grad_dir_l1_filt)
            # earth moving distance
            grad_dir_emv_filt = scipy.stats.wasserstein_distance(desc1[feature_idx], desc2[feature_idx])
            grad_dir_emv = min(grad_dir_emv, grad_dir_emv_filt)

        cost.append(grad_dir_l1)
        cost.append(grad_dir_emv)

        feature_idx = 5
        if TamperingDetection.use_chroma_diff:
            # chroma histogram L1 diff
            cost.append(np.linalg.norm(desc1[feature_idx] - desc2[feature_idx], ord=1))

        return cost

    @staticmethod
    def compute_similarities(desc_list1, desc_list2=None) -> float:
        if desc_list2 is None:
            similarities_1 = []
            for i in range(len(desc_list1)):
                for j in range(i + 1, len(desc_list1)):
                    desc1 = desc_list1[i]
                    desc2 = desc_list1[j]
                    similarities_1.append(TamperingDetection.compare_descriptors(desc1, desc2))
            return np.median(similarities_1, axis=0)
        else:
            similarities_1_to_2 = []
            for desc1 in desc_list1:
                for desc2 in desc_list2:
                    similarities_1_to_2.append(TamperingDetection.compare_descriptors(desc1, desc2))
            return np.median(similarities_1_to_2, axis=0)
