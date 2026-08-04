import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Empty
from typing import Dict, List

import cv2
import numpy as np
from future.backports.test.ssl_servers import threading
from pydantic import BaseModel

from detection.detector import DetectorFactory, DetectorType
from general.analyzer_general import log_exception, logger
# from utils.infra.cloud_logger import logger
from general.core import BDR, calculate_overlap_matrix
from general.image_encoder import ImageEncoder
from general.img_utils import crop_image_by_bbox, crop_with_minimal, is_cv2_cuda_enabled
from general.offline_analytics import decode_image_from_base64, encode_image_to_base64
from general.proj import load_configAnalytic
from level1.weapons_classifier.weapon_analyzer import WCResnet18Classifier
from offline_common import BaseBackgroundWorker, detection_to_expert_results

# defaults
OVERLAP_THRESHOLD = 0.1
RECORD_OVERLAP_HIGH_THRESHOLD = 0.7
RECORD_OVERLAP_LOW_THRESHOLD = 0.1
RECORD_ENCODING_SIM_THRESHOLD = 0.75
RECORD_EXPIRATION_TIME_MS = 24 * 60 * 60 * 1000  # 1 day
MIN_VALIDATION_CROP = (512, 512)
MIN_WEAPON_CONFIDENCE = {0.1: 0.15, 0.2: 0.22, 1: 0.35}  # dynamic based on weapon size

NUM_PROCESS_WORKERS = 6
MAX_RECORDS_NUM = 100


@dataclass
class WeaponConfig:
    enable: bool
    overlap_threshold: float
    record_overlap_high_threshold: float
    record_overlap_low_threshold: float
    record_encoding_sim_threshold: float
    record_expiration_time_ms: int
    min_validation_crop: tuple
    min_weapon_confidence: dict
    global_min_conf: float = field(init=False)
    _sorted_keys: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.global_min_conf = min(self.min_weapon_confidence.values())
        self._sorted_keys = sorted(self.min_weapon_confidence.keys())

    def get_confidence_threshold(self, weapon_size: float) -> float:
        for key in self._sorted_keys:
            if weapon_size <= key:
                return self.min_weapon_confidence[key]
        return self.min_weapon_confidence[self._sorted_keys[-1]]


@dataclass
class DetectionRecord:
    timestamp: int
    position: np.ndarray
    encoding: np.ndarray


class WeaponDetection:
    def __init__(self, camera_id: str, config: WeaponConfig):
        self.camera_id = camera_id
        self.active_alerts_data = []
        self.detector = DetectorFactory().create(DetectorType.EXPERT)
        self.classifier = WCResnet18Classifier({})
        self.encoder = ImageEncoder({})
        logger.info(
            f"initialized Expert, classifier and encoder with local {self.detector.is_local, self.classifier.is_local, self.encoder.is_local}"
        )
        class_handler = self.detector.class_handler
        self.weapons_subcls = class_handler.get_object_classes(class_handler.weapon_value)  # only gun class
        self.required_margins = [self.classifier.required_margins] * 2
        self.known_detections = []
        self.config = deepcopy(config)
        self.lock = threading.Lock()
        self.gpu_mem = None
        self.resized_mem = None
        self.detector_img_size = tuple(self.detector.args.im_size)
        self.detector_dtype = self.detector.data_type
        if os.environ.get("USE_CUDA", True) and is_cv2_cuda_enabled():
            self.detector.prepare_crops = self.preproc_cuda
            self.gpu_mem = None
            self.resized_mem = cv2.cuda_GpuMat()
            self.resized_mem.upload(np.zeros(self.detector_img_size + (3,), dtype=np.uint8))

    def preproc_cuda(self, crop_list):
        crops = []
        interp_mode = cv2.INTER_AREA if self.detector.args.antialias else cv2.INTER_LINEAR
        for element in crop_list:

            # Upload image to GPU
            if self.gpu_mem is None:
                self.gpu_mem = cv2.cuda_GpuMat(tuple(element.shape[:2]), cv2.CV_8UC3)

            self.gpu_mem.upload(element)
            cv2.cuda.resize(
                self.gpu_mem,
                (self.detector_img_size[1], self.detector_img_size[0]),
                interpolation=interp_mode,
                dst=self.resized_mem,
            )

            # Convert BGR to RGB in-place (on the same GPU memory)
            cv2.cuda.cvtColor(self.resized_mem, cv2.COLOR_BGR2RGB, dst=self.resized_mem)

            transformed_img = self.resized_mem.download()
            crops.append(np.transpose(transformed_img, (2, 0, 1)).astype(self.detector_dtype))
        return crops

    def register_detection(self, timestamp: int, position: np.ndarray, crop: np.ndarray) -> bool:
        is_existing = False
        check_encodings = False
        self.known_detections = sorted(
            [det for det in self.known_detections if timestamp - det.timestamp < self.config.record_expiration_time_ms],
            key=lambda det: det.timestamp,
            reverse=True,
        )[:MAX_RECORDS_NUM]
        exist_idx = -1
        encodings = None
        if self.known_detections:
            # first check overlap with existing detections
            overlaps = calculate_overlap_matrix(
                np.atleast_2d(position), np.array([d.position for d in self.known_detections])
            )
            if np.any(overlaps > self.config.record_overlap_high_threshold):
                is_existing = True
                exist_idx = np.argmax(overlaps)
            elif np.all(overlaps < self.config.record_overlap_low_threshold):
                is_existing = False
            else:
                check_encodings = True
        if not is_existing:
            encodings = self.encoder.forward_on_crop_list([crop])[0]
            if check_encodings:
                existing_encodings = np.array([d.encoding for d in self.known_detections])
                similarities = np.dot(existing_encodings, encodings.T)
                if np.any(similarities > self.config.record_encoding_sim_threshold):
                    is_existing = True
                    exist_idx = np.argmax(similarities)
            if not is_existing:
                logger.info(f"Camera {self.camera_id} detected weapon at {position} is a new detection")
                detection = DetectionRecord(timestamp, position, encodings)
                self.known_detections.append(detection)
        if is_existing:  # update existing detection
            logger.debug(f"Camera {self.camera_id} detected weapon at {position} with existing detection")
            if exist_idx >= 0:
                self.known_detections[exist_idx].timestamp = timestamp
                if encodings is not None:
                    new_enc = self.known_detections[exist_idx].encoding + encodings
                    self.known_detections[exist_idx].encoding = new_enc / np.linalg.norm(new_enc)
                    self.known_detections[exist_idx].position = position
        return is_existing


class WeaponPayload(BaseModel):
    camera_id: str  # id of the camera that the analysis was made for
    image: str  # Base64-encoded image
    batch_data: List[List[float]]  # 2D list representing the NumPy array
    timestamp: int  # unix timestamp in milliseconds


def weapon_task(
    payload: WeaponPayload,
    camera_states: Dict[str, WeaponDetection],
    out_queue: mp.Queue,
    expert_result_queue: mp.Queue,
):

    try:
        camera_id = payload.camera_id
        stt = camera_states[camera_id]
        if not stt.config.enable:
            return
        image = decode_image_from_base64(payload.image)
        batch_data = np.atleast_2d(np.array(payload.batch_data))
        detections = stt.detector.forward_on_crop_list([image])[0]
        try:
            w, h = image.shape[1], image.shape[0]
            result = detection_to_expert_results(detections, w, h, payload.camera_id, payload.timestamp)
            expert_result_queue.put(result)
            # logger.debug(f"Sent expert results from weapon task: {result}")
        except Exception as e:
            log_exception(logger, "Error sending expert results from weapon task", e)

        weapons = []
        h, w = image.shape[:2]
        if len(detections) > 0:
            weapons = [d for d in detections if d[5] in stt.weapons_subcls and d[4] >= stt.config.global_min_conf]

        crops = []
        for wep in weapons:
            wep_sz = max((wep[2] - wep[0]) / w, (wep[3] - wep[1]) / h)
            if wep[4] > stt.config.get_confidence_threshold(wep_sz):
                crop = crop_image_by_bbox(image, wep[:4], margins=stt.required_margins)
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if len(crops) > 0:
            logger.debug(f"Camera {camera_id} detected {len(weapons)}")
            logger.debug(f"weapons: {weapons}")
            results_cls, results_scores = stt.classifier.forward_on_crop_list(crops)
            classified = [weapons[i] for i, c in enumerate(results_cls) if c == 0]  # gun detected and classified!
            if classified:
                logger.debug(f"Camera {camera_id} {len(classified)} candidates were classified as weapon")
                weapons_normed = np.array(classified)[:, :4] / np.array([w, h, w, h])
                persons = batch_data[batch_data[:, BDR.CLASS] == 0]
                overlaps = calculate_overlap_matrix(weapons_normed, persons[:, BDR.POS])
                for threat, pos in enumerate(weapons_normed):
                    max_index = np.argmax(overlaps[threat])
                    person_id = -1
                    zoom_crop = crops[threat]
                    weapon_location = pos.tolist()
                    validation_roi = None
                    with stt.lock:
                        is_registered = stt.register_detection(payload.timestamp, pos, zoom_crop)
                    if not is_registered:  # avoid sending on the same weapon multiple times
                        if overlaps[threat][max_index] > stt.config.overlap_threshold:
                            person_id = int(persons[max_index, BDR.ID])
                            joint_roi = np.vstack([pos, persons[max_index, BDR.POS]])
                            validation_roi = np.concatenate(
                                (np.min(joint_roi[:, :2], axis=0), np.max(joint_roi[:, 2:], axis=0))
                            )
                            validation_crop = crop_with_minimal(image, validation_roi, stt.config.min_validation_crop)
                        else:
                            validation_crop = crop_with_minimal(image, pos, stt.config.min_validation_crop)
                        alert = {
                            "camera_id": payload.camera_id,
                            "timestamp": payload.timestamp,
                            "person_id": person_id,
                            "weapon_location": weapon_location,
                            "validation_crop": encode_image_to_base64(validation_crop),
                            "zoom_crop": encode_image_to_base64(cv2.cvtColor(zoom_crop, cv2.COLOR_RGB2BGR)),
                            "perimeter": validation_roi.tolist() if validation_roi is not None else weapon_location,
                        }
                        logger.info(f"Camera {camera_id} detected weapon at {weapon_location} with person {person_id}")
                        out_queue.put_nowait(alert)
    except Exception as e:
        log_exception(logger, "Error in weapon task", e)


def get_analytic_config() -> WeaponConfig:
    analytic_config = load_configAnalytic() or {}
    config = analytic_config.get("offline_analytics", {}).get("weapon_detection", {})
    min_conf_value = config.get("min_weapon_confidence", MIN_WEAPON_CONFIDENCE)
    if not isinstance(min_conf_value, dict):
        min_conf = {0.1: min_conf_value, 0.2: min_conf_value + 0.1, 1: min_conf_value + 0.3}
    else:
        min_conf = min_conf_value

    return WeaponConfig(
        enable=config.get("enable", True),
        overlap_threshold=config.get("overlap_threshold", OVERLAP_THRESHOLD),
        record_overlap_high_threshold=config.get("record_overlap_high_threshold", RECORD_OVERLAP_HIGH_THRESHOLD),
        record_overlap_low_threshold=config.get("record_overlap_low_threshold", RECORD_OVERLAP_LOW_THRESHOLD),
        record_encoding_sim_threshold=config.get("record_encoding_sim_threshold", RECORD_ENCODING_SIM_THRESHOLD),
        record_expiration_time_ms=config.get("record_expiration_time_ms", RECORD_EXPIRATION_TIME_MS),
        min_validation_crop=config.get("min_validation_crop", MIN_VALIDATION_CROP),
        min_weapon_confidence=min_conf,
    )


class WcBackgroundWorker(BaseBackgroundWorker):
    def __init__(self, in_queue: mp.Queue, out_queue: mp.Queue, expert_result_queue: mp.Queue):
        super(WcBackgroundWorker, self).__init__(in_queue, out_queue)

        self.camera_states: Dict[str, WeaponDetection] = {}
        self.config: WeaponConfig = get_analytic_config()
        self.expert_result_queue = expert_result_queue  # for posting expert detection results

    def run(self):
        logger.info("weapon Worker started")
        with ThreadPoolExecutor(max_workers=NUM_PROCESS_WORKERS) as executor:  # one for 2 cameras
            while True:
                try:
                    data: WeaponPayload = self.in_queue.get(timeout=1)  # Wait up to 1 second
                    if data is None:
                        break
                    if data.camera_id not in self.camera_states:
                        self.camera_states[data.camera_id] = WeaponDetection(data.camera_id, self.config)
                        logger.info(f"Camera {data.camera_id} initialized for weapon detection.")
                    executor.submit(weapon_task, data, self.camera_states, self.out_queue, self.expert_result_queue)
                    q_size = self.in_queue.qsize()
                    if q_size > NUM_PROCESS_WORKERS:
                        logger.warning(f"Camera {data.camera_id} buffer is full - emptying the queue")

                        # here we empty the queue safely
                        i = 0
                        while i < q_size:
                            try:
                                self.in_queue.get_nowait()
                            except Empty:
                                break
                            i += 1
                except Empty:
                    pass
                except Exception as e:
                    log_exception(logger, "Error in weapon worker", e)
                    pass
