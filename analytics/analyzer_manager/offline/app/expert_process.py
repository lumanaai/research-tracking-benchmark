import concurrent.futures
import logging
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Full
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from cython_bbox import bbox_overlaps as bbox_ious  # noqa

from detection.detector import DetectorFactory, DetectorType
from general.analyzer_general import log_exception
from general.core import BDR, calculate_overlap_matrix, ClassHandler, EndpointFactory
from general.image_encoder import ImageEncoder
from general.img_utils import crop_image_by_bbox, crop_image
from general.offline_analytics import encode_image_to_base64
from general.proj import load_configAnalytic
from general.triton_utils import is_triton_available
from level1.weapons_classifier.weapon_analyzer import WCResnet18Classifier
from offline_mq_common import OfflineAnalyzerWorker, MessageData

# defaults
OVERLAP_THRESHOLD = 0.1
RECORD_OVERLAP_HIGH_THRESHOLD = 0.7
RECORD_OVERLAP_LOW_THRESHOLD = 0.1
RECORD_ENCODING_SIM_THRESHOLD = 0.75
RECORD_EXPIRATION_TIME_MS = 24 * 60 * 60 * 1000  # 1 day
MIN_VALIDATION_CROP = (512, 512)
MIN_WEAPON_CONFIDENCE = {0.1: 0.275, 0.25: 0.17, 1: 0.2}  # dynamic based on weapon size
LEGACY_MIN_WEAPON_CONFIDENCE = {0.1: 0.10, 0.2: 0.17, 1: 0.3}
BRANDISHING_THRESHOLD = 0.05
CONFIDENCE_FACTORS = {0: 1, 1: 1.5, 2: 2}

NUM_PROCESS_WORKERS = os.environ.get("NUM_PROCESS_WORKERS", 4)
MAX_RECORDS_NUM = 100
HAND_CLASS_NAME = "hand"


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
    offline_resize: bool = False
    brandishing_threshold: float = 0.05
    confidence_factor: Dict[int, float] = field(default_factory=lambda: CONFIDENCE_FACTORS)
    global_min_conf: float = field(init=False)
    _sorted_keys: List[float] = field(init=False, default_factory=list)
    legacy_mode: bool = False

    def __post_init__(self):
        self.global_min_conf = min(self.min_weapon_confidence.values()) - 0.05
        self._sorted_keys = sorted(self.min_weapon_confidence.keys())

    def get_confidence_threshold(self, weapon_size: float) -> float:
        for key in self._sorted_keys:
            if weapon_size <= key:
                return self.min_weapon_confidence[key]
        return self.min_weapon_confidence[self._sorted_keys[-1]]

    def get_confidence_factor(self, confidence: Optional[int]) -> float:
        return 1 if self.legacy_mode else self.confidence_factor.get(confidence, self.confidence_factor.get(2))


@dataclass
class DetectionRecord:
    timestamp: int
    position: np.ndarray
    encoding: np.ndarray
    is_brandished: bool


def detection_to_expert_results(detections: np.ndarray, w: int, h: int, camera_id: str, timestamp: int) -> Dict:
    """Normalize detections and prepare results dict."""
    norm_detections = detections.astype(np.float32)
    norm_detections[:, 0] /= w
    norm_detections[:, 2] /= w
    norm_detections[:, 1] /= h
    norm_detections[:, 3] /= h
    return {
        "cameraId": camera_id,
        "timestamp": timestamp,
        "detections": norm_detections.tolist(),
    }


@dataclass
class DetectorMetadata:
    """Static metadata derived from a detector, computed once and shared."""

    weapons_subcls: list
    local_hands_idx: int
    required_margins: list
    detector_img_size: tuple
    detector_dtype: object
    person_idx: int
    remote_hands_idx: int


# ---------------------------------------------------------------------------
# Model sets: stateless Triton client stubs, one per pool thread (no sharing).
# Overhead per instance: gRPC channel + numpy input buffer (~KB, no GPU duplication).
# ---------------------------------------------------------------------------


class ExpertModelSet:
    """Stateless Triton client stubs for expert detection. One instance per thread."""

    _MAX_INIT_RETRIES = 3
    _RETRY_DELAY = 1.0  # seconds

    def __init__(self, detector_type: str, logger: logging.Logger):
        # Pre-check Triton availability with a lightweight probe before constructing
        # the full detector. This avoids building a local TRT engine (expensive, noisy
        # GPU errors) on each failed attempt.
        factory = DetectorFactory()
        args = factory.create_config(detector_type)
        endpoint_info = EndpointFactory().create(args.endpoint_model)

        for attempt in range(self._MAX_INIT_RETRIES):
            t_attempt = time.time()
            logger.info(
                f"ExpertModelSet: attempt {attempt + 1}/{self._MAX_INIT_RETRIES} "
                f"for {args.endpoint_model} starting at epoch t={t_attempt:.2f}"
            )
            available = is_triton_available(endpoint_info)
            logger.info(
                f"ExpertModelSet: attempt {attempt + 1}/{self._MAX_INIT_RETRIES} "
                f"for {args.endpoint_model} finished after {time.time() - t_attempt:.2f}s, available={available}"
            )
            if available:
                break
            if attempt < self._MAX_INIT_RETRIES - 1:
                logger.warning(
                    f"ExpertModelSet: Triton not ready for {args.endpoint_model} "
                    f"(attempt {attempt + 1}/{self._MAX_INIT_RETRIES}), retrying in {self._RETRY_DELAY}s..."
                )
                time.sleep(self._RETRY_DELAY)
        else:
            raise RuntimeError(
                f"ExpertModelSet: Triton unavailable for {args.endpoint_model} after "
                f"{self._MAX_INIT_RETRIES} attempts. Triton must be available for offline workers."
            )

        # Triton is confirmed available — create detector (will use remote path)
        logger.info(f"ExpertModelSet: building detector client for {args.endpoint_model} (factory.create)")
        t_create = time.time()
        self.detector = factory.create(detector_type)
        logger.info(
            f"ExpertModelSet: factory.create for {args.endpoint_model} returned after "
            f"{time.time() - t_create:.2f}s, is_local={self.detector.is_local}"
        )
        if self.detector.is_local:
            # Should not happen since we confirmed availability, but guard anyway
            raise RuntimeError(
                f"ExpertModelSet: detector fell back to local TRT despite Triton being available. "
                f"This indicates a race condition."
            )
        logger.info(f"ExpertModelSet initialized: detector local={self.detector.is_local}")


class WeaponModelSet(ExpertModelSet):
    """Stateless Triton stubs: detector + classifier + encoder. One instance per thread."""

    def __init__(self, detector_type: str, logger: logging.Logger):
        super().__init__(detector_type, logger)
        logger.info("WeaponModelSet: building classifier client (wc)")
        t_cls = time.time()
        self.classifier = WCResnet18Classifier({})
        logger.info(
            f"WeaponModelSet: classifier client built after {time.time() - t_cls:.2f}s, "
            f"is_local={self.classifier.is_local}"
        )
        logger.info("WeaponModelSet: building encoder client (enc)")
        t_enc = time.time()
        self.encoder = ImageEncoder({})
        logger.info(
            f"WeaponModelSet: encoder client built after {time.time() - t_enc:.2f}s, "
            f"is_local={self.encoder.is_local}"
        )
        # Validate all models are using Triton (not local TRT)
        for name, model in [("classifier", self.classifier), ("encoder", self.encoder)]:
            if model.is_local:
                raise RuntimeError(
                    f"WeaponModelSet: {name} fell back to local TRT instead of Triton. "
                    f"Triton must be available for offline workers."
                )
        logger.info(
            f"WeaponModelSet initialized: detector={self.detector.is_local}, "
            f"classifier={self.classifier.is_local}, encoder={self.encoder.is_local}"
        )


# ---------------------------------------------------------------------------
# Camera state: only detection history (shared across threads, lock-protected).
# No model references — completely decoupled from inference.
# ---------------------------------------------------------------------------


class CameraState:
    """Lightweight per-camera state for expert detection (no models)."""

    def __init__(
        self, camera_id: str, config: Union[dict, WeaponConfig], storage_path: str, logger: logging.Logger = None
    ):
        self.camera_id = camera_id
        self.logger = logger or logging.getLogger(__name__)
        self.storage_path = storage_path or "/dev/shm/expert_offline"
        self.config = deepcopy(config)


class WeaponCameraState(CameraState):
    """Per-camera weapon detection history. Lock only protects the history list."""

    def __init__(
        self,
        camera_id: str,
        config: WeaponConfig,
        storage_path: str,
        detector_metadata: DetectorMetadata,
        logger: logging.Logger = None,
    ):
        super().__init__(camera_id, config, storage_path, logger)
        # Use pre-computed metadata (no model creation here)
        self.weapons_subcls = detector_metadata.weapons_subcls
        self.local_hands_idx = detector_metadata.local_hands_idx
        self.required_margins = detector_metadata.required_margins
        self.detector_img_size = detector_metadata.detector_img_size
        self.detector_dtype = detector_metadata.detector_dtype
        self.person_idx = detector_metadata.person_idx
        self.remote_hands_idx = detector_metadata.remote_hands_idx

        # Detection history (protected by lock)
        self.known_detections: List[DetectionRecord] = []
        self.lock = threading.Lock()

        self.logger.info(
            f"WeaponCameraState initialized for camera {camera_id}, " f"weapons_subcls={self.weapons_subcls}"
        )

    def register_detection(
        self, timestamp: int, position: np.ndarray, encoding: np.ndarray, is_brandished: bool
    ) -> bool:
        """
        Register a detection using a pre-computed encoding.
        Encoding is computed OUTSIDE the lock by the caller's thread-local encoder.
        Lock only protects the history list — no inference happens under lock.
        """
        with self.lock:
            return self._register_detection_inner(timestamp, position, encoding, is_brandished)

    def _register_detection_inner(
        self, timestamp: int, position: np.ndarray, encoding: np.ndarray, is_brandished: bool
    ) -> bool:
        is_existing = False
        check_encodings = False
        self.known_detections = sorted(
            [det for det in self.known_detections if timestamp - det.timestamp < self.config.record_expiration_time_ms],
            key=lambda det: det.timestamp,
            reverse=True,
        )[:MAX_RECORDS_NUM]
        exist_idx = -1

        if self.known_detections:
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

        if not is_existing and check_encodings:
            existing_encodings = np.array([d.encoding for d in self.known_detections])
            similarities = np.dot(existing_encodings, encoding.T)
            if np.any(similarities > self.config.record_encoding_sim_threshold):
                is_existing = True
                exist_idx = np.argmax(similarities)

        if not is_existing:
            self.logger.info(f"Camera {self.camera_id} detected weapon at {position} is a new detection")
            detection = DetectionRecord(timestamp, position, encoding, is_brandished)
            self.known_detections.append(detection)

        if is_existing:
            self.logger.info(f"Camera {self.camera_id} detected weapon at {position} with existing detection")
            if exist_idx >= 0:
                self.known_detections[exist_idx].timestamp = timestamp
                new_enc = self.known_detections[exist_idx].encoding + encoding
                self.known_detections[exist_idx].encoding = new_enc / np.linalg.norm(new_enc)
                self.known_detections[exist_idx].position = position
                if not self.known_detections[exist_idx].is_brandished:
                    self.known_detections[exist_idx].is_brandished = is_brandished
                    is_existing = not is_brandished  # re-register if brandishing status changed
        return is_existing


# ---------------------------------------------------------------------------
# Task functions: stateless — receive thread-local models as parameter
# ---------------------------------------------------------------------------


@dataclass
class ExpertPayload:
    camera_id: str
    image: np.ndarray
    batch_data: List[List[float]]
    timestamp: int
    immediate: bool = False
    image_path: Optional[str] = None
    confidence: Optional[int] = None


def expert_task(payload: ExpertPayload, camera_state: CameraState, models: ExpertModelSet, logger: logging.Logger):
    try:
        image = payload.image
        detections = models.detector.forward_on_crop_list([image])[0]
        w, h = image.shape[1], image.shape[0]
        return detection_to_expert_results(detections, w, h, payload.camera_id, payload.timestamp)
    except Exception as e:
        log_exception(logger, "Error in expert task", e)
    return None


def weapon_task(
    payload: ExpertPayload, camera_state: "WeaponCameraState", models: WeaponModelSet, logger: logging.Logger
):
    try:
        camera_id = payload.camera_id
        stt = camera_state
        if not stt.config.enable:
            return
        image = payload.image
        original_image = payload.image
        detections = models.detector.forward_on_crop_list([image])[0]
        w, h = image.shape[1], image.shape[0]
        result = detection_to_expert_results(detections, w, h, payload.camera_id, payload.timestamp)
        weapons = []
        h, w = image.shape[:2]
        if len(detections) > 0:
            weapons = [d for d in detections if d[5] in stt.weapons_subcls and d[4] >= stt.config.global_min_conf]
        crops = []

        if weapons and payload.image_path is not None:
            try:
                full_path = os.path.join(stt.storage_path, payload.image_path)
                full_resolution = np.array(Image.open(full_path))
                factor_w = full_resolution.shape[1] / w
                factor_h = full_resolution.shape[0] / h
                factors = np.array([factor_w, factor_h, factor_w, factor_h, 1.0, 1.0])
                weapons = [w * factors for w in weapons]
                image = full_resolution
                w, h = image.shape[1], image.shape[0]
                logger.info(f"Loaded full resolution image from path {full_path}")
            except Exception as e:
                log_exception(logger, f"Error loading full resolution image from {payload.image_path}", e)

        for wep in weapons:
            wep_sz = max((wep[2] - wep[0]) / w, (wep[3] - wep[1]) / h)
            conf_th = stt.config.get_confidence_threshold(wep_sz) * stt.config.get_confidence_factor(payload.confidence)
            logger.info(
                f"Camera {camera_id} timestamp {payload.timestamp} detected weapon: "
                f"confidence{wep[4]} size {wep_sz} valid {wep[4] >= conf_th}"
            )
            if wep[4] > conf_th:
                crop = crop_image_by_bbox(image, wep[:4], margins=stt.required_margins)
                crops.append(crop)

        if len(crops) > 0:
            logger.info(f"Camera {camera_id} detected {len(weapons)}")
            logger.debug(f"weapons: {weapons}")
            results_cls, results_scores = models.classifier.forward_on_crop_list(crops)
            classified = [weapons[i] for i, c in enumerate(results_cls) if c == 0]
            logger.info(f"Classification results: {results_cls}, scores: {results_scores}")
            if classified:
                logger.info(f"Camera {camera_id} {len(classified)} candidates were classified as weapon")
                weapons_normed = np.array(classified)[:, :4] / np.array([w, h, w, h])
                batch_data = np.atleast_2d(np.array(payload.batch_data))
                if batch_data.shape[1] >= BDR.CLASS:
                    persons = batch_data[batch_data[:, BDR.CLASS] == stt.person_idx]
                else:
                    return result

                overlaps = calculate_overlap_matrix(weapons_normed, persons[:, BDR.POS])
                hands_remote = batch_data[batch_data[:, BDR.SUBCLASS] == stt.remote_hands_idx, :4]
                hands_local = detections[detections[:, 5] == stt.local_hands_idx, :4] / np.array([w, h, w, h])
                hands_list = [h for h in [hands_remote, hands_local] if len(h) > 0]
                hands = np.vstack(hands_list) if hands_list else np.empty((0, 4))
                if len(hands) > 0:
                    hands_overlaps = np.max(bbox_ious(weapons_normed, hands[:, BDR.POS]), axis=1)
                else:
                    hands_overlaps = np.zeros(weapons_normed.shape[0])

                for threat, pos in enumerate(weapons_normed):
                    max_index = np.argmax(overlaps[threat])
                    person_id = -1
                    zoom_crop = crops[threat]
                    weapon_location = pos.tolist()
                    validation_roi = None
                    is_brandished = hands_overlaps[threat] > stt.config.brandishing_threshold

                    # Compute encoding OUTSIDE the lock using thread-local encoder (no deadlock possible)
                    encoding = models.encoder.forward_on_crop_list([zoom_crop])[0]

                    # Lock only protects history mutation — no inference under lock
                    is_registered = stt.register_detection(payload.timestamp, pos, encoding, is_brandished)

                    if not is_registered:
                        if overlaps[threat][max_index] > stt.config.overlap_threshold:
                            person_id = int(persons[max_index, BDR.ID])
                            joint_roi = np.vstack([pos, persons[max_index, BDR.POS]])
                            validation_roi = np.concatenate(
                                (np.min(joint_roi[:, :2], axis=0), np.max(joint_roi[:, 2:], axis=0))
                            )
                            validation_crop = crop_image(image, validation_roi, margins=[0.1, 0.1], bgr_map=True)
                        else:
                            validation_crop = crop_image(image, validation_roi, margins=[0.1, 0.1], bgr_map=True)

                        alert = {
                            "camera_id": payload.camera_id,
                            "timestamp": payload.timestamp,
                            "person_id": person_id,
                            "weapon_location": weapon_location,
                            "validation_crop": encode_image_to_base64(validation_crop),
                            "zoom_crop": encode_image_to_base64(zoom_crop),
                            "snapshot": encode_image_to_base64(original_image),
                            "perimeter": validation_roi.tolist() if validation_roi is not None else weapon_location,
                            "is_brandished": bool(is_brandished),
                        }
                        logger.info(f"Camera {camera_id} detected weapon at {weapon_location} with person {person_id}")
                        result["weapon"] = alert
        return result
    except Exception as e:
        log_exception(logger, "Error in weapon task", e)


def get_analytic_config() -> WeaponConfig:
    analytic_config = load_configAnalytic() or {}
    config = analytic_config.get("offline_analytics", {}).get("weapon_detection", {})
    min_conf_value = config.get("min_weapon_confidence", None)
    legacy_mode = config.get("legacy_mode", False)
    if min_conf_value is None:
        min_conf = LEGACY_MIN_WEAPON_CONFIDENCE if legacy_mode else MIN_WEAPON_CONFIDENCE
    elif not isinstance(min_conf_value, dict):
        min_conf = {0.1: min_conf_value, 0.2: min_conf_value + 0.1, 1: min_conf_value + 0.3}
    else:
        min_conf = {float(k): float(v) for k, v in min_conf_value.items()}

    return WeaponConfig(
        enable=config.get("enable", True),
        overlap_threshold=config.get("overlap_threshold", OVERLAP_THRESHOLD),
        record_overlap_high_threshold=config.get("record_overlap_high_threshold", RECORD_OVERLAP_HIGH_THRESHOLD),
        record_overlap_low_threshold=config.get("record_overlap_low_threshold", RECORD_OVERLAP_LOW_THRESHOLD),
        record_encoding_sim_threshold=config.get("record_encoding_sim_threshold", RECORD_ENCODING_SIM_THRESHOLD),
        record_expiration_time_ms=config.get("record_expiration_time_ms", RECORD_EXPIRATION_TIME_MS),
        min_validation_crop=config.get("min_validation_crop", MIN_VALIDATION_CROP),
        min_weapon_confidence=min_conf,
        offline_resize=config.get("offline_resize", False),
        brandishing_threshold=config.get("brandishing_threshold", BRANDISHING_THRESHOLD),
        confidence_factor=config.get("confidence_factor", CONFIDENCE_FACTORS),
        legacy_mode=legacy_mode,
    )


# ---------------------------------------------------------------------------
# Worker classes
# ---------------------------------------------------------------------------


class ExpertOfflineAnalyzeWorker(OfflineAnalyzerWorker):
    name = "Expert Worker"
    use_files = False
    detector_type = DetectorType.EXPERT
    _model_set_class = ExpertModelSet
    _camera_state_class = CameraState

    def __init__(
        self,
        ipc_socket,
        timeout_ms=500,
        num_workers=6,
        max_queue_size=20,
        hwm_per_client=10,
        log_queue=None,
    ):
        super().__init__(ipc_socket, timeout_ms, num_workers, max_queue_size, hwm_per_client, log_queue)
        args = DetectorFactory().create_config(self.detector_type)

        self.im_sz = list(args.im_size) + [3]
        self.camera_states: Dict[str, CameraState] = {}
        self.config = {}
        self._task_fn = expert_task
        # Initialized in child process via _init_child_resources()
        self.expert_pool = None
        self.task_semaphore = None
        self._thread_models: Dict[int, ExpertModelSet] = {}  # thread_ident -> ModelSet

    def _init_child_resources(self):
        """Initialize resources in the child process (after fork). Eager model init."""
        self.expert_pool = ThreadPoolExecutor(max_workers=NUM_PROCESS_WORKERS)
        self.task_semaphore = threading.Semaphore(NUM_PROCESS_WORKERS)
        self._active_tasks: Dict[str, float] = {}
        self._active_tasks_lock = threading.Lock()

        # Eager init: block until all pool threads have their own models ready
        self._eager_init_models()
        self.logger.info(f"[{self.name}] {NUM_PROCESS_WORKERS} thread-local model sets ready")

    def _get_thread_models(self) -> ExpertModelSet:
        """Get the current thread's ModelSet (already initialized eagerly)."""
        return self._thread_models[threading.current_thread().ident]

    def _eager_init_models(self):
        """Initialize model sets for all pool threads.

        Model initialization is done INSIDE submitted tasks (not in the
        ThreadPoolExecutor initializer) so that transient failures don't
        permanently kill threads.

        Initialization is serialized (one thread at a time) to avoid:
        - Triton connection storms that cause transient failures
        - CUDA memory allocation contention during TRT fallback builds
        """
        init_serial_lock = threading.Lock()
        init_errors: Dict[int, Exception] = {}
        errors_lock = threading.Lock()
        barrier = threading.Barrier(NUM_PROCESS_WORKERS, timeout=180)

        def init_and_sync():
            tid = threading.current_thread().ident
            self.logger.info(f"[{self.name}] Thread {tid} init_and_sync starting, waiting for init_serial_lock")
            t_lock_wait = time.time()
            try:
                # Serialize model creation to avoid GPU/Triton contention
                with init_serial_lock:
                    self.logger.info(
                        f"[{self.name}] Thread {tid} acquired init_serial_lock after "
                        f"{time.time() - t_lock_wait:.2f}s, building model set"
                    )
                    t_build = time.time()
                    model_set = self._model_set_class(self.detector_type, self.logger)
                    self._thread_models[tid] = model_set
                self.logger.info(
                    f"[{self.name}] Thread {tid} models initialized after {time.time() - t_build:.2f}s "
                    f"(released init_serial_lock)"
                )
            except Exception as e:
                with errors_lock:
                    init_errors[tid] = e
                self.logger.error(f"[{self.name}] Thread {tid} model init failed: {e}")
                barrier.abort()
                return
            self.logger.info(f"[{self.name}] Thread {tid} entering barrier.wait()")
            t_barrier = time.time()
            try:
                barrier.wait()
                self.logger.info(f"[{self.name}] Thread {tid} passed barrier after {time.time() - t_barrier:.2f}s")
            except threading.BrokenBarrierError:
                self.logger.warning(
                    f"[{self.name}] Thread {tid} barrier BROKEN after {time.time() - t_barrier:.2f}s "
                    f"(another thread failed)"
                )

        self.logger.info(f"[{self.name}] Submitting {NUM_PROCESS_WORKERS} eager-init tasks to expert_pool")
        futures = [self.expert_pool.submit(init_and_sync) for _ in range(NUM_PROCESS_WORKERS)]
        concurrent.futures.wait(futures)
        self.logger.info(f"[{self.name}] All {NUM_PROCESS_WORKERS} eager-init futures returned")
        # Check for exceptions in the futures themselves (unexpected errors)
        for f in futures:
            if f.exception() is not None:
                raise f.exception()
        # Check for model init errors
        if init_errors:
            failed = ", ".join(f"thread {tid}: {err}" for tid, err in init_errors.items())
            raise RuntimeError(
                f"[{self.name}] {len(init_errors)}/{NUM_PROCESS_WORKERS} threads failed model init: {failed}"
            )
        self.logger.info(f"[{self.name}] Eager init complete: {len(self._thread_models)} model sets")

    _SEMAPHORE_TIMEOUT = 10.0
    _TASK_STUCK_THRESHOLD = 120.0

    def _acquire_semaphore_with_timeout(self) -> bool:
        acquired = self.task_semaphore.acquire(timeout=self._SEMAPHORE_TIMEOUT)
        if not acquired:
            self._log_stuck_tasks()
        return acquired

    def _log_stuck_tasks(self):
        now = time.time()
        with self._active_tasks_lock:
            stuck = {tid: now - ts for tid, ts in self._active_tasks.items() if now - ts > self._TASK_STUCK_THRESHOLD}
        if stuck:
            self.logger.error(
                f"[{self.name}] Detected {len(stuck)} potentially stuck tasks: "
                f"{', '.join(f'{tid}={dur:.1f}s' for tid, dur in stuck.items())}"
            )

    def run(self):
        """Override run to initialize fork-sensitive resources in the child process.

        Start the IO thread FIRST so the ZMQ socket is bound and health checks
        can connect immediately, then perform the (potentially slow) eager model init.
        """
        self.logger.info(f"Starting {self.name} process with {self.num_workers} workers")

        # Start IO thread first — binds the ZMQ socket and sets ready_event
        self.io_thread = threading.Thread(target=self.io_thread_func, daemon=False)
        self.io_thread.start()

        # Now do the heavy model initialization (may take seconds) — a watchdog thread
        # logs periodically so a permanent hang here is visible, not silent.
        t_init_start = time.time()
        self.logger.info(f"[{self.name}] Beginning _init_child_resources() at epoch t={t_init_start:.2f}")
        init_done_event = threading.Event()

        def _watchdog():
            n = 0
            while not init_done_event.wait(timeout=5):
                n += 1
                self.logger.warning(
                    f"[{self.name}] [watchdog] _init_child_resources() still not complete after "
                    f"{time.time() - t_init_start:.1f}s (check #{n})"
                )

        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()
        try:
            self._init_child_resources()
        except Exception as e:
            log_exception(self.logger, f"[{self.name}] FATAL: _init_child_resources() raised, worker thread will NOT start", e)
            raise
        finally:
            init_done_event.set()
        self.logger.info(f"[{self.name}] _init_child_resources() completed after {time.time() - t_init_start:.2f}s")

        # Start worker dispatcher thread (only after models are ready)
        self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=False)
        self.worker_thread.start()

        # Wait for stop signal
        while not self.stop_event.is_set():
            time.sleep(0.5)

        # Wait for threads to finish
        self.logger.info("Waiting for threads to complete...")
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3)
        if self.io_thread and self.io_thread.is_alive():
            self.io_thread.join(timeout=3)

        self.logger.info(f"{self.name} process stopped")

    def _create_camera_state(self, camera_id: str) -> CameraState:
        """Create camera state. Override in subclass for weapon-specific state."""
        return self._camera_state_class(camera_id, self.config, self.storage_path, self.logger)

    def _handle_job_list(self, job_list: List[MessageData]) -> List:
        results = []
        for job in job_list:
            if not self._acquire_semaphore_with_timeout():
                self.logger.warning(
                    f"[{self.name}] Semaphore acquire timed out — pool workers may be stuck. "
                    f"Dropping job for camera {job.meta.get('camera_id', 'unknown')}"
                )
                self.metrics["dropped"] = self.metrics.get("dropped", 0) + 1
                continue

            payload = ExpertPayload(
                camera_id=job.meta["camera_id"],
                image=self._decompress_image(job.image_bytes),
                batch_data=job.meta["batch_data"],
                timestamp=job.meta["timestamp"],
                immediate=job.meta.get("immediate", False),
                image_path=job.meta.get("image_path", None),
                confidence=job.meta.get("confidence", None),
            )
            if payload.camera_id not in self.camera_states:
                self.camera_states[payload.camera_id] = self._create_camera_state(payload.camera_id)
                self.logger.info(f"Camera {payload.camera_id} state initialized.")

            self.expert_pool.submit(self._wrapped_task, payload, job.ident)
        return results

    def _wrapped_task(self, payload, ident):
        task_id = f"{payload.camera_id}_{payload.timestamp}"
        with self._active_tasks_lock:
            self._active_tasks[task_id] = time.time()
        try:
            t_s = time.time()
            models = self._get_thread_models()
            camera_state = self.camera_states[payload.camera_id]
            results = self._task_fn(payload, camera_state, models, self.logger)
            if results:
                try:
                    self.response_queue.put_nowait((ident, results))
                except Full:
                    self.logger.warning(f"Response queue full in {self.name}, dropping result")
            self._update_exec_metrics(time.time() - t_s)
        except Exception as e:
            log_exception(self.logger, "Error in _wrapped_task", e)
        finally:
            with self._active_tasks_lock:
                self._active_tasks.pop(task_id, None)
            self.task_semaphore.release()
            if payload.image_path is not None:
                full_path = os.path.join(self.storage_path, payload.image_path)
                try:
                    if os.path.exists(full_path):
                        os.remove(full_path)
                except Exception as e:
                    log_exception(self.logger, f"Error cleaning up image file {payload.image_path}", e)

    def _decompress_image(self, compressed: bytes) -> np.ndarray:
        image_bytes = zlib.decompress(compressed)
        return np.frombuffer(image_bytes, dtype=np.uint8).reshape(self.im_sz)

    def cleanup(self):
        if hasattr(self, "expert_pool") and self.expert_pool:
            self.expert_pool.shutdown(wait=True)

    def flush(self):
        super().flush()
        self.camera_states.clear()
        self.logger.info("Flushed camera states")


class WeaponOfflineAnalyzeWorker(ExpertOfflineAnalyzeWorker):
    name = "Weapon Worker"
    min_batch_size: int = 2
    max_batch_size: int = NUM_PROCESS_WORKERS
    max_hold_time: float = 2
    use_files = True
    detector_type = DetectorType.WEAPON_EXPERT
    _model_set_class = WeaponModelSet
    _camera_state_class = WeaponCameraState

    def __init__(
        self,
        ipc_socket,
        timeout_ms=500,
        num_workers=6,
        max_queue_size=20,
        hwm_per_client=10,
        log_queue=None,
    ):
        super().__init__(ipc_socket, timeout_ms, num_workers, max_queue_size, hwm_per_client, log_queue)

        self.camera_states: Dict[str, WeaponCameraState] = {}
        self.config: WeaponConfig = get_analytic_config()
        self._task_fn = weapon_task
        self._detector_metadata: Optional[DetectorMetadata] = None

    def _init_child_resources(self):
        """Initialize resources, then extract detector metadata from the first model set."""
        super()._init_child_resources()
        # Extract static metadata from any thread's model set (all identical)
        first_model_set: WeaponModelSet = next(iter(self._thread_models.values()))
        detector = first_model_set.detector
        remote_class_handler = ClassHandler()
        self._detector_metadata = DetectorMetadata(
            weapons_subcls=detector.class_handler.get_object_classes(detector.class_handler.weapon_value),
            local_hands_idx=detector.class_handler.class_str_to_int(HAND_CLASS_NAME),
            required_margins=[first_model_set.classifier.required_margins] * 2,
            detector_img_size=tuple(detector.args.im_size),
            detector_dtype=detector.data_type,
            person_idx=remote_class_handler.person_value,
            remote_hands_idx=remote_class_handler.class_str_to_int(HAND_CLASS_NAME),
        )
        self.logger.info(
            f"[{self.name}] DetectorMetadata extracted: weapons_subcls={self._detector_metadata.weapons_subcls}"
        )

    def _create_camera_state(self, camera_id: str) -> WeaponCameraState:
        return WeaponCameraState(camera_id, self.config, self.storage_path, self._detector_metadata, self.logger)
