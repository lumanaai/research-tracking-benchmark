import multiprocessing as mp
import threading
import time
from queue import Empty  # Use queue.Empty for thread-safe queues
from typing import Dict

import numpy as np
from pydantic import BaseModel

# from utils.infra.cloud_logger import logger
from general.analyzer_general import logger
from general.core import EndpointFactory
from general.proj import load_config, load_configAnalytic
from general.triton_utils import is_triton_live

TRITON_CONNECTION_ATTEMPTS = 3


class BasePayload(BaseModel):
    camera_id: str
    image: str
    timestamp: int
    immediate: bool = True
    id_base: int
    id_index: int


def detection_to_expert_results(detections: np.ndarray, w: int, h: int, camera_id: str, timestamp: int) -> Dict:
    # normalize and prepare results
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


class BaseBackgroundWorker(mp.Process):
    def __init__(self, in_queue: mp.Queue, out_queue: mp.Queue):
        super(BaseBackgroundWorker, self).__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

        # handle connection to triton
        triton_ok = False
        self.app_config = load_config()
        inference_server_settings = self.app_config.get("analytics", {}).get("inferenceServerUri", {})
        endpoint_factory = EndpointFactory(inference_server_settings)
        mock_endpoint = endpoint_factory.create("")
        for aidx in range(TRITON_CONNECTION_ATTEMPTS):
            if is_triton_live(mock_endpoint):
                triton_ok = True
                break
            else:
                logger.warning(f"{aidx + 1}/{TRITON_CONNECTION_ATTEMPTS} Connection to Triton failed")
                time.sleep(5)
        if not triton_ok:
            logger.error("Triton connection failed. Exiting worker.")
            raise Exception("Triton connection failed. Exiting worker.")


class CommonBackgroundWorker(BaseBackgroundWorker):
    worker_type: str
    min_batch_size: int = 4  # Minimum batch size for processing jobs
    max_batch_size: int = 8  # Minimum batch size for processing jobs
    max_hold_time: float = 10.0  # Maximum time to hold jobs in the queue before processing in seconds

    def __init__(self, in_queue: mp.Queue, out_queue: mp.Queue):
        super().__init__(in_queue, out_queue)
        self.jobs = {True: [], False: []}  # immediate and awaiting jobs
        self.last_inserted_ts: int = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.analytic_config = load_configAnalytic() or {}

    def queue_reader(self):
        while not self.stop_event.is_set():
            try:
                data: BasePayload = self.in_queue.get(timeout=1)
                if data is None:
                    self.stop_event.set()
                    break
                logger.debug(f"Received jobs for entity {data}")
                is_added = data.immediate or len(self.jobs[True]) + len(self.jobs[False]) <= self.max_batch_size * 1.5
                if is_added:
                    with self.lock:
                        self.jobs[data.immediate].append(data)
                    self.last_inserted_ts = time.time()
            except Empty:
                continue

    def _handle_job_list(self, job_list):
        pass

    def job_handler(self):
        while not self.stop_event.is_set():
            job_list = []
            if self.jobs[True]:
                with self.lock:
                    job_list = self.jobs[True]
                    self.jobs[True] = []
            if (
                len(self.jobs[False]) > self.min_batch_size - len(job_list)
                or time.time() - self.last_inserted_ts > self.max_hold_time
                or self.max_batch_size > len(job_list) > 0
            ):
                with self.lock:
                    rem_jobs = self.max_batch_size - len(job_list)
                    job_list.extend(self.jobs[False][:rem_jobs])
                    self.jobs[False] = self.jobs[False][rem_jobs:]

            if job_list:
                self._handle_job_list(job_list)
            else:
                time.sleep(0.5)

    def _init_network(self):
        pass

    def run(self):
        logger.info(f"{self.worker_type} Worker started with 2 threads")
        self._init_network()
        reader_thread = threading.Thread(target=self.queue_reader)
        handler_thread = threading.Thread(target=self.job_handler)
        reader_thread.start()
        handler_thread.start()
        reader_thread.join()
        handler_thread.join()

    def terminate(self):
        logger.info(f"Terminating {self.worker_type} Worker")
        self.in_queue.put(None)  # Signal the reader thread to stop
        self.stop_event.set()
        logger.info(f"{self.worker_type} Worker terminated")
        super().terminate()
