import threading
import time
from queue import Full
from typing import List, Optional, Dict

import numpy as np
from pydantic import BaseModel

from general.clip_encoder import ClipVisionEncoder
from offline_mq_common import OfflineAnalyzerWorker, MessageData


class ClipPayload(BaseModel):
    camera_id: str
    id_base: int
    id_index: int
    immediate: bool = False
    extra: Optional[Dict] = None


class ClipOfflineAnalyzeWorker(OfflineAnalyzerWorker):
    name = "Clip Worker"

    def __init__(self, ipc_socket, timeout_ms=500, num_workers=1, max_queue_size=20, hwm_per_client=10, log_queue=None):
        super().__init__(ipc_socket, timeout_ms, num_workers, max_queue_size, hwm_per_client, log_queue)
        # Triton client is built in run(), inside the forked child process — NOT here.
        # Building it here would run in the parent process before fork(), and gRPC's
        # C-core is not fork-safe: every worker process forked afterward (Clip, Expert,
        # Weapon) would inherit a channel/thread state that hangs on its own first
        # gRPC call.
        self.net = None
        self.im_sz = None

    def run(self):
        """Override to build the Triton client post-fork, inside this child process."""
        self.logger.info(f"Starting {self.name} process with {self.num_workers} workers")

        # Start IO thread first — binds the ZMQ socket and sets ready_event
        self.io_thread = threading.Thread(target=self.io_thread_func, daemon=False)
        self.io_thread.start()

        self.net = ClipVisionEncoder(self.analytic_config.get("clip", {}), is_local=False)
        self.logger.info(f"initialized CLIP network for ClipBackgroundWorker with local {self.net.is_local}")
        self.im_sz = self.net.args.im_size

        # Start worker dispatcher thread (only after the model client is ready)
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

    def _handle_job_list(self, job_list: List[MessageData]) -> List:
        t_s = time.time()
        results = []
        n_jobs = len(job_list)
        if job_list:
            crops = [
                np.frombuffer(job.image_bytes, dtype=np.uint8).reshape(self.im_sz[0], self.im_sz[1], 3)
                for job in job_list
            ]
            net_results = self.net.forward_on_crop_list(crops)

            for idx, job in enumerate(job_list):
                payload: ClipPayload = ClipPayload(**job.meta)
                # logger.debug("Encodings generated for crops:", len(net_results))
                result = {
                    "cameraId": payload.camera_id,
                    "idBase": payload.id_base,
                    "idIndex": payload.id_index,
                    "encoding": net_results[idx].astype(np.float32).tobytes().hex(),
                    "extra": payload.extra or {} if net_results[idx] is not None else "",
                }
                try:
                    self.response_queue.put_nowait((job.ident, result))
                except Full:
                    self.logger.warning(f"Response queue full in Clip Worker, dropping result")

            # Update metrics
            self._update_exec_metrics(time.time() - t_s, n_jobs)
        return results
