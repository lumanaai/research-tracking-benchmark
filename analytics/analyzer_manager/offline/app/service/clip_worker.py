from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel

# from utils.infra.cloud_logger import logger
from general.analyzer_general import logger
from general.clip_encoder import ClipVisionEncoder
from general.offline_analytics import decode_image_from_base64
from offline_common import CommonBackgroundWorker


class ClipPayload(BaseModel):
    camera_id: str
    image: str
    id_base: int
    id_index: int
    immediate: bool = False
    extra: Optional[Dict] = None


class ClipBackgroundWorker(CommonBackgroundWorker):
    worker_type: str = "Clip"
    min_batch_size: int = 4  # Minimum batch size for processing jobs
    max_batch_size: int = 8  # Minimum batch size for processing jobs
    max_hold_time: float = 10.0  # Maximum time to hold jobs in the queue before processing in seconds

    def _handle_job_list(self, job_list):
        if job_list:
            crops = [decode_image_from_base64(payload.image) for payload in job_list]
            net_results = self.net.forward_on_crop_list(crops)
            logger.debug("Encodings generated for crops:", len(net_results))
            for idx, payload in enumerate(job_list):
                result = {
                    "cameraId": payload.camera_id,
                    "idBase": payload.id_base,
                    "idIndex": payload.id_index,
                    "encoding": net_results[idx].astype(np.float32).tobytes().hex(),
                    "extra": payload.extra or {} if net_results[idx] is not None else "",
                }
                self.out_queue.put(result)

    def _init_network(self):
        self.net = ClipVisionEncoder(self.analytic_config, is_local=False)
        logger.info(f"initialized CLIP network for ClipBackgroundWorker with local {self.net.is_local}")
