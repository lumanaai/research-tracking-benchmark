from pydantic import BaseModel

from detection.detector import DetectorFactory, DetectorType
# from utils.infra.cloud_logger import logger
from general.analyzer_general import logger
from general.offline_analytics import decode_image_from_base64
from offline_common import CommonBackgroundWorker, detection_to_expert_results


class ExpertPayload(BaseModel):
    camera_id: str
    image: str
    timestamp: int
    immediate: bool = True


class ExpertBackgroundWorker(CommonBackgroundWorker):
    worker_type: str = "Expert"
    min_batch_size: int = 1  # Minimum batch size for processing jobs
    max_batch_size: int = 2  # Maximum batch size for processing jobs
    max_hold_time: float = 1.5  # Maximum time to hold jobs in the queue before processing in seconds
    Payload = ExpertPayload

    def _handle_job_list(self, job_list):
        if job_list:
            crops = [decode_image_from_base64(payload.image) for payload in job_list]
            try:
                detections = self.net.forward_on_crop_list(crops)
                w, h = crops[0].shape[1], crops[0].shape[0]
                logger.debug("Detections generated for crops:", len(detections))
                for idx, payload in enumerate(job_list):
                    result = detection_to_expert_results(detections[idx], w, h, payload.camera_id, payload.timestamp)
                    self.out_queue.put(result)
            except Exception as e:
                logger.error("Error occurred while generating detections:", e)

    def _init_network(self):
        self.net = DetectorFactory(self.analytic_config).create(DetectorType.EXPERT)
        logger.info(f"initialized Expert with local {self.net.is_local}")
