import cv2
import numpy as np

from general import proj
from general.core import AnalyticImage


class SnapshotResizer:
    def __init__(self, img_size):
        self.antialias = True
        self.detector_img_size = img_size
        self.gpu_mem = None
        self.stream = None

        if proj.load_bool_from_env("USE_CV2_CUDA", default=True) and proj.is_cv2_cuda_available():
            self.stream = cv2.cuda_Stream()
            self.resized_mem = cv2.cuda_GpuMat()
            self.resized_mem.upload(np.zeros(self.detector_img_size + (3,), dtype=np.uint8), self.stream)
            self.resize = self.preproc_image_cuda
        else:
            self.resized_mem = np.zeros(self.detector_img_size + (3,), dtype=np.uint8)

    def preproc_image_cuda(self, image: AnalyticImage, antialias: bool = None, is_bgr: bool = False) -> np.ndarray:
        antialias = self.antialias if antialias is None else antialias
        interp_mode = cv2.INTER_AREA if antialias else cv2.INTER_LINEAR

        # Upload image to GPU
        if self.gpu_mem is None:
            self.gpu_mem = cv2.cuda_GpuMat()

        self.gpu_mem.upload(image.frame, self.stream)
        cv2.cuda.resize(
            self.gpu_mem,
            (self.detector_img_size[1], self.detector_img_size[0]),
            interpolation=interp_mode,
            dst=self.resized_mem,
            stream=self.stream,
        )

        # Convert BGR to RGB in-place (on the same GPU memory)
        if not is_bgr:
            cv2.cuda.cvtColor(self.resized_mem, cv2.COLOR_BGR2RGB, dst=self.resized_mem, stream=self.stream)

        self.stream.waitForCompletion()
        transformed_img = self.resized_mem.download(self.stream)
        return transformed_img

    def resize(self, image: AnalyticImage, antialias: bool = None, is_bgr: bool = False) -> np.ndarray:
        antialias = self.antialias if antialias is None else antialias
        interp_mode = cv2.INTER_AREA if antialias else cv2.INTER_LINEAR

        resized_img = cv2.resize(
            image.frame,
            (self.detector_img_size[1], self.detector_img_size[0]),
            interpolation=interp_mode,
            dst=self.resized_mem,
        )
        if not is_bgr:
            cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB, dst=self.resized_mem)
        return resized_img

    def on_input_resolution_changed(self):
        self.gpu_mem = None


class SnapshotResizerFactory:
    """
    Factory for OfflineAnalyticsClient that ensures a singleton per (server_address, camera_id) pair.
    """

    _clients = {}

    @classmethod
    def get_resizer(cls, camera_id: str, img_size) -> SnapshotResizer:
        key = f"{camera_id}_{img_size}"
        if camera_id not in cls._clients:
            cls._clients[key] = SnapshotResizer(img_size)
        return cls._clients[key]
