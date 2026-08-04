from typing import List, Dict

import cv2
import numpy as np

from detection.detector import BaseDetector
from general.analyzer_general import logger
from general.core import AnalyticImage
from general.proj import load_bool_from_env

class PreprocessorBase:
    detector_inputs_mem = None
    detector_input_mem_ptr = 0
    detector_use_cuda_mem: bool = False
    use_cuda_extract = True

    def __init__(self, config, full_resolution: List[int], detector: BaseDetector):
        self.config = config
        self.full_resolution = full_resolution
        self.detector_config = detector.args
        self.detector_dtype = np.float16 if self.detector_config.half else np.float32
        self.height, self.width = self.detector_config.im_size
        self.detector_images_bgr = np.zeros(
            shape=(self.detector_config.batch_size, *self.detector_config.im_size, 3), dtype=np.uint8
        )
        self.full_resolution_mem = np.zeros(
            (self.detector_config.batch_size, full_resolution[1], full_resolution[0], 3), dtype=np.uint8
        )
        self.detector_use_cuda_mem = self.detector_config.is_trt and detector.is_local
        if self.detector_use_cuda_mem:
            self.detector_inputs_mem = detector.model.input_buffer
        else:
            self.detector_inputs_mem = np.zeros(
                (self.detector_config.batch_size, 3, *self.detector_config.im_size), dtype=self.detector_dtype
            )
        self.batch_size = 1
        if self.use_cuda_extract:
            from general.cuda_utils import ImageExtractor

            self.image_extractor = ImageExtractor(self.batch_size, self.full_resolution)
            is_nv12 = self.config.get("inputFormat", "nv12").lower() == "nv12" or load_bool_from_env("USE_NV12", False)
            if is_nv12:
                self.image_extractor.set_nv12(is_nv12)
                logger.info("Using NV12 format for image extraction")

    def process(self, image: AnalyticImage, index: int):
        # do some processing
        pass

    def process_batch(self, images: List[AnalyticImage]):
        for i, image in enumerate(images):
            self.process(image, i)
            image.processed = self.detector_images_bgr[i]


class PreprocessorCv2Cpu(PreprocessorBase):
    use_cuda_extract = True

    def __init__(self, config, full_resolution: List[int], detector: BaseDetector):
        super().__init__(config, full_resolution, detector)
        factor = self.full_resolution[0] / 1280
        intermediate_height = int(self.full_resolution[1] / factor)
        self.intermediate_mem = np.zeros((intermediate_height, 1280, 3), dtype=np.uint8)

    def process(self, image: AnalyticImage, index: int):
        self.image_extractor.extract_images([image], out=self.full_resolution_mem[index])

        cv2.resize(
            self.full_resolution_mem[index],
            self.intermediate_mem.shape[1::-1],
            interpolation=cv2.INTER_AREA,
            dst=self.intermediate_mem,
        )

        cv2.resize(
            self.intermediate_mem,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
            dst=self.detector_images_bgr[index],
        )
        if self.detector_use_cuda_mem:
            self.detector_inputs_mem[index].set(
                cv2.cvtColor(self.detector_images_bgr[index], cv2.COLOR_BGR2RGB)
                .transpose(2, 0, 1)
                .astype(self.detector_dtype)
            )
        else:
            np.copyto(
                self.detector_inputs_mem[index],
                cv2.cvtColor(self.detector_images_bgr[index], cv2.COLOR_BGR2RGB).transpose(2, 0, 1),
            )
        if self.detector_config.normalize:
            self.detector_inputs_mem[index] /= 255.0
        return self.detector_images_bgr
