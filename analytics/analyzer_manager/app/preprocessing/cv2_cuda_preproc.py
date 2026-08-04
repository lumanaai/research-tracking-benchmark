from typing import List

import numpy as np

from detection.detector import BaseDetector
from general.core import AnalyticImage
from general.cuda_utils import ImageParserCV2
from preprocessing import PreprocessorBase
import cv2


class PreprocessorCv2Cuda(PreprocessorBase):
    use_cuda_extract = True

    def __init__(self, config, full_resolution: List[int], detector: BaseDetector):
        super().__init__(config, full_resolution, detector)
        cv2.cuda.setDevice(0)
        self.src_gpu_mat = cv2.cuda_GpuMat(full_resolution[1], full_resolution[0], cv2.CV_8UC3)
        self.dst_gpu_mat = cv2.cuda_GpuMat(self.height, self.width, cv2.CV_8UC3)
        self.dst_norm_map = cv2.cuda_GpuMat(self.height, self.width, cv2.CV_32FC3)
        self.intermediate_mem = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.cv_type = cv2.CV_32F  # cv2.CV_16F if self.detector_config.half else cv2.CV_32F

        # Create a CUDA stream
        self.stream = cv2.cuda_Stream()

    def process(self, image: AnalyticImage, index: int):
        self.image_extractor.extract_images([image], out=self.full_resolution_mem[index])
        self.src_gpu_mat.upload(self.full_resolution_mem[index])
        cv2.cuda.resize(
            self.src_gpu_mat,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
            dst=self.dst_gpu_mat,
            stream=self.stream,
        )
        self.dst_gpu_mat.download(dst=self.detector_images_bgr[index])

        if self.detector_config.normalize:
            self.dst_gpu_mat.convertTo(dst=self.dst_norm_map, rtype=self.cv_type, alpha=1 / 255.0, stream=self.stream)
        else:
            self.dst_gpu_mat.convertTo(dst=self.dst_norm_map, rtype=self.cv_type, stream=self.stream)
        self.stream.waitForCompletion()

        self.dst_norm_map.download(dst=self.intermediate_mem)
        if self.detector_use_cuda_mem:
            self.detector_inputs_mem[index].set(
                self.intermediate_mem[::-1].transpose(2, 0, 1).astype(self.detector_dtype)
            )
        else:
            np.copyto(self.detector_inputs_mem[index], self.intermediate_mem[::-1].transpose(2, 0, 1))
        return self.detector_images_bgr


class PreprocessorCv2CudaCpp(PreprocessorBase):
    use_cuda_extract = True

    def __init__(self, config, full_resolution: List[int], detector: BaseDetector):
        super().__init__(config, full_resolution, detector)
        self.intermediate_mem = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.image_parser = ImageParserCV2(
            self.full_resolution,
            self.detector_config.im_size,
            self.detector_config.antialias,
            self.detector_config.normalize,
        )

    def process(self, image: AnalyticImage, index: int):
        self.image_extractor.extract_images([image], out=self.full_resolution_mem[index])
        self.image_parser.parse_images(
            self.image_extractor.gpu_mem.data, self.detector_images_bgr[index], self.intermediate_mem
        )
        if self.detector_use_cuda_mem:
            self.detector_inputs_mem[index].set(
                self.intermediate_mem.transpose(2, 0, 1).astype(self.detector_dtype)
            )
        else:
            np.copyto(self.detector_inputs_mem[index], self.intermediate_mem.transpose(2, 0, 1))
        return self.detector_images_bgr
