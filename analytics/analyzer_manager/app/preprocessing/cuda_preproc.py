from typing import List

import cv2
import numpy as np
import cupy as cp

from detection.detector import BaseDetector
from general.core import AnalyticImage
from general.cuda_utils import ImageParser
from .preprocessor import PreprocessorBase


class PreprocessorLumCuda(PreprocessorBase):
    use_cuda_extract = True
    resize_kernel = None
    def __init__(self, config, full_resolution: List[int], detector: BaseDetector):
        super().__init__(config, full_resolution, detector)
        self._init_resize_kernel()
        self._cuda_parser_address = ImageParser().create_parser(
            resize_kernel=self.resize_kernel,
            antialias=self.detector_config.antialias,
            normalize=self.detector_config.normalize,
            kernel_size=np.shape(self.resize_kernel)[0],
            batch_size=self.batch_size,
            in_height=full_resolution[1],
            in_width=full_resolution[0],
            height=self.height,
            width=self.width,
            is_half=self.detector_config.half,
        )
        if self.detector_use_cuda_mem:
            self.process = self.process_directly_to_gpu
        else:
            self.gpu_intermediate_mem = cp.zeros((3, *self.detector_config.im_size), dtype=self.detector_dtype)

    def process(self, image: AnalyticImage, index: int):
        self.image_extractor.extract_images([image], out=self.full_resolution_mem[index])
        ImageParser().parse_images(
            self._cuda_parser_address,
            int(self.gpu_intermediate_mem.data),
            self.detector_images_bgr[index],
            int(self.image_extractor.gpu_mem.data),
        )
        self.gpu_intermediate_mem.get(out=self.detector_inputs_mem[index])
        return self.detector_images_bgr

    def process_directly_to_gpu(self, image: AnalyticImage, index: int):
        self.image_extractor.extract_images([image], out=self.full_resolution_mem[index])
        ImageParser().parse_images(
            self._cuda_parser_address,
            int(self.detector_inputs_mem[index].data),
            self.detector_images_bgr[index],
            int(self.image_extractor.gpu_mem.data),
        )
        return self.detector_images_bgr
        pass

    def _init_resize_kernel(self):
        if hasattr(self.detector_config, "resize_factor"):
            resize_factor = self.detector_config.resize_factor
        else:
            resize_factor = self.width / self.full_resolution[0]
        sigma = 1 / (3 * resize_factor)
        kernel_size = 4 * sigma
        if kernel_size % 2 != 1:  # if not uneven round to next uneven
            kernel_size = (np.floor(kernel_size / 2) + np.floor(kernel_size % 2)) * 2 + 1
        kernel_size = int(kernel_size)
        self.resize_kernel = cv2.getGaussianKernel(kernel_size, sigma).astype(np.float32)

    def __del__(self):
        if self._cuda_parser_address is not None:
            ImageParser.destroy(self._cuda_parser_address)
