from typing import List, Dict

import cv2
import numpy as np

from general.analyzer_general import ROI_SHAPE, logger
from .motion_extraction import BaseMotionExtractor, MotionExtractionConfig


class ImdiffMotionExtractionConfig(MotionExtractionConfig):
    device: str = "cpu"  # device to load weights to
    grayscale: bool = True  # work on grayscale only
    motion_th: int = 25
    min_intensity: int = 10
    resize: bool = True  # weather to resize the image
    im_size: List[int] = [160, 96]  # resize factor on DETECTOR images (detection resolution)


class ImdiffMotionEstimator(BaseMotionExtractor):
    _config_type = ImdiffMotionExtractionConfig
    args: ImdiffMotionExtractionConfig

    def __init__(self, msg_dict: Dict, image_size):
        super().__init__(msg_dict, image_size)
        self.last_image = None
        self.thresholding = self.args.motion_th / self.args.motion_levels
        if not self.args.grayscale:
            logger.error("Image-diff motion extractor only work with grayscale=True, reverting configuration")
        self.args.grayscale = True
        self.dtype = float

    def _internal_extract_mv(self, frame):
        frame = frame.astype(self.dtype)
        if self.last_image is None:
            mv_out = np.zeros(frame.shape[0:2], dtype=self.dtype)
        else:
            # subtract and normalize
            image_diff = cv2.absdiff(frame, self.last_image)
            image_max = np.maximum(frame, self.last_image)
            mv_th = (image_diff/(image_max + self.args.min_intensity)) > self.thresholding
            mv_out = mv_th.astype(np.uint8)
        self.last_image = frame
        return mv_out
