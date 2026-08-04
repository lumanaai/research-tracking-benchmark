import os
from typing import List, Dict
import numpy as np
import cv2

from general.analyzer_general import logger, ROI_SHAPE, DEFAULT_DETECTOR_IM_SIZE
from general.core import BaseConfig, MotionExtractorType


class MotionExtractionConfig(BaseConfig):
    device: str = "cpu"  # device to load weights to
    timespan_ms: int = 500  # timestamp to process motion vectors
    grayscale: bool = False  # work on grayscale only
    motion_levels = 100  # number of output levels in the motion estimations
    min_motion_th = 3  # below that it is not considered motion
    morph_kernel: int = 0  # morph kernel size (0 disabled), default on is 7
    index_percentile: int = 98  # percentile of the motion array to be calculated as motion index
    process_timespan_ms: int = timespan_ms
    resize: bool = True  # weather to resize the image
    im_size: List[int] = [320, 192]  # resize factor on DETECTOR images (detection resolution)



class BaseMotionExtractor:
    _config_type: type = MotionExtractionConfig

    def __init__(self, msg_dict: Dict, image_size):
        logger.info("Getting notion extraction arguments")
        self.args: MotionExtractionConfig = self._config_type(msg_dict)
        self.resize_resolution = None
        if self.args.resize:
            self.resize_resolution = self.args.im_size
        else:
            self.resize_resolution = image_size[-1::-1]

        if self.args.morph_kernel > 0:
            self.morph_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.args.morph_kernel, self.args.morph_kernel))
            self.post_process = self.post_process_morph
        else:
            self.morph_ker = None
            binning_filter_sz = (np.array(self.resize_resolution[-1::-1]) / np.array(ROI_SHAPE)).astype(int)
            self.binning_filter = np.ones(binning_filter_sz.astype(int), dtype=np.uint8)
            self.binning_factor = np.prod(binning_filter_sz) * 0.8 / self.args.motion_levels
            self.post_process = self._post_process_bin_sum



    def extract_motion_estimations(self, image, is_process_only: bool = False):
        mv_out = None
        motion_index = 0
        motion_max = 0
        frame = self.preprocess_frame(image)
        motion_vec = self._internal_extract_mv(frame)
        if not is_process_only:
            mv_out = self.post_process(motion_vec)
            motion_index, motion_max = self._calc_motion_index_max(mv_out)
        return mv_out, motion_index, motion_max

    def _calc_motion_index_max(self, motion_est) -> (int, int):
        res = np.percentile(motion_est, [self.args.index_percentile, 100], interpolation="nearest")
        return int(res[0]), int(res[1])

    def _internal_extract_mv(self, frame):
        pass

    def preprocess_frame(self, frame) -> List[np.array]:
        if self.args.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.args.resize:
            frame = cv2.resize(frame, tuple(self.resize_resolution), interpolation=cv2.INTER_AREA)
        return frame

    def post_process_morph(self, motion_est):
        me = cv2.resize(motion_est, ROI_SHAPE, interpolation=cv2.INTER_LINEAR)
        me = cv2.morphologyEx(np.uint8(me), cv2.MORPH_DILATE, self.morph_ker)
        me[me < self.args.min_motion_th] = 0
        return np.clip(np.asarray(me, dtype=np.uint8), a_min=0, a_max=self.args.motion_levels)

    def _post_process_bin_sum(self, motion_est):
        motion_clipped = np.clip(motion_est, a_min=0, a_max=1)
        # Reshape the matrix into a 4D array where blocks are separated

        reshaped_mot = motion_clipped.reshape(
            ROI_SHAPE[0], self.binning_filter.shape[0], ROI_SHAPE[1], self.binning_filter.shape[1]
        )
        # Find the max in each block by specifying the axes of the block
        summed = reshaped_mot.sum(axis=(1, 3))
        normed = np.clip(
            (summed.astype(float) / self.binning_factor).astype(np.uint8), a_min=0, a_max=self.args.motion_levels
        )
        normed[normed < self.args.min_motion_th] = 0
        normed.reshape(ROI_SHAPE)

        return normed


class MotionExtractorFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MotionExtractorFactory, cls).__new__(cls)
        return cls._instance

    def create(
        self, extractor_name: str = "im-diff", extractor_args: dict = None, input_size=None
    ) -> BaseMotionExtractor:

        if input_size is None:
            input_size = DEFAULT_DETECTOR_IM_SIZE
        logger.info(f"Building {extractor_name} motion detector with the arguments : {extractor_args}")
        if MotionExtractorType.IM_DIFF in extractor_name:
            from .imdiff import ImdiffMotionEstimator

            return ImdiffMotionEstimator(extractor_args, input_size)
        elif MotionExtractorType.CV2 in extractor_name:
            from .bgs import Cv2BgsExtractor

            return Cv2BgsExtractor(extractor_args, input_size)
        else:
            raise NotImplemented
