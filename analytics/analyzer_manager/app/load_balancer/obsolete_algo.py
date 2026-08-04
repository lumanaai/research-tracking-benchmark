import sys
from typing import Any, Optional, List

import cv2
import numpy as np
from general.analyzer_general import logger
import ctypes
from ctypes import POINTER, c_uint8, c_int, c_ulonglong, c_float, c_void_p, c_bool
from general import proj
from general.img_utils import yuv420_to_luma_extract
from general.core import BaseConfig, AnalyticImage


def c_ptr(array, ctype):
    return array.ctypes.data_as(POINTER(ctype))


cuda_lib1 = ctypes.CDLL(proj.cuda_path("motion_estimator.so"))
cuda_lib1.createMotionEstimator.restype = c_ulonglong
cuda_lib1.createMotionEstimator.argstype = (POINTER(c_uint8), c_int, c_int, c_int, c_int, c_int, c_int)
cuda_lib1.motionEstimationCuda.argstype = (c_ulonglong, POINTER(c_uint8), POINTER(c_uint8))

cuda_lib2 = ctypes.CDLL(proj.cuda_path("motion_estimation_extractor.so"))
cuda_lib2.createMotionEstimator.restype = c_ulonglong
cuda_lib2.createMotionEstimator.argstype = (c_int, c_int, c_int, c_int, c_int, c_int, c_uint8, c_float, c_int)
cuda_lib2.motionEstimation.argstype = (c_ulonglong, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_bool))


class LoadBalancerConfig(BaseConfig):
    algorithm: int = 2  # path for builtin weight file
    threshold: float = 0.15  # path for specific external weight file
    morph_kernel: int = 7
    icon_factor: int = 8
    norm_factor: float = 100.0
    timespan_ms: int = 1000
    min_motion_th: int = 3


class LoadBalancer:

    # hard coded static fields
    icon_dims = (32, 32)  # size of motion vector (motion estimation) image
    motion_levels = 100  # number of output levels in the motion estimations

    # state fields
    last_frames: int
    last_timestamp: int
    last_timestamp_for_info: int
    lib_address: Optional[Any]
    last_image: Any

    def __init__(self, init_dict, width, height, mv_duration):
        self.mv_duration = mv_duration
        self.config = LoadBalancerConfig(init_dict)

        self.icon_dims_l1 = (self.icon_dims[0] * self.config.icon_factor, self.icon_dims[1] * self.config.icon_factor)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_kernel, self.config.morph_kernel))
        self.w = width
        self.h = height
        self.last_image = None
        self.last_mv_out = np.zeros(self.icon_dims, dtype=np.uint8)
        self.last_frames = 0
        self.last_timestamp = 0
        self.last_timestamp_for_info = 0
        self.lib_address = None

        if self.config.algorithm == 0:
            self._internal_extract_mv = self._extract_motion_estimations_cuda1
        elif self.config.algorithm == 1:
            self._internal_extract_mv = self._extract_motion_estimations_cuda2
        else:
            self._internal_extract_mv = self._extract_motion_estimations_cpu

    def __del__(self):
        if self.lib_address is not None:
            if self.config.algorithm == 0:
                cuda_lib1.destroyMotionEstimator(c_ulonglong(self.lib_address))
            elif self.config.algorithm == 1:
                cuda_lib2.destroyMotionEstimator(c_ulonglong(self.lib_address))

    def is_require_gpu_mem(self):
        return self.config.algorithm == 1

    def extract_motion_estimations(self, frames: List[AnalyticImage], bgr_gpu=None):
        if len(frames) == 0:
            logger.error("LoadBalancer: received empty list of images")
            return []
        b_size = len(frames)
        to_process = None
        if self.config.timespan_ms > 0:
            to_process = [False] * b_size
            for i in range(b_size):
                if frames[i].timestamp - self.last_timestamp > self.config.timespan_ms:
                    to_process[i] = True
                    self.last_timestamp = frames[i].timestamp
        mv_out = self._internal_extract_mv(frames, to_process, bgr_gpu)

        return self._build_motion_info(frames, mv_out, self.mv_duration)

    def close_and_normalize(self, me, image_resized):
        me_morph = cv2.morphologyEx(np.uint8(me), cv2.MORPH_CLOSE, self.kernel)
        image_resized_float = np.asarray(image_resized, dtype=float) + 0.5
        me_normed = np.asarray(me_morph, dtype=float) / image_resized_float * self.motion_levels
        me_normed[me_morph < self.config.min_motion_th] = 0
        return np.clip(np.asarray(me_normed, dtype=np.uint8), a_min=0, a_max=self.motion_levels)

    def _build_motion_info(self, frames, mv_out, duration):
        timestamp = frames[-1].timestamp
        self.last_frames += len(frames)
        max_mv_out = np.maximum(np.array(mv_out).max(axis=0), self.last_mv_out)
        if (timestamp - self.last_timestamp_for_info) > duration:
            motion_info = {}
            motion_info["startTimestamp"] = frames[0].timestamp
            motion_info["endTimestamp"] = timestamp
            motion_info["numberOfFrames"] = self.last_frames
            motion_info["vector"] = max_mv_out.flatten().astype(int).tolist()
            motion_info["maxMotion"] = int(np.max(max_mv_out))

            # reset internal info
            self.last_frames = 0
            self.last_mv_out = np.zeros(self.icon_dims, dtype=np.uint8)
            # update timestamp of sending valid info
            self.last_timestamp_for_info = timestamp
        else:
            self.last_mv_out = max_mv_out
            motion_info = None

        return motion_info, mv_out

    def _extract_motion_estimations_cuda1(self, frames, skip_filter: List[bool] = None, _=None):
        mv_out = []
        start_idx = 0
        if self.last_image is None:
            self.last_image = frames[0].frame
            start_idx += 1
            mv_out.append(np.zeros(self.icon_dims, dtype=np.uint8))
            self.lib_address = cuda_lib1.createMotionEstimator(
                c_ptr(self.last_image, POINTER(c_uint8)),
                c_int(self.icon_dims_l1[1]),
                c_int(self.icon_dims_l1[0]),
                c_int(self.w),
                c_int(self.h),
                c_int(self.icon_dims[1]),
                c_int(self.icon_dims[0]),
            )

        me = np.ndarray(self.icon_dims, dtype=np.uint8)
        image_resized = np.ndarray(self.icon_dims, dtype=np.uint8)

        for idx in range(start_idx, len(frames)):
            if skip_filter is None or skip_filter[idx]:
                image = frames[idx].frame
                cuda_lib1.motionEstimationCuda(
                    c_ulonglong(self.lib_address),
                    c_ptr(me, c_uint8),
                    c_ptr(image_resized, c_uint8),
                    c_ptr(image, c_uint8),
                )
                cuda_lib1.synchronizeCudaStream(c_ulonglong(self.lib_address))

                # apply closing after normalization
                me_norm = self.close_and_normalize(me, image_resized)
                mv_out.append(me_norm)
                self.last_image = image
            else:
                mv_out.append(np.zeros(self.icon_dims, dtype=np.uint8))
        return mv_out

    def _extract_motion_estimations_cuda2(self, frames, skip_filter: List[bool] = None, bgr_gpu=None):
        batch_size = len(frames)
        if self.lib_address is None:
            self.lib_address = cuda_lib2.createMotionEstimator(
                c_int(self.w),
                c_int(self.h),
                c_int(self.icon_dims[1]),
                c_int(self.icon_dims[0]),
                c_int(self.config.morph_kernel),
                c_int(self.config.morph_kernel),
                c_uint8(self.motion_levels - 1),
                c_float(self.config.norm_factor),
                c_int(batch_size),
            )
        np_skip_filter = np.array(skip_filter)
        me = np.zeros(shape=[batch_size, self.icon_dims[0], self.icon_dims[1]], dtype=np.uint8)
        cuda_lib2.motionEstimation(
            c_ulonglong(self.lib_address),
            c_ptr(me, c_uint8),
            c_void_p(bgr_gpu.data_ptr()),
            c_ptr(np_skip_filter, c_bool),
        )

        mv_out = []
        for i in range(batch_size):
            mv_out.append(me[i])

        return mv_out

    def _extract_motion_estimations_cpu(self, frames, skip_filter: List[bool] = None, _=None):
        mv_out = []

        def convert_to_gray(image_to_convert):
            return cv2.cvtColor(image_to_convert, cv2.COLOR_BGR2GRAY)

        start_idx = 0
        if self.last_image is None:
            self.last_image = convert_to_gray(frames[0].frame)
            start_idx += 1
            mv_out.append(np.zeros(self.icon_dims, dtype=np.uint8))

        for idx in range(start_idx, len(frames)):
            if skip_filter is None or skip_filter[idx]:
                # work on gray images
                image = convert_to_gray(frames[idx].frame)

                # resize for normalization purposes
                image_resized = cv2.resize(image, self.icon_dims, interpolation=cv2.INTER_LINEAR)

                # subtract and resize to icon size
                image_diff = cv2.absdiff(image, self.last_image)
                image_diff = cv2.resize(image_diff, self.icon_dims_l1, interpolation=cv2.INTER_LINEAR)
                me = cv2.resize(image_diff, self.icon_dims, interpolation=cv2.INTER_LINEAR)

                # apply closing after normalization
                me_norm = self.close_and_normalize(me, image_resized)
                self.last_image = image
                mv_out.append(me_norm)
            else:
                mv_out.append(np.zeros(self.icon_dims, dtype=np.uint8))
        return mv_out