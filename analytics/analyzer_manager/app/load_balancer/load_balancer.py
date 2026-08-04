from copy import copy

from general.core import BaseConfig, MotionExtractorType, MotionData
from load_balancer.motion_extraction import MotionExtractorFactory


class LoadBalancerConfig(BaseConfig):
    enable: bool = True  # Enables motion extraction
    timespan_ms: int = 500
    min_motion_th = 3  # below that it is not considered motion
    extractor: str = MotionExtractorType.IM_DIFF  # type of motion extractor
    process_timespan_ms: int = None

    def __init__(self, args_dict: dict = None):
        super().__init__(args_dict)
        if self.process_timespan_ms is None:
            self.process_timespan_ms = self.timespan_ms


class LoadBalancer:

    # hard coded static fields
    motion_levels = 100  # number of output levels in the motion estimations

    def __init__(self, init_dict, detector_resolution):
        self.config = LoadBalancerConfig(init_dict)
        self.resolution = detector_resolution
        self.initialization_msg = copy(init_dict)
        self.extractor = MotionExtractorFactory().create(self.config.extractor, init_dict, detector_resolution)
        self.last_timestamp = 0

    def update_resolution(self, resolution):
        self.resolution = resolution
        self.extractor = MotionExtractorFactory().create(self.config.extractor, self.initialization_msg, resolution)
    def extract_motion_estimations(self, timestamps, detector_cpu, detector_gpu) -> MotionData:
        mv_per_frame = [None] * len(timestamps)
        motion_index = [0] * len(timestamps)
        motion_max = [0] * len(timestamps)
        for idx, timestamp in enumerate(timestamps):
            if timestamp - self.last_timestamp > self.config.timespan_ms:
                self.last_timestamp = timestamp
                mv_per_frame[idx], motion_index[idx], motion_max[idx] = self.extractor.extract_motion_estimations(detector_cpu[idx])
            elif timestamp - self.last_timestamp > self.config.process_timespan_ms:
                self.extractor.extract_motion_estimations(detector_cpu[idx], True)

        return MotionData(mv_per_frame, motion_index, motion_max)

    '''
    def extract_motion_estimations_old(self, timestamps, detector_cpu, detector_gpu):
        mv_per_frame = []
        for idx, timestamp in enumerate(timestamps):
            if timestamp - self.last_timestamp > self.config.timespan_ms:
                self.last_timestamp = timestamp
                mv_per_frame.append(self.extractor.extract_motion_estimations(detector_cpu[idx]))
            elif timestamp - self.last_timestamp > self.config.process_timespan_ms:
                self.extractor.extract_motion_estimations(detector_cpu[idx], True)
                mv_per_frame.append(np.zeros(ROI_SHAPE, dtype=np.uint8))
            else:
                mv_per_frame.append(np.zeros(ROI_SHAPE, dtype=np.uint8))
        return self._build_motion_info(timestamps, mv_per_frame, self.mv_duration)

    def _build_motion_info_old(self, timestamps, mv_out, duration):
        timestamp = timestamps[-1]
        self.last_frames += len(timestamps)
        max_mv_out = np.maximum(np.array(mv_out).max(axis=0), self.last_mv_out)
        if (timestamp - self.last_timestamp_for_info) > duration:
            motion_info = {}
            motion_info["startTimestamp"] = timestamps[0]
            motion_info["endTimestamp"] = timestamp
            motion_info["numberOfFrames"] = self.last_frames
            motion_info["vector"] = max_mv_out.flatten().astype(int).tolist()
            motion_info["maxMotion"] = int(np.max(max_mv_out))

            # reset internal info
            self.last_frames = 0
            self.last_mv_out *= 0
            # update timestamp of sending valid info
            self.last_timestamp_for_info = timestamp
        else:
            self.last_mv_out = max_mv_out
            motion_info = None

        return motion_info, mv_out
    '''
