import os
from typing import Dict

import cv2

from general.analyzer_general import logger
from general.cuda_utils import NvJpegEncoder


class ImageHandler:
    def __init__(
        self,
        dest_path: str,
        limit_per_period,
        prefix="thumbnail",
        period_sec: float = 100.0,
        name_builder=None,
        nvjpeg_encoder_address=None,
        jpg_quality=70,
        count_offset: int = 0,
    ):
        self.limit_per_period = int(limit_per_period)
        self.period_index = 0
        self._period_frame_counter = 0
        self.period: float = period_sec * 1000
        self.prefix = prefix
        self.dest_path = dest_path
        self.jpg_quality = jpg_quality
        self.nvjpeg_encoder_address = nvjpeg_encoder_address
        self.count_offset = count_offset
        if name_builder == "timestamp":
            def name_function(*args):
                return f"{self.prefix}-{args[2]}.jpg"

        elif name_builder == "norm_timestamp":

            def name_function(*args):
                return f"{self.prefix}-{args[0]}.jpg"

        elif name_builder == "custom":

            def name_function(*args):
                return f"{self.prefix}-{args[3]}.jpg"

        elif name_builder == "custom_counter":

            def name_function(*args):
                return f"{self.prefix}-{args[3]}-{self._period_frame_counter + self.count_offset}.jpg"

        elif name_builder == "thumb_counter":

            def name_function(*args):
                frame_counter = self._period_frame_counter if len(args) < 5 else args[-1]
                return f"{self.prefix}-{args[0]}-{frame_counter}-0.jpg"

        else:

            def name_function(*args):
                return f"{self.prefix}-{args[0]}-{args[1]}-{self._period_frame_counter + self.count_offset}.jpg"

        self.build_name = name_function

    def set_period(self, period_sec: float) -> float:
        self.period = period_sec * 1000
        self.period_index = 0
        self._period_frame_counter = 0

    def update_sent(self, timestamp):
        period_index = int(timestamp / self.period)
        if self.period_index == period_index:
            self._period_frame_counter += 1
        else:
            self._period_frame_counter = 1
            self.period_index = period_index

    def _encode_image(self, image):
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
        if image.size == 0:
            return None
        if self.nvjpeg_encoder_address is not None:
            im_enc = NvJpegEncoder().encode(self.nvjpeg_encoder_address, image.copy(), encode_param[1])
            # im_enc = self.nvjpeg_encoder_address.encode(image.copy(), encode_param[1])
        else:
            im_enc = cv2.imencode(".jpg", image, encode_param)[1].tobytes()
        return im_enc

    def can_send(self, timestamp):
        period_index = int(timestamp / self.period)
        return self._period_frame_counter < self.limit_per_period or period_index > self.period_index

    def send(self, im, timestamp, thumb_count=0, req_name: str = None, force: bool = False):
        period_index = self._check_advance(timestamp)
        normalized_ts = int(period_index * self.period)
        file_name = None
        if self._period_frame_counter < self.limit_per_period or force:
            file_name = self.build_name(normalized_ts, thumb_count, timestamp, req_name)
            im_enc = self._encode_image(im)
            if im_enc:
                with open(os.path.join(self.dest_path, file_name), "wb") as output:
                    output.write(im_enc)
                self._period_frame_counter += 1
            else:
                logger.warning(f"receive empty image to encode. shape: {im.shape}")

        else:
            logger.warning("Crossed Thumbnail limit")
        return file_name, normalized_ts

    def generate_name(self, timestamp, thumb_count=0, req_name: str = None):
        normalized_ts = self.normalize(timestamp)
        return self.build_name(normalized_ts, thumb_count, timestamp, req_name, thumb_count), normalized_ts

    def skip(self, timestamp):
        self._check_advance(timestamp)
        if self._period_frame_counter < self.limit_per_period:
            self._period_frame_counter += 1

    def _check_advance(self, timestamp):
        period_index = int(timestamp / self.period)
        if self.period_index < period_index:
            self._period_frame_counter = 0
            self.period_index = period_index
        return period_index

    def send_out_of_cycle(self, im, file_name):
        im_enc = self._encode_image(im)
        with open(os.path.join(self.dest_path, file_name), "wb") as output:
            output.write(im_enc)

    def normalize(self, timestamp: int) -> int:
        return int(int(timestamp / self.period) * self.period)

    def is_in_period(self, timestamp) -> bool:
        period_index = int(timestamp / self.period)
        return self.period_index <= period_index

    @property
    def count(self):
        return self._period_frame_counter


class BwLimitImageHandler(ImageHandler):
    def __init__(
        self,
        dest_path: str,
        prefix="thumbnail",
        name_builder=None,
        nvjpeg_encoder_address=None,
        count_offset: int = 0,
        bw_rate_limits: Dict[int, int] = None,
    ):
        # period will always be 60 sec
        # jpg quality will be determined dynamically by bw_rate_limits

        super().__init__(dest_path, 0, prefix, 60, name_builder, nvjpeg_encoder_address, 0, count_offset)
        # jpg quality for given BW consumption per minute
        self.bw_rate_limit_th = []
        self.bw_rate_limit_quality = []

        for b, q in bw_rate_limits.items():
            self.bw_rate_limit_th.append(int(b) * 1000)
            self.bw_rate_limit_quality.append(q)
        self.bw_limit_idx = 0
        self.bw_limit = self.bw_rate_limit_th[self.bw_limit_idx]
        self.jpg_quality = self.bw_rate_limit_quality[self.bw_limit_idx]
        self.current_bw = 0

    def _check_advance(self, timestamp):
        period_index = int(timestamp / self.period)
        if self.period_index < period_index:
            self._period_frame_counter = 0
            self.period_index = period_index
            self.bw_limit_idx = 0
            self.jpg_quality = self.bw_rate_limit_quality[self.bw_limit_idx]
            self.current_bw = 0
        return period_index

    def send(self, im, timestamp, thumb_count=0, req_name: str = None, force: bool = False):
        # sending always happens, so its like force
        return super().send(im, timestamp, thumb_count, req_name, force=True)

    def _encode_image(self, image):
        enc_image = super()._encode_image(image)
        self.current_bw += len(enc_image)
        if (
            self.current_bw > self.bw_rate_limit_th[self.bw_limit_idx]
            and self.bw_limit_idx < len(self.bw_rate_limit_th) - 1
        ):
            self.bw_limit_idx += 1
            self.jpg_quality = self.bw_rate_limit_quality[self.bw_limit_idx]
        return enc_image

    def can_send(self, timestamp):
        return True
