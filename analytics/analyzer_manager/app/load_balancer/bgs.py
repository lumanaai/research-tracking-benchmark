from typing import Dict
import cv2

from .motion_extraction import BaseMotionExtractor, MotionExtractionConfig


def set_uniform_lighting(img, value=128):
    """
    Set the Value channel of an HSV image to a uniform value.

    :param img: img (np.array)
    :param value: The value to set for the Value channel (0-255).
    :return: Image with uniform lighting in BGR color space.
    """

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split into the H, S, and V channels
    h, s, _ = cv2.split(hsv)

    # if night vision return the original image
    if h.mean() < 1 and s.mean() < 1:
        return img

    # Create a uniform V channel
    v_uniform = s.copy()
    v_uniform.fill(value)

    # Merge channels back
    hsv_uniform = cv2.merge([h, s, v_uniform])

    # Convert back to BGR
    bgr_uniform = cv2.cvtColor(hsv_uniform, cv2.COLOR_HSV2BGR)

    return bgr_uniform


class BaseBgsExtractor(BaseMotionExtractor):
    def __init__(self, msg_dict: Dict, image_size):
        super().__init__(msg_dict, image_size)
        self.engine = self._init_engine()
        # self.post_process = self.post_process_neighbors

    def _init_engine(self):
        pass


class Cv2MotionExtractionConfig(MotionExtractionConfig):
    history: int = 10
    varThreshold: int = 25
    detectShadows: bool = False
    algorithm: str = "mog2"
    process_timespan_ms = 0
    morph_kernel = 0


class Cv2BgsExtractor(BaseBgsExtractor):
    _config_type: type = Cv2MotionExtractionConfig
    args: Cv2MotionExtractionConfig

    def _init_engine(self):
        if self.args.algorithm == "mog2":
            return cv2.createBackgroundSubtractorMOG2(
                self.args.history, self.args.varThreshold, self.args.detectShadows
            )

    def _internal_extract_mv(self, frame):
        frame_mask = self.engine.apply(frame)
        return frame_mask

    def preprocess_frame(self, frame):
        return super().preprocess_frame(set_uniform_lighting(frame))
