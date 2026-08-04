from enum import Enum

from general.inference import InferenceWrapper, BaseInferenceConfig


class FaceDetectorsType(str, Enum):
    BASE = "base"
    YUNET = "YuNet"
    RETINAFACE = "RetinaFace"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class FaceDetectorConfig(BaseInferenceConfig):
    conf_threshold = 0.9
    nms_threshold = 0.3
    top_k = 5000
    num_landmarks = 5


class BaseFaceDetector(InferenceWrapper):
    _config_type = FaceDetectorConfig
    args: FaceDetectorConfig

    _input_size = None

    # returns bounding box, landmarks and confidence
    def detect(self, image):
        pass

    @property
    def input_size(self):
        return tuple(self._input_size)

    @property
    def num_landmarks(self):
        return self.args.num_landmarks


class FaceDetectorFactory:
    @staticmethod
    def create(detector_type: str, config: dict) -> BaseFaceDetector:
        detector_type = detector_type.lower()
        if detector_type == FaceDetectorsType.YUNET.lower():
            from .models.yunet import YuNet

            detector = YuNet(config)
        elif detector_type == FaceDetectorsType.RETINAFACE.lower():
            from .models.retinaface import RetinaFace

            detector = RetinaFace(config)
        else:
            raise ValueError(f"detector_name must be a supported detector:{FaceDetectorsType.list()}")
        return detector
