from enum import Enum

import cv2
from skimage import transform as trans
import numpy as np

from general.inference import InferenceWrapper


class FaceRecognizerType(str, Enum):
    BASE = "base"
    SFACE = "sface"
    MAGFACE = "magface"
    EDIFFIQA = "ediffiqa"
    ARCFACE = "arcface"
    WEBFACE = "webface"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class BaseFaceRecognizer(InferenceWrapper):
    _input_size = None
    _canonical_locations: np.array = None
    descriptor_length = 128

    # returns bounding box, landmarks and confidence
    def infer(self, image):
        pass

    @property
    def input_size(self):
        return self._input_size

    def set_canonical_locations(self):
        self._canonical_locations = (
                np.array(
                    [
                        [0.34191, 0.46157],
                        [0.65653, 0.45983],
                        [0.50022, 0.64050],
                        [0.37097, 0.82469],
                        [0.63151, 0.82325]
                    ],
                    dtype=np.float32,
                )
                 * self._input_size
        )

    def canonize_face(self, crop, landmark):
        tform = trans.SimilarityTransform()
        tform.estimate(landmark.astype(np.float32), self._canonical_locations)
        affine = tform.params[0:2, :]
        # affine = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        warped = cv2.warpAffine(crop, affine, self._input_size, borderValue=0.0)
        return warped

    def preprocess_image(self, image):
        return cv2.resize(image, self._input_size)


class FaceRecognizerFactory:
    @staticmethod
    def create(recognizer_type: str, config) -> BaseFaceRecognizer:
        recognizer_type = recognizer_type.lower()
        if recognizer_type == FaceRecognizerType.SFACE.lower():
            from level1.face.models import SFace
            recognizer = SFace(config)
        elif recognizer_type == FaceRecognizerType.MAGFACE.lower():
            from level1.face.models import MagFace
            recognizer = MagFace(config)
        elif recognizer_type == FaceRecognizerType.EDIFFIQA.lower():
            from level1.face.models import EDifFIQA
            recognizer = EDifFIQA(config)
        elif recognizer_type == FaceRecognizerType.ARCFACE.lower():
            from level1.face.models import Arcface
            recognizer = Arcface(config)
        elif recognizer_type == FaceRecognizerType.WEBFACE.lower():
            from level1.face.models import WebFace
            recognizer = WebFace(config)
        else:
            raise ValueError(f"recognizer_name must be a supported recognizer:{FaceRecognizerType.list()}")
        return recognizer
