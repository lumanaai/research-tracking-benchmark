from typing import Dict, Optional
import cv2
import albumentations as alb

from general.analyzer_general import InferenceType
from general.common_models import CalibratedModelWrapper, BinaryNetConfig


class DoorsConfig(BinaryNetConfig):
    weights: str = "doors_resnet18.pt"  # path for builtin weight file
    margins: float = 0.0
    classifier_threshold: float = 0.5
    use_calibration = True
    name: InferenceType = InferenceType.DOORS_CLASSIFICATION
    antialias = True
    half: bool = False
    grayscale: bool = False


class DoorsClassifier(CalibratedModelWrapper):
    result_mapping: Dict[int, str] = {1: "open", 0: "close"}

    _config_type = DoorsConfig
    args: DoorsConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):

        super(DoorsClassifier, self).__init__(msg_dict, is_local)
        print("Finished building Doors classification model")

    @property
    def classification_threshold(self):
        return self.args.classifier_threshold

    def _build_transform(self):
        if self.args.grayscale:
            interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR
            self.transform = alb.Compose(
                [
                    alb.ToGray(),
                    alb.Resize(self.args.im_size[0], self.args.im_size[1], interpolation=interp_mode),
                    alb.ChannelShuffle(p=1),
                    alb.Normalize(self.args.mean, self.args.std),
                ]
            )
        else:
            super()._build_transform()
