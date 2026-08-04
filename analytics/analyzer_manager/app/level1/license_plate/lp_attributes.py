from typing import Tuple
import cv2
import albumentations as alb

from general.analyzer_general import InferenceType
from level1.common_classifier.classifier_attributes import ClassifierAConfig, AttributeInterpreter, BaseAttributesModel


class LPaConfig(ClassifierAConfig):
    weights: str = "lpc_resnet18_1_1.pt"  # path for builtin weight file
    attr_desc_file: str = "lpc_net_1_0.csv"  # path for network description file
    im_size: Tuple[int, int] = (224, 224)
    no_sqr_pad: bool = False  # Anti aliasing when down sampling
    antialias: bool = False
    minimal_crop_size: int = 20
    use_calibration: bool = True
    has_calibration: bool = False
    global_conf_th: float = 0.5
    name: InferenceType = InferenceType.LPC_ATTR


class LPAttributesModel(BaseAttributesModel):
    _config_type = LPaConfig
    args: LPaConfig
    interpreter_class: type = AttributeInterpreter
    _interpreter: AttributeInterpreter  

    def _build_transform(self):
        interp_mode = cv2.INTER_LINEAR
        self.resize_transform = alb.Resize(self.args.im_size[0], self.args.im_size[1], interpolation=interp_mode)
        self.transform = alb.Compose(
            [alb.Normalize(self.args.mean, self.args.std), self.resize_transform]
        ) 


