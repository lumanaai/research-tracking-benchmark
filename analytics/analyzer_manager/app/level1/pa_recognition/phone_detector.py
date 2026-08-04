import numpy as np
from typing import List, Tuple

from level1.common_classifier.classifier_attributes import AttributeInterpreter
from level1.pa_recognition.person_attributes import PaConfig
from general.analyzer_general import InferenceType, logger
from .hands import HandsModel

class PhoneConfig(PaConfig):
    weights: str = "phone_resnet18_0_1.pt"  # path for builtin weight file
    attr_desc_file: str = "phone.csv"  # path for network description file
    im_size: Tuple[int, int] = (192, 192)
    minimal_crop_size: int = 25
    use_calibration: bool = False
    has_calibration: bool = False
    num_att: int = 0  # auto loaded from attribute description file
    name: InferenceType = InferenceType.PHONE
    arch: str = "resnet18"
    half: bool = True
    antialias = True
    margins = 0.5
    thresholds: List[float] = [-0.3, -0.1]  # [global, phone] logit offsets: sigmoid(logit + thr) > 0.5 ↔ logit > -thr


class PhoneInterpreter(AttributeInterpreter):
    def __init__(self, args: PhoneConfig):
        super().__init__(args)
        for c in self._category_list:
            if c.label == "globalType":
                self.hands_index = c.binary_fields[0].index
            if c.label == "phone":
                self.phone_index = c.binary_fields[0].index

    def compile_requirements(self) -> List[np.ndarray]:
        mask = np.ones(self._num_attr, dtype=int)  # 1 at every position = all attributes active
        return [mask]


class PhoneModel(HandsModel):
    _config_type = PhoneConfig
    args: PhoneConfig
    logger.info("Phone analytics is enabled")
    interpreter_class: type = PhoneInterpreter

    def compile_classification_requirements(self) -> List[np.ndarray]:
        return self.interpreter.compile_requirements()
    
    def post_infer(self, outputs, inputs):
        for idx, image_scores in enumerate(outputs):
            outputs[idx] = image_scores + self.calibration_thresholds
        return super(HandsModel, self).post_infer(outputs, inputs)

    def classify(self, crop_list, classification: np.ndarray):
        mask = classification != -1
        m_class = classification[mask]
        results = self.forward_on_crop_list(crop_list)
        high_confs = [self.interpreter._mapping[i].high_conf for i, use in enumerate(mask) if use]
        scores = np.array([r["scores"][mask] for r in results])
        res = np.all((scores > high_confs) == m_class, axis=1)
        confs = np.prod(np.abs(scores - high_confs) / 0.5, axis=1)
        return res, confs
    