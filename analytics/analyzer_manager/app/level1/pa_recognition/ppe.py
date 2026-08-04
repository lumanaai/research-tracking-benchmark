import numpy as np
from scipy.special import expit

from general.analyzer_general import InferenceType, logger
from level1.pa_recognition.person_attributes import PaConfig, AttributesModel
from level1.common_classifier.classifier_attributes import AttributeInterpreter, AttributeType


class PpeConfig(PaConfig):
    weights: str = "ppe_resnet34_1_2.pt"  # path for builtin weight file
    attr_desc_file: str = "ppe.csv"  # path for network description file
    no_sqr_pad: bool = False  # Anti aliasing when down sampling
    minimal_crop_size: int = 80
    use_calibration: bool = True
    has_calibration: bool = True
    global_conf_th: float = 0.5
    num_att: int = 0  # auto loaded from attribute description file
    name: InferenceType = InferenceType.PPE_ATTR
    arch: str = "resnet34"
    half: bool = False
    max_dynamic_batch: int = 4

undefined_scores = np.array([-1, 1])


class PpeInterpreter(AttributeInterpreter):
    def calc_confidence_score(self, scores):
        undef_scores = []
        for field in self._mapping:
            if self._mapping[field].type == AttributeType.UNDEFINED:
                if scores[field] > self._mapping[field].high_conf:  # if not defined
                    return -1
                undef_scores.append(scores[field])
        return 1 - np.mean(undef_scores)


class PpeModel(AttributesModel):
    _config_type = PpeConfig
    args: PpeConfig
    logger.info("PPE is enabled")
    interpreter_class: type = PpeInterpreter

    def post_infer(self, outputs, inputs):
        result_list = []
        for ind, image_scores in enumerate(outputs):
            if self.is_crop_reliable(inputs[ind]):
                result_list.append({"scores": expit(image_scores)})
            else:
                result_list.append({"scores": undefined_scores})
        return result_list
