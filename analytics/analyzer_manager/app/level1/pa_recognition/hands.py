from typing import Tuple, Dict, Optional, List

import numpy as np

from general.analyzer_general import InferenceType, logger
from level1.common_classifier.classifier_attributes import AttributeInterpreter, BaseAttributesModel
from level1.pa_recognition.person_attributes import PaConfig


class HandsConfig(PaConfig):
    weights: str = "hands_resnet18_0_7.pt"  # path for builtin weight file
    attr_desc_file: str = "hands.csv"  # path for network description file
    im_size: Tuple[int, int] = (192, 192)
    minimal_crop_size: int = 50
    use_calibration: bool = False
    has_calibration: bool = False
    num_att: int = 0  # auto loaded from attribute description file
    name: InferenceType = InferenceType.HANDS
    arch: str = "resnet18"
    half: bool = True
    antialias = True
    margins = 0.5
    # thresholds: List[float] = [-1.2, -1.36, 0] # v3 thresholds
    # thresholds: List[float] = [-1.5, -1, 0.5]  # v5 thresholds
    # thresholds: List[float] = [-0.1, -0.1, 1]  # v6 thresholds
    thresholds: List[float] = [-0.3, 0.0, 1]  # v7 thresholds


class HandsInterpreter(AttributeInterpreter):
    def __init__(self, args: HandsConfig):
        super().__init__(args)
        for c in self._category_list:
            if c.label == "gloveType":
                self.gloves_index = c.binary_fields[0].index
            if c.label == "gloveColor":
                self.colored_index = c.binary_fields[0].index

    def compile_requirements(
        self, is_wearing: Optional[bool] = None, is_colored: Optional[bool] = None
    ) -> List[np.array]:
        masks = []
        # compile the requirements for the classification
        mask = np.ones(self._num_attr, dtype=int) * -1
        mask[self._global_field.index] = 0
        if is_wearing is None:
            # No requirements beyond hand presence
            masks.append(mask)
            return masks

        mask[self.gloves_index] = int(not is_wearing)

        # wearing gloves - colored or transparent - need just one mask
        if is_wearing:
            if is_colored is not None:
                mask[self.colored_index] = int(not is_colored)
            masks.append(mask)
        else:  # need to account for both not wearing gloves and wearing the wrong color
            masks.append(np.copy(mask))
            if is_colored is not None:
                mask[self.gloves_index] = int(is_wearing)
                mask[self.colored_index] = int(is_colored)
                masks.append(mask)
        return masks


class HandsModel(BaseAttributesModel):
    _config_type = HandsConfig
    args: HandsConfig
    logger.info("Hands analytics is enabled")
    interpreter_class: type = HandsInterpreter

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)
        self.calibration_thresholds = np.array(self.args.thresholds)

    def _checkpoint_from_file(self):
        import torch

        ckpt = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        return ckpt

    def post_infer(self, outputs, inputs):
        for idx, image_scores in enumerate(outputs):
            outputs[idx] = -1 * (image_scores + self.calibration_thresholds)  # for "calibration" replacement
            # image_scores *= -1
        return super().post_infer(outputs, inputs)

    def _build_transform(self):
        super(BaseAttributesModel, self)._build_transform()

    def classify(self, crop_list, classification: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # wrapper function to check if the crops meets the required classification. return nparray of boolean values
        mask = classification != -1
        m_class = classification[mask]
        results = self.forward_on_crop_list(crop_list)
        scores = np.array([r["scores"][mask] for r in results])
        confs = np.prod(np.abs(0.5 - scores) / 0.5, axis=1)
        res = np.all(np.round(scores) == m_class, axis=1)
        return res, confs

    def compile_classification_requirements(
        self, is_wearing: bool, is_colored: Optional[bool] = None
    ) -> List[np.array]:
        # wrapper function to compile the classification requirements
        masks = self.interpreter.compile_requirements(is_wearing, is_colored)
        return masks

    @property
    def required_margins(self):
        return self.args.margins

    @property
    def image_size(self):
        return self.args.im_size

    @property
    def minimal_crop_size(self):
        return self.args.minimal_crop_size
