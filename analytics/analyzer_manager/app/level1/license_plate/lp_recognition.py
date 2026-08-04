from typing import Dict, List, Optional, Tuple

import numpy as np

from general import proj
from general.analyzer_general import InferenceType
from general.inference import InferenceWrapper, BaseInferenceConfig
from level1.license_plate.PaddleOCR2Pytorch.tools.infer.predict_rec_model_batch_modular import (
    ModelBuilder,
    Preprocessor,
    InferenceEngine,
    Postprocessor,
    warmup,
)


class LPConfig(BaseInferenceConfig):

    rec_algorithm: str = "CRNN"
    rec_char_dict_path: str = "en_dict.txt"
    rec_char_type: str = "en"
    rec_image_shape: str = "3,48,320"
    rec_yaml_path: str = "en_PP-OCRv4_rec_LPR.yml"
    use_space_char: bool = True
    rec_image_inverse: bool = True
    limited_max_width: int = 1280
    limited_min_width: int = 16
    max_text_length: int = 10

    name: InferenceType = InferenceType.LICENSE_PLATE
    use_half: bool = True
    use_gpu: bool = True

    weights: str = "lp_paddle_v4.pt"

    # Not sure is needed
    scales: list = [8, 16, 32]
    show_log: bool = False
    min_crop_width: float = 0.1
    max_crop_width: float = 0.9
    min_crop_height: float = 0.1
    max_crop_height: float = 0.9

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        self.rec_yaml_path = proj.info_path(self.name, self.rec_yaml_path)
        self.rec_char_dict_path = proj.info_path(self.name, self.rec_char_dict_path)
        self.rec_model_path = self.weights
        # self.weights = self.rec_model_path


class LicensePlateRecognition(InferenceWrapper):
    _config_type = LPConfig
    args: LPConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        self.args = self._config_type(msg_dict)
        self.preprocessor = Preprocessor(self.args)
        self.postprocessor = Postprocessor(self.args)
        super().__init__(msg_dict, is_local)
        self.args.im_size = self.input_shape[1::-1]

    def build_full_model(self):

        model_builder = ModelBuilder(self.args)
        self.model = InferenceEngine(model_builder.net, self.args)
        warmup(self.preprocessor, self.model, self.args)

    def prepare_crops(self, crop_list):
        crops = self.preprocessor.preprocess(crop_list)
        return crops

    def post_infer(self, model_result: np.ndarray, crops: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        results = self.postprocessor.postprocess(model_result)
        preds_str = [r[0] for r in results]
        confidence = [r[1] for r in results]
        return preds_str, confidence

    def _build_transform(self):
        pass

    @property
    def input_shape(self):
        return self.preprocessor.rec_image_shape[::-1]
