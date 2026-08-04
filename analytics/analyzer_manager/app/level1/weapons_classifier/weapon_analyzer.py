from typing import List, Any, Dict, Optional
import numpy as np

from level1.advanced import BaseAdvanceAnalyzer, AttrData
from general.core import BatchDataResolver, AdvanceAnalyzerType, AttrConfidence, AttrProperty, WorkItem
from general.analyzer_general import InferenceType
from general.common_models import CalibratedModelWrapper, BinaryNetConfig


class WCConfig(BinaryNetConfig):
    weights: str = "wc_resnet18_cal_1_7.pt"  # path for builtin weight file
    margins: float = 0.5
    n_classes: int = 2  # number of classes to classify
    classifier_threshold: float = 0.5
    name: InferenceType = InferenceType.WEAPONS_CLASSIFICATION
    antialias = True
    half: bool = False
    low_conf_th: float = 0.2  # low confidence threshold in case of a small area detection or too large area


class WCResnet18Classifier(CalibratedModelWrapper):
    result_mapping: Dict[int, str] = {0: "gun", 1: "None"}
    _config_type = WCConfig
    args: WCConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):

        super(WCResnet18Classifier, self).__init__(msg_dict, is_local)
        print("Finished building weapons model")


class WeaponAnalyzer(BaseAdvanceAnalyzer):
    analyze_untracked = False
    _type = AdvanceAnalyzerType.WEAPONS
    name = "weaponAnalyzer"
    enabled = False

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)
        self.supported_classes = [self.class_handler.weapon_value]
        self.supported_subclasses = self.class_handler.get_object_classes(self.class_handler.weapon_value)
        self.history_tbl = {}
        if msg is None:
            msg = {}
        self.model = WCResnet18Classifier(msg)
        self.crop_margins = [0.5, 0.5]  # 50 % in each side

    def calculate_pre_score(self, batch_data: BatchDataResolver, image_batch=None) -> Dict[int, float]:
        detections = batch_data.query(batch_data.SUBCLASS, self.supported_subclasses)
        scores = {}

        # scores based on crop width: should be higher than th and lower than width (orientation)
        if len(detections) > 0:
            # scores based on size only
            shape = image_batch[0].frame.shape
            height = shape[0]
            width = shape[1]
            net_sz = self.model.args.im_size[0] * self.model.args.im_size[1]
            pos = detections[:, BatchDataResolver.POS]
            index = (detections[:, BatchDataResolver.INDEX]).astype(int)
            confs = detections[:,batch_data.CONFIDENCE]
            norm_factor = width * height / net_sz
            norm_area_pix = np.multiply((pos[:, 2] - pos[:, 0]), (pos[:, 3] - pos[:, 1])) * norm_factor
            crop_scores = np.minimum(norm_area_pix, 1)
            low_conf_s = np.logical_and(
                norm_area_pix < 0.3, confs < self.model.args.low_conf_th
            )
            low_conf_l = np.logical_and(
                norm_area_pix > 5, confs < self.model.args.low_conf_th
            )
            crop_scores[np.logical_or(low_conf_s, low_conf_l)] = -1
            scores.update({index[i]: scr for i, scr in enumerate(crop_scores)})
        return scores

    def calculate_post_score(self, obj_id: int, attr_data: AttrData) -> float:
        return attr_data.attributes["score"]

    def _run_model(self, batch: List[WorkItem]) -> List[Any]:
        results_cls, results_scores = self.model(batch)
        l1_res = []
        for i, result in enumerate(results_cls):
            l1_res.append({"desc": WCResnet18Classifier.result_mapping[result], "score": results_scores[i]})
        return l1_res

    def to_properties(self, attr_data: Dict) -> Dict:
        score = attr_data["score"]
        if score > self.post_score_success_th:
            conf = AttrConfidence.HIGH
        else:
            conf = AttrConfidence.LOW
        return {"weaponType": AttrProperty(attr_data["desc"], score, conf, self.priority)}
