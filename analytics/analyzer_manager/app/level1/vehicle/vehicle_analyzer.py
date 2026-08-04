from typing import List, Any, Dict

import numpy as np

from general.analyzer_general import GLOBAL_SUCCESS, GLOBAL_FAILURE
from general.core import (
    AdvanceAnalyzerType,
    DescriptorVector,
    WorkItem,
    AttrConfidence,
    AttrProperty,
    BatchDataResolver,
)
from general.img_utils import motion_score
from level1.advanced import BaseAdvanceAnalyzer, AttrData
from .vehicle_parsing import BoTWithClassifier


class VehicleAnalyzer(BaseAdvanceAnalyzer):

    post_score_failure_th = -1.0
    retry_on_failure = 1
    analyzer_batch_limit = 2
    min_blur_th = 0.3
    attribute_num: int
    keys_to_filter = ["scores", "blur", "size_factor", "subclass"]
    night_keys_to_filter = ["colors"]
    name = "vehicleAnalyzer"
    _type = AdvanceAnalyzerType.VEHICLE
    hist_factor = 4

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)
        if msg is None:
            msg = {}
        config_dict = msg["vehicle"] if "vehicle" in msg else msg
        self.model = BoTWithClassifier(config_dict)
        self.supported_classes = [self.class_handler.vehicle_value]
        self.supported_subclasses = [
            self.class_handler.class_str_to_int(s)
            for s in ["car", "truck", "motorcycle", "bicycle", "forklift", "boat", "bus"]  #  "bus",
            if s in self.class_handler.classes_names
        ]
        self.small_subclasses = [
            self.class_handler.class_str_to_int(s)
            for s in ["motorcycle", "bicycle"]
            if s in self.class_handler.classes_names
        ]
        self.reid_only_subclasses = {
            self.class_handler.class_str_to_int(s): s for s in ["boat", "bus"] if s in self.class_handler.classes_names
        }
        self.make_filter = ["motorcycle", "bicycle", "forklift", "bus"]
        self.filter = {}
        self.min_crop_size = np.array(self.model.args.min_crop_size)
        self.min_crop_size_for_small = self.min_crop_size / 2
        self.norm_crop_size = np.array(self.model.args.im_size) * 2 - self.min_crop_size
        self.high_conf_crop_size = np.array(self.model.args.im_size)
        self.is_export_invalid_crops = True
        self.crop_ar = context.analytic_config.get("trainThumbPolicy", {}).get("crops_ar", {}).get("vehicle", None)
        self.reid_version = self.model.version

    def _calc_area_score(self, crop_sz, sub_class: int = -1):
        min_crop_sz = self.min_crop_size_for_small if sub_class in self.small_subclasses else self.min_crop_size
        norm_size = (np.array(crop_sz) - min_crop_sz) / self.norm_crop_size
        return np.clip(np.mean(norm_size), 0, 1)

    def calculate_post_score(self, obj_id: int, attr_data: AttrData):
        scores = attr_data.attributes.get("scores", [])
        net_score = self.model.calculate_post_score(scores)
        blur_score = np.clip((attr_data.attributes.get("blur", 0) - self.min_blur_th) / (self.min_blur_th), 0, 1)
        size_score = np.clip(attr_data.attributes.get("size_factor", 1), 0.5, 1)
        return net_score * size_score * blur_score

    def get_metadata(self, crop_info):
        return {"subclass": int(crop_info[BatchDataResolver.SUBCLASS])}

    def _run_model(self, batch: List[WorkItem]) -> List[Any]:

        blurriness = [motion_score(image.image, crop_sz=0.9) for image in batch]
        size_factor = [np.min(image.image.shape[:2] / self.high_conf_crop_size) for image in batch]

        results = self.model(batch)
        for idx, res in enumerate(results):
            descriptor = res.pop("descriptor", None)
            if descriptor is not None:
                res["descriptorHex"] = DescriptorVector(descriptor)
                res["reid_version"] = self.reid_version
            res["blur"] = blurriness[idx]
            res["size_factor"] = size_factor[idx]
            if res["blur"] > self.min_blur_th:
                res["use_image"] = True
            res["subclass"] = batch[idx].extra.metadata.get("subclass", -1)
        return results

    def merge_with_hist(self, attr: Dict, obj_id: int) -> (Dict, float):
        if len(self.processed.get(obj_id, [])) < 2:
            return attr, self.processed[obj_id][-1].post_score
        scores = np.array([res.post_score for res in self.processed[obj_id]])
        s = scores**self.hist_factor
        if np.sum(s) == 0:
            s = np.ones_like(s)
        factors = (s / sum(s))[:, np.newaxis]
        new_scores = {}
        for key in self.model.categories:
            new_scores[key] = np.sum(
                np.array([res.attributes["scores"][key] for res in self.processed[obj_id]]) * factors,
                axis=0,
            )
        desc = np.sum(
            np.array([res.attributes["descriptorHex"].data for res in self.processed[obj_id]]) * factors,
            axis=0,
        )
        blur = np.sum(np.array([res.attributes.get("blur", 0) for res in self.processed[obj_id]]) @ factors)
        size_factor = np.max(np.array([res.attributes.get("size_factor", 1) for res in self.processed[obj_id]]))
        new_attr = {
            "descriptorHex": DescriptorVector(desc / np.linalg.norm(desc)),
            "scores": new_scores,
            "blur": blur,
            "size_factor": size_factor,
        }
        return new_attr, np.sum(scores @ factors)

    def to_properties(self, attr_data: Dict) -> Dict:
        scores = attr_data.pop("scores")
        blur = attr_data.pop("blur", 0)
        size_factor = attr_data.pop("size_factor", 1)
        max_conf = AttrConfidence.HIGH
        if blur < self.min_blur_th:
            max_conf = AttrConfidence.LOW
        elif size_factor < 0.75:
            max_conf = AttrConfidence.MEDIUM

        subclass = attr_data.pop("subclass", -1)
        if subclass in self.reid_only_subclasses:
            attr_props = super().to_properties(attr_data)
            attr_props["type"] = [
                AttrProperty(self.reid_only_subclasses[subclass], 1, AttrConfidence.HIGH, self.priority)
            ]
        else:
            if max_conf is not AttrConfidence.HIGH:
                attr_data.pop("descriptorHex", None)
            attr_props = self.model.interpreter.interpret_scores(scores, self.priority, max_conf)
            other_props = super().to_properties(attr_data)
            attr_props.update(other_props)

        # bug fix
        keys_to_rename = {"color": "colors"}
        for old_key, new_key in keys_to_rename.items():
            if old_key in attr_props:
                attr_props[new_key] = attr_props.pop(old_key)
        return attr_props

    def export_results(self, att_data):
        prop_data = super().export_results(att_data)
        t = prop_data.get("type", None)
        if t and t[0].value in self.make_filter:
            prop_data.pop("make", "")
        if t is not None:
            if t[0].value == "no-vehicle":
                prop_data.pop("type", "")
                prop_data["globalType"] = [AttrProperty(GLOBAL_FAILURE, 1, AttrConfidence.HIGH, self.priority)]
            elif t[0].value not in self.reid_only_subclasses.values():
                prop_data["globalType"] = [AttrProperty(GLOBAL_SUCCESS, 1, AttrConfidence.HIGH, self.priority)]
        return prop_data
