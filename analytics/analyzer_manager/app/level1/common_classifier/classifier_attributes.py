import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit, logit

from general import proj
from general.analyzer_general import InferenceType, GLOBAL_FAILURE, GLOBAL_TYPE_KEY
from general.core import AttrProperty, AttrConfidence, softmax
from general.inference import InferenceWrapper, BaseInferenceConfig
from level1.common_classifier.model.DeepMAR import DeepMar
from level1.common_classifier.utils import fetch_transforms
from level1.reid.BoT import BotConfig, BoT


class ClassifierAConfig(BaseInferenceConfig):
    weights: str = ""                       # path for built-in weight file
    attr_desc_file: str = ""                # path for network description csv file
    im_size: Tuple[int, int] = (224, 224)
    no_sqr_pad: bool = False                # Anti aliasing when down sampling
    minimal_crop_size: int = 80
    use_calibration: bool = True
    has_calibration: bool = False
    global_conf_th: float = 0.5
    name: InferenceType

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        default_folder = self.name.split("_")[0]
        if Path(self.attr_desc_file).is_absolute():
            attr_file = self.attr_desc_file
        elif self.unique_weights:
            attr_file = Path(self.unique_weights).with_suffix(".csv")
        else:
            attr_file = proj.info_path(default_folder, self.attr_desc_file)
        if not os.path.exists(attr_file):
            candidates = sorted(glob.glob(proj.info_path(default_folder, "*.csv")), key=os.path.getmtime)
            if candidates:
                attr_file = candidates[-1]
            else:
                attr_file = proj.info_path(default_folder, self.attr_desc_file)
        self.attr_desc_file = attr_file

        if self.use_calibration:
            self.has_calibration = True


class AttributeType(str, Enum):
    NONE = "none"
    MULTI_SELECT = "multi_select"
    SINGLE_SELECT = "single_select"
    BINARY = "binary"
    UNDEFINED = "undefined"


@dataclass
class ClassAttribute:
    index: int
    label: str
    negative_label: str
    category: str
    type: AttributeType
    weight: float = 1
    undefined_index: int = -1
    high_conf: float = 0.5
    med_conf: float = 0.5
    low_conf: float = 0.5
    calibrated: int = 0

    def __post_init__(self):
        if type(self.index) is not int:
            self.index = int(self.index)
        if type(self.type) is not AttributeType:
            self.type = AttributeType(self.type)
        if type(self.weight) is not float:
            self.weight = float(self.weight)
        if type(self.undefined_index) is not int:
            self.undefined_index = int(self.undefined_index)
        if type(self.high_conf) is not float:
            self.high_conf = float(self.high_conf)
        if type(self.med_conf) is not float:
            self.med_conf = float(self.med_conf)
        if type(self.low_conf) is not float:
            self.low_conf = float(self.low_conf)
        if type(self.calibrated) is not bool:
            self.calibrated = int(self.calibrated)


class AttributesCategory:
    label: str
    undefined_field_index: int
    attr_type: AttributeType
    fields: List[ClassAttribute]
    binary_fields: List[ClassAttribute]

    def __init__(self, label, fields: List[ClassAttribute], global_fields_map: Dict[int, ClassAttribute]):
        self.label = label
        self.undefined_field_index = -1
        self.fields = []
        self.indices = []
        self.binary_fields = []
        self.attr_type = AttributeType.NONE
        self.fields_map = global_fields_map
        for field in fields:
            if field.type is AttributeType.BINARY:
                self.binary_fields.append(field)
            elif field.type is not AttributeType.UNDEFINED:
                self.fields.append(field)
                self.indices.append(field.index)
                self.undefined_field_index = field.undefined_index
                self.attr_type = field.type

    def is_defined(self, undefined_index: int, scores: np.array) -> bool:
        return undefined_index < 0 or scores[undefined_index] < self.fields_map[undefined_index].high_conf

    @staticmethod
    def gen_binary_attr(field: ClassAttribute, score: float, priority) -> AttrProperty:
        if score >= field.high_conf:
            return AttrProperty(field.label, max(score, 1 - field.calibrated), AttrConfidence.HIGH, priority)
        elif score >= field.med_conf:
            return AttrProperty(field.label, max(score, 1 - field.calibrated), AttrConfidence.MEDIUM, priority)
        return AttrProperty(field.negative_label, max(1 - score, 1 - field.calibrated), AttrConfidence.HIGH, priority)

    @staticmethod
    def gen_multiselect_attr(field, score, priority):
        if score >= field.high_conf:
            return AttrProperty(field.label, max(score, 1 - field.calibrated), AttrConfidence.HIGH, priority)
        if score >= field.med_conf:
            return AttrProperty(field.label, max(score, 1 - field.calibrated), AttrConfidence.MEDIUM, priority)
        if score >= field.low_conf:
            return AttrProperty(field.label, max(score, 1 - field.calibrated), AttrConfidence.LOW, priority)
        return None

    def interpret_category_scores(self, scores, priority) -> List[AttrProperty]:
        results = []
        for field in self.binary_fields:
            if self.is_defined(field.undefined_index, scores):
                results.append(self.gen_binary_attr(field, scores[field.index], priority))

        if self.is_defined(self.undefined_field_index, scores):
            if self.attr_type == AttributeType.SINGLE_SELECT:
                cat_scores = logit(scores[self.indices])
                probs = softmax(cat_scores)
                mx_field = self.fields[np.argmax(probs)]
                scores[self.indices] = probs
                # once its probability compare against th
                attr = self.gen_multiselect_attr(mx_field, scores[mx_field.index], priority)
                if attr is not None:
                    results.append(attr)
                # results.append(
                #    AttrProperty(mx_field.label, max(scores[ind_max], 1 - mx_field.calibrated), AttrConfidence.HIGH)
                #)
            elif self.attr_type == AttributeType.MULTI_SELECT:
                for field in self.fields:
                    attr = self.gen_multiselect_attr(field, scores[field.index], priority)
                    if attr is not None:
                        results.append(attr)
                # if len(results) == 0:
                #    results.append("unknown")
        # else:
        # results.append("unknown")
        return results

    def get_possible_attributes(self) -> List[str]:
        results = []
        for field in self.binary_fields:
            results.append(field.label)
            results.append(field.negative_label)

        for field in self.fields:
            results.append(field.label)

        if self.undefined_field_index > -1:
            results.append("unknown")
        return results


class AttributeInterpreter:
    _mapping: Dict[int, ClassAttribute]
    _category_list: List[AttributesCategory]
    interpreter_version: str = "1.0"
    failure_kw = GLOBAL_FAILURE
    global_kw = GLOBAL_TYPE_KEY

    def __init__(self, args: ClassifierAConfig):
        self._category_list = []
        self._mapping, categories = AttributeInterpreter.import_mapping_csv(args.attr_desc_file)
        self._num_attr: int = len(self._mapping)
        if not args.use_calibration:
            for key in self._mapping:
                self._mapping[key].calibrated = 0
                self._mapping[key].high_conf = args.global_conf_th
                self._mapping[key].low_conf = args.global_conf_th
        self._build_categories(categories)

        self.undefined_fields = [f for f in self._mapping if self._mapping[f].type == AttributeType.UNDEFINED]
        undef_mapper = defaultdict(list)
        for key in self._mapping:
            if self._mapping[key].undefined_index >=0:
                undef_mapper[self._mapping[key].undefined_index].append(key)
        self.undefined_field_mapping = {f: np.array(undef_mapper[f]).astype(int) for f in self.undefined_fields}

        global_category = [cat for cat in self._category_list if cat.label == self.global_kw]
        if len(global_category) > 0:
            self._global_field: ClassAttribute = global_category[0].binary_fields[0]
        else:
            self._global_field = self._mapping[self.undefined_fields[0]]

    @staticmethod
    def import_mapping_csv(file_name: str) -> Tuple[Dict, Dict]:
        import csv

        mapping: Dict[int, ClassAttribute] = {}
        categories: Dict[str, List[ClassAttribute]] = {}

        with open(file_name, "rt") as f:
            table = csv.DictReader(f)
            for row in table:
                attr = ClassAttribute(**row)
                mapping[attr.index] = attr
                if attr.category not in categories:
                    categories[attr.category] = []
                categories[attr.category].append(attr)

        return mapping, categories

    def _build_categories(self, categories: Dict[str, List[ClassAttribute]]):
        self._category_list.clear()
        for cat in categories:
            self._category_list.append(AttributesCategory(cat, categories[cat], self._mapping))

    def interpret_scores(self, sig_scores: np.array, priority) -> Dict[str, List[AttrProperty]]:
        results = {}
        for cat in self._category_list:
            cat_results = cat.interpret_category_scores(sig_scores, priority)
            if len(cat_results):
                results[cat.label] = cat_results
        # self.resolve_conflicts(results)
        return results

    def merge_scores(self, sig_scores: List[np.array]) -> np.array:
        score_mat = np.array(sig_scores)
        if len(sig_scores) == 1:
            return sig_scores[0]
        global_scores = score_mat[:, self._global_field.index]
        good_idxs = np.where(global_scores <= self._global_field.high_conf)[0]
        if len(good_idxs) == 0:
            return sig_scores[-1]
        elif len(good_idxs) == 1:
            return sig_scores[good_idxs[0]]

        score_mat = score_mat[good_idxs]
        weights = np.ones_like(score_mat)
        for fidx, fields in self.undefined_field_mapping.items():
            def_mask = (score_mat[:, fidx] < self._mapping[fidx].high_conf).astype(float)
            weights[:, fields] = (def_mask * (1 - score_mat[:, fidx]) ** 2)[:, np.newaxis]

        re_weight = score_mat * weights
        sum_weights = np.sum(weights, axis=0)
        result_vec = np.where(sum_weights > 0, np.sum(re_weight, axis=0) / sum_weights, 0)

        ### legacy algorithm, use max per binary attribute, not using the weights at all

        # result_vec = np.mean(score_mat, axis=0)
        # for field in self._mapping:
        #     if self._mapping[field].type in [AttributeType.BINARY, AttributeType.UNDEFINED]:
        #         scores = score_mat[:, field]
        #         neg = np.where(scores < self._mapping[field].high_conf)[0]
        #         scores[neg] = 1 - scores[neg]
        #         max_ind = np.argmax(scores)
        #         if max_ind in neg:
        #             result_vec[field] = 1 - scores[max_ind]
        #         else:
        #             result_vec[field] = scores[max_ind]
        return result_vec

    def calc_confidence_score(self, scores) -> float:
        score = -1
        if scores[self._global_field.index] <= self._global_field.high_conf:
            score =  1 - np.mean(scores[self.undefined_fields])
        return score

    def interpret_success(self, scores) -> Dict:
        if scores[self._global_field.index] < self._global_field.high_conf:
            return {self.global_kw: self._global_field.negative_label}
        else:
            return {self.global_kw: self._global_field.label}

    def unknown_results(self) -> Dict[str, List[str]]:
        results = {}
        # for cat in self._category_list:
        #    results[cat.label] = ["unknown"]
        return results

    def build_category_mapping(self) -> Dict[str, List[str]]:
        category_mapping = {}
        for c in self._category_list:
            category_mapping[c.label] = c.get_possible_attributes()
        return category_mapping

    @staticmethod
    def resolve_conflicts(results: Dict[str, List[str]]):
        # global failure should nullify results
        global_kw = AttributeInterpreter.global_kw
        failure_kw = AttributeInterpreter.failure_kw

        if global_kw in results and failure_kw in results[global_kw]:
            return {global_kw: failure_kw}
        return results

    def _export_mapping(self) -> Dict:
        fields_dict = []
        for attr in self._mapping.keys():
            attr_data = self._mapping[attr]
            fields_dict.append(
                {
                    "index": attr_data.index,
                    "label": attr_data.label,
                    "negative_label": attr_data.negative_label,
                    "category": attr_data.category,
                    "type": attr_data.type.value,
                    "weight": attr_data.weight,
                    "undefined_index": attr_data.undefined_index,
                }
            )
        return {"version": AttributeInterpreter.interpreter_version, "fields": fields_dict}

    @property
    def num_attr(self):
        return self._num_attr


class BaseAttributesModel(InferenceWrapper):
    _config_type = ClassifierAConfig
    args: ClassifierAConfig
    interpreter_class: type = AttributeInterpreter
    _interpreter: AttributeInterpreter

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        print("Beginning attributes network init...")

        self.args = self._config_type(msg_dict)
        self._interpreter = self.interpreter_class(self.args)
        self.num_att = self._interpreter.num_attr

        super().__init__(msg_dict, is_local)
        print(f"Finished {self.args.name} attributes network init")

    def _build_transform(self):
        if self.args.has_calibration:
            super()._build_transform()
        else:
            self.transform = fetch_transforms(self.args)

    def _checkpoint_from_file(self):
        import torch

        ckpt = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        return ckpt["state_dicts"][0]

    def build_full_model(self):
        self.model = DeepMar(self.num_att, self.args.im_size, self.args.has_calibration, self.args.weights)
        self.model.load_state_dict(self._checkpoint_from_file())
        self.model.to(self.args.device)
        self.model = self.model.eval()

        if self.args.half:
            self.model = self.model.half()
        #   model warmup
        self.model.warmup(self.args.device, self.args.half)

    def post_infer(self, outputs, inputs):
        result_list = []
        for ind, image_scores in enumerate(outputs):
            sig_scores = expit(image_scores)  # noqa
            is_reliable = self.is_crop_reliable(inputs[ind])

            result_dict = self._interpreter.interpret_success(sig_scores)
            if not is_reliable:
                sig_scores.fill(-1)
            result_dict["scores"] = sig_scores
            result_dict["confidence"] = self._interpreter.calc_confidence_score(sig_scores)
            result_list.append(result_dict)
        return result_list

    def is_crop_reliable(self, crop: np.array) -> bool:
        sz = crop.shape[:2]
        return any([s > self.args.minimal_crop_size for s in sz])

    @property
    def interpreter(self):
        return self._interpreter


class PersonReidConfig(BotConfig):
    weights: str = "pa-reid_resnet50_ibn_a_1_0.pt"  # path for builtin weight file
    name: InferenceType = InferenceType.PERSON_REID
    reid_version = 2

class PersonReid(BoT):
    _config_type = PersonReidConfig


