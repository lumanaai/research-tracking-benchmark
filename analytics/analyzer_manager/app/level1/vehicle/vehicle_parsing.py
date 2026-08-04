import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import albumentations as alb
import cv2
import numpy as np
import torch

from general import proj
from general.analyzer_general import InferenceType, logger
from general.core import AttrProperty, AttrConfidence, softmax
from general.img_utils import ResizeAndPadToTarget
from level1.reid.BoT import BoT, BotConfig
from level1.reid.model import BaselineWithClassifier


class BotWithClassifierConfig(BotConfig):
    weights: str = "vehicle_resnet50_ibn_a_2_2.pt"  # path for builtin weight file
    im_size: Tuple[int, int] = (256, 256)
    half: bool = True  # default with false for compatability with xavier
    last_stride: int = 1  # last convolution stride
    model_name: str = "resnet50_ibn_a"
    pretrain_path: Optional[str] = ""
    model_neck: str = "bnneck"
    neck_feat: str = "after"
    model_pretrain_choice = "self"
    feature_dim: int = 512
    thresh_conf: float = 0.3  # confidence threshold
    name: InferenceType = InferenceType.VEHICLE
    attributes: str = "vehicle_attributes_2.2.csv"
    antialias: bool = True
    min_crop_size: Tuple[int, int] = (200, 200)


@dataclass
class VehicleCategory:
    name: str
    labels: List[str]
    high_conf: np.array
    med_conf: np.array
    low_conf: np.array


def norm_score(score: np.array, low: float, high: float) -> np.array:
    """maps score to 0-1 range, with low and high as 0.25 and 0.75 respectively"""
    return np.clip(((score - low) / (high - low)) * 0.5 + 0.25, 0, 1)


def norm_softmax(score: np.array, axis=0) -> np.array:
    return softmax(score - np.max(score, axis=axis))


class AttributeParser:
    def __init__(self, attribute_file: str, use_max: bool = True):
        self.categories = []
        categories_fields = defaultdict(list)
        with open(attribute_file, "rt") as f:
            table = csv.DictReader(f)
            for row in table:
                categories_fields[row["category"]].append(row)

        for category in categories_fields:
            labels = [row["label"] for row in categories_fields[category]]
            high_conf = np.array([float(row["high_conf"]) for row in categories_fields[category]])
            med_conf = np.array([float(row["med_conf"]) for row in categories_fields[category]])
            low_conf = np.array([float(row["low_conf"]) for row in categories_fields[category]])
            self.categories.append(VehicleCategory(category, labels, high_conf, med_conf, low_conf))
        if use_max:
            self.interpret_scores = self.parse_max
        else:
            self.interpret_scores = self.parse_all

    def parse_all(
        self, scores: Dict[str, np.array], priority, max_conf: AttrConfidence = AttrConfidence.HIGH
    ) -> Dict[str, List[AttrProperty]]:
        results = {}
        for cat in self.categories:
            cat_scores = scores[cat.name]
            high_confs = cat_scores >= cat.high_conf
            med_confs = (cat_scores >= cat.med_conf) & ~high_confs
            low_confs = (cat_scores >= cat.low_conf) & (cat_scores < cat.med_conf)
            high_idxs = np.where(high_confs)[0]
            med_idxs = np.where(med_confs)[0]
            low_idxs = np.where(low_confs)[0]
            attrs = []
            for i in high_idxs:
                sc = norm_score(cat_scores[i], cat.low_conf[i], cat.high_conf[i])
                attrs.append(AttrProperty(cat.labels[i], sc, min(AttrConfidence.HIGH, max_conf), priority))
            for i in med_idxs:
                sc = norm_score(cat_scores[i], cat.low_conf[i], cat.high_conf[i])
                attrs.append(AttrProperty(cat.labels[i], sc, min(AttrConfidence.MEDIUM, max_conf), priority))
            for i in low_idxs:
                sc = norm_score(cat_scores[i], cat.low_conf[i], cat.high_conf[i])
                attrs.append(AttrProperty(cat.labels[i], sc, AttrConfidence.LOW, priority))
            if attrs:
                results[cat.name] = attrs
        return results

    def parse_max(
        self, scores: Dict[str, np.array], priority, max_conf: AttrConfidence = AttrConfidence.HIGH
    ) -> Dict[str, List[AttrProperty]]:
        results = {}
        for cat in self.categories:
            cat_scores = scores[cat.name]
            mx_idx = np.argmax(cat_scores)
            score = cat_scores[mx_idx]
            if score > cat.low_conf[mx_idx]:
                sc = norm_score(score, cat.low_conf[mx_idx], cat.high_conf[mx_idx])
                if score >= cat.high_conf[mx_idx]:
                    conf = AttrConfidence.HIGH
                elif score >= cat.med_conf[mx_idx]:
                    conf = AttrConfidence.MEDIUM
                else:
                    conf = AttrConfidence.LOW
                results[cat.name] = [AttrProperty(cat.labels[mx_idx], sc, min(conf, max_conf), priority)]
        return results

    @property
    def attribute_len(self):
        return [len(cat.labels) for cat in self.categories]


def _load_attributes(args: BotWithClassifierConfig) -> AttributeParser:
    file_to_load = proj.info_path(args.name, args.attributes)
    if Path(args.attributes).exists():
        file_to_load = args.attributes
        logger.info(f"Loading attributes from {file_to_load}")
    elif Path(file_to_load).exists():
        logger.info(f"Loading attributes from {file_to_load}")
    else:
        file_to_load = proj.info_path(args.name, "vehicle_attributes.csv")
        logger.warning(f"Couldn't find requested file, Loading attributes from {file_to_load}")
    return AttributeParser(file_to_load, use_max=False)


class BoTWithClassifier(BoT):
    _config_type = BotWithClassifierConfig
    args: BotWithClassifierConfig
    categories: List[str]
    confidence_field = "confidence"

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        args = self._config_type(msg_dict)

        self.interpreter = _load_attributes(args)
        self.categories = [cat.name for cat in self.interpreter.categories]
        self.output_names = ["features"] + self.categories

        super().__init__(msg_dict, is_local)
        self.post_infer = self.attr_post_infer

    def _build_model(self):
        model = BaselineWithClassifier(
            self.args.last_stride,
            self.args.model_neck,
            self.args.neck_feat,
            self.args.model_name,
            self.args.feature_dim,
            self.interpreter.attribute_len,
        )
        model.to(self.args.device)
        return model

    def infer_full(self, crops: List[np.array]):
        crop_stack = torch.tensor(np.stack(crops, axis=0)).to(self.args.device)  # b,c,h,w
        with torch.no_grad():  # transform all crops
            feats, attrs = self.model(crop_stack)  # noqa
        return feats.cpu().numpy(), *[a.cpu().numpy() for a in attrs]

    def attr_post_infer(self, outputs, inputs):
        feats, attrs = outputs[0], outputs[1:]
        results = []
        feats_norm = self._normalize_output_func(feats, None)
        for idx, feat in enumerate(feats_norm):
            scores = {cat: attrs[i][idx] for i, cat in enumerate(self.categories)}
            scores[self.confidence_field] = np.array([np.max(norm_softmax(scores[cat])) for cat in self.categories])
            results.append({"descriptor": feat, "scores": scores})
        return results

    def _build_transform(self):
        interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR
        self.transform = alb.Compose(
            [
                alb.Normalize(self.args.mean, self.args.std),
                # Resize the longest side to image size, maintaining aspect ratio
                ResizeAndPadToTarget(
                    target_height=self.args.im_size[0], target_width=self.args.im_size[1], interpolation=interp_mode
                ),
            ]
        )

    def build_trt_model(self):
        super().build_trt_model()
        self._fix_output_order()

    def calculate_post_score(self, scores: Dict[str, np.array]) -> float:
        return np.mean(scores[self.confidence_field])
