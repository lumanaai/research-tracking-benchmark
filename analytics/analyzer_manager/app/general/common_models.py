from typing import Dict, Optional

import numpy as np
import torch
from torch import nn as nn

from general.inference import InferenceWrapper, BaseInferenceConfig


class BinaryNetConfig(BaseInferenceConfig):
    margins: float = 0.05  # margin to use when cropping (minus means dilation)
    classifier_threshold: float = 0  # threshold for which classification is considered True\False
    use_calibration: bool = True  # indication if to use calibration or not
    has_calibration: bool = True  # indication if to use calibration or not
    n_classes: int  # number of classes, determined by whether there is calibration or not

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        self.n_classes = 1 if self.has_calibration else 2


class CalibratedModel(nn.Module):
    def __init__(self, args: BinaryNetConfig):
        super(CalibratedModel, self).__init__()
        from torchvision import models

        if "resnet18" in args.weights:
            model = models.resnet18()
        elif "resnet34" in args.weights:
            model = models.resnet34()
        elif "resnet50" in args.weights:
            model = models.resnet50()
        else:
            raise NotImplementedError
        self.features = nn.Sequential(*list(model.children())[:-1])

        self.classifier = nn.Linear(model.fc.in_features, args.n_classes)
        self.calib_w = nn.Parameter(torch.tensor(1.0, requires_grad=False), requires_grad=False)
        self.calib_b = nn.Parameter(torch.tensor(0.0, requires_grad=False), requires_grad=False)

        if not args.use_calibration:
            self.forward = self.forward_uncalibrated

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        # Apply linear layers
        x1 = self.classifier(x)

        # Multiply by scalars and apply sigmoid
        x1 = torch.sigmoid(self.calib_w * x1 + self.calib_b)
        return x1

    def forward_uncalibrated(self, x):
        # uncalibrated
        x = self.features(x)
        x = torch.flatten(x, 1)

        # Apply linear layers
        return torch.sigmoid(self.classifier(x))


class CalibratedModelWrapper(InferenceWrapper):
    _config_type = BinaryNetConfig
    args: BinaryNetConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)
        if not self.args.has_calibration:
            self.post_infer = self.post_infer_uncalibrated

    def build_full_model(self):
        self.model = CalibratedModel(self.args)
        ckpt = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.args.half:
            self.model = self.model.half()
        self.model = self.model.eval()
        self.model.to(self.args.device)

    def post_infer(self, outputs, inputs):
        arr = np.squeeze(outputs, axis=1)
        preds = (arr > self.args.classifier_threshold).astype(int)
        scores = np.where(
            preds == 0,
            1 - (arr / self.args.classifier_threshold),
            (arr - self.args.classifier_threshold) / (1 - self.args.classifier_threshold),
        )
        return preds.tolist(), scores.tolist()

    def post_infer_uncalibrated(self, outputs, inputs):
        preds = np.argmax(outputs, axis=1)
        sorted_scores = np.sort(outputs, axis=1)
        scores = np.clip(
            np.squeeze((sorted_scores[:, -1] - sorted_scores[:, -2]) / self.args.classifier_threshold), 0, 1
        )
        return preds.tolist(), scores.tolist()

    @property
    def required_margins(self):
        return self.args.margins


class ClassificationConfig(BaseInferenceConfig):
    margins: float = 0.0  # margin to use when cropping (minus means dilation)
    n_classes: int = 0  # number of classes, determined by whether there is calibration or not
    use_calibration: bool = False  # indication if to use calibration or not
    has_calibration: bool = False  # indication if to use calibration or not
    antialias = True


class ClassificationWrapper(CalibratedModelWrapper):
    _config_type = ClassificationConfig
    args: ClassificationConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        # skip calibration stuff
        super(CalibratedModelWrapper, self).__init__(msg_dict, is_local)

    def post_infer(self, outputs, inputs):
        preds = np.argmax(outputs, axis=1)
        scores = outputs.max(axis=1)
        return preds.tolist(), scores.tolist()
