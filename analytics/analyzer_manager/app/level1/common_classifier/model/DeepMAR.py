from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torch.autograd import Variable

from .resnet import resnet50, resnet34, resnet18, resnet101


class DeepMar(nn.Module):
    def __init__(self, num_att: int, im_size: Tuple[int, int], has_calibration: bool = True, weights: str = "resnet50"):
        super(DeepMar, self).__init__()
        # init the necessary parameter for network structure
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop_pool5: bool = True
        self.drop_pool5_rate: float = 0.5
        self.use_calibration: bool = True
        self.last_conv_stride: int = 2
        self.image_size = im_size
        last_layer = 0
        if "resnet18" in weights:
            self.base = resnet18(pretrained=False, last_conv_stride=self.last_conv_stride)
            last_layer = 512
        elif "resnet34" in weights:
            self.base = resnet34(pretrained=False, last_conv_stride=self.last_conv_stride)
            last_layer = 512
        elif "resnet101" in weights:
            self.base = resnet101(pretrained=False, last_conv_stride=self.last_conv_stride)
        elif "resnext50_32x4d" in weights:
            from torchvision.models import resnext50_32x4d

            self.base = resnext50_32x4d(pretrained=False)
            self.base = torch.nn.Sequential(*list(self.base.children())[:-1])
        elif "efficientnet_b4" in weights:
            from torchvision.models import efficientnet_b4

            self.base = efficientnet_b4(pretrained=False)
            self.base = torch.nn.Sequential(*list(self.base.children())[:-1])
            last_layer = 1792
        else:  # default assume resnet50
            self.base = resnet50(pretrained=False, last_conv_stride=self.last_conv_stride)
            last_layer = 2048

        self.classifier = nn.Linear(last_layer, num_att)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

        if has_calibration:
            self.calib_w = nn.Parameter(torch.ones((1, num_att), requires_grad=False), requires_grad=False)
            self.calib_b = nn.Parameter(torch.zeros((1, num_att), requires_grad=False), requires_grad=False)
            self.calibrated = nn.Parameter(torch.tensor(False), requires_grad=False)
        else:
            self.calibrated = False

    def warmup(self, device, half: bool = False):
        b_size = 2  # batch size warmup
        runs = 5  # warmup runs

        print("Warming up...")
        warmup_image = torch.zeros(b_size, 3, self.image_size[0], self.image_size[1]).to(device)
        if half:
            warmup_image = warmup_image.half()
        for i in range(runs):
            self.forward(warmup_image)

    def forward(self, x):
        x = self.base(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.drop_pool5:
            x = functional.dropout(x, p=self.drop_pool5_rate, training=False)
        x = self.classifier(x)

        if self.calibrated and self.use_calibration:
            x = x * self.calib_w + self.calib_b
        return x


class DeepMarExtractFeature(object):
    """
    A feature extraction function
    """

    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, imgs):
        old_train_eval_model = self.model.training

        # set the model to be eval
        self.model.eval()

        # imgs should be Variable
        if not isinstance(imgs, Variable):
            print("imgs should be type: Variable")
            raise ValueError
        score = self.model(imgs)
        score = score.data.cpu().numpy()

        self.model.train(old_train_eval_model)

        return score
