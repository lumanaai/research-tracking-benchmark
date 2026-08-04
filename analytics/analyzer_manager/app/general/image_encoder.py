from math import floor
from typing import Dict, Optional

import numpy as np
from .analyzer_general import InferenceType
from .inference import BaseInferenceConfig, InferenceWrapper


class EncoderConfig(BaseInferenceConfig):
    weights: str = "resnet18.pt"  # path for builtin weight file
    vector_size: Optional[int] = None
    name = InferenceType.IMAGE_ENCODER


class ImageEncoder(InferenceWrapper):
    _config_type = EncoderConfig
    args: EncoderConfig

    _feature_length: int = 128
    stride: int = 1

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super(ImageEncoder, self).__init__(msg_dict, is_local=is_local)

        self.post_infer = self._normalize_output_func
        # set the output vector size
        self.stride = 1

        # warm up

        dummy_input = np.zeros((300, 300, 3))
        res = self.forward_on_crop_list([dummy_input])
        model_out_size = res.shape[1]

        if self.args.vector_size is not None:
            self.stride = floor(model_out_size / self.args.vector_size)
        self._feature_length = int(model_out_size / self.stride)

    def build_full_model(self):
        import torch
        import torch.nn as nn
        import torchvision.models as models

        # Load pre-trained backbone and remove the last layer
        if "resnet18" in self.args.weights:
            backbone = models.resnet18()
        elif "efficientnet_b0" in self.args.weights:
            backbone = models.efficientnet_b0()
        elif "mobilenet_v3" in self.args.weights:
            backbone = models.mobilenet_v3_small()
        elif "squeezenet1_1" in self.args.weights:
            backbone = models.squeezenet1_1()

        else:
            raise NotImplementedError
        ckpt = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        backbone.load_state_dict(ckpt)
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        model = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        model.eval()
        model.to(self.args.device)
        if self.args.half:
            model = model.half()
        self.model = model

    @property
    def feature_length(self):
        return self._feature_length

    @property
    def image_size(self):
        return self.args.im_size

    def encode_patch(self, image_patch):
        enc = self.forward_on_crop_list([image_patch])
        return np.squeeze(enc)

    @staticmethod
    def match_encoding_list(encoding, query):
        results = []
        for vec in query:
            results.append(np.dot(encoding, vec))
        return np.array(results)

    @staticmethod
    def match_encoding(encoding, query):
        return np.dot(encoding, query)

