from typing import Tuple, Optional, Dict

import torch
from general.analyzer_general import InferenceType, logger
from general.inference import BaseInferenceConfig, InferenceWrapper
from .model import Baseline


class BotConfig(BaseInferenceConfig):
    weights: str = ""  # path for builtin weight file
    reid_version = 0
    im_size: Tuple[int, int] = (256, 128)
    half: bool = True  # default with false for compatability with xavier
    last_stride: int = 1  # last convolution stride
    model_name: str = "resnet50_ibn_a"
    pretrain_path: Optional[str] = ""
    model_neck: str = "bnneck"
    neck_feat: str = "after"
    model_pretrain_choice = "self"
    feature_dim: int = 512
    thresh_conf: float = 0.3  # confidence threshold
    name: InferenceType = InferenceType.PERSON_REID



class BoT(InferenceWrapper):
    _config_type = BotConfig
    args: BotConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        logger.info("Initializing ReID with model...")
        super(BoT, self).__init__(msg_dict, is_local)
        self.post_infer = self._normalize_output_func

    def build_full_model(self):
        logger.info("Initializing preproccessing for ReID BoT model...")
        self.model = self._build_model()
        self.model.load_param(self.args.weights)
        self.model.eval()
        if self.args.half:
            self.model = self.model.half()
            for idx, layer in enumerate(self.model.modules()):
                if 'Norm' in layer.__class__.__name__:
                    logger.info(f"Changing layer number {idx}: {layer.__class__.__name__} to FP32")
                    layer.float()  # to FP32
        self._warmup()

    def _warmup(self):
        logger.info("Warming up ReID BoT model...")
        warm_up_batch_size = 2
        images = torch.zeros(
            (warm_up_batch_size, 3, self.args.im_size[0], self.args.im_size[1]),
            device=self.args.device,
        )
        if self.args.half:
            images = images.half()
        for i in range(5):
            self.model(images)

    def _build_model(self):
        model = Baseline(
            self.args.last_stride,
            self.args.model_neck,
            self.args.neck_feat,
            self.args.model_name,
            self.args.feature_dim
        )
        model.to(self.args.device)
        return model

    @property
    def version(self):
        return self.args.reid_version