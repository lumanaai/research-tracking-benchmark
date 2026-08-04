from collections import deque
from typing import Dict, Optional, List, Tuple

import albumentations as alb
import cv2
import numpy as np

from .analyzer_general import InferenceType, log_exception, logger
from .img_utils import LowerRightCrop
from .inference import BaseInferenceConfig, InferenceWrapper
from .offline_analytics import check_expert_availability, OfflineAnalyticsClient, OfflineAnalyticsClientFactory

models = {
    "ViT-B-32": "laion2b_s34b_b79k",
    "ViT-B-16": "laion2b_s34b_b88k",
    "ViT-L-14": "laion2b_s32b_b82k",
    "ViT-H-14": "laion2b_s32b_b79k",
    "ViT-g-14": "laion2b_s34b_b88k",
    "ViT-H-14-CLIPA-336": "datacomp1b",
    "ViT-bigG-14-CLIPA": "datacomp1b",
    "ViT-bigG-14-CLIPA-336": "datacomp1b",
    "convnext_base_w": "laion2b_s13b_b82k_augreg",
    "convnext_large_d": "laion2b_s26b_b102k_augreg",
    "convnext_large_d_320": "laion2b_s29b_b131k_ft_soup",
    "convnext_xxlarge": "laion2b_s34b_b82k_augreg_soup",
    "EVA02-B-16": "merged2b_s8b_b131k",
    "EVA02-L-14": "merged2b_s4b_b131k",
    "EVA02-L-14-336": "merged2b_s6b_b61k",
    "EVA01-g-14-plus": "merged2b_s11b_b114k",
    "EVA02-E-14": "laion2b_s4b_b115k",
    "EVA02-E-14-plus": "laion2b_s9b_b144k",
    "ViT-B-16-SigLIP": "webli",
    "ViT-B-16-SigLIP-256": "webli",
    "ViT-B-16-SigLIP-384": "webli",
    "ViT-B-16-SigLIP-512": "webli",
    "coca_ViT-B-32": "laion2b_s13b_b90k",
    "coca_ViT-L-14": "laion2b_s13b_b90k",
}


class EncoderConfig(BaseInferenceConfig):
    weights: str = "clip_ViT-B-16-SigLIP2-wise.pt"  # path for builtin weight file
    model_arch: str = "ViT-B-16-SigLIP2"
    vector_size: int = 768
    logit_scale_exp = 106.1456

    im_size = (224, 224)
    name = InferenceType.CLIP_ENCODER
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]


class ClipVisionEncoder(InferenceWrapper):
    _config_type = EncoderConfig
    args: EncoderConfig
    full_model = None

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super(ClipVisionEncoder, self).__init__(msg_dict, is_local=is_local)
        self.post_infer = self._normalize_output_func
        self.aspect_ratio = self.args.im_size[1] / self.args.im_size[0]

    def _build_transform(self):
        # assume resize will be done per application, so the basic is to just crop the center
        self.transform = alb.Compose(
            [
                # Resize the longest side to image size, maintaining aspect ratio
                # alb.SmallestMaxSize(max_size=max(self.args.im_size), interpolation=cv2.INTER_AREA),
                alb.CenterCrop(height=self.args.im_size[0], width=self.args.im_size[1]),
                alb.Normalize(self.args.mean, self.args.std),
            ]
        )

    def build_full_model(self):
        # Load pre-trained backbone and remove the last layer
        import open_clip
        import torch

        class EncodeImageModule(torch.nn.Module):
            def __init__(self, base_model):
                super(EncodeImageModule, self).__init__()
                self.base_model = base_model

            def forward(self, x):
                return self.base_model.encode_image(x)

        self.full_model = open_clip.create_model(
            self.args.model_arch, pretrained=self.args.weights, precision="fp32", device=self.args.device
        )

        # carefully convert to half
        if self.args.half:
            for name, param in self.full_model.named_parameters():
                with torch.no_grad():
                    data = param.data
                    subnormals = (data.abs() < 6e-5) & (data != 0)
                    if subnormals.any():
                        data[subnormals] = torch.sign(data[subnormals]) * 6e-5

            self.full_model = self.full_model.half()

        self.model = EncodeImageModule(self.full_model)

    def _infer_full(self, crops: List[np.array]):
        import torch

        crop_stack = torch.tensor(np.stack(crops, axis=0)).to(self.args.device)  # b,c,h,w
        with torch.no_grad():  # transform all crops
            feats = self.model.encode_image(crop_stack)  # noqa
        return feats.cpu().numpy()

    def forward_on_prepared_crops(self, crop_list):
        crops = []
        for element in crop_list:
            crops.append(np.transpose(element, (2, 0, 1)).astype(self.data_type))
        feats = self.infer(crops)
        return self.post_infer(feats, crop_list)

    @property
    def descriptor_size(self):
        return self.args.vector_size

    @property
    def logit_scale_exp(self):
        return self.args.logit_scale_exp

    @staticmethod
    def match_encoding_list(encoding, query):
        results = []
        for vec in query:
            results.append(np.dot(encoding, vec))
        return np.array(results)

    @staticmethod
    def match_encoding(encoding, query):
        return np.dot(encoding, query)

    def encode_text(self, prompts: List[str]):
        import torch
        import open_clip

        if self.full_model is None:
            raise NotImplementedError("Requires full model")
        tokenizer = open_clip.get_tokenizer(self.args.model_arch)
        tokens = tokenizer(prompts).to(self.args.device)
        with torch.no_grad():
            vectors = self.full_model.encode_text(tokens).to("cpu").numpy()
        return self._normalize_output_func(vectors, None)


class ClipDispatcher:

    _crops_enabled: bool = False
    _thumbs_enabled: bool = False
    use_offline = True
    offline_errors = 0
    offline_client: Optional[OfflineAnalyticsClient] = None

    def __init__(self, config: Dict, context=None):
        self.enabled = config.get("enable", True)
        if not self.enabled:
            self.pop_descriptors = lambda: []
            self.encode_object_crops = lambda x: []
            self.encode_thumbnail_async = lambda x, y, immediate: None
            return

        self.config = config
        self.encoder = ClipVisionEncoder(config)
        self.thumbnails_queue: List = []
        self.descriptors_queue: List = []
        self.last_descriptors = deque(maxlen=10)
        self.waiting_request_ts: Optional[int] = None
        self.process_mode = self.config.get("preprocess_thumbnail_mode", "zoom_pad")
        self.desc_per_thumbnail = min(self.config.get("descriptors_per_thumbnail", 1), 3)
        self.min_thumb_for_process = self.config.get("min_thumbnail_for_process", 3)
        self.thum_croppers = []
        im_sz = self.encoder.args.im_size

        self.context = context
        try:
            offline_analytics_settings = self.context.app_config.get("analytics", {}).get("offlineAnalyticsUri", {})
            expert_url = check_expert_availability(offline_analytics_settings)
            self.offline_client = OfflineAnalyticsClientFactory.get_client(expert_url, self.context.camera_id)
        except Exception as e:
            self.use_offline = False
            log_exception(logger, "could not use offline analytics", e)

        normalizer = alb.Normalize(self.encoder.args.mean, self.encoder.args.std)
        crop_resizer = alb.LongestMaxSize(max_size=max(im_sz), interpolation=cv2.INTER_AREA)
        crop_pad = alb.PadIfNeeded(min_height=im_sz[0], min_width=im_sz[1], border_mode=0, value=(0, 0, 0))
        if self.process_mode == "zoom":
            resizer = alb.SmallestMaxSize(max_size=max(im_sz), interpolation=cv2.INTER_AREA)
            thumb_crop_locs = [self.desc_per_thumbnail] if self.desc_per_thumbnail < 3 else [1, 2]
            if 1 in thumb_crop_locs:
                self.thum_croppers.append(alb.CenterCrop(height=im_sz[0], width=im_sz[1]))
            if 2 in thumb_crop_locs:
                self.thum_croppers += [
                    alb.Crop(x_min=0, y_min=0, x_max=im_sz[1], y_max=im_sz[0], always_apply=True),
                    LowerRightCrop(target_width=im_sz[1], target_height=im_sz[0], always_apply=True),
                ]
        elif self.process_mode == "zoom_pad":
            self.desc_per_thumbnail = 1
            resizer = alb.Compose(
                [alb.LongestMaxSize(max_size=int(max(im_sz) * 1.2), interpolation=cv2.INTER_AREA), crop_pad]
            )
            self.thum_croppers = [alb.CenterCrop(height=im_sz[0], width=im_sz[1])]
        else:  # self.process_mode == "pad":
            resizer = crop_resizer
            self.thum_croppers = [crop_pad]
        self.thumb_transform = alb.Compose([resizer, normalizer])
        self.crop_transform = alb.Compose([crop_resizer, normalizer, crop_pad])
        self.crop_resizer = alb.Compose([crop_resizer, crop_pad])
        self.set_crops_enabled(self.config.get("encode_crops", False))
        self.set_thumbs_enabled(self.config.get("encode_thumbnails", True))

    def _encode_thumbnail_async(self, thumbnails: np.ndarray, timestamp: int, immediate: bool = False):
        norm_resized = self.thumb_transform(image=thumbnails)["image"]
        crops = [cropper(image=norm_resized)["image"] for cropper in self.thum_croppers]
        self.thumbnails_queue += [(crop, timestamp) for crop in crops]
        if immediate or len(self.thumbnails_queue) > self.min_thumb_for_process:
            crops, tss = zip(*self.thumbnails_queue)
            encodings = self.encoder.forward_on_prepared_crops(crops)
            self._processed_thumbnails_encoding(encodings, tss)

    def _encode_object_crops(self, crop_list: List[np.array]) -> List[np.array]:
        crops = [self.crop_transform(image=crop)["image"] for crop in crop_list]
        if self.thumbnails_queue:
            obj_crops_n = len(crops)
            thumb_crops, timestamps = zip(*self.thumbnails_queue)
            crops += thumb_crops
            encodings = self.encoder.forward_on_prepared_crops(crops)
            self._processed_thumbnails_encoding(encodings[obj_crops_n:], timestamps)
            return encodings[:obj_crops_n]
        else:
            return self.encoder.forward_on_prepared_crops(crops)

    def encode_object_crop_async(
        self, crop: np.ndarray, id_base, id_index, immediate: bool = False, extra: Dict = None
    ) -> bool:
        # crop should already be in the right size
        if not self.use_offline:
            return False

        return self.offline_client.send_clip_request(crop, id_base, id_index, immediate, extra)

    def pop_clip_object_async(self) -> List[Dict]:
        if not self.use_offline:
            return []

        encodings = self.offline_client.get_clip_encodings()
        if encodings is None:
            # meaning there was an error
            self._log_and_check_offline_error()
            return []
        self.offline_errors = 0
        return encodings

    def _log_and_check_offline_error(self):
        self.offline_errors += 1
        if self.offline_errors > 3:
            logger.error("Failed to get clip encodings from offline analytics server. Disabling Clip on Objects")
            self.use_offline = False

    def _processed_thumbnails_encoding(self, encodings: np.ndarray, timestamps: List[int]):
        combined = [(encoding, ts) for encoding, ts in zip(encodings, timestamps)]
        self.descriptors_queue += combined
        self.thumbnails_queue.clear()
        for item in combined:
            self.last_descriptors.append(item)

    def _pop_descriptors(self) -> List[Tuple[np.array, int]]:
        descriptors = self.descriptors_queue
        self.descriptors_queue = []
        return descriptors

    @property
    def aspect_ratio(self):
        return self.encoder.aspect_ratio

    @property
    def logit_scale_exp(self):
        return self.encoder.logit_scale_exp

    @property
    def crops_enabled(self) -> bool:
        return self._crops_enabled

    @property
    def thumbs_enabled(self) -> bool:
        return self._thumbs_enabled

    @property
    def image_shape(self) -> List[int]:
        return list(self.encoder.args.im_size) + [3]

    def set_crops_enabled(self, crops_enable: bool):
        if self.enabled and crops_enable:
            self._crops_enabled = True
            self.encode_object_crops = self._encode_object_crops
        else:
            self.encode_object_crops = lambda x: []
            self._crops_enabled = False

    def set_thumbs_enabled(self, thumbs_enabled: bool):
        if self.enabled and thumbs_enabled:
            self.encode_thumbnail_async = self._encode_thumbnail_async
            self.pop_descriptors = self._pop_descriptors
            self._thumbs_enabled = True
        else:
            self.pop_descriptors = lambda: []
            self.encode_thumbnail_async = lambda x, y, immediate: None
            self._thumbs_enabled = False
