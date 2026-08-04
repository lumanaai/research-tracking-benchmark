import os
from collections import OrderedDict
from typing import List, Tuple, Type, Dict, Optional

import albumentations as alb
import cv2
import numpy as np

from . import proj
from .analyzer_general import logger, InferenceType, DEFAULT_TRT_MAX_BATCH_ALLOWED
from .core import BaseConfig, WorkItem, EndpointInfo, EndpointFactory


class BaseInferenceConfig(BaseConfig):
    weights: str = ""  # path for builtin weight file
    unique_weights: str = ""  # path for specific external weight file
    device: str = "cuda"  # device to load weights to
    half: bool = True  # use fp 16
    is_trt: bool = False  # use tensor-rt for inference
    im_size: Tuple[int, int] = (224, 224)
    mean: List[float] = [0.485, 0.456, 0.406]  # normalizing factors
    std: List[float] = [0.229, 0.224, 0.225]  # normalizing factors
    enable: bool = True  # is model enabled
    name: InferenceType = InferenceType.NONE  # name of the inference engine, should be InferenceType
    endpoint_model: str = None  # optional name of the remote endpoint model, default is name if kept None
    antialias: bool = True  # Anti aliasing when down sampling
    force_full: bool = False  # force full model inference
    max_dynamic_batch: int = DEFAULT_TRT_MAX_BATCH_ALLOWED  # max dynamic batch size
    is_bgr: bool = False  # input images are in RGB format

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        self.init_weight()

    def init_weight(self, resource_dir: str = None):
        if resource_dir is None:
            name = self.name.split("_")[0]
            resource_dir = proj.local_weights_path(name)
        default_weights = os.path.join(resource_dir, self.weights)
        if self.unique_weights:
            self.weights = self.unique_weights
        elif not self.force_full:  # default weights
            engine_path = os.path.join(resource_dir, "model.plan")
            if os.path.exists(engine_path):
                self.weights = engine_path

        if not os.path.exists(self.weights):
            # logger.error(f"Could not find weights {self.weights}, reverting to default weights {default_weights}")
            self.weights = default_weights
        # logger.info(f"using weights file {self.weights}")
        self.is_trt = self.weights.endswith(".engine") or self.weights.endswith(".plan")

        if self.endpoint_model is None:
            self.endpoint_model = self.name


class InferenceWrapper:
    _config_type: Type = BaseInferenceConfig
    remote_info: EndpointInfo
    model = None
    output_names: List = []
    input_name: str = None
    is_dynamic = True

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        logger.info("Getting arguments...")
        self.args: BaseInferenceConfig = self._config_type(msg_dict)
        self.remote_info = EndpointFactory().create(self.args.endpoint_model)

        if is_local is False or (is_local is None and proj.load_bool_from_env("TRITON_ENABLED", True)):
            from .triton_utils import is_triton_available

            is_local = not is_triton_available(self.remote_info)
        else:
            is_local = True
        self.is_local = is_local

        if not self.args.enable:
            return
        if self.is_local:
            logger.info(f"building  model with weighs {self.args.weights}")
            if self.args.is_trt:
                self.build_trt_model()
                self.infer = self.model.infer
                self.is_dynamic = self.model.is_dynamic

            else:
                self.build_full_model()
                self.infer = self.infer_full
                self.is_dynamic = True
        else:
            logger.info(f"using remote server for inference with endpoint {self.remote_info.model_name}")
            self.build_triton_model()
            self.infer = self.model.infer
            self.is_dynamic = self.model.is_dynamic

        if self.args.half:
            self.data_type = np.float16
        else:
            self.data_type = np.float32

        self._build_transform()

    def build_trt_model(self):
        from general.trt_utils import TrtInfer

        self.model = TrtInfer(self.args.weights)
        self.args.im_size = self.model.im_size
        self.args.half = self.model.is_half

    def build_triton_model(self):
        from .triton_utils import TritonInfer

        self.model = TritonInfer(self.remote_info)
        self.args.im_size = self.model.im_size
        self.args.half = self.model.is_half

    def build_full_model(self):
        pass

    def _build_transform(self):
        interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR
        self.resize_transform = alb.Resize(self.args.im_size[0], self.args.im_size[1], interpolation=interp_mode)
        self.transform = alb.Compose([self.resize_transform, alb.Normalize(self.args.mean, self.args.std)])

    def apply_resize(self, image: np.array, target: np.array = None):
        resized_crop = self.resize_transform(image=image)["image"]
        if target is None:
            return resized_crop
        np.copyto(target, resized_crop)
        return target

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, work_batch: List[WorkItem]):
        crop_list = [item.image for item in work_batch]
        return self.forward_on_crop_list(crop_list)

    def prepare_crops(self, crop_list):
        crops = []
        for element in crop_list:
            transformed_img = self.transform(image=element)["image"]
            crops.append(np.transpose(transformed_img, (2, 0, 1)).astype(self.data_type))
        return crops

    def forward_on_crop_list(self, crop_list):
        crops = self.prepare_crops(crop_list)
        feats = self.infer(crops)
        return self.post_infer(feats, crop_list)

    def post_infer(self, outputs, inputs):
        return outputs

    def infer_full(self, crops: List[np.array]):
        import torch

        crop_stack = torch.tensor(np.stack(crops, axis=0)).to(self.args.device)  # b,c,h,w
        with torch.no_grad():  # transform all crops
            feats = self.model(crop_stack)  # noqa
        return feats.cpu().numpy()

    def _normalize_output_func(self, outputs, inputs):  # noqa
        # Calculate the L2 norm along axis=1
        norm = np.linalg.norm(outputs, ord=2, axis=-1, keepdims=True)

        # Normalize the array
        return outputs / norm

    def _fix_output_order(self):
        output_shapes = OrderedDict()
        for name in self.output_names:
            output_shapes[name] = self.model.output_shapes[name]
        self.model.output_shapes = output_shapes

    @property
    def export_is_half(self):
        return self.args.half
