import warnings
from typing import List, Dict, Optional

import numpy as np
import albumentations as alb
import cv2
from general.analyzer_general import InferenceType
from general.inference import BaseInferenceConfig, InferenceWrapper
from general.ort_utils import ort_type_to_numpy


class ResizeSeq:
    def __init__(self, resize_transform):
        self.resize_transform = resize_transform

    def __call__(self, sequance):
        """
        :param frames: array of shape [T*3, H, W, 3]
        :return: array of shape [T*3, H', W', 3]
        """
        return {"sequance": np.stack([self.resize_transform(image=frame)["image"] for frame in sequance])}


class GrayscaleAndPack:
    def __call__(self, sequance):
        """
        :param frames: array of shape [T*3, H, W, 3]
        :return: array of shape [T, H, W, 3]
        """
        # Convert to grayscale
        frames = np.stack(
            [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in sequance]
        )  # Convert each frame to grayscale

        # Reshape and stack frames into 3 channels
        T = frames.shape[0] // 3
        frames = frames.reshape(T, 3, frames.shape[1], frames.shape[2])  # [T, 3, H, W]
        return {"sequance": frames / 255.0}


class ViolenceDetectorConfig(BaseInferenceConfig):
    weights = "mobilenet_lstm_T5_v3_balanced.onnx"
    T = 5
    im_size = (224, 384)
    name: InferenceType = InferenceType.VIOLENCE
    max_dynamic_batch = 8
    half: bool = True


class ViolenceDetector(InferenceWrapper):
    _config_type = ViolenceDetectorConfig
    args: ViolenceDetectorConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)
        self.sequence_length = self.args.T * 3


    def build_full_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(self.args.weights, providers=["CUDAExecutionProvider"])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.args.half = ort_type_to_numpy[self.onnx_session.get_inputs()[0].type] is np.float16

        self.output_name = self.onnx_session.get_outputs()[0].name
        self.output_names = [self.output_name]

    def infer_full(self, crops: List[np.array]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.onnx_session.run(
                self.output_names, {self.input_name: np.stack(crops).astype(self.data_type)}
            )
        return results[0]

    def _build_transform(self):
        interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR
        self.resize_transform = alb.Resize(self.args.im_size[-2], self.args.im_size[-1], interpolation=interp_mode)
        self.transform = alb.Compose([ResizeSeq(self.resize_transform), GrayscaleAndPack()])

    def transform_single_image(self, image, mem_ptr):
        resized = self.resize_transform(image=image)["image"]
        mem_ptr[:] = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(self.data_type) / self.data_type(255.0)
        return resized


    def prepare_crops(self, crop_list):
        crops = []
        for element in crop_list:
            if element.shape[0] != self.sequence_length:  # each elements should be with shape [T*3, H, W, 3]
                raise ValueError(f"Expected {self.sequence_length} frames, got {element.shape[0]}")
            transformed_img = self.transform(sequance=element)["sequance"]
            crops.append(transformed_img.astype(self.data_type))
        return crops

    def forward_on_crop_list(self, crop_list):  # each elements should be with shape [T*3, H, W, 3]
        crops = self.prepare_crops(crop_list)  # shape [B, T, 3, im_size, im_size]
        feats = self.infer(crops)
        return self.post_infer(feats, crop_list)

    def forward_on_sequence(self, sequence_list):
        feats = self.infer(sequence_list)
        return self.post_infer(feats, sequence_list)

    @property
    def export_is_half(self):
        return True

    @property
    def inference_buffer_size(self):
        return [self.args.T, 3, self.args.im_size[-2], self.args.im_size[-1]]
