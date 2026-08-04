import warnings
from typing import List, Dict, Optional

import albumentations as alb
import cv2
import numpy as np

from general.analyzer_general import InferenceType, logger
from general.inference import BaseInferenceConfig, InferenceWrapper
from general.ort_utils import ort_type_to_numpy
from level1.violence.mobilenet import MobilenetLSTM
from level1.violence.violence_detector import GrayscaleAndPack, ResizeSeq


# ============================================================================
# YOLO Violence Detector (Triplet-frame based)
# ============================================================================


class ViolenceDetectorConfig(BaseInferenceConfig):
    weights = "yolov8s-violence_eff_full_union_triplet_w.onnx"
    im_size = (384, 640)
    name: InferenceType = InferenceType.VIOLENCE_DETECT
    half: bool = True
    max_dynamic_batch = 2
    min_conf: float = 0.3


class ViolenceDetector(InferenceWrapper):
    """
    YOLO-based violence detector that operates on triplet frames.
    Each input is a 3-channel image where each channel is a grayscale frame
    sampled at (t - offset, t, t + offset), encoding temporal motion.
    Outputs bounding boxes of detected violent interactions.
    """

    _config_type = ViolenceDetectorConfig
    args: ViolenceDetectorConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)

    def build_trt_model(self):
        from general.trt_utils import TrtInfer

        self.model = TrtInfer(self.args.weights, input_name="images")
        self.args.im_size = self.model.im_size
        self.args.half = self.model.is_half

    def build_full_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(self.args.weights, providers=["CUDAExecutionProvider"])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.args.half = ort_type_to_numpy[self.onnx_session.get_inputs()[0].type] is np.float16

        self.output_names = [o.name for o in self.onnx_session.get_outputs()]

    def infer_full(self, crops: List[np.ndarray]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.onnx_session.run(
                self.output_names, {self.input_name: np.stack(crops).astype(self.data_type)}
            )
        return results[0]

    def _build_transform(self):
        h, w = self.args.im_size
        self.resize_transform = alb.Resize(h, w, interpolation=cv2.INTER_LINEAR)

    def transform_single_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = self.resize_transform(image=gray)["image"].astype(self.data_type) / self.data_type(255.0)
        return resized

    def prepare_crops(self, crop_list):
        """Transform a list of frame triplets into model-ready inputs.
        Each element: list/array of 3 BGR frames -> [3, H, W] grayscale."""
        crops = []
        for triplet in crop_list:
            channels = np.stack([self.transform_single_image(frame) for frame in triplet])
            crops.append(channels)
        return crops

    def forward_on_sequence(self, sequence_list):
        feats = self.infer(sequence_list)
        return self.post_infer(feats, sequence_list)

    def post_infer(self, outputs, inputs):
        """
        Parse raw YOLO output into a list of detections.
        Returns list of (x1, y1, x2, y2, confidence) arrays, one per batch element.
        """
        # Adjust parsing based on actual export format.
        if outputs.ndim == 3 and outputs.shape[1] < outputs.shape[2]:
            # Transposed format [batch, 5, num_preds] -> [batch, num_preds, 5]
            outputs = np.transpose(outputs, (0, 2, 1))

        batch_detections = []
        for preds in outputs:
            # preds: [num_preds, 5+] where columns are [cx, cy, w, h, conf, ...]
            if preds.shape[-1] >= 5:
                conf = preds[:, 4]
                mask = conf > self.args.min_conf
                filtered = preds[mask]
                if len(filtered) > 0:
                    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    confs = filtered[:, 4]
                    dets = np.stack([x1, y1, x2, y2, confs], axis=-1)
                    batch_detections.append(dets)
                else:
                    batch_detections.append(np.empty((0, 5), dtype=preds.dtype))
            else:
                batch_detections.append(np.empty((0, 5), dtype=preds.dtype))
        return batch_detections


# ============================================================================
# MobilenetLSTM Violence Classifier V2 (crop-based, ImageNet-normalized)
# ============================================================================


class ViolenceClassifierConfig(BaseInferenceConfig):
    weights = "mobilenet_lstm_on_det_crop_eff_384_384_v1.pt"
    T = 5
    im_size = (384, 384)
    name: InferenceType = InferenceType.VIOLENCE
    max_dynamic_batch = 2
    half: bool = True
    # CLIP normalization
    mean: List[float] = [0.48145466, 0.4578275, 0.40821073]
    std: List[float] = [0.26862954, 0.26130258, 0.27577711]
    min_conf: float = 0.5


class ViolenceClassifier(InferenceWrapper):
    """
    MobilenetLSTM-based violence classifier (V2).
    Operates on detection crops rather than full-scene frames.
    Input: 15 frames cropped to the union bbox of detections,
           packed into [T=5, 3, H, W] grayscale with CLIP normalization.
    Output: scalar violence score.
    """

    _config_type = ViolenceClassifierConfig
    args: ViolenceClassifierConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        logger.info("Initializing Violence Classifier...")
        super().__init__(msg_dict, is_local)
        self.sequence_length = self.args.T * 3

    def build_full_model(self):
        import torch

        logger.info("Building MobilenetLSTM violence classifier model...")
        self.model = self._build_model()
        state_dict = torch.load(self.args.weights, map_location=self.args.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if self.args.half:
            self.model = self.model.half()
            for idx, layer in enumerate(self.model.modules()):
                if "Norm" in layer.__class__.__name__:
                    layer.float()
        self._warmup()

    def _build_model(self):
        model = MobilenetLSTM(num_classes=1, seq_length=self.args.T)
        model.to(self.args.device)
        return model

    def _warmup(self):
        import torch

        logger.info("Warming up Violence Classifier...")
        warm_up_batch_size = 2
        dummy = torch.zeros(
            (warm_up_batch_size, self.args.T, 3, self.args.im_size[-2], self.args.im_size[-1]),
            device=self.args.device,
        )
        if self.args.half:
            dummy = dummy.half()
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy)

    def _build_transform(self):
        interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR
        self.resize_transform = alb.Resize(self.args.im_size[-2], self.args.im_size[-1], interpolation=interp_mode)
        self.transform = alb.Compose([ResizeSeq(self.resize_transform), GrayscaleAndPack()])

    def transform_single_image(self, image: np.ndarray, mem_ptr: np.ndarray, channel_idx: int = -1):
        """
        Resize and grayscale a single frame into the pre-allocated buffer.

        Args:
            image: RGB frame
            mem_ptr: target slice of the buffer [H, W]
            channel_idx: which of the 3 packed channels (0/1/2) this frame maps to.
                         Applies per-channel ImageNet normalization inline.
                         If -1, stores raw /255 without normalization.
        """
        if image.shape[:2] != tuple(self.args.im_size[-2:]):
            resized = self.resize_transform(image=image)["image"]
        else:
            resized = image
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(self.data_type) / self.data_type(255.0)
        if 0 <= channel_idx <= 2:
            mean = self.data_type(self.args.mean[channel_idx])
            std = self.data_type(self.args.std[channel_idx])
            mem_ptr[:] = (gray - mean) / std
        else:
            mem_ptr[:] = gray
        return mem_ptr

    def prepare_crops(self, crop_list):
        """Transform a list of frame sequences into model-ready inputs.
        Each element: array of shape [T*3, H, W, 3] (BGR frames) -> [T, 3, H, W] normalized."""
        crops = []
        for element in crop_list:
            if element.shape[0] != self.sequence_length:
                raise ValueError(f"Expected {self.sequence_length} frames, got {element.shape[0]}")
            transformed = self.transform(sequance=element)["sequance"]  # [T, 3, H, W], /255
            for c in range(3):
                mean = self.data_type(self.args.mean[c])
                std = self.data_type(self.args.std[c])
                transformed[:, c] = (transformed[:, c] - mean) / std
            crops.append(transformed.astype(self.data_type))
        return crops

    def forward_on_sequence(self, sequence_list):
        feats = self.infer(sequence_list)
        return self.post_infer(feats, sequence_list)

    def post_infer(self, outputs, inputs):
        scores = []
        for logit in outputs.reshape(-1):
            score = float(logit)
            if score > self.args.min_conf:
                scores.append(score)
        return scores

    @property
    def export_is_half(self):
        return True

    @property
    def inference_buffer_size(self):
        return [self.args.T, 3, self.args.im_size[-2], self.args.im_size[-1]]
