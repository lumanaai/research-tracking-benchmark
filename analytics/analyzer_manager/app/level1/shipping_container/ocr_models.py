"""
Container OCR model wrapper following the InferenceWrapper pattern.

Provides ContainerOcrConfig + ContainerOcr, analogous to
level1/license_plate/lp_recognition.py and level1/pa_recognition/ppe.py.

The underlying model is a PaddleOCR PP-OCRv5 recognition net exported to ONNX,
served via TensorRT (.plan) in production or ONNX fallback locally.
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from general import proj
from general.analyzer_general import InferenceType, logger
from general.inference import InferenceWrapper, BaseInferenceConfig
from general.ort_utils import ort_type_to_numpy
from level1.license_plate.PaddleOCR2Pytorch.pytorchocr.postprocess.rec_postprocess import CTCLabelDecode


class ContainerOcrConfig(BaseInferenceConfig):
    """Config for container OCR recognition model."""
    weights: str = "container_ocr.onnx"
    name: InferenceType = InferenceType.CONTAINER_OCR
    max_dynamic_batch: int = 4
    im_size: Tuple[int, int] = (48, 320)
    rec_char_dict_path: str = "char_dict.txt"   # path to char dict for CTC decoding (digits + uppercase)
    use_space_char: bool = False                # if True, adds space char to dict for CTC decoding
    rotate_vertical_crops: bool = True          # whether to rotate tall crops 90 CW before OCR
    vertical_crop_ratio_th: float = 1.5         # height/width ratio threshold for vertical crop rotation

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        if self.rec_char_dict_path:
            self.rec_char_dict_path = proj.info_path(self.name, self.rec_char_dict_path)


class ContainerOcr(InferenceWrapper):
    """
    Container OCR recognition model following InferenceWrapper pattern.

    Preprocessing: aspect-ratio-preserving resize + normalize to [-1, 1] + zero-pad.
    Inference: ONNX / TRT / Triton (auto-selected by InferenceWrapper).
    Postprocessing: CTC greedy decode via CTCLabelDecode.
    """
    _config_type = ContainerOcrConfig
    args: ContainerOcrConfig

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        self.args = self._config_type(msg_dict)
        self._ctc_decoder = CTCLabelDecode(
            character_dict_path=self.args.rec_char_dict_path or None,
            use_space_char=self.args.use_space_char,
        )
        super().__init__(msg_dict, is_local)

    def build_full_model(self):
        """Load ONNX model via onnxruntime."""
        import onnxruntime as ort

        logger.info(f"Container OCR: loading ONNX weights from {self.args.weights}")
        with warnings.catch_warnings():
            ort.set_default_logger_severity(3)
            warnings.simplefilter("ignore")
            _providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
            self.ort_session = ort.InferenceSession(self.args.weights, providers=_providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            inp_info = self.ort_session.get_inputs()[0]
            self.args.half = ort_type_to_numpy.get(inp_info.type, np.float32) is np.float16
            # Auto-detect im_size from ONNX input if shape is static
            if inp_info.shape and all(isinstance(d, int) for d in inp_info.shape):
                _, c, h, w = inp_info.shape
                self.args.im_size = (h, w)

    def build_trt_model(self):
        super().build_trt_model()

    def infer_full(self, crops: List[np.ndarray]):
        """Run ONNX inference on a batch of preprocessed crops."""
        batch = np.stack(crops, axis=0).astype(self.data_type, copy=False)
        ort_inputs = {self.input_name: batch}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs[0]

    # --- Preprocessing ---

    def _build_transform(self):
        """Override default albumentations transform (not used for OCR)."""
        pass

    def prepare_crops(self, crop_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Resize crops preserving aspect ratio, normalize to [-1, 1], pad to fixed width.
        Applies vertical rotation for tall crops before resizing.
        """
        imgH, imgW = self.args.im_size
        preprocessed = []
        for crop in crop_list:
            if crop is None or crop.size == 0:
                preprocessed.append(np.zeros((3, imgH, imgW), dtype=np.float32))
                continue
            oriented = self._maybe_rotate(crop)
            norm_img = self._resize_norm_img(oriented, imgH, imgW)
            preprocessed.append(norm_img)
        return preprocessed

    def _maybe_rotate(self, crop: np.ndarray) -> np.ndarray:
        """Rotate tall crops 90 CCW to make them horizontal for OCR."""
        if not self.args.rotate_vertical_crops:
            return crop
        h, w = crop.shape[:2]
        if w > 0 and h / w >= self.args.vertical_crop_ratio_th:
            return np.rot90(crop, k=1)
        return crop

    @staticmethod
    def _resize_norm_img(img: np.ndarray, imgH: int, imgW: int) -> np.ndarray:
        """PaddleOCR-style resize: keep aspect ratio, normalize [-1, 1], zero-pad."""
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = min(imgW, max(16, math.ceil(imgH * ratio)))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32").transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        return padding_im

    # --- Postprocessing ---

    def post_infer(self, model_result: np.ndarray, crops: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """CTC greedy decode -> list of (text, confidence) pairs."""
        results = self._ctc_decoder(model_result)
        texts = [r[0] for r in results]
        scores = [float(r[1]) for r in results]
        return texts, scores
