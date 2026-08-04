import glob
from typing import List, Dict, Optional
import cv2
import numpy as np
import torch
import cupy as cp

from general import proj
from general.analyzer_general import InferenceType
from general.inference import BaseInferenceConfig, InferenceWrapper
from general.trt_utils import TrtInfer
from .models import vgg_model
from .utils import CTCLabelConverter, load_text_nn

language_models = {
    "english_g2": {
        "symbols": "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €",
        "characters": "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",  # noqa
    },
}

g2_network_params = {"input_channel": 1, "output_channel": 256, "hidden_size": 256}


def custom_mean(x):
    return x.prod() ** (2.0 / np.sqrt(len(x)))


class CrnnConfig(BaseInferenceConfig):
    weights: str = "crnn_english_g2.pt"
    name: InferenceType = InferenceType.CRNN
    half = False
    model_type = "english_g2"
    model_height = 64
    min_char_sz = 10
    language_model: Dict[str, Dict] = None
    separator_list: Dict = {}
    decoder = "greedy"
    beam_width = 5
    batch_size = -1
    allow_list = None
    block_list = None
    output_format = "standard"
    max_input_width = 640
    max_dynamic_batch = 16

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        self.language_model = language_models.get(self.model_type)


def input_width_to_output_width(input_width: int) -> int:
    return input_width // 4 - 1


class CrnnTrt(TrtInfer):  # not in use since using max size always
    def infer(self, data: List[np.array]):
        batch = len(data)
        w = data[0].shape[-1] // 4 - 1
        feats = super().infer(data)
        chars = feats.shape[-1]
        feats = np.reshape(feats.ravel()[: batch * w * chars], [batch, w, chars])
        return feats

    def run_engine(self, crop_stack: np.array):
        if self.is_dynamic:
            partial_array = self.bindings[self.input_name].data[
                : crop_stack.shape[0], : crop_stack.shape[1], : crop_stack.shape[2], : crop_stack.shape[3]
            ]
            cp.copyto(partial_array, cp.asarray(crop_stack))
            self.context.set_binding_shape(0, crop_stack.shape)
        else:
            self.bindings[self.input_name].data[:] = cp.asarray(crop_stack)
        self.context.execute_v2(list(self.binding_addrs.values()))
        return len(crop_stack)


class CrnnTextRecognition(InferenceWrapper):
    _config_type = CrnnConfig
    args: CrnnConfig

    _input_size = None

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):

        self.args = self._config_type(msg_dict)
        character = self.args.language_model["characters"]
        dictionaries = glob.glob(proj.info_path("text", "*.txt"))
        dict_list = {}
        for i, d in enumerate(dictionaries):
            dict_list[i] = d
        self.converter = CTCLabelConverter(character, separator_list={}, dict_pathlist=dict_list)
        self.character = self.converter.character
        self.num_classes = len(self.character)
        if self.args.allow_list:
            ignore_char = "".join(set(self.character) - set(self.args.allow_list))
        elif self.args.block_list:
            ignore_char = "".join(set(self.args.block_list))
        else:
            ignore_char = ""
        self.ignore_idx = [self.character.index(char) for char in ignore_char if char in character]

        super().__init__(msg_dict, is_local)
        self.use_max_size = self.args.is_trt or not self.is_local

    def build_full_model(self):
        self.model = load_text_nn(
            vgg_model.Model(num_class=self.num_classes, **g2_network_params),
            self.args.weights,
            self.args.half,
            self.args.device,
        )

    @property
    def max_size_allowed(self):
        return {
            "input": [self.args.max_dynamic_batch, 1, self.args.model_height, self.args.max_input_width],
            "output": [
                self.args.max_dynamic_batch,
                input_width_to_output_width(self.args.max_input_width),
                len(self.character),
            ],
        }

    def build_trt_model(self):
        self.model = TrtInfer(self.args.weights, max_size_allowed=self.max_size_allowed)
        self.args.im_size = self.model.im_size
        self.args.half = self.model.is_half

    def build_triton_model(self):
        from general.triton_utils import TritonInfer

        self.model = TritonInfer(self.remote_info, max_size_allowed=self.max_size_allowed)
        self.args.im_size = self.model.im_size
        self.args.half = self.model.is_half

    def prepare_crops(self, crop_list):
        # assume the crops are already warped if necessary and aligned
        new_widths = []
        for crop in crop_list:
            aspect_ratio = crop.shape[1] / crop.shape[0]
            resized_width = int(self.args.model_height * aspect_ratio)
            new_widths.append(resized_width)
        max_width = self.args.max_input_width if self.use_max_size else np.max(new_widths)
        crops = []
        # alpha = 1.3  # Contrast control (1.0-3.0)
        # beta = 20  # Brightness control (0-100)
        for i, crop in enumerate(crop_list):
            resized_width = new_widths[i]
            resized_crop = cv2.resize(crop, (resized_width, self.args.model_height))
            # adjusted = cv2.convertScaleAbs(resized_crop, alpha=alpha, beta=beta)
            if max_width > resized_width:
                padded_crop = cv2.copyMakeBorder(resized_crop, 0, 0, 0, max_width - resized_width, cv2.BORDER_REPLICATE)
            crops.append((padded_crop[np.newaxis, :] / 127.5 - 1).astype(self.data_type))
        return crops

    def infer_full(self, crops: List[np.array]):
        crop_stack = torch.tensor(np.stack(crops, axis=0)).to(self.args.device)  # b,c,h,w
        if self.args.batch_size < 0:
            with torch.no_grad():  # transform all crops
                feats = self.model(crop_stack)  # noqa
            return feats.cpu().numpy()
        else:
            all_feat = []
            with torch.no_grad():  # transform all crops
                for idx in range(0, len(crops), self.args.batch_size):
                    feats = self.model(crop_stack[idx : idx + self.args.batch_size])  # noqa
                    all_feat.append(feats.cpu().numpy())
            return all_feat

    def post_infer(self, outputs, inputs):

        # Compute softmax along axis 2
        exp_outputs = np.exp(outputs)
        preds_prob = exp_outputs / np.sum(exp_outputs, axis=2, keepdims=True)

        # Set specific indices to 0
        preds_prob[:, :, self.ignore_idx] = 0.0

        # Normalize
        preds_prob /= preds_prob.sum(axis=2, keepdims=True)

        return self.decode(preds_prob)

    def decode(self, preds_prob: np.array) -> (List[str], List[float]):
        # Get max values and corresponding indices
        values = preds_prob.max(axis=2)
        indices = preds_prob.argmax(axis=2)

        if self.args.decoder == "greedy":
            # Select max probability (greedy decoding) then decode index to character
            preds_index = indices.ravel()
            preds_size = [preds_prob.shape[1]] * preds_prob.shape[0]
            preds_str = self.converter.decode_greedy(preds_index, preds_size)
        elif self.args.decoder == "beamsearch":
            preds_str = self.converter.decode_beamsearch(preds_prob, beamWidth=self.args.beam_width)
        elif self.args.decoder == "wordbeamsearch":
            preds_str = self.converter.decode_wordbeamsearch(preds_prob, beamWidth=self.args.beam_width)
        else:
            raise NotImplemented

        # Filter values where indices are non-zero and handle the case where all indices are zero
        preds_max_prob = [v[i != 0] if np.any(i != 0) else np.array([0]) for v, i in zip(values, indices)]

        # Compute confidence scores and pair them with preds_str
        confidence = [custom_mean(pred_max_prob) for pred_max_prob in preds_max_prob]

        return preds_str, confidence

    def _build_transform(self):
        pass
