import csv
import warnings
from typing import List, Dict, Optional

import numpy as np

from general import proj
from general.analyzer_general import InferenceType
from general.inference import BaseInferenceConfig, InferenceWrapper
from general.ort_utils import ort_type_to_numpy


class StgcnConfig(BaseInferenceConfig):
    weights = "stgcn_w25_1_0.onnx"  # "stgcn_1_0_w20.onnx"
    expected_skeleton_joints = 26
    joints_used = [19, 18, 17, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
    im_size = (1, 1, 25, 15, 3)
    name: InferenceType = InferenceType.STGCN
    classes_metadata: str = "actions.csv"
    extra_frames: int = 0


class StgcnActionRecognition(InferenceWrapper):
    _config_type = StgcnConfig
    args: StgcnConfig
    input_name = "input"

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)
        self.classes = self._load_classes()
        self.extra_frames = self.args.extra_frames
        self.actual_len = self.args.im_size[-3]

    def build_trt_model(self):
        from general.trt_utils import TrtInfer

        self.model = TrtInfer(self.args.weights, self.input_name)

    def build_full_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(self.args.weights, providers=["CPUExecutionProvider"])
        self.input_name = self.onnx_session.get_inputs()[0].name

        self.args.half = ort_type_to_numpy[self.onnx_session.get_inputs()[0].type] is np.float16

        output_name_01 = self.onnx_session.get_outputs()[0].name
        self.output_names = [output_name_01]
        self.output_dict = {}
        for n in self.output_names:
            self.output_dict[n] = []

    def prepare_crops(self, crop_list):
        crops = []
        for skeletons_batch in crop_list:
            skeletons_batch = skeletons_batch[:, self.args.joints_used, :]
            reps = np.ceil(self.activity_len / len(skeletons_batch)).astype(int)
            if reps > 1:
                sk = np.repeat(skeletons_batch, reps, axis=0)
            else:
                sk = skeletons_batch
            crp = np.expand_dims(np.expand_dims(sk, axis=0), axis=0)
            if self.extra_frames:
                for i in range(self.extra_frames):
                    crops.append(crp[:, :, i : i + self.actual_len, :, :])
            else:
                crops.append(crp[:, :, : self.actual_len, :, :])
        return crops

    def infer_full(self, crops: List[np.array]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crops_stack = np.stack(crops)
            crop_res = self.onnx_session.run(self.output_names, {self.input_name: crops_stack.astype(self.data_type)})

        return crop_res[0]

    def post_infer(self, outputs, inputs):
        # features = np.squeeze(outputs, axis=1)
        features = outputs
        out = []
        if self.extra_frames:
            for idx in range(0, len(features), self.extra_frames):
                arr = features[idx : idx + self.extra_frames]
                r, c = np.unravel_index(np.argmax(arr), arr.shape)
                out.append(
                    {"scores": np.max(arr, axis=0), "action": self.classes.get(c, "Unknown"), "confidence": arr[r, c]}
                )

        else:
            max_inds = np.argmax(features, axis=1)

            for idx, feat in enumerate(features):
                mx_idx = int(max_inds[idx])
                out.append({"scores": feat, "action": self.classes.get(mx_idx, "Unknown"), "confidence": feat[mx_idx]})
        return out

    def _load_classes(self) -> Dict[int, str]:
        file_to_load = proj.info_path("action", self.args.classes_metadata)
        with open(file_to_load, mode="r") as file:
            csv_reader = csv.reader(file)
            # Skip the header
            next(csv_reader, None)

            return {int(row[0]): row[1].strip() for row in csv_reader}

    def _build_transform(self):
        pass

    @property
    def activity_len(self):
        return self.actual_len + self.args.extra_frames

    @property
    def expected_skeleton_joints(self):
        return self.args.expected_skeleton_joints

    def export_is_half(self):
        return True
