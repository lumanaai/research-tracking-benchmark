import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import albumentations as alb
import cv2
from general.analyzer_general import InferenceType
from general.img_utils import ResizeAndPadToTarget
from general.inference import BaseInferenceConfig, InferenceWrapper
from general.ort_utils import ort_type_to_numpy


def get_simcc_maximum(simcc_x: np.array, simcc_y: np.array, apply_softmax: bool = False) -> Tuple[np.array, np.array]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    # no need to check the shape of simcc_x and simcc_y
    # assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    # assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    # assert simcc_x.ndim == 2 or simcc_x.ndim == 3, f"Invalid shape {simcc_x.shape}"
    # assert simcc_y.ndim == 2 or simcc_y.ndim == 3, f"Invalid shape {simcc_y.shape}"
    # assert simcc_x.ndim == simcc_y.ndim, f"{simcc_x.shape} != {simcc_y.shape}"

    # if simcc_x.ndim == 3:
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    # else:
    #    N = None

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1

    # if N:
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


class RtmPoseConfig(BaseInferenceConfig):
    weights = "rtmpose_s26_1_0.onnx"
    im_size = (256, 192)
    name: InferenceType = InferenceType.RTMPOSE
    margins = [0.05, 0.05]
    pad_image: bool = False  # not supported at the moment
    max_dynamic_batch = 20
    half: bool = False  # dont use fp 16



class RtmPose(InferenceWrapper):
    _config_type = RtmPoseConfig
    args: RtmPoseConfig
    input_name = "input"

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)
        self.norm_divisor = np.array(self.args.im_size[-1::-1])[np.newaxis, np.newaxis, :] * 2

    @property
    def max_size_allowed(self):
        return {
            self.input_name: [self.args.max_dynamic_batch, 3, self.args.im_size[0], self.args.im_size[1]],
        }

    def build_trt_model(self):
        from general.trt_utils import TrtInfer

        self.model = TrtInfer(self.args.weights, self.input_name, max_size_allowed=self.max_size_allowed)

    def build_full_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(self.args.weights, providers=["CPUExecutionProvider"])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.args.half = ort_type_to_numpy[self.onnx_session.get_inputs()[0].type] is np.float16

        output_name_01 = self.onnx_session.get_outputs()[0].name
        output_name_02 = self.onnx_session.get_outputs()[1].name
        self.output_names = [output_name_01, output_name_02]
        self.output_dict = {}
        for n in self.output_names:
            self.output_dict[n] = []

    def infer_full(self, crops: List[np.array]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = [[] for _ in range(len(self.output_names))]
            for crop in crops:
                crop_res = self.onnx_session.run(
                    self.output_names, {self.input_name: np.expand_dims(crop.astype(self.data_type), axis=0)}
                )
                for i in range(len(self.output_names)):
                    results[i].append(crop_res[i])
        return np.vstack(results[0]), np.vstack(results[1])

    def post_infer(self, outputs, inputs):
        res_x, res_y = outputs
        points, scores = self.decode(res_x, res_y)
        return np.concatenate((points, scores[:, :, np.newaxis]), axis=2)  # N x n_joints x 3
        # points = points[:,self.args.joints_used_idx_26,:]
        # return np.concatenate((points, scores[:, self.args.joints_used_idx_26, np.newaxis]), axis=2)  # N x n_joints x 3

    def _build_transform(self):
        interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR

        if self.args.pad_image:
            self.resize_transform = ResizeAndPadToTarget(
                        target_height=self.args.im_size[0], target_width=self.args.im_size[1], interpolation=interp_mode
                    )

        else:
            self.resize_transform = alb.Resize(self.args.im_size[0], self.args.im_size[1], interpolation=interp_mode)
        self.transform = alb.Compose(
            [self.resize_transform, alb.Normalize(self.args.mean, self.args.std)]
        )

    def apply_resize(self, image: np.array, target: np.array = None, bgr: bool = False):
        resized_crop = self.resize_transform(image=image)["image"]
        if bgr:
            resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_RGB2BGR)
        if target is None:
            return resized_crop
        np.copyto(target, resized_crop)
        return target

    def decode(self, simcc_x: np.array, simcc_y: np.array) -> Tuple[np.array, np.array]:
        """Decode keypoint coordinates from SimCC representations. The decoded
        coordinates are in the input image space.

        Args:
            simcc_x (np.ndarray): SimCC label for x-axis
            simcc_y (np.ndarray): SimCC label for y-axis

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)

        # Unsqueeze the instance dimension for single-instance results
        # if keypoints.ndim == 2:
        #     keypoints = keypoints[None, :]
        #     scores = scores[None, :]

        keypoints /= self.norm_divisor

        return keypoints, scores

    @property
    def margins(self):
        return self.args.margins

    @property
    def aspect_ratio(self):
        return self.args.im_size[1] / self.args.im_size[0]

    @property
    def image_size(self):
        return self.args.im_size
