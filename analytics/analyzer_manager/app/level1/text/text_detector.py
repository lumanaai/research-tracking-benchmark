from typing import List, Dict, Optional
import cv2
import numpy as np
import torch
import albumentations as alb

from general.analyzer_general import InferenceType
from general.inference import BaseInferenceConfig, InferenceWrapper
from .models.craft import CRAFT
from .models.craft_utils import getDetBoxes
from .utils import group_text_box, diff, load_text_nn


class CraftConfig(BaseInferenceConfig):
    weights: str = "craft_mlt_25k.pt"
    name: InferenceType = InferenceType.CRAFT
    im_size = [384, 640]
    half: bool = True
    input_sz: List[int] = None
    min_size = 20
    text_threshold = 0.7
    low_text = 0.4
    poly = False
    link_threshold = 0.4
    mag_ratio = 1.0
    slope_ths = 0.1
    ycenter_ths = 0.5
    height_ths = 0.5
    width_ths = 0.5
    add_margin = 0.1
    optimal_num_chars = None
    estimate_num_chars = False
    batch_size: int = 8
    normalize = False

    def __init__(self, args_dict=None):
        super().__init__(args_dict)
        self.estimate_num_chars = self.optimal_num_chars is not None


class CraftTextDetector(InferenceWrapper):
    _config_type = CraftConfig
    args: CraftConfig
    output_names = ["text", "features"]

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)
        if self.args.input_sz is not None:
            ar_out = self.args.im_size[0]/self.args.im_size[1]
            ar_in = self.args.input_sz[0]/self.args.input_sz[1]
            if ar_in > ar_out:
                intermediate_size = [self.args.im_size[0], int(ar_out / ar_in * self.args.im_size[1])]
            else:
                intermediate_size = [int(ar_in / ar_out * self.args.im_size[0]), self.args.im_size[1]]
            self._build_transform_padding(intermediate_size)

    def build_full_model(self):
        self.model = load_text_nn(CRAFT(), self.args.weights, self.args.half, self.args.device)

    def infer_full(self, crops: List[np.array]):
        crop_stack = torch.tensor(np.stack(crops, axis=0)).to(self.args.device)  # b,c,h,w
        with torch.no_grad():  # transform all crops
            y, feature = self.model(crop_stack)  # noqa
        return y.cpu().numpy(), feature.cpu().numpy()

    def build_trt_model(self):
        super().build_trt_model()
        self._fix_output_order()

    def _build_transform_padding(self, intermediate_size):
        interp_mode = cv2.INTER_AREA if self.args.antialias else cv2.INTER_LINEAR
        self.transform = alb.Compose(
            [
                alb.Resize(intermediate_size[0], intermediate_size[1], interpolation=interp_mode),
                alb.PadIfNeeded(min_height=self.args.im_size[0], min_width=self.args.im_size[1], border_mode=cv2.BORDER_REPLICATE),
                alb.Normalize(self.args.mean, self.args.std),
            ]
        )

    def build_triton_model(self):
        super().build_triton_model()
        self._fix_output_order()

    def get_textboxes(self, network_output):
        result = []
        boxes_list, polys_list = [], []
        for out in network_output:
            # make score and link map
            score_text = out[:, :, 0]
            score_link = out[:, :, 1]

            # Post-processing
            boxes, polys, mapper = getDetBoxes(
                score_text,
                score_link,
                self.args.text_threshold,
                self.args.link_threshold,
                self.args.low_text,
                self.args.poly,
                self.args.estimate_num_chars,
            )
            if self.args.estimate_num_chars:
                boxes = list(boxes)
                polys = list(polys)
            for k in range(len(polys)):
                if self.args.estimate_num_chars:
                    boxes[k] = (boxes[k], mapper[k])
                if polys[k] is None:
                    polys[k] = boxes[k]
            boxes_list.append(boxes)
            polys_list.append(polys)

        if self.args.estimate_num_chars:
            polys_list = [
                [p for p, _ in sorted(polys, key=lambda x: abs(self.args.optimal_num_chars - x[1]))]
                for polys in polys_list
            ]

        for polys in polys_list:
            single_img_result = []
            for i, box in enumerate(polys):
                poly = np.array(box).reshape((-1))
                single_img_result.append(poly)
            result.append(single_img_result)
        return result

    def post_infer(self, outputs, inputs):
        y, feature = outputs
        return self.get_textboxes(y)

    def group_text(self, text_box_list):
        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(
                text_box,
                self.args.slope_ths,
                self.args.ycenter_ths,
                self.args.height_ths,
                self.args.width_ths,
                self.args.add_margin,
                sort_output=not self.args.estimate_num_chars,
            )
            if self.args.min_size:
                horizontal_list = [i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > self.args.min_size]
                free_list = [
                    i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > self.args.min_size
                ]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg
