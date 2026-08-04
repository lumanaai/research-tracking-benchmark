from typing import List, Any, Dict, Optional, Tuple
import numpy as np

from general.analyzer_general import (
    InferenceType,
    GLOBAL_TYPE_KEY,
    GLOBAL_FAILURE,
    GLOBAL_SUCCESS,
    logger,
    PLATE_NATIONALITY,
)
from level1.advanced import BaseAdvanceAnalyzer, AttrData, batch_data_convert_parts_to_object, BaseAdvanceAnalyzerConfig
from general.core import (
    BatchDataResolver,
    AdvanceAnalyzerType,
    WorkItem,
    AttrProperty,
    AdvanceAnalyticResults,
    AttrConfidence,
)
from level1.common_classifier.classifier_attributes import BaseAttributesModel
from level1.license_plate.lp_recognition import LicensePlateRecognition
from level1.license_plate.lp_attributes import LPAttributesModel


class LPRECConfig(BaseAdvanceAnalyzerConfig):

    preferred_conf: float = 0.885
    min_conf: float = 0.8
    max_batch_size: int = 8
    skew_th: float = 0.5
    min_lp_vehicle_overlap: float = 0.5
    min_crop_width = 75
    max_crop_width = 360


class LPRAnalyzer(BaseAdvanceAnalyzer):
    crop_on_success_only = True
    name = "license_plate_recognition"
    _type = AdvanceAnalyzerType.LPREC
    use_low_prio_set = True
    max_retries = 10
    low_prio_max_time = 5
    analyzer_batch_limit = 2
    min_require_results: AdvanceAnalyticResults = AdvanceAnalyticResults.SUCCESS
    _config_type = LPRECConfig
    args: LPRECConfig

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)

        self.lp_subclass = self.class_handler.plate_subclass_value

        self.supported_classes = [self.class_handler.vehicle_value]
        self.supported_subclasses = [self.class_handler.get_object_classes(self.class_handler.vehicle_value)]

        self.recognizer = LicensePlateRecognition(msg)
        self.args.max_crop_width = max(self.args.max_crop_width, self.recognizer.input_shape[0])
        self.filter = {}
        self._additional_models: Dict[InferenceType, BaseAttributesModel] = {}

        self.min_conf = self.args.min_conf
        self.preferred_conf = self.args.preferred_conf
        self.score_map = {"high": 1, "medium": 0.75, "low": 0.5, "none": 0}
        self.post_score_success_th = self.args.preferred_conf
        self.post_score_failure_th = self.args.min_conf
        self.keys_to_filter = [GLOBAL_TYPE_KEY]
        self.license_plate_statistics = {"confidence": 0}

    def prepare_batch_data(self, batch_data: BatchDataResolver) -> BatchDataResolver:
        if self.enabled:
            return batch_data_convert_parts_to_object(
                batch_data, self.lp_subclass, self.class_handler.vehicle_value, self.args.min_lp_vehicle_overlap
            )
        return batch_data

    def calculate_pre_score(self, batch_data: BatchDataResolver, image_batch=None) -> Dict[int, float]:
        detections = batch_data.query(batch_data.SUBCLASS, self.supported_subclasses)
        if self.filter_untracked:
            detections = detections[detections[:, BatchDataResolver.ID] >= 0]
        scores = {}
        # scores based on crop width: should be higher than th and lower than width (orientation)
        if len(detections) > 0:
            width, height = batch_data.full_resolution
            # area is normalized pos * ROI size, so we need to divide by roi size and multiply by image size
            pos = detections[:, BatchDataResolver.POS]
            crop_widths = (pos[:, 2] - pos[:, 0]) * width
            norm_widths = np.minimum((crop_widths - self.args.min_crop_width) / self.args.max_crop_width, 1)
            crop_scores = np.where(norm_widths < 0, -1, norm_widths)
            # cancel out image-edges overlapping plates
            x1_px = pos[:, 0] * width
            y1_px = pos[:, 1] * height
            x2_px = pos[:, 2] * width
            y2_px = pos[:, 3] * height
            touches_edges = (x1_px <= 0) | (y1_px <= 0) | (x2_px >= (width - 1)) | (y2_px >= (height - 1))
            crop_scores = np.where(touches_edges, -1, crop_scores)
            scores.update({int(detections[i, BatchDataResolver.INDEX]): scr for i, scr in enumerate(crop_scores)})
        return scores

    def calculate_post_score(self, obj_id: int, attr_data: AttrData) -> float:
        att = attr_data.attributes
        if GLOBAL_FAILURE in att[GLOBAL_TYPE_KEY]:
            return -1
        return att["scores"].get("plate", 0) * 0.9 + attr_data.pre_score * 0.1

    def _run_model(self, batch: List[WorkItem]) -> List[Any]:
        proc_fail = {GLOBAL_TYPE_KEY: [GLOBAL_FAILURE]}
        results = [proc_fail] * len(batch)
        crops = [item.image for item in batch]
        good_indices = [i for i in range(len(crops)) if crops[i] is not None]
        valid_crops = [crops[i] for i in good_indices]  # can add sharpness test here
        texts, confs = self.recognizer.forward_on_crop_list(valid_crops)
        add_model_results = {}
        for model in self._additional_models:
            add_model_results[model.value] = self._additional_models[model]([batch[i] for i in good_indices])

        for i, orig_ind in enumerate(good_indices):
            plate_metrics = {"location": batch[orig_ind].extra.location}
            results[orig_ind] = {
                GLOBAL_TYPE_KEY: [GLOBAL_SUCCESS],
                "scores": {"plate": confs[i]},
                "plate": texts[i],
                "zoom_image": crops[orig_ind],
                "plate_metrics": plate_metrics,
            }
            if add_model_results:
                results[orig_ind][model.value] = add_model_results[model.value][i]
        # print(f"Timestamp: {batch[0].extra.timestamp} LPR results: {results}")    # for debug
        return results

    def to_properties(self, attr_data: Dict) -> Dict:
        prop_data = {}
        metrics = []
        plate_score = attr_data.get("scores", {}).get("plate", 0)
        att_conf = self.plate_conf_to_attr_conf(plate_score)
        prop_data["plate"] = [AttrProperty(attr_data.get("plate", ""), plate_score, att_conf, self.priority)]
        input_metrics = attr_data.pop("plate_metrics", {})
        for k, v in input_metrics.items():
            metrics.append(AttrProperty(k, v, att_conf, self.priority))
        prop_data["plateMetrics"] = metrics
        lpc_type = InferenceType.LPC_ATTR
        if lpc_type in attr_data:
            if attr_data[lpc_type].get(GLOBAL_TYPE_KEY, GLOBAL_FAILURE) == GLOBAL_SUCCESS:
                lpc_data = self._additional_models[InferenceType.LPC_ATTR].interpreter.interpret_scores(
                    attr_data[lpc_type]["scores"], 0
                )
                lpc_res = lpc_data.get(PLATE_NATIONALITY, None)
                if lpc_res is not None:
                    prop_data[PLATE_NATIONALITY] = lpc_res

        return prop_data  # should be regular dict

    def merge_with_hist(self, attr: Dict, obj_id: int) -> Tuple[Optional[Dict], float]:
        if len(self.processed[obj_id]) < 2:
            return attr, self.processed[obj_id][-1].post_score
        else:
            max_idx = np.argmax([res.attributes["scores"].get("plate", 0) for res in self.processed[obj_id]])
            return self.processed[obj_id][max_idx].attributes, self.processed[obj_id][max_idx].post_score

    def add_to_low_prio_set(self, obj_id, metadata):
        can_add = True
        if obj_id in self.processed:
            max_conf = np.max([res.attributes["scores"].get("plate", 0) for res in self.processed[obj_id]])
            if max_conf >= self.preferred_conf:
                can_add = False
        if can_add:
            self.low_prio_set.add(obj_id)

    def get_metadata(self, crop_info):
        return {}

    def plate_conf_to_attr_conf(self, score) -> AttrConfidence:
        if score > self.preferred_conf:
            return AttrConfidence.HIGH
        elif score > self.min_conf:
            return AttrConfidence.MEDIUM
        return AttrConfidence.LOW

    def update_filters(self, enabled, filters):
        if enabled is not None:
            self.enabled = enabled
        if filters is not None:
            self.filter.update(filters if isinstance(filters, dict) else vars(filters))
            lpc = InferenceType.LPC_ATTR
            if len(self.filter.get(lpc, [])) > 0 and lpc not in self._additional_models:
                self._additional_models[lpc] = LPAttributesModel({})


# raise ValueError(
#     "minimal resolution-from slack? is it your net? its in the config up there (max/min width/height) use sum?"
# )
# raise ValueError("aspect ration- from slack?")
# raise ValueError("confidence threhsold- use conf_thresh")
# raise ValueError("what to return if below threshold")
