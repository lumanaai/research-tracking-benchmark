from copy import copy
from typing import List, Any, Dict, Tuple

from general.analyzer_general import InferenceType, GLOBAL_TYPE_KEY, GLOBAL_SUCCESS, GLOBAL_FAILURE
from level1.advanced import BaseAdvanceAnalyzer, AttrData
from level1.common_classifier.utils import *
from .person_attributes import AttributeInterpreter, AttributesModel, PersonReid
from general.core import  AdvanceAnalyzerType, DescriptorVector, WorkItem


def update_dict_with_lists(d:Dict, u: Dict):
    for k, v in u.items():
        if isinstance(v, list):
            d[k] = d.get(k, []) + v
        else:
            d[k] = v


class PersonAttrAnalyzer(BaseAdvanceAnalyzer):
    post_score_failure_th = -1.0
    retry_on_failure = 1
    analyzer_batch_limit = 2

    attribute_num: int
    base_keys_to_filter = ["poseType", "specialType", "confidence"]
    base_night_keys_to_filter = ["upperbodyColor", "lowerbodyColor", "hairColor", "footwearColor"]
    name = "personAttrAnalyzer"
    _type = AdvanceAnalyzerType.HUMAN_PARSING

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)

        pa_config_dict = msg["pa"] if "pa" in msg else msg
        self.pa_model = AttributesModel(pa_config_dict)
        reid_config_dict = msg["reid"] if "reid" in msg else msg
        self.reid_model = PersonReid(reid_config_dict)
        self.reid_version = self.reid_model.version

        self.supported_classes = [self.class_handler.person_value]
        self.supported_subclasses = self.class_handler.get_object_classes(self.class_handler.person_value)
        self.history_tbl = {}

        self.config_msg = msg
        self._additional_models: Dict[InferenceType, AttributesModel] = {}
        self.keys_to_filter = copy(self.base_keys_to_filter)
        self.night_keys_to_filter = copy(self.base_night_keys_to_filter)
        self.filter = {}
        self.min_crop_size = np.array([self.pa_model.args.minimal_crop_size] * 2)
        self.norm_crop_size = np.array(self.pa_model.args.im_size) * 2 - self.min_crop_size
        self.is_export_invalid_crops = True
        self.crop_ar = context.analytic_config.get("trainThumbPolicy", {}).get("crops_ar",{}).get("person", None)


    def _calc_area_score(self, crop_sz, sub_class: int = -1):
        norm_size = np.clip((np.array(crop_sz) - self.min_crop_size) / self.norm_crop_size, 0, 1)
        return norm_size[0] * norm_size[1]

    def calculate_post_score(self, obj_id: int, attr_data: AttrData):
        if attr_data.attributes.get("globalType", "success") == "failure":
            attr_data.post_score = -1
        else:
            pa_score = self.pa_model.interpreter.calc_confidence_score(attr_data.attributes["scores"])
            additional_scores = [pa_score]
            for model in self._additional_models:
                scores = attr_data.attributes[model]["scores"]
                conf = self._additional_models[model].interpreter.calc_confidence_score(scores)
                if conf < 0:
                    attr_data.post_score = -1
                    return attr_data.post_score
                else:
                    additional_scores.append(conf)
            attr_data.post_score = np.mean(additional_scores)
        return attr_data.post_score

    def _run_model(self, batch: List[WorkItem]) -> List[Any]:
        results = [None] * len(batch)
        net_res = self.pa_model(batch)
        reid_res = self.reid_model(batch)
        additional_res = {}
        for model in self._additional_models:
            additional_res[model] = self._additional_models[model](batch)
        for ind in range(len(results)):
            results[ind] = net_res[ind]
            if reid_res[ind] is not None:
                results[ind]["descriptorHex"] = DescriptorVector(reid_res[ind])
                results[ind]["descriptor"] = reid_res[ind].astype(np.float32)
                results[ind]["reid_version"] = [self.reid_version]
                for model in self._additional_models:
                    results[ind][model] = additional_res[model][ind]
        return results

    def merge_with_hist(self, attr: Dict, obj_id: int) -> Tuple[Dict, float]:
        score = self.processed[obj_id][-1].post_score
        if len(self.processed[obj_id]) > 1:
            new_attr = copy(attr)
            results_list = [attr.attributes["scores"] for attr in self.processed[obj_id]]
            new_scores = self.pa_model.interpreter.merge_scores(results_list)
            new_attr["scores"] = new_scores
            merged_pa_score = self.pa_model.interpreter.calc_confidence_score(new_scores)

            # descriptor merge
            confs = [att.attributes.get("confidence", -1) for att in self.processed[obj_id]]
            good_desc = [idx for idx, conf in enumerate(confs) if conf > 0]
            if len(good_desc) < 2:
                max_idx = np.argmax(confs)
                desc = self.processed[obj_id][max_idx].attributes.get("descriptor")
            else:
                confs = np.clip(confs,0,1)
                desc = np.average([att.attributes.get("descriptor") for att in self.processed[obj_id]], axis=0, weights=confs)
                desc = desc / np.linalg.norm(desc)
            new_attr["descriptor"] = desc
            new_attr["descriptorHex"] = DescriptorVector(desc)
            new_attr[GLOBAL_TYPE_KEY] = GLOBAL_SUCCESS if len(good_desc) > 0 else GLOBAL_FAILURE

            models_scores = [merged_pa_score]
            additional_valid = True
            for model in self._additional_models:
                results_list = [attr.attributes[model]["scores"] for attr in self.processed[obj_id]]
                new_scores = self._additional_models[model].interpreter.merge_scores(results_list)
                new_attr[model]["scores"] = new_scores
                merged_score = self._additional_models[model].interpreter.calc_confidence_score(new_scores)
                additional_valid &= merged_score > 0
                models_scores.append(max(merged_score, 0))
            score = np.mean(models_scores) if additional_valid else self.post_score_failure_th
            attr = new_attr
        return attr, score

    def to_properties(self, attr_data: Dict) -> Dict:
        scores = attr_data.pop("scores")
        attr_data.pop(AttributeInterpreter.global_kw)
        attr_props = self.pa_model.interpreter.interpret_scores(scores, self.priority)

        for model in self._additional_models:
            scores = attr_data.pop(model)
            model_props = self._additional_models[model].interpreter.interpret_scores(scores["scores"], self.priority)
            update_dict_with_lists(attr_props, model_props)
        other_props = super().to_properties(attr_data)
        attr_props.update(other_props)
        return attr_props

    def update_filters(self, enabled, filters):
        if enabled is not None:
            self.enabled = enabled
        if filters is not None:
            self.filter.update(filters if isinstance(filters, dict) else vars(filters))

            att_list = self.filter.get("attribute_filter", [])
            if len(att_list) > 0:
                self.keys_to_filter = att_list + self.base_keys_to_filter

            ppe = InferenceType.PPE_ATTR
            if len(self.filter.get(ppe, [])) > 0 and ppe not in self._additional_models:
                from .ppe import PpeModel

                ppe_config_dict = self.config_msg.get(ppe, {})
                ppe_config_dict["enable"] = True
                self._additional_models[ppe] = PpeModel(ppe_config_dict)
