from copy import copy
from typing import List, Dict
import numpy as np

from general.analyzer_general import logger
from general.common_models import ClassificationWrapper
from general.core import AlertInfo, AnalyticImage
from general.img_utils import crop_image
from general.inference import InferenceWrapper
from .alerts_utils import StateMachine, ObjectInfo, read_object_info_from_form, UNKNOWN_STATE, NEGATIVE_STATE, \
    read_transition_duration_from_form
from .base_alerts import BaseAlert, AlertCandidate, duration_unit_to_sec

MIN_ALLOWED_DURATION = 1000


class ClassificationBaseAlert(BaseAlert):
    period_ms = 1000
    state_ids: List[int]
    conf_to_latency_conversion = {0: 1, 1: 3, 2: 5}
    duration = MIN_ALLOWED_DURATION
    object_info: ObjectInfo
    bbox: np.array = None
    num_of_crops_required: int = 2  # bulk mode
    validation_crop_num = 0
    transition_duration: Dict[int, Dict[int, int]] = None  # transition duration between states

    def __init__(self, alert_dict: dict, context):
        super(ClassificationBaseAlert, self).__init__(alert_dict, context)
        self.set_flow_values()
        self.object_states = self.object_info.states
        self.state_ids = [s.id for s in self.object_states]
        if len(self.object_states) == 0:
            raise ValueError("No states defined")
        elif len(self.object_states) == 1:
            self.parse_object_data_to_state = self.parse_object_data_single_state
        else:
            self.parse_object_data_to_state = self.parse_object_data_multi_state

        confidence = self.settings.get("confidence", 2)  # default is high confidence
        self.latency = self.conf_to_latency_conversion.get(confidence, 5)
        self.extra_fields = {"extra_fields": {"query": self.special_filter}}

        self.object_data_buffer = []
        self.object_data_ts_buffer = []

        self.crops = []
        self.crops_ts = []
        self.is_crop_mode = self.bbox is not None
        self.last_checked_image = 0

        # state parameters
        self.state_machine = StateMachine(self.object_info, self.latency, self.duration, self.transition_duration)
        # self.validation_crop_size = VALIDATION_FULL_IMAGE_SIZE
        # if self.validation_crop_num > 1 and self.routing != AlertRouting.NO_ROUTING:
        #     self.validator_poses = divide_n_into_s_parts_inclusive(self.latency, self.validation_crop_num)
        #     srt = math.ceil(math.sqrt(self.validation_crop_num))
        #     self.validation_grid = (math.ceil(self.validation_crop_num / srt), srt)

    def set_flow_values(self):
        if self.formValue:
            duration = int(self.formValue.get("duration", 0))  # to ms
            units = self.apply_from_dict("durationUnit", self.formValue, 0)
            self.duration = max(1000, duration * duration_unit_to_sec[units] * 1000)  # at least a sec
            self.object_info = read_object_info_from_form(self.formValue)
            self.transition_duration = read_transition_duration_from_form(self.formValue)
            self.bbox = self.object_info.coordinates
            self.period_ms = self.formValue.get("period", self.period_ms)


    def get_crops(self, images, timestamps):
        for tidx, ts in enumerate(timestamps):
            if ts > self.last_checked_image + self.period_ms:
                crop = crop_image(images[tidx].frame, self.bbox, margins=(0,0), bgr_map=False)
                self.crops.append(crop)
                self.crops_ts.append(ts)
                self.last_checked_image = ts

    def is_active_batch(self, images, motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []

        # extract the object data into results buffer
        self.extract_object_data_from_images(images)

        # if we have new encoding - parse the encoding and check the state
        if self.object_data_buffer:
            # this fills the state buffer and returns the current state
            alert_valid_idx = []
            for idx, enc in enumerate(self.object_data_buffer):
                ts = self.object_data_ts_buffer[idx]
                state = self.parse_object_data_to_state(enc)
                if self.state_machine.add_measurement(state, ts):
                    alert_valid_idx.append(idx)
                    extra = copy(self.extra_fields)
                    extra["extra_fields"]["state"] = self.state_machine.state
                    alert_candidates.append(self.build_alert_candidate(ts, extra=extra))
                # debug
                logger.debug(f"curr_state: {state}, state: {self.state_machine.state}, ts: {ts}")

            # clean the buffers
            self._clear_buffers(self.last_checked_image)
        return alert_candidates

    def _clear_buffers(self, last_checked_image: int):
        self.crops.clear()
        self.crops_ts.clear()
        self.object_data_buffer.clear()
        self.object_data_ts_buffer.clear()
        self.last_checked_image = last_checked_image

    def clear_state(self):
        self.state_machine.clear()

    def parse_object_data_multi_state(self, obj_data) -> int:
        pass

    def parse_object_data_single_state(self, obj_data) -> int:
        pass

    def extract_object_data_from_images(self, images: List[AnalyticImage]):
        pass

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super(ClassificationBaseAlert, self).build_alert_info(candidate)
        # custom object fields here?

        # attach the message
        self.generate_alert_message(candidate, alert_info)
        return alert_info

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        alert_info.alertMessage = (
            "new state detected: " + self.object_info.get_state(candidate.extra["extra_fields"]["state"]).description
        )


class ClassificationModelAlert(ClassificationBaseAlert):
    type_name = "classification"
    classification_model: InferenceWrapper
    cls_threshold: float = -1

    def __init__(self, alert_dict: Dict, context):
        super(ClassificationModelAlert, self).__init__(alert_dict, context)

        # get the model name from the config
        cls_alert_config = self.context.get_config().get("cls-alerts",{})
        model_config = None
        object_id = self.object_info.id
        for k in cls_alert_config.keys():
            if object_id in k:
                model_config = cls_alert_config[k]
                break
        if model_config is None:
            raise ValueError(f"Model config not found for object id {object_id}")
        else:
            self.classification_model = ClassificationWrapper(model_config)
        self.is_crop_mode = True
        if self.bbox is None:
            self.bbox = np.array([0,0,1,1])


    def set_flow_values(self):
        super(ClassificationModelAlert, self).set_flow_values()
        self.cls_threshold = self.formValue.get("classification_threshold", -1)
        if self.cls_threshold < 0:
            self.cls_threshold = 1.0 / len(self.object_info.states)


    def parse_object_data_multi_state(self, obj_data) -> int:
        c, score = obj_data
        if score > self.cls_threshold:
            return self.object_states[c].id
        return UNKNOWN_STATE

    def parse_object_data_single_state(self, obj_data) -> int:
        c, score = obj_data
        if score > self.cls_threshold:
            return self.object_states[c].id
        return NEGATIVE_STATE


    def extract_object_data_from_images(self, images: List[AnalyticImage]):
        timestamps = [img.timestamp for img in images]
        self.get_crops(images, timestamps)
        if len(self.crops) >= self.num_of_crops_required:
            cl, scores = self.classification_model.forward_on_crop_list(self.crops)
            for idx,c in enumerate(cl):
                self.object_data_buffer.append((c, scores[idx]))
                self.object_data_ts_buffer.append(self.crops_ts[idx])
                # debug
                # import cv2
                # cv2.imwrite(f"/mnt/d/light_detection/out/cls/{self.crops_ts[idx]}_{c}_{scores[idx]}.jpg", cv2.cvtColor(self.crops[idx],cv2.COLOR_RGB2BGR))