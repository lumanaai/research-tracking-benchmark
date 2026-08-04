import json
from collections import deque, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union, Tuple

import numpy as np

from general.analyzer_general import logger, log_exception, ROI_SHAPE
from general.common_models import ClassificationWrapper, CalibratedModelWrapper
from general.core import BDR, AnalyticImage, MotionData, ObjectManager
from general.db_handler import CustomObjectsState, CustomObjectsStateRepresentative
from general.image_encoder import ImageEncoder
from general.img_utils import crop_image

UNKNOWN_STATE = -1
DEFAULT_LATENCY = 3


@dataclass
class ObjectState:
    id: int
    description: str
    encodings: Optional[np.ndarray]
    threshold: Optional[float]


@dataclass
class StateChangeRecord:
    timestamp: int
    stateIdFrom: int
    stateIdTo: int
    duration: float


class ObjectHandler:
    id: int
    name: str
    coordinates: np.ndarray
    states_data: Dict[int, ObjectState]
    _model: Union[ClassificationWrapper, ImageEncoder]

    # state handling parameters
    sampling_rate_factor: float = 1

    # configuration parameters
    sample_period_ms: int = 1000
    latency: int = DEFAULT_LATENCY
    motion_sensitivity: int = 10
    model_sensitivity: float = 0.35

    def __init__(
        self,
        id_: int,
        name: str,
        coordinates: np.ndarray,
        states: List[ObjectState],
        configuration: Optional[Dict] = None,
    ):
        self.id = id_
        self.name = name
        self.coordinates = coordinates
        self.state_ids = [s.id for s in states]

        self.states_data = {s.id: s for s in states}

        # update configuration from dict if exists
        if configuration:
            self.sample_period_ms = configuration.get("sample_period_ms", self.sample_period_ms)
            self.latency = configuration.get("latency", self.latency)
            self.motion_sensitivity = configuration.get("motion_sensitivity", self.motion_sensitivity)
            self.model_sensitivity = configuration.get("model_sensitivity", self.model_sensitivity)
        self.default_latency = self.latency
        self.min_requested_latency = None

        self.valid_count_th = np.ceil(self.latency / 2)
        self.state_queue = deque(maxlen=self.latency)
        self.state_changes = []
        self.last_state_change = -1
        self.current_state = UNKNOWN_STATE
        self.last_sample_ts: int = -1
        self.batch_state_changes: List[StateChangeRecord] = []

        # for motion handling
        coords = self.coordinates * [*ROI_SHAPE, *ROI_SHAPE]
        coords[:2] = np.floor(coords[:2])
        coords[2:] = np.ceil(coords[2:])
        coords = np.clip(coords, 0, ROI_SHAPE[0]).astype(int)
        self.slice = coords
        self.margins = [0, 0]
        self.state_thresholds = np.array([self.states_data[s].threshold for s in self.state_ids])
        self._model = ImageEncoder({})
        self._run_model = self.run_encoder_model

        # for reports
        self.report_period = 60 * 1000  # 1 minute

        self.low_model_th = self.model_sensitivity
        self.high_model_th = 1 - self.model_sensitivity
        self.max_unsampled_period = 4 * self.sample_period_ms

    def set_model(self, model: ClassificationWrapper):
        self._model = model
        self._run_model = self.run_classification_model

    def denoise_state(self, state) -> Tuple[int, bool]:
        self.state_queue.append(state)
        denoised = UNKNOWN_STATE
        counts = Counter(self.state_queue)
        max_state, max_val = max(counts.items(), key=lambda x: x[1])
        if max_val >= self.valid_count_th:
            denoised = max_state
        return denoised, max_val == self.latency

    def _has_motion(self, motion_data: MotionData) -> bool:
        return bool(
            np.any(
                motion_data.unified[self.slice[1] : self.slice[3], self.slice[0] : self.slice[2]]
                >= self.motion_sensitivity
            )
        )

    def track_state(self, batch_data: BDR, image_batch: List[AnalyticImage], motion_data: MotionData):
        next_ts = self.last_sample_ts + self.sample_period_ms
        crops = []
        timestamps = []
        self.batch_state_changes.clear()
        if (
            self._has_motion(motion_data)
            or image_batch[-1].timestamp - self.last_sample_ts >= self.max_unsampled_period
            or self.sampling_rate_factor < 1
        ):
            for image in image_batch:
                if image.timestamp >= next_ts:
                    crops.append(crop_image(image.frame, self.coordinates, margins=self.margins, bgr_map=False))
                    timestamps.append(image.timestamp)
                    self.last_sample_ts = image.timestamp
                    next_ts = self.last_sample_ts + self.sample_period_ms * self.sampling_rate_factor
        else:  #  no motion detected, skip sampling
            next_ts += self.sample_period_ms

        if crops:
            new_states = self._run_model(crops)
            for state, ts in zip(new_states, timestamps):
                denoised_state, is_consensus = self.denoise_state(state)
                if denoised_state != self.current_state and denoised_state != UNKNOWN_STATE:
                    compensated_ts = ts - self.sample_period_ms
                    record = StateChangeRecord(
                        timestamp=compensated_ts,
                        stateIdFrom=self.current_state,
                        stateIdTo=denoised_state,
                        duration=compensated_ts - self.last_state_change if self.last_state_change > 0 else 0,
                    )
                    self.state_changes.append(record)
                    self.current_state = denoised_state
                    self.batch_state_changes.append(record)
                    self.last_state_change = compensated_ts
                self.sampling_rate_factor = 1 if is_consensus else 0.5

    def set_latency(self, latency: int = 0):
        if latency < self.default_latency:
            latency = self.default_latency
            logger.info(f"cannot set latency of state object {self.name} to below the default latency {latency}")

        if self.min_requested_latency is None:
            self.min_requested_latency = latency
        else:
            self.min_requested_latency = min(self.min_requested_latency, latency)
        if self.min_requested_latency != self.latency:
            logger.info(f"changing latency of state object {self.name} to new latency: {self.min_requested_latency}")
            self.latency = self.min_requested_latency
            self.valid_count_th = np.ceil((self.latency + 1) / 2)  # to make sure it will have "majority"
            old_items = list(self.state_queue)
            self.state_queue = deque(old_items, maxlen=self.latency)

    def run_encoder_model(self, crops: List[np.ndarray]) -> List[int]:
        encodings = self._model.forward_on_crop_list(crops)
        state_ids = []
        for enc in encodings:
            scores = np.array([self.states_data[s].encodings @ enc for s in self.state_ids])
            # Sort indices by scores in descending order
            sorted_indices = np.argsort(scores)[::-1]
            norm_distances = scores / self.state_thresholds
            # Check states in descending order of scores
            state_id = UNKNOWN_STATE
            for idx in sorted_indices:
                if scores[idx] >= self.state_thresholds[idx]:
                    state_id = self.state_ids[idx]
                    break
            if state_id == UNKNOWN_STATE:
                sorted_norm_inds = np.argsort(norm_distances)[::-1]
                id1 = sorted_norm_inds[0]
                id2 = sorted_norm_inds[1]
                # clear distinction
                if norm_distances[id1] > 0.95 and norm_distances[id1] - norm_distances[id2] > 0.05:
                    state_id = self.state_ids[id1]
            state_ids.append(state_id)
        return state_ids

    def run_classification_model(self, crops: List[np.ndarray]) -> List[int]:
        cl, scores = self._model.forward_on_crop_list(crops)
        states = []
        for c, score in zip(cl, scores):
            states.append(UNKNOWN_STATE if score < self.high_model_th else self.state_ids[c])
        return states

    @property
    def has_proprietary_model(self) -> bool:
        return isinstance(self._model, CalibratedModelWrapper)

    def generate_report(self, base_ts) -> Dict:
        timeline_report = []
        changes_report = []
        next_ts = base_ts + self.report_period
        self.state_changes = [c for c in self.state_changes if c.timestamp >= base_ts]
        changes = [c for c in self.state_changes if c.timestamp < next_ts]
        if not changes:
            if len(self.state_queue) == self.state_queue.maxlen:  # queue is full - we have a valid state
                timeline_report.append(
                    {
                        "baseMin": base_ts,
                        "customObjectStateId": self.id,
                        "stateId": self.current_state,
                        "dwell": (next_ts - self.last_state_change) / 1000 if self.last_state_change > 0 else 0,
                        "value": int(self.report_period / 1000),
                        "toUpdate": 0,
                    }
                )
        else:
            last_ts = base_ts
            for change in changes:
                timeline_report.append(
                    {
                        "baseMin": base_ts,
                        "customObjectStateId": self.id,
                        "stateId": change.stateIdFrom,
                        "dwell": change.duration / 1000,
                        "value": int((change.timestamp - last_ts) / 1000),
                        "toUpdate": 0,
                    }
                )
                changes_report.append({**asdict(change), "customObjectStateId": self.id, "toUpdate": 1})
                last_ts = change.timestamp

            timeline_report.append(
                {
                    "baseMin": base_ts,
                    "customObjectStateId": self.id,
                    "stateId": changes[-1].stateIdTo,
                    "dwell": (next_ts - changes[-1].timestamp) / 1000,
                    "value": int((next_ts - last_ts) / 1000),
                    "toUpdate": 0,
                }
            )
        return {"timeline": timeline_report, "changes": changes_report}


def roi_to_coords(roy: str) -> np.ndarray:
    roy_dict = json.loads(roy)
    return np.array([roy_dict[0]["x"], roy_dict[0]["y"], roy_dict[1]["x"], roy_dict[1]["y"]]).astype(np.float32)


class StaticObjectManager(ObjectManager):

    def __init__(self, context):
        self.context = context
        self.objects: Dict[int, ObjectHandler] = {}
        self.config = context.config.get("state_objects", {})

    def on_db_update(self):
        metadata: List[CustomObjectsState] = self.context.analytic_db.get("states", {}).get("metadata", [])
        reps: List[CustomObjectsStateRepresentative] = self.context.analytic_db.get("states", {}).get(
            "representatives", []
        )

        new_keys = set([obj_data.id for obj_data in metadata])
        old_keys = set(self.objects.keys())

        for key in old_keys:
            if key not in new_keys:
                self.objects.pop(key, None)
                logger.warning(f"Removed state object {key} due to missing metadata")

        for obj_data in metadata:
            if obj_data.id in self.objects:
                # update states metadata only
                try:
                    self.objects[obj_data.id].name = obj_data.name
                    for sid, state_name in enumerate(obj_data.states_dict):
                        self.objects[obj_data.id].states_data[sid].description = state_name
                except Exception as e:
                    log_exception(logger, f"Failed to update state object {obj_data.id}", e)
            else:
                obj_states = []
                object_reps = [r for r in reps if r.customObjectsStateId == obj_data.id]
                state_ids = set([r.state for r in object_reps])
                coords = roi_to_coords(obj_data.ROI)

                # Sort state_ids with -1 (UNKNOWN_STATE) at the end
                sorted_state_ids = sorted([s for s in state_ids if s != -1]) + ([-1] if -1 in state_ids else [])
                for s in sorted_state_ids:
                    encoding = np.mean([r.embedding_np for r in object_reps if r.state == s], axis=0)
                    obj_states.append(
                        ObjectState(
                            id=s,
                            description=obj_data.states_dict[s] if s >= 0 else "Undefined",
                            encodings=encoding / np.linalg.norm(encoding),
                            threshold=obj_data.per_state_thresholds_dict[s],
                        )
                    )
                self.objects[obj_data.id] = ObjectHandler(obj_data.id, obj_data.name, coords, obj_states, self.config)

            if not self.objects[obj_data.id].has_proprietary_model:
                self._on_model_update(obj_data.id)

    def _on_model_update(self, object_id: int):

        state_objects_config = self.context.config.get("state-objects", {})
        classification_model = None
        try:
            model_config = None
            object_id_str = str(object_id)
            for k in state_objects_config.keys():
                if object_id_str in k:
                    model_config = state_objects_config[k]
                    n_classes = len(self.objects[object_id].states_data)
                    if n_classes == 2:
                        n_classes = 1  # binary classification can be done with single output
                    model_config["n_classes"] = n_classes
                    break
            if model_config is not None:
                if model_config["n_classes"] == 1:
                    model_config["classifier_threshold"] = 0.5
                    classification_model = CalibratedModelWrapper(model_config)
                else:
                    classification_model = ClassificationWrapper(model_config)
        except Exception as e:
            log_exception(logger, f"Error loading model for object {object_id}", e)

        if classification_model is None:
            logger.warning(f"Reverting to use encoding instead of model for object {object_id}")
        else:
            logger.info(f"using model for object {object_id}")
            self.objects[object_id].set_model(classification_model)

    def track(self, batch_data: BDR, image_batch: List[AnalyticImage], motion_data: MotionData):
        for obj in self.objects.values():
            obj.track_state(batch_data, image_batch, motion_data)

    def sync_and_cleanup(self, base: int) -> Dict:
        report = []
        for obj in self.objects.values():
            obj_report = obj.generate_report(base)
            report.extend(obj_report["timeline"])
            report.extend(obj_report["changes"])
        if report:
            return {"custom_objects_state": {"custom_objects_state": report, "timestamp": base}}
        else:
            return {}

    @property
    def enabled(self) -> bool:
        return len(self.objects) > 0

    def update_model(self, model_key):
        for obj_id in self.objects.keys():
            if str(obj_id) in model_key:
                self._on_model_update(obj_id)
                return
        logger.error(f"Model {model_key} not found in state objects")
