import copy
import time
from argparse import Namespace
from dataclasses import dataclass
from io import BytesIO
from typing import List, Dict, Tuple, Any, Set, Union, Optional

import cv2
import numpy as np
from PIL import Image
from cython_bbox import bbox_overlaps as bbox_ious  # noqa

from general.analyzer_general import logger, ROI_SHAPE, log_exception
from general.core import (
    BatchDataResolver,
    AdvanceAnalyzerType,
    AttrProperty,
    AttrConfidence,
    DescriptorVector,
    AdvanceAnalyticPriority,
    WorkItem,
    AdvanceAnalyticResults,
    advance_analyzer_priority,
    BDR,
    BaseConfig,
    AnalyticImage,
    MotionData,
)
from general.img_utils import crop_image_by_bbox_and_ar, fit_bbox_with_margins, crop_image_by_bbox


@dataclass
class AttrData:
    pos: List[float]
    timestamp: int
    confidence: float
    pre_score: float
    attributes: Dict[str, Any]
    post_score: float
    metadata: Dict[str, Any] = None


@dataclass
class Candidate:
    pos: List[float]
    timestamp: int
    confidence: float
    score: float
    crop: np.ndarray
    priority: int = 0
    metadata: Dict[str, Any] = None
    location: int = -1
    crop_pos: List[float] = None


def batch_data_convert_parts_to_object(batch_data: BDR, part_id: int, obj_id: int, min_overlap: float) -> BDR:
    # this function will replace the object position with the part position for all parts that are inside an object
    # a single part(1:1). other parts and objects will be invalidated
    new_batch_data = copy.deepcopy(batch_data)
    new_batch_data.data[:, BatchDataResolver.ID] = -1
    detections = batch_data.query(batch_data.SUBCLASS, part_id)
    if len(detections) == 0:
        return new_batch_data
    # first match face with person
    parts_idxs = detections[:, BatchDataResolver.INDEX].astype(int)
    obj_ids = batch_data.data[:, BatchDataResolver.ID].astype(int)
    obj_mask = (batch_data.data[:, batch_data.CLASS] == obj_id).astype(int)
    parts_overlaps = batch_data.all_overlaps[parts_idxs]
    part_m, obj_m = np.where(parts_overlaps * obj_mask > min_overlap)
    unique, counts = np.unique(part_m, return_counts=True)
    mapping = dict(zip(unique, counts))
    matches = {
        parts_idxs[part_m[i]]: obj_m[i]
        for i in range(len(part_m))
        if mapping.get(part_m[i], 0) == 1 and obj_ids[obj_m[i]] > -1
    }
    for lp, pidx in matches.items():
        new_batch_data.data[pidx, BatchDataResolver.POS] = batch_data.data[lp, batch_data.POS]
        new_batch_data.data[pidx, BatchDataResolver.ID] = batch_data.data[pidx, BatchDataResolver.ID]
    return new_batch_data


class BaseAdvanceAnalyzerConfig(BaseConfig):
    min_sec_between_calls: int = 0  # unlimited
    iou_similarity_th: float = 0.75
    max_candidates_same_place: int = 2


class BaseAdvanceAnalyzer:
    name: str
    enabled: bool = True
    filter: Union[Namespace, Dict]
    _config_type = BaseAdvanceAnalyzerConfig
    args: BaseAdvanceAnalyzerConfig

    is_night_mode: bool = False
    # history: Dict[int, History]
    supported_classes: list
    supported_subclasses: list
    candidates: Dict[int, List[Candidate]]
    processed: Dict[int, List[AttrData]]
    batch_process_list: List[Tuple[int, Candidate]]
    low_prio_set: Set[int] = set()
    use_low_prio_set: bool = False
    max_candidates: int = 5
    analyzer_batch_limit = 6
    expiration_time_sec: float = 3.0  # time to retire id after it wasn't seen
    max_history_duration: float = 180.0  # max time an id can stay in history without running the analysis
    low_prio_max_time: float = 5.0  # max time to spend in low priority set
    low_prio_ts_ms: float = 0.0  # when the last timestamp an object was taken from low priority set
    post_score_success_th: float = 0.5
    process_batch_size: int = 2
    keys_to_filter: List[str] = []
    night_keys_to_filter: List[str] = []
    occlusion_threshold: float = 0.4
    distance_threshold: float = 0.05
    post_score_failure_th: float = -1.0
    retry_on_failure = 0
    encoder_address = None
    low_prio_index = 0
    max_retries = 3
    crop_on_success_only = False
    _type = AdvanceAnalyzerType.BASE
    is_export_invalid_crops: bool = False
    filter_untracked: bool = True
    min_require_results: AdvanceAnalyticResults = AdvanceAnalyticResults.NOT_GOOD
    rate_limit: float = 0  # 0 unlimited, > 0 number of seconds between calls. not applied to high priority calls

    def __init__(self, msg: dict, context, instance_id: str = None):
        self.history = {}
        self.candidates = {}
        self.processed = {}
        self.batch_process_list = []
        self.low_prio_set = set()
        self.count = 0
        self.debug = False
        self.roi_filter = False
        self.filter = Namespace(roi=None, roi_filter=False)
        self.class_handler = context.class_handler
        self.low_prio_ts_ms = 0.0
        self.stats = {}
        self.crop_margins = None
        self.crop_ar = None
        self.zoom_ar = None
        self.zoom_crop_margins = None
        self.priority = advance_analyzer_priority[self._type]
        self.instance_id = instance_id
        if msg is None:
            msg = {}
        self.args = self._config_type(msg)
        self.last_call = 0
        self.rate_limit = self.args.min_sec_between_calls * 1000

    @staticmethod
    def calc_area(crop_info):
        return (crop_info[3] - crop_info[1]) * (crop_info[2] - crop_info[0])

    @staticmethod
    def calc_intersection_area(crop_info1, crop_info2):
        dx = min(crop_info1[2], crop_info2[2]) - max(crop_info1[0], crop_info2[0])
        dy = min(crop_info1[3], crop_info2[3]) - max(crop_info1[1], crop_info2[1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        return 0

    def track(self, image_batch, batch_data: BatchDataResolver, motion_data: MotionData):
        relevant_batch_ids = batch_data.unique(batch_data.ID, batch_data.SUBCLASS, self.supported_subclasses).astype(
            int
        )
        if self.enabled and len(relevant_batch_ids) > 0:
            score_table = self.calculate_pre_score(batch_data, image_batch)
            for obj_id in relevant_batch_ids:
                obj_id = int(obj_id)
                if obj_id != -1:
                    self.update_candidates(obj_id, score_table, batch_data, image_batch)

    def run(
        self, image_batch, batch_data: BatchDataResolver, ids_to_process: Dict[int, AdvanceAnalyticPriority]
    ) -> Tuple[List[Dict], Dict]:

        timestamps = [i.timestamp for i in image_batch]
        tidx = 0
        info_batch = {}
        images = {}
        score_table = {}  # default
        for ent_id in ids_to_process:
            # force new candidate if the entity wasn't processed already and there is no good candidate
            # also don't force if it's a regular flow and not part of alert
            force_new_candidate = ids_to_process[ent_id] > AdvanceAnalyticPriority.REQUIRED and (
                not (self.candidates.get(ent_id, None) or self.processed.get(ent_id, []))
            )
            if ids_to_process[ent_id] == AdvanceAnalyticPriority.IMMEDIATE or force_new_candidate:
                # force run L1 on what we have
                if len(batch_data) > 0:
                    self.update_candidates(ent_id, score_table, batch_data, image_batch, force_update=True)
            if (
                ids_to_process[ent_id] >= AdvanceAnalyticPriority.HIGH
                or timestamps[tidx] - self.last_call >= self.rate_limit
            ):
                if self.add_process_job(ent_id, is_last=ids_to_process[ent_id] == AdvanceAnalyticPriority.IMMEDIATE):
                    self.last_call = timestamps[tidx]
                    tidx = min(tidx + 1, len(timestamps) - 1)

        additional_work = list(ids_to_process.keys())
        while (
            len(self.batch_process_list) % self.analyzer_batch_limit > 0
            and additional_work
            and timestamps[tidx] - self.last_call >= self.rate_limit
        ):
            ent_id = additional_work.pop(0)
            if ent_id in self.candidates and self.candidates[ent_id]:
                if self.add_process_job(ent_id):
                    additional_work.append(ent_id)
                    self.last_call = timestamps[tidx]
                    tidx = min(tidx + 1, len(timestamps) - 1)

        work_batch = []
        while len(self.batch_process_list) > 0:
            id_job, candidate = self.batch_process_list.pop(0)
            crop = candidate.crop
            if candidate.crop_pos is not None:
                [x1, y1, x2, y2] = candidate.crop_pos
                crop = crop[y1:y2, x1:x2]
            work_batch.append(WorkItem(id_job, crop, candidate))
            if not self.crop_on_success_only and id_job not in images:
                images[id_job] = {"image": cv2.cvtColor(candidate.crop, cv2.COLOR_RGB2BGR)}

        ents_results = {}
        results = []
        if len(work_batch) > 0:
            results = self._run_model(work_batch)
        for ind, work_item in enumerate(work_batch):
            obj_id = work_item.id
            candidate = work_item.extra
            if results[ind].pop("use_image", False) is True:
                if obj_id not in images:
                    images[obj_id] = {}
                # make sure we always take the first image and zoom from the batch result - highest pre score
                if "image" not in images[obj_id]:
                    images[obj_id]["image"] = cv2.cvtColor(candidate.crop, cv2.COLOR_RGB2BGR)

            zoom_image = results[ind].pop("zoom_image", None)
            if zoom_image is not None:
                if obj_id not in images:
                    images[obj_id] = {}
                # make sure we always take the first image and zoom from the batch result - highest pre score
                if "zoom_image" not in images[obj_id]:
                    images[obj_id]["zoom_image"] = cv2.cvtColor(zoom_image, cv2.COLOR_RGB2BGR)

            raw_attr = results[ind]
            attr_data = AttrData(
                candidate.pos,
                candidate.timestamp,
                candidate.confidence,
                candidate.score,
                raw_attr,
                0,
                candidate.metadata,
            )
            post_score = self.calculate_post_score(obj_id, attr_data)
            attr_data.post_score = post_score
            self.add_results(obj_id, attr_data)
            merged, merged_score = self.merge_with_hist(raw_attr, obj_id)
            if merged is not None:
                info_batch[obj_id] = self.export_results(merged)
                ents_results[obj_id] = merged_score
            self.update_candidates_score(obj_id, attr_data)

        for ent_id in ids_to_process:
            success = AdvanceAnalyticResults.NO_VALID_CANDIDATE
            if ent_id in ents_results:
                if ents_results[ent_id] > self.post_score_success_th:
                    success = AdvanceAnalyticResults.SUCCESS
                elif ents_results[ent_id] > self.post_score_failure_th:
                    success = AdvanceAnalyticResults.NOT_GOOD
                else:
                    success = AdvanceAnalyticResults.FAILURE
            ents_results[ent_id] = success
        return info_batch, images, self.stats, ents_results

    def cleanup(self, ids_to_delete: List[int]):
        for obj_id in ids_to_delete:
            self.candidates.pop(obj_id, None)
            self.processed.pop(obj_id, None)

    def add_from_low_prio(self, work_batch, cur_time_ms):
        if not self.use_low_prio_set:
            return
        safe_check = 0  # to avoid infinite loop
        while (
            len(self.low_prio_set) > 0
            and (len(work_batch) == 0 or len(work_batch) % self.analyzer_batch_limit != 0)
            and safe_check < self.analyzer_batch_limit
        ):
            # iterate over the copy of the set to avoid changing it while iterating
            low_prio_list = list(self.low_prio_set)
            for i in range(len(low_prio_list)):
                obj_id = low_prio_list[self.low_prio_index % len(low_prio_list)]
                if self.history[obj_id].retries >= self.max_retries:
                    self.low_prio_set.discard(obj_id)  # no need to continue checking it
                    continue
                candidates = self.candidates[obj_id]
                if len(candidates) > 0:
                    scores = [cand.score for cand in candidates]
                    max_ind = np.argmax(scores)
                    best_candidate = self.candidates[obj_id].pop(max_ind)
                    work_batch.append(Namespace(candidate=best_candidate, id=obj_id))
                    self.low_prio_ts_ms = cur_time_ms
                    self.history[obj_id].retries += 1
                if len(self.candidates[obj_id]) == 0:
                    self.low_prio_set.discard(obj_id)
                self.low_prio_index += 1
            safe_check += 1

    def _retire_id(self, obj_id: int):
        if self.history[obj_id].active:
            # check if there are enough results for this id:
            processed = self.processed[obj_id]
            scores = np.asarray([attr.post_score for attr in processed])
            if not self.use_low_prio_set:
                if max(scores) < self.post_score_success_th:
                    self.add_process_job(obj_id)
            else:
                if max(scores) >= self.post_score_success_th:
                    self.low_prio_set.discard(obj_id)
            # remove id from active ids
            self.history[obj_id].active = False

    def merge_with_hist(self, attr: Dict, obj_id: int) -> (Dict, float):
        return attr, self.processed[obj_id][-1].post_score

    def generate_candidate(
        self, index, pre_score, batch_data, image_batch, priority: int = 0, metadata=None
    ) -> Candidate:
        info = batch_data[index]
        image = image_batch[int(info[BDR.FRAME_ID])].frame
        pos = info[BDR.POS]
        crop, crop_pos = self.crop_function(image, pos)
        timestamp = info[BDR.TIMESTAMP]
        location = int(info[BDR.LOCATION])
        return Candidate(pos, timestamp, info[BDR.CONFIDENCE], pre_score, crop, priority, metadata, location, crop_pos)

    def add_candidate(self, obj_id: int, candidate: Candidate):
        if obj_id in self.candidates:
            candidates = self.candidates[obj_id]
            if len(candidates) < self.max_candidates:
                self.candidates[obj_id].append(candidate)
            else:
                scores = [cand.score for cand in candidates]
                min_ind = np.argmin(scores)
                self.candidates[obj_id][min_ind] = candidate
        else:
            self.candidates[obj_id] = [candidate]

    def add_process_job(self, obj_id: int, is_last: bool = False) -> bool:
        is_added = False
        if obj_id in self.candidates:
            candidates = self.candidates[obj_id]
            if len(candidates) > 0:
                if is_last:
                    best_candidate = self.candidates[obj_id].pop()
                    self.batch_process_list.append((obj_id, best_candidate))
                else:
                    scores = [cand.score for cand in candidates]
                    max_ind = np.argmax(scores)
                    best_candidate = self.candidates[obj_id].pop(max_ind)
                    self.batch_process_list.append((obj_id, best_candidate))
            is_added = True
        return is_added

    def add_results(self, obj_id: int, att_data: AttrData):
        if obj_id not in self.processed:
            self.processed[obj_id] = []
        self.processed[obj_id].append(copy.deepcopy(att_data))

    def _calc_area_score(self, crop_sz, sub_class: int = -1):
        return 1

    def calculate_pre_score(self, batch_data: BatchDataResolver, image_batch=None):
        detections = batch_data.query(batch_data.SUBCLASS, self.supported_subclasses)
        if self.filter_untracked:
            detections = detections[detections[:, BatchDataResolver.ID] > 0]
        scores = {}
        img_sz = image_batch[0].frame.shape
        occlusions = batch_data.overlaps
        for idx, crop_info in enumerate(detections):
            crop_ind = int(crop_info[batch_data.INDEX])
            crop_id = int(crop_info[batch_data.ID])

            # tracked criteria
            if crop_id < 0:
                scores[crop_ind] = -1
                continue

            confidence = crop_info[batch_data.CONFIDENCE]

            # area criteria
            crop_pos = crop_info[BatchDataResolver.POS]
            crop_sz = [(crop_pos[2] - crop_pos[0]) * img_sz[1], (crop_pos[3] - crop_pos[1]) * img_sz[0]]
            area_score = self._calc_area_score(crop_sz, crop_info[batch_data.SUBCLASS])
            if area_score == 0:
                scores[crop_ind] = -1
                continue

            # proximity to previous crop
            if int(crop_info[batch_data.ID]) in self.processed:
                poses = np.array([p.pos for p in self.processed[int(crop_info[batch_data.ID])]])
                centers = np.column_stack(((poses[:, 0] + poses[:, 2]) / 2, (poses[:, 1] + poses[:, 3]) / 2))
                distances = np.linalg.norm(centers - crop_info[batch_data.CENTER] / ROI_SHAPE, axis=1)
                if np.min(distances) < self.distance_threshold:
                    scores[crop_ind] = -1
                    continue

            # occlusion criteria
            if occlusions[crop_ind] > self.occlusion_threshold:
                scores[crop_ind] = -1
                continue
            scores[crop_ind] = (
                area_score * confidence * (1 - occlusions[crop_ind] * float(self.occlusion_threshold < 0))
            )
        return scores

    def calculate_post_score(self, obj_id: int, attr_data: AttrData) -> float:
        return 0

    def update_candidates_score(self, obj_id: int, attr_data: AttrData):
        cur_x = (attr_data.pos[0] + attr_data.pos[2]) / 2
        cur_y = (attr_data.pos[1] + attr_data.pos[3]) / 2
        ind_to_del = []
        for ind, candidate in enumerate(self.candidates[obj_id]):
            candidate_x = (candidate.pos[0] + candidate.pos[2]) / 2
            candidate_y = (candidate.pos[1] + candidate.pos[3]) / 2
            distance = ((cur_x - candidate_x) ** 2 + (cur_y - candidate_y) ** 2) ** 0.5
            if distance < self.distance_threshold:
                ind_to_del.append(ind)
        for i in range(len(ind_to_del)):
            del self.candidates[obj_id][ind_to_del[-i - 1]]

    def _run_model(self, batch: List[WorkItem]) -> List[Any]:
        pass

    def get_crop_margins(self):
        return None

    def export_results(self, att_data):

        prop_data = self.to_properties(att_data)

        # filter non relevant attributes
        for key in self.keys_to_filter:
            if key in prop_data:
                del prop_data[key]
        if self.is_night_mode and self.night_keys_to_filter:
            for key in self.night_keys_to_filter:
                if key in prop_data:
                    del prop_data[key]
        return prop_data

    def _add_metrics_to_properties(self, attr_data: Dict, prop_data: Dict):
        # updates prop_data with metrics from attr_data
        metrics = attr_data.pop("metrics", {})
        if metrics:
            metrics_list = []
            for metric, value in metrics.items():
                metrics_list.append(AttrProperty(metric, value, AttrConfidence.HIGH, self.priority))
            prop_data["metrics"] = metrics_list

    def to_properties(self, attr_data: Dict) -> Dict:

        if "scores" in attr_data:
            scores: Dict = attr_data.pop("scores")
        else:
            scores = {}

        prop_data = {}

        self._add_metrics_to_properties(attr_data, prop_data)

        for key in attr_data:
            if isinstance(attr_data[key], DescriptorVector):
                prop_data[key] = attr_data[key]
            else:
                score = scores.get(key, 1)
                prop_data[key] = (
                    [AttrProperty(val, score, AttrConfidence.HIGH, self.priority) for val in attr_data[key]]
                    if isinstance(attr_data[key], list)
                    else AttrProperty(attr_data[key], score, AttrConfidence.HIGH, self.priority)
                )
        return prop_data

    def get_metadata(self, crop_info):
        return {}

    def add_to_low_prio_set(self, obj_id, metadata):
        self.low_prio_set.add(obj_id)

    def update_candidates(
        self,
        obj_id: int,
        score_table: Dict[int, float],
        batch_data: BatchDataResolver,
        image_batch,
        force_update: bool = False,
    ) -> Optional[Candidate]:
        indices = batch_data.query(batch_data.ID, obj_id, batch_data.INDEX).astype(int)
        best_index = -1
        best_score = -1
        for row_ind in indices:
            if score_table.get(row_ind, -1) > best_score:
                best_index = row_ind
                best_score = score_table[row_ind]

        if best_index < 0 or (best_score < 0 and not force_update):
            return None

        is_validated, new_score = self.validate_candidate(best_index, batch_data, image_batch)
        if not is_validated:
            return None
        elif new_score is not None:
            cur_score = new_score
        else:
            cur_score = best_score
        row_ind = best_index

        # this ensures we always have a candidate
        cur_candidates = batch_data.data[row_ind]
        add_candidate = False
        # if there are vacant candidates slots
        if (
            obj_id not in self.candidates
            or len(self.candidates[obj_id]) == 0
            or (cur_score > 0 and len(self.candidates[obj_id]) < self.max_candidates)
        ):
            add_candidate = True
        # if score is legal and
        elif cur_score > 0:
            cand_scores = np.array([c.score for c in self.candidates[obj_id]])
            if np.any(cur_score > cand_scores):
                can_bboxes = np.array([c.pos for c in self.candidates[obj_id]])
                ious = bbox_ious(np.atleast_2d(cur_candidates[BDR.POS]), np.atleast_2d(can_bboxes))[0]
                same_place = ious > self.args.iou_similarity_th
                if sum(same_place > 0) < self.args.max_candidates_same_place:
                    add_candidate = True
                else:
                    min_idx = np.argmin(np.where(same_place, cand_scores, 1))
                    if same_place[min_idx]:
                        del self.candidates[obj_id][min_idx]
                        add_candidate = True

        if add_candidate:
            metadata = self.get_metadata(cur_candidates)
            candidate = self.generate_candidate(int(row_ind), cur_score, batch_data, image_batch, metadata=metadata)
            self.add_candidate(obj_id, candidate)
            return candidate
        else:
            return None

    @property
    def analyzer_type(self):
        return self._type

    def update_filters(self, enabled, filters):
        if enabled is not None:
            self.enabled = enabled
        if filters is not None:
            self.filter = filters

    def prepare_batch_data(self, batch_data: BatchDataResolver) -> BatchDataResolver:
        return batch_data

    def crop_function(self, image: np.ndarray, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos_with_margin = fit_bbox_with_margins(pos, image.shape[1::-1], self.crop_margins)
        if self.crop_ar is None:
            [x1, y1, x2, y2] = pos_with_margin
            crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crop_pos = None
        else:
            im_size = image.shape[1::-1]
            bb = pos * np.array([*im_size, *im_size])
            crop, crop_pos = crop_image_by_bbox_and_ar(
                image, bb, self.crop_ar, self.crop_margins, bgr_map=False, return_pos=True
            )
            crop_pos = np.maximum(np.array(pos_with_margin) - np.hstack((*crop_pos[:2], crop_pos[:2])), 0)
        return crop, crop_pos

    def crop_zoom_function(self, image: np.ndarray, pos: np.ndarray) -> np.ndarray:
        if self.zoom_ar is None:
            return crop_image_by_bbox(image, pos, self.zoom_crop_margins)
        return crop_image_by_bbox_and_ar(image, pos, self.zoom_ar, self.zoom_crop_margins)

    def validate_candidate(
        self, best_index: int, batch_data: BDR, image_batch: List[AnalyticImage]
    ) -> Tuple[bool, Optional[float]]:
        return True, None


class ExtApiConfig(BaseAdvanceAnalyzerConfig):

    api_token: str = None
    api_url: str = None
    jpeg_quality: int = 80
    timeout_sec: int = 2


class ExtApiAnalyzer(BaseAdvanceAnalyzer):
    _config_type = ExtApiConfig
    args: ExtApiConfig

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)
        self.stats = {"api_errors": 0, "api_calls": 0}

    def _run_model(self, batch: List[WorkItem]) -> List:
        results = [{}] * len(batch)
        for i, work_item in enumerate(batch):

            # Create a JPEG bytes buffer directly from the image
            jpeg_bytes = BytesIO()
            Image.fromarray(work_item.image).save(jpeg_bytes, format="JPEG", quality=self.args.jpeg_quality)

            # Reset the buffer pointer to the beginning (no need to re-encode or copy data)
            jpeg_bytes.seek(0)

            try:
                t0 = time.time()
                self.stats["api_calls"] += 1
                api_res = self._run_api(jpeg_bytes)
            except Exception as e:
                log_exception(logger, "ALPR Connection error", e)
                logger.warning(f"Time for unsuccessful ALPR attempt: {time.time() - t0}")
                api_res = None
            if api_res is None:
                self.stats["api_errors"] += 1
            else:
                results[i] = self._parse_api_results(api_res, work_item)
        return results

    def _run_api(self, jpeg_bytes: BytesIO) -> Optional[Dict]:
        pass

    def _parse_api_results(self, api_res: Dict, work_item: WorkItem) -> Dict:
        pass


class AdvanceFactory:
    _instance = None
    analyzers: List[BaseAdvanceAnalyzer] = []
    use_static_analyzers: bool = False

    def __new__(cls, use_static_detector: bool = False):
        if cls._instance is None:
            cls._instance = super(AdvanceFactory, cls).__new__(cls)
            cls._instance.use_static_detector = use_static_detector
        return cls._instance

    def create(self, analyzer_name: str, msg: dict, context, instance_id: str = "") -> BaseAdvanceAnalyzer:
        if msg is None:
            msg = {}
        if self.use_static_detector:
            print("Static mode (shared) Not supported - producing a new detector")
        if analyzer_name == AdvanceAnalyzerType.HUMAN_PARSING:
            from .pa_recognition import pa_analyzer

            logger.info("Initialize person attributes")
            return pa_analyzer.PersonAttrAnalyzer(msg, context)
        elif analyzer_name == AdvanceAnalyzerType.ALPR:
            from .alpr import plate_analyzer

            logger.info("Initialize ALPR")
            return plate_analyzer.AlprAnalyzer(msg, context, instance_id)
        elif analyzer_name == AdvanceAnalyzerType.WEAPONS:
            logger.info("Initialize weapons classifier")
            from .weapons_classifier import weapon_analyzer

            return weapon_analyzer.WeaponAnalyzer(msg, context)
        elif analyzer_name == AdvanceAnalyzerType.FACE:
            logger.info("Initialize face id")
            from .face import face_analyzer

            return face_analyzer.FaceAnalyzer(msg, context)
        elif analyzer_name == AdvanceAnalyzerType.VEHICLE:
            logger.info("Initialize vehicle red and attributes")
            from .vehicle import vehicle_analyzer

            return vehicle_analyzer.VehicleAnalyzer(msg, context)
        elif analyzer_name == AdvanceAnalyzerType.LPREC:
            logger.info("Initialize vehicle red and attributes")
            from .license_plate import lp_analyzer

            return lp_analyzer.LPRAnalyzer(msg, context)

        elif analyzer_name == AdvanceAnalyzerType.CONTAINER:
            logger.info("Initialize container analyzer")
            from .shipping_container.container_analyzer import create_container_analyzer

            return create_container_analyzer(msg, context, instance_id)
        else:
            raise ValueError(
                f"analyzer_name({analyzer_name}) must be a supported analyzers:{AdvanceAnalyzerType.list()}"
            )
