import json
from collections import deque, defaultdict
from copy import deepcopy
from math import floor, ceil
from typing import List, Dict, Optional

import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious  # noqa
from numpy import ndarray

from level1.alpr.plate_analyzer import get_alpr_data
from .analyzer_general import ROI_SHAPE, logger, log_exception
from .cloud_converter import convert_to_trackers
from .core import (
    BatchDataResolver,
    Event,
    ClassHandler,
    AdvanceAnalyzerType,
    AnalyticImage,
    is_l1_not_failure,
    MotionData,
    DescriptorVector,
    ObjectManager,
)
from .entity import ActiveEntityData
from .image_encoder import ImageEncoder
from .img_utils import crop_image, crop_with_minimal
from .state_objects import StaticObjectManager
from .state_shelves import StaticShelvesManager
from .stats_collector import RobustStatsCollector

BLACKLIST_UNMATCH = -1
BLACKLIST_MOTION = 0
BLACKLIST_CONFIDENCE = 0  # no difference between motion and confidence


def _update_db_dict(db_dict, obj_data, mean_key, neg_key, thr_value):
    if obj_data.object_type not in db_dict:
        db_dict[obj_data.object_type] = {"pos": (None, None, None), "neg": (None, None, None)}
    pos_data = db_dict[obj_data.object_type]["pos"]
    array_thr = np.array([thr_value])
    object_id = np.array([obj_data.custom_object.customObjectId])
    obj_pos_data = np.atleast_2d(getattr(obj_data, mean_key))
    if obj_pos_data is not None and obj_pos_data.size > 0:
        if pos_data[0] is None:
            db_dict[obj_data.object_type]["pos"] = (obj_pos_data, array_thr, object_id)
        else:
            db_dict[obj_data.object_type]["pos"] = (
                np.vstack([pos_data[0], obj_pos_data]),
                np.hstack([pos_data[1], array_thr]),
                np.hstack([pos_data[2], object_id]),
            )
    if neg_key is not None:
        neg_data = db_dict[obj_data.object_type]["neg"]
        neg_val = np.atleast_2d(getattr(obj_data, neg_key))
        if neg_val is not None and neg_val.size > 0:
            array_thr = np.repeat(array_thr, neg_val.shape[0], axis=0)
            object_id = np.repeat(object_id, neg_val.shape[0], axis=0)
            if neg_data[0] is None:
                db_dict[obj_data.object_type]["neg"] = (neg_val, array_thr, object_id)
            else:
                db_dict[obj_data.object_type]["neg"] = (
                    np.vstack([neg_data[0], neg_val]),
                    np.hstack([neg_data[1], array_thr]),
                    np.hstack([neg_data[2], object_id]),
                )


def _update_text_db_dict(db_dict: Dict, custom_object, object_type: int):
    if object_type not in db_dict:
        db_dict[object_type] = {"pos": (None, None, None), "neg": (None, None, None)}
    pos_data = db_dict[object_type]["pos"]
    array_thr = np.array([custom_object.clipTextThr])
    object_id = np.array([custom_object.customObjectId])
    if pos_data[0] is None:
        db_dict[object_type]["pos"] = (np.atleast_2d(custom_object.prompt_descriptor), array_thr, object_id)
    else:
        db_dict[object_type]["pos"] = (
            np.vstack([pos_data[0], custom_object.prompt_descriptor]),
            np.hstack([pos_data[1], array_thr]),
            np.hstack([pos_data[2], object_id]),
        )


def _match_custom_object(ent_descriptor, obj_db) -> Dict:
    if not obj_db or ent_descriptor is None:
        return {}

    pos_mean_reid, match_th, obj_ids = obj_db["pos"]
    if pos_mean_reid is None:
        return {}
    scores = ent_descriptor @ pos_mean_reid.T
    pos_objects = {int(obj_ids[i]): scores[i] for i in range(len(scores)) if scores[i] > match_th[i]}

    neg_mat, match_th, obj_ids = obj_db["neg"]
    neg_objects = set()
    if neg_mat is not None:
        neg_scores = ent_descriptor @ neg_mat.T
        valid_neg = np.argwhere(neg_scores > match_th).flatten()
        for idx in valid_neg:
            obj_id = int(obj_ids[idx])
            if obj_id not in neg_objects and neg_scores[idx] > pos_objects.get(obj_id, 0):
                neg_objects.add(obj_id)
                pos_objects.pop(obj_id, None)

    return {"pos": list(pos_objects.keys()), "neg": list(neg_objects)}


def merge_custom_object_data(data1: Dict, data2: Dict) -> Dict:
    pos = set(data1.get("pos", [])).union(set(data2.get("pos", [])))
    neg = set(data1.get("neg", [])).union(set(data2.get("neg", [])))
    is_clip = data1.get("is_clip", False) or data2.get("is_clip", False)
    is_reid = data1.get("is_reid", False) or data2.get("is_reid", False)
    return {"pos": list(pos - neg), "neg": list(neg), "is_clip": is_clip, "is_reid": is_reid}


class EntityDB:
    active_objects: Dict[int, ActiveEntityData]  # id: data
    inactive_objects: Dict[int, ActiveEntityData]  # location: data
    blacklist_objects: Dict[int, ActiveEntityData]  # location: data
    blacklist_id2orig_id: Dict[int, int]  # save the original id of reappering blacklist object
    sync_period: int
    inactivity_period: int
    _area_maps: List
    on_entities_update: Event
    on_reid_match: Event
    on_entities_removed: Event
    on_entities_purged: Event
    on_entities_stats_update: Event
    zoom_images_collection: deque

    def __init__(self, context, inactivity_period_sec=5, sync_period_ms: int = 5000):
        config = context.config.get("entityManagement", {})
        self.collection_len = 20
        self.sync_period = sync_period_ms
        self.inactivity_period = int(inactivity_period_sec * 1000)
        self.stats_sync_period = config.get("stats_collection", {}).get("period_sec", 600) * 1000

        self.removal_period = 10 * self.inactivity_period + 1  # to make sure we always have data available
        self.active_objects = {}
        self.inactive_objects = {}
        self.blacklist_objects = {}
        self.blacklist_id2orig_id = {}
        self.update_timestamp = 0
        # self._route_solver = RouteSolver()
        self._area_maps = []
        self.on_entities_update = Event()
        self.on_reid_match = Event()
        self.on_entities_removed = Event()
        self.on_entities_purged = Event()
        self.on_entities_stats_update = Event()
        self._next_sync_ts = 0
        self._next_max_hist_sync_ts = 0
        self.entities_attributes_report = set()
        self.context = context
        self._roi_filter = []
        self._obj_to_roi_filter = {}
        self._class_filters = []
        self.objects_ids = context.class_handler.object_ids
        self.zero_appearance = {obj: 0 for obj in self.objects_ids}
        self.zoom_images_collection = deque(maxlen=self.collection_len)
        self.small_images_collection = {}
        self.minimal_crop_size = [150, 150]
        self.minimal_crop_ar = {}
        self.blacklist_enabled = config.get("enable_blacklist", False)
        self.blacklist_iou_th = config.get("blacklist_iou", 0.8)
        self.tag_blacklist = config.get("flag_blacklist", True)
        self.blacklist_span_th = config.get("blacklist_span", 2)
        self.blacklist_encode_th = config.get("blacklist_encode", 0.8)
        self.blacklist_expiration_ms = config.get("blacklist_expiration_min", 60 * 8) * 60 * 1000
        self.blacklist_use_classifier = config.get("blacklist_use_classifier", False)
        self.blacklist_low_conf_objects = set(config.get("blacklist_low_confidence_objects", [0]))
        self.blacklist_confidence_th = config.get("blacklist_conf_thresh", context.tracker.get_confidence_th())
        self.blacklist_low_confidence_th = config.get(
            "blacklist_low_confidence_th", context.tracker.get_confidence_th() * 2
        )
        self.blacklist_motion_th = config.get("blacklist_motion_thresh", 10)
        bounds = config.get("blacklist_bounds", 2.5)
        self.center_bound = [bounds, ROI_SHAPE[0] - bounds]
        self.tracker_update = context.tracker.merge_ids
        if self.blacklist_encode_th > 0:
            self.image_encoder = ImageEncoder(config.get("encoderParams", {}))
            self.encoding_function = self.image_encoder.encode_patch
        else:
            self.encoding_function = lambda *args: None

        self.stat_collector = RobustStatsCollector(self)
        thumb_config = context.config.get("trainThumbPolicy", {}).get("crops_ar", {})
        class_handler = context.class_handler
        for k, v in thumb_config.items():
            obj_val = class_handler.object_str_to_int(k)
            if obj_val >= 0:
                self.minimal_crop_ar[obj_val] = v

        self.state_object_manager = StaticObjectManager(context)
        self.state_shelves_manager = StaticShelvesManager(context)
        self.additional_managers: List[ObjectManager] = []

        self.custom_objects_db_reids = {}
        self.custom_objects_db_clips = {}
        self.custom_objects_db_texts = {}
        self.on_db_update()
        context.on_db_update += self.on_db_update

    @property
    def class_handler(self) -> ClassHandler:
        return self.context.class_handler

    def encode_matching(self, encoding, query):
        if encoding is None or query is None:
            return True
        return self.image_encoder.match_encoding(encoding, query) > self.blacklist_encode_th

    def sync_and_cleanup(self, end_timestamp: int = None, force: bool = False) -> Optional[Dict]:

        if end_timestamp > self._next_max_hist_sync_ts:
            stats = self.stat_collector.collect()
            self._next_max_hist_sync_ts = (end_timestamp // self.stats_sync_period + 1) * self.stats_sync_period
            if stats:
                self.on_entities_stats_update(stats)

        if not force and end_timestamp < self._next_sync_ts:
            return None

        obj_info = []
        keys = list(self.active_objects.keys())
        removed = []
        att_dict = {}
        objects_count = {**dict.fromkeys(self.objects_ids, 0)}

        for entity_id in self.entities_attributes_report:
            att_dict[entity_id] = self.get_entity(entity_id).get_attribute_report()
        self.entities_attributes_report.clear()

        for oid in range(len(keys)):
            obj = self.active_objects[keys[oid]]
            sync_report = obj.sync(end_timestamp, roi_filter=self._obj_to_roi_filter[obj.object_id])
            if sync_report is not None and obj.class_id in self._class_filters:
                obj_info.append(sync_report)
                objects_count[obj.object_id] += 1
            if end_timestamp - obj.last_seen > self.inactivity_period:
                self.apply_to_blacklist(obj)
                self.inactive_objects[keys[oid]] = obj
                obj.deactivate()
                del self.active_objects[keys[oid]]
                removed.append(keys[oid])
        if len(removed) > 0:
            self.on_entities_removed(removed)

        # handle expired inactives
        time_th = end_timestamp - self.removal_period
        expired_ids = [
            id_
            for id_ in self.inactive_objects
            if self.inactive_objects[id_].last_seen < time_th and id_ not in removed
        ]
        for oid in expired_ids:
            del self.inactive_objects[oid]
        if len(expired_ids) > 0:
            self.on_entities_purged(expired_ids)
            for eid in expired_ids:
                self.small_images_collection.pop(eid, None)
                self.blacklist_id2orig_id.pop(eid, None)

        self._next_sync_ts = floor(end_timestamp / self.sync_period) * self.sync_period + self.sync_period

        obj_count_names = {}
        obj_total_count = 0
        for k in objects_count:
            if objects_count[k] > 0:
                obj_count_names[self.class_handler.get_object_name(k)] = objects_count[k]
                obj_total_count += objects_count[k]
        if len(att_dict) > 0 or obj_total_count > 0 or len(obj_info) > 0:
            trackers = convert_to_trackers(obj_info, att_dict)
            report = {"attributes": att_dict, "info": obj_info, "objects": obj_count_names, "trackers": trackers}
        else:
            report = None
        if self.blacklist_enabled:
            bl_ids = list(self.blacklist_objects.keys())
            expiration_th = end_timestamp - self.blacklist_expiration_ms
            for id_ in bl_ids:
                if self.blacklist_objects[id_].last_seen < expiration_th:
                    self.blacklist_objects.pop(id_, None)
                    logger.info(f"Remove id {id_} from blacklist due to expiration time")

        return report

    def query(self, object_ids, class_ids=None, filters: Dict = None) -> List[int]:
        candidates = list(self.active_objects.keys())

        is_dis = filters is None
        candidates = [idx for idx in candidates if self.active_objects[idx].match_filters(object_ids, filters, is_dis)]

        if class_ids is not None:
            candidates = [idx for idx in candidates if self.active_objects[idx].class_id in class_ids]
        return candidates

    def force_get_ent_description_json(self, ent_id, crop):
        desc = {}
        if ent_id in self.active_objects:
            ent_data = self.active_objects[ent_id]
            if ent_data.object_id == self.class_handler.vehicle_value:
                # for now, run alpr anyway
                desc_dict = get_alpr_data(crop)
                if "plate" not in desc_dict and "description" in ent_data.attributes:
                    desc_dict = ent_data.attributes["description"]
                    pass
            else:
                desc_dict = deepcopy(ent_data.attributes)
                if "descriptor" in desc_dict:
                    del desc_dict["descriptor"]
            desc = json.dumps(desc_dict)
        return desc

    def register_zoom_image(self, ent_id, zoom_image):
        self.zoom_images_collection.append({"ent_id": ent_id, "zoom_image": zoom_image})

    def query_zoom_image(self, ent_id):
        zoom_dict = None
        for d in self.zoom_images_collection:
            if d["ent_id"] == ent_id:
                zoom_dict = d
                break
        if zoom_dict is not None:
            self.zoom_images_collection.remove(zoom_dict)
            return zoom_dict["zoom_image"]
        return None

    def add_small_image(self, ent_id: int, bbox: np.array, image: ndarray):
        ent_data = self.get_entity(ent_id)
        if ent_data is not None:
            ar = self.minimal_crop_ar.get(ent_data.object_id, None)
            self.small_images_collection[ent_id] = crop_with_minimal(image, bbox, self.minimal_crop_size, ar)

    def get_largest_image(self, ent_id):
        # ent_data = self.get_entity(ent_id)
        # if ent_data is None or ent_data.is_blacklisted:
        #    return None
        return self.small_images_collection.pop(ent_id, None)

    def visual_search(self, visual_key) -> Optional[int]:
        return None
        raise NotImplementedError

    def get_ent_crop_info(self, ent_id, analyzer_type: AdvanceAnalyzerType, is_zoom: bool = False):
        if ent_id in self.active_objects:
            ent_data = self.active_objects[ent_id]
        elif ent_id in self.inactive_objects:
            ent_data = self.inactive_objects[ent_id]
        else:
            return None
        if analyzer_type == AdvanceAnalyzerType.FACE:
            crop_type = "face"
        elif analyzer_type == AdvanceAnalyzerType.ALPR and is_zoom:
            crop_type = "plate"
        else:
            crop_type = self.class_handler.get_object_name(ent_data.object_id)

        info = {
            "id_base": ent_data.first_seen,
            "id_index": ent_data.track_id,
            "type": crop_type,
            "crop_index": ent_data.crop_index,
        }
        ent_data.crop_index += 1
        return info

    def add_ent_image(self, ent_id, image_name, is_zoom: bool = False, image_data: np.array = None):
        ent_data = self.get_entity(ent_id)
        if ent_data is not None:
            if is_zoom:
                ent_data.zoom_image = image_name
            else:
                ent_data.best_image = image_name
                self.small_images_collection.pop(ent_id, None)
            self.entities_attributes_report.add(ent_id)

    def add_ent_encoding(self, ent_id, encoding):
        ent_data = self.get_entity(ent_id)
        if ent_data is not None:
            ent_data.clip = DescriptorVector(encoding)
            self.entities_attributes_report.add(ent_id)

    def get_entity(self, ent_id) -> Optional[ActiveEntityData]:
        if ent_id in self.active_objects:
            return self.active_objects[ent_id]
        elif ent_id in self.inactive_objects:
            return self.inactive_objects[ent_id]
        return None

    def get_ent_image(self, ent_id, is_zoom: bool = False) -> Optional[str]:
        image_name = None
        ent_data = self.get_entity(ent_id)
        if ent_data is not None:
            if is_zoom:
                image_name = ent_data.zoom_image
            else:
                image_name = ent_data.best_image
        return image_name

    def track(self, batch_data: BatchDataResolver, image_batch: List[AnalyticImage], motion_data: MotionData):
        changed_entities = []
        ids = batch_data.unique(batch_data.ID).astype(int).tolist()
        objects_appearance = batch_data.count_objects(self.zero_appearance)
        sizes_data = []
        for ent_id in ids:
            if ent_id < 0:
                continue
            id_data = batch_data.query(batch_data.ID, ent_id)
            if ent_id not in self.active_objects:
                if ent_id in self.inactive_objects:
                    # take out from inactive objects
                    ent_data = self.inactive_objects.pop(ent_id)
                    ent_data.reactivate()
                    self.active_objects[ent_id] = ent_data
                else:
                    # search descriptor in l1 results:
                    image = image_batch[int(id_data[-1, BatchDataResolver.FRAME_ID])].frame
                    im_crop = crop_image(image, id_data[-1, BatchDataResolver.POS])
                    encoding = self.encoding_function(im_crop)
                    blacklist_id = self.match_ent_to_blacklist(id_data, motion_data, encoding)
                    timestamp = id_data[0, BatchDataResolver.TIMESTAMP]
                    init_pos = id_data[0, BatchDataResolver.POS]
                    self.active_objects[ent_id] = ActiveEntityData(ent_id, int(timestamp), self.class_handler, init_pos)
                    self.active_objects[ent_id].encoding = encoding
                    if blacklist_id != BLACKLIST_UNMATCH:
                        self.active_objects[ent_id].flags.add("blacklist")
                    else:
                        pass
                        # found_prev = False
                        # if "descriptor" in attr:
                        #    match = self.visual_search(attr["descriptor"])
                        #    if match is not None:
                        #        self.on_reid_match(ent_id, match)
                        #        ent_id = match
                        #        found_prev = True
                        # if not found_prev:
                        # timestamp = id_data[0, BatchDataResolver.TIMESTAMP]
                        # self.active_objects[ent_id] = ActiveEntityData(ent_id, int(timestamp), self.class_handler)
                        # self.active_objects[ent_id].encoding = encoding

            active_ent = self.active_objects[ent_id]
            is_type_changed, is_moved, enlarged = active_ent.update(id_data, objects_appearance)
            if is_type_changed:
                changed_entities.append(ent_id)
            if is_moved and active_ent.is_blacklisted:
                max_span = active_ent.locations.calc_max_span()
                factor = 1 if active_ent.max_confidence > self.blacklist_confidence(active_ent.object_id) else 2
                if max_span > self.blacklist_span_th * factor:
                    active_ent.flags.remove("blacklist")
                    changed_entities.append(ent_id)

            if is_moved:
                sizes_data.append(id_data[-1, :])

            if enlarged >= 0 and active_ent.best_image is None:  # not active_ent.is_blacklisted and
                self.add_small_image(ent_id, id_data[enlarged, BatchDataResolver.POS], image_batch[enlarged].frame)
        if sizes_data:
            self.stat_collector.add_id_data(np.vstack(sizes_data))

        if len(changed_entities) > 0:
            self.on_entities_update(changed_entities)
            self.entities_attributes_report.update(changed_entities)

        # tracking additional requirements
        for manager in self.additional_managers:
            manager.track(batch_data, image_batch, motion_data)

    def update_l1_attributes(self, l1_results: dict):
        changed_entities = []
        obj_attr = defaultdict(dict)
        obj_analyzers = defaultdict(set)

        # preprocess l1
        for l1_res in l1_results:
            for obj in l1_res["results"]:
                obj_attr[obj].update(l1_res["results"][obj])
                if l1_res["success"].get(obj, 1) >= 0:
                    obj_analyzers[obj].add(l1_res["analyzer"].value)

        for ent_id in obj_attr:
            ent_data = self.get_entity(ent_id)
            if ent_data is not None:
                has_desc = "descriptor" in obj_attr[ent_id]
                ent_data.update_attr(obj_attr[ent_id], obj_analyzers[ent_id])
                if obj_analyzers[ent_id]:
                    changed_entities.append(ent_id)
                if has_desc:
                    obj_db = self.custom_objects_db_reids.get(ent_data.object_id, {})
                    custom_obj = _match_custom_object(ent_data.visual_id, obj_db)
                    custom_obj["is_reid"] = True
                    ent_data.custom_object_data.update(custom_obj)
        if len(changed_entities) > 0:
            self.on_entities_update(changed_entities)
            self.entities_attributes_report.update(changed_entities)

    def is_blacklist_by_motion(self, ent_data: ndarray, motion_data: MotionData) -> bool:
        if self.blacklist_motion_th == 0:
            return False
        if motion_data.index < self.blacklist_motion_th:
            return True
        x1_min_floor = floor(np.min(ent_data[:, 0]) * ROI_SHAPE[0])  # floor of min x1
        y1_min_floor = floor(np.min(ent_data[:, 1]) * ROI_SHAPE[1])  # floor of min y1
        x2_max_ceil = ceil(np.max(ent_data[:, 2]) * ROI_SHAPE[0])  # ceil of max x2
        y2_max_ceil = ceil(np.max(ent_data[:, 3]) * ROI_SHAPE[1])  # ceil of max y2

        motion_region = motion_data.unified[y1_min_floor:y2_max_ceil, x1_min_floor:x2_max_ceil]
        max_motion = 0 if motion_region.size == 0 else np.max(motion_region)
        return max_motion < self.blacklist_motion_th

    def match_ent_to_blacklist(self, ent_data: ndarray, motion_data: MotionData, encoding: Optional[np.array] = None):

        # check if object moved
        center = ent_data[:, BatchDataResolver.CENTER]
        spans = np.ptp(center, axis=0)
        if np.any(spans > self.blacklist_span_th):
            return BLACKLIST_UNMATCH

        # check if object is in blacklist
        if len(self.blacklist_objects) > 0:
            bbox = ent_data[-1:, BatchDataResolver.POS]
            for id_, bl in self.blacklist_objects.items():
                if np.isin(bl.object_id, ent_data[:, BatchDataResolver.CLASS]):
                    iou = bbox_ious(bl.last_bbox[np.newaxis, :], bbox)
                    if iou.squeeze() > self.blacklist_iou_th:
                        if self.encode_matching(encoding, bl.encoding):
                            return id_

        obj_id = ent_data[-1, BatchDataResolver.CLASS]

        # check if object is in blacklist by confidence
        confidence = np.max(ent_data[:, BatchDataResolver.CONFIDENCE])
        is_edge = np.any((center < self.center_bound[0]) | (center > self.center_bound[1]))
        should_factor = not is_edge or self.context.is_night_mode
        bl_conf_th = self.blacklist_confidence(obj_id)
        conf_th = min(0.95, (bl_conf_th * 1.5) if should_factor else bl_conf_th)
        if confidence < conf_th:
            return BLACKLIST_CONFIDENCE
        elif obj_id not in self.blacklist_low_conf_objects:
            return BLACKLIST_UNMATCH

        # check if object is in blacklist by motion
        if self.is_blacklist_by_motion(ent_data[:, BatchDataResolver.POS], motion_data):
            return BLACKLIST_MOTION

        return BLACKLIST_UNMATCH

    def apply_to_blacklist(self, ent_data: ActiveEntityData):
        if self.blacklist_enabled:
            if not ent_data.is_blacklisted:
                # confidence test
                if ent_data.mean_confidence > self.blacklist_confidence_th:
                    return

                # location history test
                max_span = ent_data.locations.calc_max_span()
                if max_span > self.blacklist_span_th:
                    return

                if self.blacklist_use_classifier and is_l1_not_failure(ent_data.attributes):
                    return

                # best thing if we can encode it somehow and save the encoding vector, for later comparison
                if self.tag_blacklist:
                    ent_data.flags.add("blacklist")

            is_exist_in_bl = False
            for id_, bl in self.blacklist_objects.items():
                if bl.object_id == ent_data.object_id:
                    iou = bbox_ious(bl.last_bbox[np.newaxis, :], ent_data.last_bbox[np.newaxis, :])
                    if iou.squeeze() > 0.95:
                        is_exist_in_bl = True
                        # add connection between new id and original id
                        self.blacklist_id2orig_id[ent_data.track_id] = self.blacklist_id2orig_id.get(
                            bl.track_id, bl.track_id
                        )
                        break
            if not is_exist_in_bl:
                self.blacklist_objects[ent_data.track_id] = ent_data
                self.blacklist_id2orig_id[ent_data.track_id] = ent_data.track_id

    def get_area_map(self):
        area_map = np.stack(self._area_maps)
        area_map[area_map == 0] = np.nan
        return np.reshape(np.nanmean(area_map, axis=0), ROI_SHAPE)

    def get_heat_map(self):
        sum_map = []
        for obj_id in self.active_objects:
            sum_map.append(self.active_objects[obj_id].heat_map(0))
        return np.reshape(np.sum(np.stack(sum_map), axis=0), ROI_SHAPE)

    def set_filters(self, roi_filters, classes_filter):
        self._class_filters = classes_filter
        self._roi_filter = roi_filters
        new_filts = {}

        for filt in roi_filters:
            for obj in filt:
                if obj not in new_filts:
                    if filt[obj].roiFilter:
                        new_filts[obj] = filt[obj].roi
                    else:
                        new_filts[obj] = None
                else:
                    if filt[obj].roiFilter and filt[obj].roi is not None:
                        if new_filts[obj] is not None:
                            new_filts[obj] = np.logical_or(new_filts[obj], filt[obj].roi)

        for obj in self.class_handler.object_ids:
            if obj in new_filts:
                if new_filts[obj] is not None:
                    new_filts[obj] = set(np.ravel_multi_index(np.where(new_filts[obj]), ROI_SHAPE))
            else:
                new_filts[obj] = None
        self._obj_to_roi_filter = new_filts

    @property
    def roi_filter(self):
        return self._roi_filter

    @property
    def class_filter(self):
        return self._class_filters

    def blacklist_confidence(self, ent_object_id):
        if ent_object_id in self.blacklist_low_conf_objects:
            return self.blacklist_low_confidence_th
        return self.blacklist_confidence_th

    def on_db_update(self):
        custom_objects_db_reids = {}
        custom_objects_db_clips = {}
        custom_objects_db_texts = {}

        def is_valid_thr(thr):
            return thr is not None and thr > 0

        for obj_data in self.context.analytic_db.get("custom_objects", []):
            if is_valid_thr(obj_data.custom_object.reidThr):
                _update_db_dict(
                    custom_objects_db_reids, obj_data, "reid_pos_mean", "reid_neg", obj_data.custom_object.reidThr
                )
            if is_valid_thr(obj_data.custom_object.clipImageThr):
                _update_db_dict(
                    custom_objects_db_clips, obj_data, "clip_pos_mean", "clip_neg", obj_data.custom_object.clipImageThr
                )
            if is_valid_thr(obj_data.custom_object.clipTextThr) and obj_data.custom_object.promptEncHex is not None:
                _update_text_db_dict(custom_objects_db_texts, obj_data.custom_object, obj_data.object_type)

        self.custom_objects_db_reids = custom_objects_db_reids
        self.custom_objects_db_clips = custom_objects_db_clips
        self.custom_objects_db_texts = custom_objects_db_texts

        try:
            self.state_object_manager.on_db_update()
            self.additional_managers.clear()
            if self.state_object_manager.enabled:
                self.additional_managers.append(self.state_object_manager)
            self.state_shelves_manager.on_db_update()
            if self.state_shelves_manager.enabled:
                self.additional_managers.append(self.state_shelves_manager)
        except Exception as e:
            log_exception(logger, "Failed to update state object manager from db", e)
            self.context.synced_status = False

    def match_clip_to_custom_object(self, encodings, object_type: int) -> Dict:
        from_clip = {}
        from_text = {}
        if object_type in self.custom_objects_db_clips:
            from_clip = _match_custom_object(encodings, self.custom_objects_db_clips[object_type])
        if object_type in self.custom_objects_db_texts:
            from_text = _match_custom_object(encodings, self.custom_objects_db_texts[object_type])
        if from_clip and from_text:
            merged = merge_custom_object_data(from_clip, from_text)
        elif from_clip:
            merged = from_clip
        else:
            merged = from_text
        merged["is_clip"] = True
        return merged

    def generate_additional_timeline_report(self, base_period) -> Dict:
        report = {}
        for manager in self.additional_managers:
            report.update(manager.sync_and_cleanup(base_period))
        return report

    def update_model(self, model_type, model_key):
        if model_type == "state-object":
            if self.state_object_manager.enabled:
                self.state_object_manager.update_model(model_key)
            else:
                logger.error("state object is not enabled")
        else:
            logger.error("Unknown model type {}".format(model_type))
