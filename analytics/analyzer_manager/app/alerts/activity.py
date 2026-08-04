from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious  # noqa

from general.analyzer_general import logger, CameraType
from general.core import (
    BatchDataResolver,
    BDR,
    AnalyticImage,
    MotionData,
    AlertInfo,
    AlertRouting,
    divide_n_into_s_parts_inclusive,
    MemoryHandler,
)
from general.img_utils import fix_bbox_to_ar
from level1.skeleton.rtmpose import RtmPose
from level1.skeleton.stgcn import StgcnActionRecognition
from .base_alerts import AlertCandidate, ObjectAlert, BaseAlert, duration_unit_to_sec


@dataclass
class PoseData:
    timestamp: int
    crop: np.ndarray
    location: np.ndarray
    aspect_ratio: float
    pose: np.ndarray = None


@dataclass
class TrackingData:
    poses: List[PoseData] = field(default_factory=list)
    last_seen: int = 0
    alert_sent: bool = False
    next_test: int = -1
    num_samples: int = 0
    buffer_idx: int = -1


# def localize_skeleton(skeleton: np.array, location: np.array, image_size):
#     top_left_x, top_left_y, bottom_right_x, bottom_right_y = location

#     # Extract normalized x and y coordinates
#     nx, ny = skeleton[:, 0], skeleton[:, 1]

#     # Calculate the image coordinates
#     ix = (top_left_x + nx * (bottom_right_x - top_left_x)) / image_size[0]
#     iy = (top_left_y + ny * (bottom_right_y - top_left_y)) / image_size[1]

#     # Combine ix and iy back with the third column from the original matrix
#     image_coordinates = np.column_stack((ix, iy, skeleton[:, 2]))

#     return image_coordinates


def localize_skeleton(skeleton: np.ndarray, location: np.ndarray, image_size: Tuple[int, int]):
    """
    Convert normalized bounding-box coords to center-based normalized
    image coords in [-1, 1].
    """
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = location
    w, h = image_size

    # skeleton is (n_joints, 3) => [n_x, n_y, score]
    # or (n_joints, 2) => [n_x, n_y] (depends on your usage)

    # 1) Extract normalized bounding-box coords
    nx, ny = skeleton[:, 0], skeleton[:, 1]

    # 2) Convert to absolute image coords
    box_w = bottom_right_x - top_left_x
    box_h = bottom_right_y - top_left_y
    x_abs = top_left_x + nx * box_w
    y_abs = top_left_y + ny * box_h

    # 3) Shift and scale to [-1,1]
    x_norm = (x_abs - (w / 2)) / (w / 2)
    y_norm = (y_abs - (h / 2)) / (h / 2)

    # If skeleton has a third column (scores), keep it:
    if skeleton.shape[1] == 3:
        scores = skeleton[:, 2]
        skeleton_out = np.column_stack((x_norm, y_norm, scores))
    else:
        skeleton_out = np.column_stack((x_norm, y_norm))

    return skeleton_out


MAX_AVAILABLE_TRACKERS = 30


class ActivityAlert(ObjectAlert):
    type_name = "activity"
    default_ar_threshold = 0
    default_activity_len = None
    min_size_to_start_tracking = 0.05
    max_num_of_trackers = 10
    boundary_margin = -1  # set to -1 to disable
    validation_grid = [3, 4]
    default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE
    def_confidence = 0.75

    def __init__(self, alert_dict: Dict, context):
        super(ActivityAlert, self).__init__(alert_dict, context)

        analytic_config = self.context.get_config()
        self.configs = {}
        for config in ["pose", "action"]:
            self.configs[config] = {}
            if "l1_models" in analytic_config and "attributes" in analytic_config["l1_models"]:
                self.configs[config] = analytic_config["l1_models"]["attributes"].get(config, {})
                self.configs[config]["enable"] = True
        self.skeleton_detector = RtmPose(self.configs["pose"])
        self.activity_classifier = StgcnActionRecognition(self.configs["action"])
        self.person_obj = self.context.get_class_handler().object_str_to_int("person")
        self.active_ids: Dict[int, TrackingData] = {}
        self.margins = self.skeleton_detector.margins
        self.max_activity_len = self.activity_classifier.activity_len
        self.activity_len = self.max_activity_len if self.default_activity_len is None else self.default_activity_len
        self.ar_threshold = (
            0.1 + 0.4 * (1 - self.sensitivity) if self.sensitivity is not None else self.default_ar_threshold
        )
        extra_frames = self.activity_classifier.extra_frames
        self.next_step_size = extra_frames if extra_frames > 0 else self.activity_len // 2
        self.conf_threshold = self.def_confidence if self.sensitivity is None else 0.5 + 0.5 * (1 - self.sensitivity)
        self.crop_ar = self.skeleton_detector.aspect_ratio
        self.object_ids = [self.person_obj]
        alert_config = analytic_config.get("alertConfig", {}).get("activity", {})
        if "min_size_to_start_tracking" in alert_config:
            self.min_size_to_start_tracking = alert_config.get("min_size_to_start_tracking")
        if "max_num_of_trackers" in alert_config:
            self.max_num_of_trackers = alert_config.get("max_num_of_trackers")
        self.required_l1 = False  # no need for filters and such
        self.l1_required_type[self.person_obj] = None
        activity_buffer_size = list(self.skeleton_detector.image_size) + [3]
        self.memory_handler = MemoryHandler(
            num_buffers=self.max_num_of_trackers,
            num_crops=self.max_activity_len,
            crop_size=activity_buffer_size,
            dtype=np.uint8,
        )
        self.validator_poses = divide_n_into_s_parts_inclusive(self.max_activity_len, np.prod(self.validation_grid))

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BatchDataResolver
    ) -> List[AlertCandidate]:
        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            return []
        alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()

        # filter ids that we already alerted on
        inactives = set([id_ for id_ in self.active_ids if self.active_ids[id_].alert_sent])
        alert_ids = [id_ for id_ in alert_ids if id_ not in inactives]

        ids_to_check = []
        image_size_norm = [*batch_data.image_size, *batch_data.image_size]
        for id_ in alert_ids:
            id_data = vars_data[vars_data[:, BatchDataResolver.ID] == id_]
            if id_ in self.active_ids:
                active_id = self.active_ids[id_]
            else:
                max_size = np.max(id_data[:, BatchDataResolver.POS[2]] - id_data[:, BatchDataResolver.POS[0]])
                buffer_idx = -1
                if max_size >= self.min_size_to_start_tracking:
                    buffer_idx = self.memory_handler.acquire()
                if buffer_idx < 0:
                    continue
                else:
                    active_id = TrackingData(next_test=self.activity_len, buffer_idx=buffer_idx)
            n_samples = len(id_data)
            poses = id_data[:, BatchDataResolver.POS]
            data_valid = np.all(np.min(poses, axis=0) > self.boundary_margin) and np.all(
                np.max(poses, axis=0) < 1 - self.boundary_margin
            )
            if not data_valid:
                active_id.next_test = active_id.num_samples + self.activity_len
                self.active_ids[id_] = active_id
                continue
            for d in id_data:
                image_pos = d[BatchDataResolver.POS] * image_size_norm
                orig_ar = (image_pos[2] - image_pos[0]) / (image_pos[3] - image_pos[1])
                ts = d[BatchDataResolver.TIMESTAMP]
                pos = fix_bbox_to_ar(image_pos, self.crop_ar, batch_data.image_size, self.margins)
                image = images[int(d[BatchDataResolver.FRAME_ID])].frame
                crop = self.skeleton_detector.apply_resize(
                    image[pos[1] : pos[3], pos[0] : pos[2]], self.memory_handler.next(active_id.buffer_idx), bgr=True
                )
                pose_data = PoseData(ts, crop, pos, orig_ar)
                active_id.poses.append(pose_data)

            active_id.last_seen = id_data[-1, BatchDataResolver.TIMESTAMP]
            active_id.poses = active_id.poses[-self.max_activity_len :]
            active_id.num_samples += n_samples
            self.active_ids[id_] = active_id
            if active_id.num_samples >= active_id.next_test:
                average_aspect_ratio = np.mean([p.aspect_ratio for p in active_id.poses[:-n_samples]])
                curr_aspect_ratio = np.median([p.aspect_ratio for p in active_id.poses[-n_samples:]])
                aspect_ratio_diff = np.abs(average_aspect_ratio - curr_aspect_ratio) / average_aspect_ratio
                if aspect_ratio_diff > self.ar_threshold:
                    ids_to_check.append(id_)
                elif active_id.num_samples > self.max_activity_len:
                    ent_data = self.entity_db.get_entity(id_)
                    if ent_data is None or not ent_data.is_valid(self.success_required_filter):
                        # mark as sent to avoid future processing
                        self.nullify_id(id_)
                        logger.info(f"ID {id_} is not valid, nullifying for activity tracking")

        alert_candidates = []
        # update alert info in case of active
        if len(ids_to_check) > 0:
            activities_results, poses_by_id = self.check_activities(ids_to_check, batch_data.image_size)
            for idx, result in enumerate(activities_results):
                active_id = self.active_ids[result["ent_id"]]
                if result["action"] == self.type_name:
                    # if the confidence is not high enough, we will retry next batch if data exists
                    # crops are expected as BGR here
                    if result["confidence"] > self.conf_threshold:
                        candidate = self.build_alert_candidate(
                            active_id.last_seen,
                            result["ent_id"],
                            crops=[cv2.cvtColor(active_id.poses[-1].crop, cv2.COLOR_RGB2BGR)],
                            validation_images=[self.build_collage(active_id.poses)],
                        )
                        if self.activate_ddata and self.internal_alert:
                            # get poses for this specific entity
                            entity_poses = poses_by_id.get(result["ent_id"], [])
                            candidate = self.add_debug_data(candidate, result, entity_poses)
                        alert_candidates.append(candidate)
                        self.nullify_id(result["ent_id"])
                else:
                    # push next update forward
                    active_id.next_test = active_id.num_samples + self.next_step_size
                logger.debug(
                    f"time: {images[-1].timestamp} checked activity for id {result['ent_id']}: {result['action']} with confidence {result['confidence']}"
                )
        # skel_stat = {id_: len([p for p in self.active_ids[id_].poses if p.crop is not None]) for id_ in self.active_ids}
        # logger.warn(f"active skeletons total: {sum(list(skel_stat.values()))} desc: {skel_stat}")
        return alert_candidates

    def add_debug_data(
        self, candidate: AlertCandidate, result: Dict, input_poses_lst: List[np.array]
    ) -> AlertCandidate:
        candidate.extra = candidate.extra or {}
        # add debug data for internal alerts
        debug_data = {
            "ent_id": result["ent_id"],
            "action": result["action"],
            "activity_confidence": result["confidence"].item(),
            "activity_scores": result.get("scores", []),
            "input_crops": [],
            "timestamp": [],
            "location": [],
            "rtm_skeletons": [],
        }
        for posedata in input_poses_lst:
            debug_data["input_crops"].append(cv2.cvtColor(posedata.crop.copy(), cv2.COLOR_RGB2BGR))
            debug_data["rtm_skeletons"].append(posedata.pose)
            debug_data["timestamp"].append(int(posedata.timestamp))
            debug_data["location"].append(posedata.location)
        candidate.extra["debug_data"] = self.serialize_debug_data(debug_data)
        return candidate

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            track_data = self.active_ids.pop(ent, None)
            if track_data is not None and track_data.buffer_idx >= 0:
                self.memory_handler.release(track_data.buffer_idx)

    def on_entities_update(self, ent_ids: List[int]):
        pass

    def check_activities(self, ids_to_check: List[int], image_size) -> Tuple[List[Dict], Dict[int, List[PoseData]]]:
        data_for_activity = {}
        poses_by_id = {}  # Store poses for each entity ID
        for id_ in ids_to_check:
            ent_data = self.entity_db.get_entity(id_)
            if ent_data is None or not ent_data.is_valid(self.success_required_filter):
                # mark as sent to avoid future processing
                self.nullify_id(id_)
                continue
            poses = self.active_ids[id_].poses
            inds_for_pose = [idx for idx, data in enumerate(poses) if data.pose is None]
            crops = [poses[idx].crop for idx in inds_for_pose]
            if crops:
                skeletons = self.skeleton_detector.forward_on_crop_list(crops)
                for i, idx in enumerate(inds_for_pose):
                    poses[idx].pose = localize_skeleton(skeletons[i], poses[idx].location, image_size)
            skeletons = np.stack([pose.pose for pose in poses])
            skeletons[:, :, :2] -= np.mean(skeletons[0, skeletons[0, :, 2] > 0.5, :2], axis=0)
            data_for_activity[id_] = skeletons
            poses_by_id[id_] = poses  # Store poses for this specific ID
        activity_res = []
        if len(data_for_activity) > 0:
            ids = list(data_for_activity.keys())
            activity_res = self.activity_classifier.forward_on_crop_list([data_for_activity[id_] for id_ in ids])
            for idx, res in enumerate(activity_res):
                res["ent_id"] = ids[idx]
        return activity_res, poses_by_id

    def nullify_id(self, ent_id):
        self.active_ids[ent_id].alert_sent = True
        self.active_ids[ent_id].next_test = 0

        # release memory
        if self.active_ids[ent_id].buffer_idx >= 0:
            self.memory_handler.release(self.active_ids[ent_id].buffer_idx)
            self.active_ids[ent_id].buffer_idx = -1

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        BaseAlert.generate_alert_message(self, candidate, alert_info)

    def build_collage(self, poses: List[PoseData]) -> np.ndarray:
        collage_shape = np.array(self.skeleton_detector.image_size) * np.array(self.validation_grid)
        collage = np.zeros(collage_shape.tolist() + [3], dtype=np.uint8)
        for i, pidx in enumerate(self.validator_poses):
            if pidx >= len(poses):
                break
            pose = poses[pidx]
            x, y = divmod(i, self.validation_grid[1])
            collage[
                x * self.skeleton_detector.image_size[0] : (x + 1) * self.skeleton_detector.image_size[0],
                y * self.skeleton_detector.image_size[1] : (y + 1) * self.skeleton_detector.image_size[1],
            ] = pose.crop
        return cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)


class FallAlert(ActivityAlert):
    type_name = "Falling down"
    default_ar_threshold = 0.20
    default_activity_len = None
    alert_message = "Person falling is detected"
    boundary_margin = 0.025  # set to -1 to disable
    default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE
    def_confidence = 0.65
    pos_change_th: float = 0.15  # iou threshold to consider object's position had changed enough
    fall_recovery_time_ms: int = 0

    def __init__(self, alert_dict: Dict, context):
        super(FallAlert, self).__init__(alert_dict, context)
        if self.context.camera_type in [
            CameraType.CEILING_MOUNTED,
            CameraType.CEILING_MOUNTED_FISHEYE,
            CameraType.WALL_MOUNTED_FISHEYE,
        ]:
            raise ValueError(
                f"Fall alert is not supported in {str(self.context.camera_type).lower().replace('-', ' ')}"
            )
        self.set_flow_values()
        cfg = self.alerts_config.get(self.type_name, {})
        self.pos_change_th = cfg.get("pos_change_th", 0.15)
        self.first_activity_time = 0
        self.activity_monitor = {}
        self.set_advance_fall(context.advance_fall)

    def set_flow_values(self):
        if self.formValue:
            count = self.apply_from_dict("duration", self.formValue, 0)
            units = self.apply_from_dict("durationUnit", self.formValue, 0)

            # time to consider an actual fall if person does not recover
            self.fall_recovery_time_ms = count * duration_unit_to_sec[units] * 1000

    def _short_activity_debounce(self, ent_id: int, current_time: int, id_bbox: np.ndarray) -> bool:
        # check duration of contious fall activity
        duration = current_time - self.activity_monitor[ent_id][0]
        iou = bbox_ious(self.activity_monitor[ent_id][1], id_bbox)[0][0]
        if iou >= 1 - self.pos_change_th:
            # person has not moved enough, continue monitoring
            if duration >= self.fall_recovery_time_ms:
                # person hasn't recovered in time, consider as fall
                return True
        else:
            # position changed enough, reset monitoring
            self.activity_monitor.pop(ent_id, None)
            logger.info(
                f"Fall detection for ID {ent_id} was suppressed due to a movement from initial position (IOU: {iou:.2f})"
            )
        return False

    def is_active_batch(self, images, motion_data, batch_data):
        alert_candidates = []
        raw_alert_cand = super().is_active_batch(images, motion_data, batch_data)
        for cand in raw_alert_cand:
            id_bbox = batch_data.query(BDR.ID, cand.ent_ids[0], BDR.POS)
            if id_bbox is None or id_bbox.size == 0:
                continue
            id_bbox = np.ascontiguousarray(id_bbox[-1].reshape(1, 4).astype(np.float64))
            if cand.ent_ids[0] not in self.activity_monitor:
                self.activity_monitor[cand.ent_ids[0]] = [cand.timestamp, id_bbox, cand]

        for ent_id in list(self.activity_monitor.keys()):
            id_bbox = batch_data.query(BDR.ID, ent_id, BDR.POS)
            if id_bbox is None or id_bbox.size == 0:
                continue
            id_bbox = np.ascontiguousarray(id_bbox[-1].reshape(1, 4).astype(np.float64))
            if self._short_activity_debounce(ent_id, images[-1].timestamp, id_bbox):
                alert_candidates.append(self.activity_monitor[ent_id][2])
                self.activity_monitor.pop(ent_id, None)
        return alert_candidates

    def set_advance_fall(self, is_advanced: bool):
        if not self.internal_alert and is_advanced and self.max_num_of_trackers < MAX_AVAILABLE_TRACKERS:
            self.max_num_of_trackers = MAX_AVAILABLE_TRACKERS
            self.memory_handler.extend_num_buffers(MAX_AVAILABLE_TRACKERS)
            logger.info("Fall alert max_num_of_trackers increased to handle advance fall alerts")
