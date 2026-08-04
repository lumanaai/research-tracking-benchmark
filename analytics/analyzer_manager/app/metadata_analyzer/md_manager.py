import os
import time
from collections import defaultdict
from collections import deque, OrderedDict
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Any

import cv2
import numpy as np

from detection.yolov5.utils.plots import Annotator, colors
from detection.yolov5.yolov8_detector import YoloV8ExpertArgs
from general.analyzer_general import logger, ROI_SHAPE, log_exception
from general.cloud_converter import convert_custom_object
from general.core import (
    AlertType,
    ThumbnailType,
    BatchDataResolver,
    Metadata,
    MotionData,
    ThumbnailInfo,
    AlertInfo,
    ClassHandler,
    DescriptorVector,
    MemoryHandler,
    BoundedPriorityQueue,
    DashboardsType,
)
from general.entity import ActiveEntityData
from general.entity_db import merge_custom_object_data
from general.img_utils import scale_image_by_width
from general.perf_utils import time_sync
from metadata_analyzer.saver_selection import SaverSelector
from preprocessing.snapshot_resizer import SnapshotResizerFactory
from .image_handler import ImageHandler, BwLimitImageHandler
from .timeline_builder import TimelineBuilder


def bbox_to_perim(bbox) -> List[dict]:
    # If bbox is a 1D array (single bounding box)
    if bbox.ndim == 1:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        tlwh = [np.array([x1, y1, width, height]).astype(int).tolist()]

    # If bbox is a 2D array (multiple bounding boxes)
    else:
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        width = x2 - x1
        height = y2 - y1
        tlwh = np.stack([x1, y1, width, height], axis=1).astype(int).tolist()

    return [{"x": b[0], "y": b[1], "dx": b[2], "dy": b[3]} for b in tlwh]


def find_max_motion_blocks(mv, grid_size=4):
    # Reshape the matrix into a 4D array where blocks are separated
    reshaped_matrix = mv.reshape(grid_size, mv.shape[0] // grid_size, grid_size, mv.shape[1] // grid_size)
    # Find the max in each block by specifying the axes of the block
    max_values = reshaped_matrix.max(axis=(1, 3))
    return max_values


@dataclass
class ClipTask:
    id_base: int
    id_index: int
    memory_index: int = -1  # pointer to the mem buffer
    memory_buffer: Optional[np.array] = None
    custom_object: Optional[Dict] = None


class MetadataManager:
    ZOOM_GB = 0.1

    def __init__(self, context):
        self.context = context
        self.alert_manager = context.alert_manager
        self.entity_db = context.entity_db
        self.timeline_builder = TimelineBuilder(entity_db=context.entity_db)
        self.analytic_config = context.config
        self.config = context.config
        self.thumb_width = int(self.config["thumbnailPolicy"]["thum_width"])
        self.thumb_height = self.thumb_width
        self.snapshot_width = int(self.config["thumbnailPolicy"]["snap_width"])
        self.snapshot_height = round(self.snapshot_width * self.context.resolution[1] / self.context.resolution[0])
        self.magnitude = self.config["post_process"]["pixelateThumbnail"]["magnitude"]
        self.pixelate_regions = []
        self.pixelate_classes = []
        self.pixelate = False
        self.object_to_exclude_annotations = self.config["post_process"].get("object_to_exclude_from_annotation", [])
        self.last_snapshot = ""
        self.snapshot_cycle = False
        self.last_thumbnail: ThumbnailInfo = ThumbnailInfo("", -1, -1)
        self.last_thumbnail_ts = -1
        self.last_frame_number = -1
        self.old_alert_crop_duration = 2000  # above that we will try to send a more updated crop
        self.update_empty_json: bool = bool(self.config["jsonPolicy"]["updateEmptyInfo"])
        self.counting_aggregator = defaultdict(lambda: defaultdict(lambda: {"in": 0, "out": 0}))
        self.region_counting = {}
        self.dashboard_aggregator = {}
        self.counting_sync_period = self.config["countingPolicy"]["updateRate"] * 1000
        self.send_counting_as_alerts = self.context.app_config["analytics"]["countingAsAlerts"]
        self._next_counting_sync_period = 0
        self.low_priority_alerts = []
        self.clip_min_dwell = 1.0  # sec
        self.clip_max_dwell = 15.0  # sec

        # clip encodings
        self.clip_dispatcher = context.clip_dispatcher
        self.clip_encode_entities = bool(self.config["trainThumbPolicy"].get("clip_encode_crops", False))
        if self.clip_encode_entities and self.clip_dispatcher.enabled and not self.clip_dispatcher.crops_enabled:
            self.clip_dispatcher.set_crops_enabled(True)
        if not self.clip_dispatcher.use_offline:
            logger.error("Clip encoding is enabled but the dispatcher is not set to use offline mode.")
            self.clip_encode_entities = False
        if self.clip_encode_entities:
            self.clip_queue_size = self.config["trainThumbPolicy"].get("clip_crop_queue_size", 30)  # to ms
            self.clip_period = 1000 / self.config["trainThumbPolicy"].get("clip_crop_rate_limit_sec", 0.5)  # to ms
            self.clip_mem_buffer = MemoryHandler(self.clip_queue_size, 1, self.clip_dispatcher.image_shape)
            self.clip_queue = BoundedPriorityQueue(self.clip_queue_size)
            self.last_clip_object_ts = 0
            self.clip_object_priority_factor = {
                self.class_handler.person_value: 1,
                self.class_handler.vehicle_value: 0.75,
                self.class_handler.pet_value: 0.5,
            }
        self.clip_objects_results = []

        # motion handling
        self.mv_duration = int(self.config["motionPolicy"].get("motionVectorsDuration", 30000))
        self.min_motion_output = int(self.config["motionPolicy"].get("min_motion_output", 10))
        self.log_thumbnails = bool(self.config["thumbnailPolicy"].get("logThumbnails", False))

        self.last_motion_frames = 0
        self.last_motion_timestamp_for_info = 0
        self.last_mv_out = np.zeros(ROI_SHAPE, dtype=np.uint8)
        self.motion_index = 0
        self.motion_window = int(self.config["thumbnailPolicy"].get("motionWindow", 2))
        self.skipped_thumbnail_queue = deque(maxlen=self.motion_window)
        self.last_skipped = False
        self.block_skip_counter = 0

        self.alert_thumb_buffer: OrderedDict[int, np.ndarray] = OrderedDict()
        self.alert_snapshot_buffer: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_alert_buffer_size = 5

        expert_args = YoloV8ExpertArgs({})
        expert_img_size = tuple(expert_args.im_size)
        if expert_img_size[1] == self.snapshot_width:
            self.resizer = SnapshotResizerFactory.get_resizer(context.camera_id, expert_img_size)
            self.resizer.antialias = True
            self.snapshot_resizer = lambda x: self.resizer.resize(x, is_bgr=True)
        else:
            self.snapshot_resizer = lambda x: scale_image_by_width(x.frame, self.snapshot_width)

        # image handling
        train_path = str(
            os.path.join(context.app_config["locations"]["trainingThumbnails"], context.edge_id, context.camera_id)
        )

        self.training_selector = SaverSelector(context)
        self.training_sender = ImageHandler(
            train_path,
            1,
            period_sec=self.training_selector.get_save_period(),
            prefix="training",
            name_builder="timestamp",
            nvjpeg_encoder_address=context.nvjpeg_encoder_address,
            jpg_quality=int(self.config["trainThumbPolicy"]["trainJpegquality"]),
        )
        if "crop_bw_limits_kb" in self.config["alertThumbPolicy"]:
            self.alert_sender = BwLimitImageHandler(
                os.path.join(context.app_config["locations"]["alertThumbnails"], context.edge_id, context.camera_id),
                nvjpeg_encoder_address=context.nvjpeg_encoder_address,
                name_builder="custom",
                count_offset=1,
                bw_rate_limits=self.config["alertThumbPolicy"]["crop_bw_limits_kb"],
            )
        else:
            self.alert_sender = ImageHandler(
                os.path.join(context.app_config["locations"]["alertThumbnails"], context.edge_id, context.camera_id),
                limit_per_period=self.config["alertThumbPolicy"]["maxAlertThumbnailsIn100Seconds"],
                nvjpeg_encoder_address=context.nvjpeg_encoder_address,
                name_builder="custom",
                jpg_quality=int(self.config["alertThumbPolicy"]["alertJpegquality"]),
                count_offset=1,
            )

        self.crop_sender = BwLimitImageHandler(
            train_path,
            prefix="attributes",
            name_builder="custom",
            nvjpeg_encoder_address=context.nvjpeg_encoder_address,
            bw_rate_limits=self.config["trainThumbPolicy"]["crop_bw_limits_kb"],
        )

        self.thumb_handler = ImageHandler(
            os.path.join(context.app_config["locations"]["thumbnails"], context.edge_id, context.camera_id),
            limit_per_period=1,
            period_sec=int(self.config["thumbnailPolicy"]["thumbnailsDuration"]) / 1000,
            name_builder="thumb_counter",
            nvjpeg_encoder_address=context.nvjpeg_encoder_address,
            jpg_quality=int(self.config["thumbnailPolicy"]["jpegquality"]),
        )
        self.mandatory_thumbnail_period = int(self.config["thumbnailPolicy"]["mandatoryThumbnailsMultiplier"]) * int(
            self.thumb_handler.period
        )
        self.min_motion_for_thumb = int(self.config["thumbnailPolicy"]["minMotionForOptionalThumbnail"])
        self.min_obj_motion_for_thumb = int(self.config["thumbnailPolicy"].get("minObjMotionForOptionalThumbnail", 2))
        self.snapshot_handler = ImageHandler(
            train_path,
            limit_per_period=1,
            prefix="snapshot",
            period_sec=int(self.config["thumbnailPolicy"]["snapshotDuration"]) / 1000,
            name_builder="norm_timestamp",
            nvjpeg_encoder_address=context.nvjpeg_encoder_address,
            jpg_quality=int(self.config["thumbnailPolicy"]["snapshotquality"]),
        )
        self.alerted_ents = set()
        self.entity_db.on_entities_purged += self.on_entities_purged
        self.on_db_update()
        self.context.on_db_update += self.on_db_update

    def on_db_update(self):
        if self.context.analytic_db.get("shelves") and DashboardsType.shelves not in self.dashboard_aggregator:
            self.dashboard_aggregator[DashboardsType.shelves] = {}

    def run(
        self,
        image_batch,
        batch_data,
        detector_images,
        l1_results,
        performance,
        is_night_mode,
        motion_data,
        skip_ts,
        alerted_ents: Set[int],
    ) -> Tuple[Metadata, Dict, Optional[Dict], List]:
        t0 = time_sync()

        # generate relevant reports
        timestamps = [image.timestamp for image in image_batch]
        self.last_frame_number = image_batch[-1].frame_number
        timeline_report = self.timeline_builder.track(batch_data, timestamps)
        alert_results = self.alert_manager.get_active_alerts(timestamps[-1])
        sync_report = self.entity_db.sync_and_cleanup(timestamps[-1])
        motion_res = self._build_motion_info(motion_data, timestamps)

        self._add_motion_index(motion_data)
        self.alerted_ents.update(alerted_ents)

        self.training_selector.add_images(image_batch, batch_data, is_night_mode)

        # manage alerts that are not in this batch but may be in next batches
        for candidate in self.alert_manager.suspended_alerts:
            ts = candidate.timestamp
            alert = self.alert_manager.get_alert(candidate.alert_id)
            if ts in timestamps:
                img, annotations = self.alert_thumb_buffer.get(ts, (detector_images[timestamps.index(ts)], []))
                self.alert_thumb_buffer[ts] = self.annotate_thumbnail(img, ts, batch_data, candidate.ent_ids)
                if ts not in self.alert_snapshot_buffer and alert is not None and alert.is_store_snapshot:
                    self.alert_snapshot_buffer[ts] = self.snapshot_resizer(image_batch[timestamps.index(ts)])

        # manage all the batch thumbnails
        b_thumb_res = self.handle_thumbnails(image_batch, detector_images, batch_data, l1_results, alert_results)
        thumbnails, snapshots, trainings, crops, best_images, zoom_images = b_thumb_res

        # maintenance for the alert and snapshot buffer
        self._maintain_alert_buffer(self.alert_thumb_buffer)
        self._maintain_alert_buffer(self.alert_snapshot_buffer)

        alerts = self.handle_alerts_flows(alert_results)
        counting_res = self._build_dashboard_info(timestamps[-1])

        batch_alert = len(alerts) > 0 and any([alert.priority > 0 for alert in alert_results])

        if sync_report is None and (batch_alert or timeline_report):
            sync_report = self.entity_db.sync_and_cleanup(timestamps[-1], force=True)
            if sync_report is None:
                sync_report = {}

        if sync_report is not None:
            all_alerts = alerts + self.low_priority_alerts
            if len(all_alerts):
                sync_report["alerts"] = all_alerts
                self.low_priority_alerts.clear()

            if self.snapshot_cycle:
                sync_report["mainSnapshot"] = self.last_snapshot
                self.snapshot_cycle = False
            sync_report["mainThumbnail"] = self.last_thumbnail.thumbnail
            sync_report["mainThumbnailTimestamp"] = self.last_thumbnail.timestamp
            sync_report["timestamp"] = str(timestamps[-1])
            sync_report["misc"] = ""
            if timeline_report:
                sync_report["timeline"] = timeline_report
            for ent_id in best_images:
                if ent_id in sync_report["attributes"]:
                    sync_report["attributes"][ent_id]["bestImage"] = best_images[ent_id]
            for ent_id in zoom_images:
                if ent_id in sync_report["attributes"]:
                    sync_report["attributes"][ent_id]["zoomImage"] = zoom_images[ent_id]
        elif len(alerts) > 0:
            self.low_priority_alerts.extend(alerts)

        t1 = time_sync()
        performance["md analyzer"] = f"{round((t1 - t0) * 1000 / len(image_batch), 1)} [ms]"
        results = Metadata()

        if sync_report is None and (self.update_empty_json or self.snapshot_cycle):
            sync_report = {
                "mainThumbnail": self.last_thumbnail.thumbnail,
                "mainThumbnailTimestamp": self.last_thumbnail.timestamp,
                "timestamp": str(timestamps[-1]),
                "misc": "",
                "info": [],
                "trackers": [],
                "objects": {},
            }

            if self.snapshot_cycle:
                sync_report["mainSnapshot"] = self.last_snapshot
                self.snapshot_cycle = False

        if sync_report is not None:
            sync_report["last_frame_number"] = self.last_frame_number
            metadata = {
                "performance": performance,
                "timestamp": str(int(round(time.time() * 1000))),
                "batchAlert": batch_alert,
                "smartStorage": batch_alert,
                "misc": {"trainingInfo": trainings, "snapshotInfo": snapshots},
                "snapshots": snapshots,
            }
            if skip_ts is not None:
                metadata["misc"]["skipped_frames"] = skip_ts
            results.info = {"metadata": metadata, "results": [sync_report], "encodings": []}
        results.training.extend(trainings + crops)
        results.snapshots.extend(snapshots)
        results.thumbnails.extend(thumbnails)
        results.trafficMsgs.extend([])
        encodings = self.clip_dispatcher.pop_descriptors()
        if encodings:
            enc_dicts = []
            for enc in encodings:
                encode_name, encode_norm_ts = self.thumb_handler.generate_name(enc[1], thumb_count=0)
                enc_dicts.append(
                    {"timestamp": encode_norm_ts, "encoding": DescriptorVector(enc[0]), "thumbnail_name": encode_name}
                )
            if results.info is None:
                results.info = {"metadata": {}, "results": [], "encodings": []}
            results.info["encodings"] = enc_dicts
        obj_encodings = None
        if self.clip_objects_results:
            obj_encodings = self.clip_objects_results
            self.clip_objects_results = []
        return results, motion_res, counting_res, obj_encodings

    def _build_dashboard_info(self, end_timestamp: int = None) -> Optional[Dict]:
        dashboard_info = {}
        if end_timestamp > self._next_counting_sync_period:

            last_ts = self._next_counting_sync_period - self.counting_sync_period
            dashboard_info.update(self.entity_db.generate_additional_timeline_report(last_ts))

            self._next_counting_sync_period = (
                end_timestamp // self.counting_sync_period + 1
            ) * self.counting_sync_period
            ts_bucket = (end_timestamp // self.counting_sync_period) * self.counting_sync_period

            if self.counting_aggregator:
                counting_info = {"timestamp": ts_bucket, "events": []}
                for eventId in self.counting_aggregator:
                    counting_data = {"eventId": eventId, "trackers": []}
                    for trackerClass, alertData in self.counting_aggregator[eventId].items():
                        counting_data["trackers"].append(
                            {
                                "trackerClass": trackerClass,
                                "trackerTypeId": self.class_handler.entity_class_to_cloud_type_mapping.get(
                                    trackerClass
                                ),
                                "alertData": alertData,
                            }
                        )
                    counting_info["events"].append(counting_data)
                # Reset for next sync
                self.counting_aggregator.clear()
                dashboard_info["counting"] = counting_info

            if self.region_counting:
                rcounting_info = {"timestamp": ts_bucket, "events": []}
                for zone_id, counting_info in self.region_counting.items():
                    rcounting_data = {"eventId": zone_id, "trackers": []}
                    for obj_id, count in counting_info.get("regionCount", {}).items():
                        rcounting_data["trackers"].append(
                            {
                                "trackerClass": self.class_handler.object_to_default_entity_class[obj_id],
                                "trackerTypeId": obj_id,
                                "alertData": {"count": count, "in": 0, "out": 0},
                            }
                        )
                    thumbs = counting_info.get("validationThumbnails", None)
                    if thumbs is not None:
                        rcounting_data["validationThumbnails"] = thumbs
                        rcounting_data["roi"] = counting_info.get("roi", [])
                    rcounting_info["events"].append(rcounting_data)
                if "counting" not in dashboard_info:
                    dashboard_info["counting"] = rcounting_info
                else:
                    dashboard_info["counting"]["events"].extend(rcounting_info["events"])
                self.region_counting.clear()

        return dashboard_info

    def handle_alerts_flows(self, alert_results: List[AlertInfo]) -> List[Dict]:
        alerts = []
        for alert_info in alert_results:
            alert_json = alert_info.to_dict()
            if alert_info.detectionType == AlertType.counting.value:
                value = alert_info.alertData["value"]
                if value > 0:
                    self.counting_aggregator[alert_info.eventId][alert_info.trackerClass]["in"] += value
                else:
                    self.counting_aggregator[alert_info.eventId][alert_info.trackerClass]["out"] += abs(value)
                if not self.send_counting_as_alerts:
                    continue
            # elif alert_info.detectionType == AlertType.emptyShelfCounter.value:
            #     shelf_id = alert_info.extra.get("shelf_id")
            #     new_value = int(alert_info.alertData * 100)  # to percentage
            #     upd = {"value": new_value, "in": 0, "out": 0}
            #     last_report = self.dashboard_aggregator[DashboardsType.shelves].get(shelf_id, None)
            #     if last_report:
            #         last_value = last_report["value"]
            #         delta = new_value - last_value
            #         if delta > 0:
            #             upd["in"] = delta
            #         else:
            #             upd["out"] = abs(delta)
            #     self.dashboard_aggregator[DashboardsType.shelves][shelf_id] = upd
            #     continue  # skip further handling
            if alert_info.detectionType == AlertType.regionCount.value:
                zone_id = alert_info.eventId

                counting_info: Dict[str, Any] = {
                    "regionCount": alert_info.extra.get("regionCount", {}).copy(),
                    "timestamp": alert_info.timestamp,
                }
                if alert_info.validationThumbnails:
                    counting_info["validationThumbnails"] = alert_info.validationThumbnails
                    counting_info["roi"] = alert_info.extra.get("polygons", [])
                self.region_counting[zone_id] = counting_info
                continue

            debug_data = alert_json.pop("debugData", None)
            logger.info(f"ALERT RAISED: {alert_json}")
            if debug_data:
                alert_json["debugData"] = debug_data
            alerts.append(alert_json)
        return alerts

    def handle_thumbnails(self, image_batch, detector_images, batch_data, l1_results, alert_results: List[AlertInfo]):
        thumbnails = []
        snapshots = []
        trainings = []
        crops = []
        best_images = {}
        zoom_images = {}

        timestamps = [im.timestamp for im in image_batch]
        last_ts = image_batch[-1].timestamp

        # handle crops
        object_encoding_batch = {}
        if self.crop_sender.can_send(last_ts):
            for analyzer_res in l1_results:
                obj_crops = analyzer_res["images"]
                analyzer_type = analyzer_res["analyzer"]
                for obj_id in obj_crops:
                    for img_key in obj_crops[obj_id]:
                        is_zoom = img_key == "zoom_image"
                        ent_crop = obj_crops[obj_id][img_key]
                        ent_img = self.entity_db.get_ent_image(obj_id, is_zoom)

                        # no need to add image if its from unsuccessful l1
                        if analyzer_res["success"].get(obj_id, 0) < 0 and ent_img is not None:
                            continue
                        ent_info = self.entity_db.get_ent_crop_info(obj_id, analyzer_type, is_zoom)
                        if ent_info is not None and 0 not in ent_crop.shape:
                            crop_name = f"{ent_info['id_base']}-{int(ent_info['id_index'])}-{ent_info['type']}-{ent_info['crop_index']}"
                            thumb_name, _ = self.crop_sender.send(ent_crop, last_ts, req_name=crop_name)
                            if thumb_name is None:  # cant send anymore - limit reached
                                break
                            self.entity_db.add_ent_image(obj_id, thumb_name, is_zoom)

                            if is_zoom:
                                self.entity_db.register_zoom_image(obj_id, ent_crop)
                                zoom_images[obj_id] = thumb_name
                            else:
                                best_images[obj_id] = thumb_name
                                object_encoding_batch[obj_id] = ent_crop
                            crops.append(
                                {"thumbnail": thumb_name, "timestamp": last_ts, "thumType": ThumbnailType.Crop.value}
                            )
        if self.clip_encode_entities:
            self._handle_clip_for_objects(object_encoding_batch, timestamps)

        # get the alerts thumbnails
        alerts_thumbs = {}
        alerts_grouped = {}
        if alert_results:
            # handle alert zooms
            for alert_info in alert_results:
                if alert_info.idIndex > 0:
                    zoom = self.entity_db.small_images_collection.get(alert_info.idIndex)
                    if zoom is not None:
                        if not alert_info.crops:
                            alert_info.crops.append(zoom)
                        elif timestamps[-1] - alert_info.timestamp > self.old_alert_crop_duration:
                            alert_info.crops = [zoom]

                if alert_info.idIndex > 0:
                    ent_zoom_image = self.entity_db.query_zoom_image(alert_info.idIndex)
                    if ent_zoom_image is not None:
                        alert_info.crops.append(ent_zoom_image)

                for idx, crop in enumerate(alert_info.crops):
                    thumb_name = f"{alert_info.eventId}-{alert_info.timestamp}-{idx}"
                    zoom_name, _ = self.alert_sender.send(crop, alert_info.timestamp, req_name=thumb_name)
                    if zoom_name is None:  # cant send anymore - limit reached
                        break
                    alert_info.thumbnails.append(zoom_name)

                # handle validation crops
                offset = len(alert_info.crops)
                for idx, crop in enumerate(alert_info.validation_images):
                    thumb_name = f"{alert_info.eventId}-{alert_info.timestamp}-{idx + offset}"
                    zoom_name, _ = self.alert_sender.send(crop, alert_info.timestamp, req_name=thumb_name)
                    if zoom_name is None:  # cant send anymore - limit reached
                        break
                    alert_info.validationThumbnails.append(zoom_name)
                if len(alert_info.thumbnails) == 0 and len(alert_info.validationThumbnails) > 0:
                    alert_info.thumbnails.append(alert_info.validationThumbnails[0])
                elif (
                    len(alert_info.validationThumbnails) == 0
                    and alert_info.routing > 0
                    and len(alert_info.thumbnails) > 0
                ):
                    alert_info.validationThumbnails.append(alert_info.thumbnails[0])

                alert = self.alert_manager.get_alert(alert_info.eventId)
                if alert is not None and alert.is_store_snapshot:
                    if alert_info.timestamp in timestamps:
                        snapshot = self.snapshot_resizer(image_batch[timestamps.index(alert_info.timestamp)])
                    else:
                        snapshot = self.alert_snapshot_buffer.pop(alert_info.timestamp, None)

                    if snapshot is not None:
                        snapshot_name = (
                            f"{self.snapshot_handler.prefix}-{alert_info.timestamp}-ooc-{alert_info.eventId}.jpg"
                        )
                        self.snapshot_handler.send_out_of_cycle(snapshot, snapshot_name)
                        alert_info.snapshots = [snapshot_name]

                        snapshots.append(
                            {
                                "thumbnail": snapshot_name,
                                "timestamp": alert_info.timestamp,
                                "thumType": ThumbnailType.Snapshot.value,
                            }
                        )
                    if alert_info.perimeter is not None and not isinstance(alert_info.perimeter, dict):
                        # bbox to x,y,width,height scaled by snapshot size
                        perim = np.array(alert_info.perimeter) * np.array(
                            [self.snapshot_width, self.snapshot_height, self.snapshot_width, self.snapshot_height]
                        )

                        alert_info.perimeter = bbox_to_perim(perim)
                    elif alert_info.perimeter is None and alert_info.snapshots:
                        perim = np.array([0, 0, self.snapshot_width - 1, self.snapshot_height - 1])
                        alert_info.perimeter = bbox_to_perim(perim)

            # generate zoom per timestamp with alert entities
            alerts_sorted = sorted([alert for alert in alert_results if alert], key=attrgetter("timestamp"))
            alerts_grouped = {k: list(g) for k, g in groupby(alerts_sorted, key=attrgetter("timestamp"))}
            for ts, alerts in alerts_grouped.items():
                if ts in timestamps:
                    alerts_id_index = [a.idIndex for a in alerts if a.idIndex is not None]
                    index = timestamps.index(ts)
                    annotated_image = self.annotate_thumbnail(detector_images[index], ts, batch_data, alerts_id_index)
                    alerts_thumbs[ts] = annotated_image

                    if ts in self.alert_thumb_buffer:
                        self.alert_thumb_buffer[ts] = self.annotate_thumbnail(
                            self.alert_thumb_buffer[ts][0], ts, batch_data, alerts_id_index
                        )
                    continue

                if ts in self.alert_thumb_buffer:  # timestamp not in the list meaning it has happened before the batch
                    # either we can send or there is an alert, so force it
                    alert_id = alerts[0].eventId
                    normalized_ts = self.thumb_handler.normalize(ts)
                    img, annotations = self.alert_thumb_buffer.pop(ts)
                    thumb_name = f"{self.thumb_handler.prefix}-{normalized_ts}-ooc-{alert_id}.jpg"
                    self.thumb_handler.send_out_of_cycle(img, thumb_name)
                    thumbnail_info = ThumbnailInfo(thumb_name, ts, 0, normalized_ts, annotations=annotations)
                    thumbnails.append(thumbnail_info)
                else:
                    thumb_name, normalized_ts = self.thumb_handler.generate_name(ts, thumb_count=0)

                for alert_info in alerts:
                    alert_info.mainThumbnail = thumb_name
                    alert_info.mainThumbnailTimestamp = normalized_ts

        for i, im in enumerate(image_batch):
            if im.timestamp in alerts_thumbs or self.thumb_handler.can_send(im.timestamp):
                # check if we need to extract thumbnail
                if im.timestamp in alerts_thumbs:
                    annotated_image, annotations = alerts_thumbs[im.timestamp]
                    can_skip = False
                else:
                    annotated_image, annotations = self.annotate_thumbnail(
                        detector_images[i], im.timestamp, batch_data, []
                    )
                    can_skip, is_mandatory = self.can_skip(im.timestamp, alert_results)

                    if can_skip and not is_mandatory:
                        self.thumb_handler.skip(im.timestamp)
                        self.skipped_thumbnail_queue.append(
                            (im.timestamp, im.frame_number, annotated_image, annotations)
                        )
                        annotated_image = None

                if not can_skip:
                    self.block_skip_counter = self.motion_window + 1  # reset the counter

                    # send skipped thumbnails
                    while len(self.skipped_thumbnail_queue):
                        skipped_ts, skipped_frame_num, skipped_img, annotations = self.skipped_thumbnail_queue.popleft()
                        thumb_name, normalized_ts = self.thumb_handler.generate_name(skipped_ts, thumb_count=0)
                        self.thumb_handler.send_out_of_cycle(skipped_img, thumb_name)
                        thumbnail_info = ThumbnailInfo(
                            thumb_name, skipped_ts, skipped_frame_num, normalized_ts, annotations
                        )
                        thumbnails.append(thumbnail_info)

                # zero motion indication
                if self.thumb_handler.can_send(im.timestamp):
                    self.motion_index = 0

                    if not can_skip:
                        # send thumbnail to be encoded at its proper time - use original image and not annotated
                        self.clip_dispatcher.encode_thumbnail_async(
                            cv2.cvtColor(detector_images[i], cv2.COLOR_BGR2RGB), im.timestamp, immediate=False
                        )

                if annotated_image is not None:
                    self.block_skip_counter -= 1

                    # either we can send or there is an alert, so force it
                    thumb_name, normalized_ts = self.thumb_handler.send(annotated_image, im.timestamp, force=True)
                    thumbnail_info = ThumbnailInfo(
                        thumb_name, im.timestamp, im.frame_number, normalized_ts, annotations
                    )
                    thumbnails.append(thumbnail_info)

                    # add main thumbnail to alert annotation
                    if im.timestamp in alerts_grouped:
                        for alert_info in alerts_grouped[im.timestamp]:
                            alert_info.mainThumbnail = thumb_name
                            alert_info.mainThumbnailTimestamp = normalized_ts

                    self.last_thumbnail = thumbnail_info

        # handle snapshots
        if self.snapshot_handler.can_send(last_ts):
            scaled_image = self.snapshot_resizer(image_batch[-1])
            im_size = scaled_image.shape[1::-1]
            snap_annotator = Annotator(scaled_image, line_width=1, pil=not ascii)
            im_data = batch_data.query(BatchDataResolver.TIMESTAMP, last_ts)
            scale_factors = [*im_size, *im_size]
            magnitude = self.magnitude * self.thumb_width / self.snapshot_width
            for annot in im_data:
                bboxes = (np.clip(annot[BatchDataResolver.POS], 0, 1) * scale_factors).astype(int)
                snap_annotator.pixelate(
                    bboxes,
                    int(annot[BatchDataResolver.SUBCLASS]) in self.pixelate_classes,
                    magnitude,
                    self.pixelate,
                )

            new_regions = self.scale_pixelation_regions(im_size)
            self.apply_pixelate_regions(scaled_image, new_regions)

            thumb_name, normalized_ts = self.snapshot_handler.send(scaled_image, last_ts)
            snapshots.append({"thumbnail": thumb_name, "timestamp": last_ts, "thumType": ThumbnailType.Snapshot.value})

            self.last_snapshot = thumb_name
            self.snapshot_cycle = True

        # handle training images
        if self.training_sender.can_send(last_ts) and len(self.training_selector):
            image_data = self.training_selector.pop()
            thumb_name, _ = self.training_sender.send(image_data.image, last_ts)
            trainings.append(
                {
                    "thumbnail": thumb_name,
                    "timestamp": last_ts,
                    "thumType": ThumbnailType.Training.value,
                    "edgeMetadata": {
                        "detections": image_data.detections.astype(float).tolist(),
                        "confidence": image_data.conf,
                        "night_mode": image_data.is_night_mode,
                        "background": image_data.is_background,
                        "blacklist": image_data.is_blacklist,
                    },
                }
            )
            new_period = self.training_selector.get_save_period()
            if self.training_sender.period != new_period * 1000:
                self.training_sender.set_period(new_period)

        # handle local action thumbnail - generate a hard copy of the main thumbnail
        for alert_info in alert_results:
            if alert_info.storeLocalThumbnail and alert_info.mainThumbnail:
                try:
                    org_file = Path(os.path.join(self.thumb_handler.dest_path, alert_info.mainThumbnail))
                    new_name = org_file.with_name(f"{org_file.stem}.local{org_file.suffix}")
                    os.link(org_file, new_name)
                except Exception as e:
                    log_exception(logger, "could not rename file for local action", e)

        return thumbnails, snapshots, trainings, crops, best_images, zoom_images

    def set_pixelate_filters(self, pixelate_regions, pixelate_classes, pixelate, thumb_input_size):
        self.pixelate_regions = pixelate_regions
        self.pixelate_classes = pixelate_classes
        self.pixelate = pixelate
        self.thumb_height = thumb_input_size[1]

    def annotate_thumbnail(self, frame, timestamp, batch_results: BatchDataResolver, alerts_id_index):
        im_scaled = scale_image_by_width(frame, self.thumb_width)
        # handle object annotation and pixelation
        scale_factors = np.array([*im_scaled.shape[1::-1], *im_scaled.shape[1::-1]])
        annotator = Annotator(im_scaled, line_width=1, pil=not ascii)
        im_data = batch_results.query(BatchDataResolver.TIMESTAMP, timestamp)
        for annot in im_data:
            bboxes = (np.clip(annot[BatchDataResolver.POS], 0, 1) * scale_factors).astype(int)
            annotator.pixelate(
                bboxes, int(annot[BatchDataResolver.CLASS]) in self.pixelate_classes, self.magnitude, self.pixelate
            )
            det_id = int(annot[BatchDataResolver.ID])
            # label = str(det_id) if det_id > 0 else "D"
            closed_bbx = det_id >= 0 and det_id in alerts_id_index
            obj_id = int(annot[BatchDataResolver.CLASS])
            if obj_id not in self.object_to_exclude_annotations:
                annotator.box_label(bboxes, color=colors(obj_id, True), isClosed=closed_bbx)
        self.apply_pixelate_regions(im_scaled)
        annotation = im_data[:, BatchDataResolver.POS + [BatchDataResolver.CLASS, BatchDataResolver.ID]]

        return im_scaled, annotation.tolist()

    def apply_pixelate_regions(self, im_scaled, regions=None):
        if regions is None:
            regions = self.pixelate_regions

        # handle pixelation regions
        for region in regions:
            bbox = region["bbox"]
            crop = im_scaled[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
            if self.pixelate:  # if not pixelate but has region - it means blackening
                scale_percent = self.magnitude / 100
                downscaled = cv2.resize(
                    im_scaled[bbox[1] : bbox[3], bbox[0] : bbox[2], :],
                    None,
                    fx=scale_percent,
                    fy=scale_percent,
                    interpolation=cv2.INTER_AREA,
                )
                pixelated = cv2.resize(downscaled, crop.shape[1::-1], interpolation=cv2.INTER_NEAREST)
                crop *= region["mask"]
                crop += pixelated * (1 - region["mask"])
            else:
                crop *= region["mask"]

    def scale_pixelation_regions(self, new_size):
        new_regions = []
        if new_size[0] == self.thumb_width:
            return self.pixelate_regions
        scale_factor = np.array(new_size) / np.array([self.thumb_width, self.thumb_height])
        for region in self.pixelate_regions:
            new_bbox = (region["bbox"] * np.array([*scale_factor, *scale_factor])).astype(int)
            new_mask = cv2.resize(
                region["mask"], (new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]), interpolation=cv2.INTER_NEAREST
            )
            new_regions.append({"bbox": new_bbox, "mask": new_mask[:, :, np.newaxis]})
        return new_regions

    def _build_motion_info(self, motion_data: MotionData, timestamps: List[int]) -> Dict:
        norm_ts = np.array(timestamps) - self.last_motion_timestamp_for_info
        new_period_idx = np.where(norm_ts > self.mv_duration)
        curr_mv_out = self.last_mv_out
        motion_info = None
        motion_est = motion_data.per_frame_motion
        if new_period_idx[0].size > 0:
            end_id = int(new_period_idx[0][0])
            curr_mv = [motion_est[i] for i in range(end_id) if motion_est[i] is not None]
            self.last_motion_frames += end_id
            if curr_mv:
                curr_mv_out = np.max(np.stack([*curr_mv, curr_mv_out], axis=2), axis=2)
            motion_grid_fine = find_max_motion_blocks(curr_mv_out, 16)
            motion_grid = find_max_motion_blocks(motion_grid_fine, 4)
            max_motion = np.max(motion_grid)
            if max_motion >= self.min_motion_output:
                motion_info = {
                    "startTimestamp": self.last_motion_timestamp_for_info,
                    "endTimestamp": timestamps[end_id],
                    "numberOfFrames": int(self.last_motion_frames),
                    # "vector": curr_mv_out.flatten().astype(int).tolist(),
                    # "maxMotionGrid": motion_grid.flatten().astype(int).tolist(),
                    # "maxMotionGridFine": motion_grid_fine.flatten().astype(int).tolist(),
                    "maxMotion": int(max_motion),
                    "maxMotionGridJson": {
                        f"b{index}": int(value)
                        for index, value in enumerate(motion_grid.flatten())
                        if value > self.min_motion_output
                    },
                    "maxMotionGridFineJson": {
                        f"b{index}": int(value)
                        for index, value in enumerate(motion_grid_fine.flatten())
                        if value > self.min_motion_output
                    },
                    "vectorJson": {
                        f"b{index}": int(value)
                        for index, value in enumerate(curr_mv_out.flatten())
                        if value > self.min_motion_output
                    },
                }

            # reset internal info
            self.last_motion_frames = 0
            self.last_mv_out *= 0
            self.last_motion_timestamp_for_info = timestamps[end_id]

        else:
            end_id = 0
            self.last_motion_frames += len(timestamps)
        curr_mv = [mv for mv in motion_est[end_id:] if mv is not None]
        if curr_mv:
            self.last_mv_out = np.max(np.stack([*curr_mv, curr_mv_out], axis=2), axis=2)
        return motion_info

    def _add_motion_index(self, motion_data: MotionData):
        self.motion_index = max(motion_data.index, self.motion_index)

    def can_skip(self, timestamp, alerts):
        norm_ts = self.thumb_handler.normalize(timestamp)
        is_mandatory = norm_ts % self.mandatory_thumbnail_period == 0 or self.block_skip_counter > 0
        can_skip = self.motion_index <= self.min_motion_for_thumb
        if alerts:
            can_skip = False

        if can_skip:
            obj_motion = self.timeline_builder.calc_objects_motion(
                self.last_thumbnail.timestamp, self.min_obj_motion_for_thumb
            )
            can_skip = obj_motion < self.min_obj_motion_for_thumb

        if self.log_thumbnails:
            logger.info(f"timestamp: {timestamp}, norm_ts: {norm_ts}, motion_index: {self.motion_index}")
        return can_skip, is_mandatory

    def _maintain_alert_buffer(self, buffer):
        entries_to_delete = len(buffer) - self.max_alert_buffer_size
        if entries_to_delete > 0:
            for _ in range(entries_to_delete):
                buffer.popitem(last=False)

    @property
    def class_handler(self) -> ClassHandler:
        return self.context.class_handler

    def _handle_clip_for_objects(self, object_encoding_batch, timestamps):
        immediate_tasks = []

        for id, img in object_encoding_batch.items():
            ent_data = self.entity_db.get_entity(id)
            priority = self._calc_clip_priority(ent_data, img)
            if priority > 0:
                custom_obj = {"object_id": ent_data.object_id, "is_reid": False}
                custom_obj.update(ent_data.custom_object_data)
                is_immediate = ent_data.track_id in self.alerted_ents if ent_data else False
                to_push = ClipTask(ent_data.id_base, ent_data.track_id, -1, None, custom_obj)
                if is_immediate:
                    immediate_tasks.append((to_push, img))
                else:
                    is_pushed, popped = self.clip_queue.push_pop(to_push, priority)
                    if popped:
                        self.clip_mem_buffer.release(popped.memory_index)

                    if is_pushed:
                        mem_idx = self.clip_mem_buffer.acquire()
                        if mem_idx < 0:
                            logger.error("memory buffer is full, cannot process clip task")
                        else:
                            mem = self.clip_mem_buffer.next(mem_idx)
                            to_push.memory_index = mem_idx
                            to_push.memory_buffer = mem
                            np.copyto(
                                mem,
                                cv2.cvtColor(self.clip_dispatcher.crop_resizer(image=img)["image"], cv2.COLOR_BGR2RGB),
                            )
                    else:
                        logger.info(f"clip task for {ent_data.id_base} rejected due to low priority, skipping")

        while immediate_tasks:
            task, img = immediate_tasks.pop()
            img = cv2.cvtColor(self.clip_dispatcher.crop_resizer(image=img)["image"], cv2.COLOR_BGR2RGB)
            self.clip_dispatcher.encode_object_crop_async(
                img, task.id_base, task.id_index, extra=task.custom_object, immediate=True
            )

            # for rate control
            self.last_clip_object_ts += self.clip_period

        while np.any(np.array(timestamps) > self.last_clip_object_ts + self.clip_period):
            task: ClipTask = self.clip_queue.pop()
            if task is None:
                # empty queue, nothing to process
                self.last_clip_object_ts = timestamps[0]  # reset to the first timestamp
                break
            else:
                mem_idx = task.memory_index
                if mem_idx < 0:
                    logger.error(f"clip task for object {task.id_index} has no memory, skipping")
                    continue
                else:
                    self.clip_mem_buffer.release(mem_idx)
                    self.clip_dispatcher.encode_object_crop_async(
                        task.memory_buffer, task.id_base, task.id_index, extra=task.custom_object
                    )
                    self.last_clip_object_ts += self.clip_period

        self.clip_objects_results += self.clip_dispatcher.pop_clip_object_async()
        for res in self.clip_objects_results:
            custom_object_data = res.pop("extra", {})
            encodings = res.get("encoding", None)
            object_type = custom_object_data.get("object_id", None)
            ent_data = self.entity_db.get_entity(res["idIndex"])
            if ent_data is not None:  # it can only be updated from the last time
                object_type = ent_data.object_id
                custom_object_data = ent_data.custom_object_data

            from_clip = {}

            # if we have encodings and we haven't run it yet
            if encodings is not None and object_type is not None:
                encodings_vec = np.frombuffer(bytes.fromhex(encodings), dtype=np.float32)
                from_clip["is_clip"] = True
                if not custom_object_data.get("is_clip", False):
                    from_clip = self.entity_db.match_clip_to_custom_object(encodings_vec, object_type)

            # summarize results:
            merged = merge_custom_object_data(custom_object_data, from_clip)
            if ent_data is not None:
                ent_data.custom_object_data = merged
            pos = merged.get("pos", [])
            if pos:
                res.update(convert_custom_object(pos))
                res["timestamp"] = timestamps[-1]

    def _calc_clip_priority(self, ent_data: ActiveEntityData, img: np.ndarray) -> float:
        if ent_data is None or img is None or ent_data.clip is not None:
            return 0
        obj_priority = self.clip_object_priority_factor.get(ent_data.object_id, 0)
        norm_size = img.shape / np.array(self.clip_dispatcher.image_shape)
        min_sz_factor = 0.66
        sz_factors = np.where(norm_size < min_sz_factor, 0, np.clip(norm_size, min_sz_factor, 1.25))
        sz_priority = np.prod(sz_factors)

        dwell = float(ent_data.last_seen - ent_data.first_seen) / 1000
        dwell_priority = 0.0 if dwell < self.clip_min_dwell else min(dwell / self.clip_max_dwell, 1.0)

        alert_priority = 1.0 if ent_data.track_id in self.alerted_ents else 0.0
        total_score = obj_priority * 0.5 + sz_priority * 0.3 + dwell_priority * 0.2 + alert_priority
        return total_score if sz_priority > 0 else alert_priority

    def on_entities_purged(self, ent_ids: List[int]):
        # called every time entity is deleted
        self.alerted_ents -= set(ent_ids)
