from __future__ import absolute_import, division, print_function

import copy
import io
import json
import time
from collections import OrderedDict
from typing import List, Any

import cv2
import numpy as np
import requests
from PIL import Image

from general import proj
from general.analyzer_general import logger, inRoi, Position, ALPR_UNKNOWN, log_exception
from general.core import BatchDataResolver, AdvanceAnalyzerType, WorkItem
from general.img_utils import motion_score
from level1.advanced import BaseAdvanceAnalyzer, AttrData, BaseAdvanceAnalyzerConfig

jetson = proj.load_bool_from_env("JETSON", False)

_session = None
_config = None
TIMEOUT_SECONDS = 2

container_label_map = {
    "Size and Type Codes": "sizeCode",
    "Owner Code and Category Identifier": "ownerCode",
    "Serial Number": "serialNumber",
}


class LprConfig(BaseAdvanceAnalyzerConfig):
    min_success_conf = 0.9
    crop_pixesls_th = 500000
    alprtoken = None
    containertoken = None
    sdk_url = None
    mmc = True
    alprTokenEnable = False
    containerTokenEnable = False
    alprOnPermise = False
    min_dimensions = [240, 200]
    confidenceLimit = {"detection": 0, "plate": 0.7, "mmc": 0.1}
    post_score_weights = {"plate": 6, "mmc": 4, "containerId": 1}
    mmc_success_thresh = {"make": 0.3, "model": 0.3, "colors": 0.7}
    cropFactor = {"width": 0.05, "height": 0.05}
    alprCloudURL = "https://lpr.lumix.ai/v1/plate-reader/"
    containerCloudURL = "https://container-api.parkpow.com/api/v1/predict/"
    min_blur_th = 0.375
    min_alpr_y_gb = 20
    min_alpr_x_gb = 20
    boundry_y_gb = 0.15
    boundry_x_gb = 1
    jpegQuality = 80
    alprURL = "http://localhost:8080"
    margins = [0.1, 0.1]
    min_lp_crop_width = 75
    min_lp_vehicle_overlap = 0.5
    regions = None

    def __init__(self, args_dict=None):
        if args_dict is None:
            args_dict = {}
        super().__init__(args_dict)
        if self.alprOnPermise:
            self.sdk_url = self.alprURL
        self.regions = self.regions or []


class AlprAnalyzer(BaseAdvanceAnalyzer):

    fields = [
        "plate",
        "type",
        "region",
        "model",
        "make",
        "colors",
        "orientation",
        "sizeCode",
        "serialNumber",
        "ownerCode",
    ]
    analyzer_batch_limit = 6
    alpr_batch_limit = 6
    container_batch_limit = 2
    collage_ratio_thresh = 3
    use_low_prio_set = True
    _type = AdvanceAnalyzerType.ALPR
    max_retries = 5
    base_night_keys_to_filter = ["colors"]
    base_keys_to_filter = ["scores", "pscore", "mscore"]
    is_lpr = True
    is_container_id = False
    name = "alprAnalyzer"
    _config_type = LprConfig
    args: LprConfig

    def __init__(self, msg: dict, context, instance_id: str = None):
        super().__init__(msg, context, instance_id)
        print("Beginning alpr Analyzer init...")

        global _config
        _config = copy.deepcopy(self.args)
        self.post_score_failure_th = self.args.confidenceLimit["plate"]
        self.post_score_success_th = self.args.min_success_conf
        self.alprCloudURL = self.args.alprCloudURL
        self.is_export_invalid_crops = False  # should be true once the mechanism is inside entity db

        if msg["alprTokenEnable"]:
            logger.info("ALPR Liscense is used")
        else:
            logger.info("Running in dummy ALPR mode")

        self.supported_classes = []
        if self.class_handler.vehicle_value >= 0:
            self.supported_classes = [self.class_handler.vehicle_value]
            self.supported_subclasses = [
                self.class_handler.class_str_to_int(s)
                for s in ["car", "motorcycle", "bus", "truck"]
                if s in self.class_handler.classes_names
            ]

        self.stats["lpr"] = {}
        self.stats["lpr"]["api_calls"] = 0
        self.stats["lpr"]["api_errors"] = 0
        self.stats["lpr"]["hits"] = 0
        self.stats["lpr"]["miss"] = 0

        self.keys_to_filter = copy.copy(self.base_keys_to_filter)
        self.night_keys_to_filter = copy.copy(self.base_night_keys_to_filter)
        self.zoom_ar = (
            context.analytic_config.get("trainThumbPolicy", {}).get("crops_ar", {}).get("license_plate", None)
        )
        self.zoom_crop_margins = self.args.margins
        self.crop_ar = context.analytic_config.get("trainThumbPolicy", {}).get("crops_ar", {}).get("vehicle", None)
        self.lp_subclass = self.class_handler.plate_subclass_value

    def alpr_frame_check(self, crop_info, conf, shape):
        height = shape[0]
        width = shape[1]
        bboxes = [crop_info[0] * width, crop_info[1] * height, crop_info[2] * width, crop_info[3] * height]
        # first we test boundary and confidence
        confidence = round(float(conf), 2)
        crop_width = bboxes[2] - bboxes[0]
        crop_height = bboxes[3] - bboxes[1]

        edge_x = (bboxes[0] <= self.args.min_alpr_x_gb) or (bboxes[2] >= width - self.args.min_alpr_x_gb)
        edge_y = (bboxes[1] <= self.args.min_alpr_y_gb) or (bboxes[3] >= height - self.args.min_alpr_y_gb)

        pos = Position(bboxes, shape)

        edge = (edge_x and not (self.args.boundry_x_gb < pos["x"] <= 1 - self.args.boundry_x_gb)) or (
            edge_y and not (self.args.boundry_y_gb < pos["y"] <= 1 - self.args.boundry_y_gb)
        )

        # TBD: set more limits to containers
        global_hit = (
            crop_width > self.args.min_dimensions[0]
            and crop_height > self.args.min_dimensions[1]
            and confidence > self.args.confidenceLimit["detection"]
            and not edge
        )
        if not global_hit:
            return 0

        filter_hit = False
        for _, filter in enumerate(self.filter.union):
            if inRoi(filter, pos):
                filter_hit = True
                break
        if not (filter_hit):
            return 0

        return min(crop_width * crop_height, self.args.crop_pixesls_th) / (height * width)

    def get_metadata(self, crop_info):
        location = int(crop_info[BatchDataResolver.LOCATION])
        md = {
            "plate": True if location < len(self.filter.plate) and self.filter.plate[location] else False,
            "mmc": True if location < len(self.filter.mmc) and self.filter.mmc[location] else False,
        }
        # is this crop in ROI of plate, mmc, or containerId?
        return md

    def get_crop_margins(self):
        return [self.args.cropFactor["width"], self.args.cropFactor["height"]]

    def _get_vehicle_indices_with_valid_plates(self, batch_data: BatchDataResolver):
        """Return set of vehicle detection indices that have a valid paired license plate,
        or None if plate filtering should be skipped."""
        if not self.is_lpr or self.lp_subclass < 0:
            return None

        plates = batch_data.query(BatchDataResolver.SUBCLASS, self.lp_subclass)
        if len(plates) == 0:
            return set()

        width, height = batch_data.full_resolution
        # quality filters on plate detections
        plate_pos = plates[:, BatchDataResolver.POS]
        plate_widths = (plate_pos[:, 2] - plate_pos[:, 0]) * width
        valid = plate_widths >= self.args.min_lp_crop_width
        # reject plates touching image edges
        x1_px = plate_pos[:, 0] * width
        y1_px = plate_pos[:, 1] * height
        x2_px = plate_pos[:, 2] * width
        y2_px = plate_pos[:, 3] * height
        touches_edge = (x1_px <= 0) | (y1_px <= 0) | (x2_px >= (width - 1)) | (y2_px >= (height - 1))
        valid &= ~touches_edge

        valid_plates = plates[valid]
        if len(valid_plates) == 0:
            return set()

        # find vehicle detections overlapping with valid plates
        plate_indices = valid_plates[:, BatchDataResolver.INDEX].astype(int)
        vehicle_mask = (batch_data.data[:, BatchDataResolver.CLASS] == self.class_handler.vehicle_value).astype(int)
        plate_overlaps = batch_data.all_overlaps[plate_indices]
        _, vehicle_indices = np.where(plate_overlaps * vehicle_mask > self.args.min_lp_vehicle_overlap)
        return set(vehicle_indices.tolist())

    def calculate_pre_score(self, batch_data: BatchDataResolver, image_batch=None):
        cars_detections = batch_data.query(batch_data.SUBCLASS, self.supported_subclasses)
        scores = {}

        # filter vehicles that don't have a valid paired license plate
        vehicles_with_plates = self._get_vehicle_indices_with_valid_plates(batch_data)

        for idx, crop_info in enumerate(cars_detections):
            obj_id = int(crop_info[batch_data.ID])
            crop_ind = int(crop_info[batch_data.INDEX])
            if obj_id == -1:
                scores[crop_ind] = -1
                continue
            # if we have a plate filter, and this vehicle doesn't have a valid paired plate, give it a score of -1 to filter it out
            if vehicles_with_plates is not None and crop_ind not in vehicles_with_plates:
                scores[crop_ind] = -1
                continue

            # check if the crop is too close to the processed crop
            x_crop = (crop_info[0] + crop_info[2]) / 2
            y_crop = (crop_info[1] + crop_info[3]) / 2
            if int(crop_info[batch_data.ID]) in self.processed:
                processed_crop = self.processed[int(crop_info[batch_data.ID])][-1]
                x_processed_crop = (processed_crop.pos[0] + processed_crop.pos[2]) / 2
                y_processed_crop = (processed_crop.pos[1] + processed_crop.pos[3]) / 2
                if (
                    (x_crop - x_processed_crop) ** 2 + (y_crop - y_processed_crop) ** 2
                ) ** 0.5 < self.distance_threshold:
                    scores[crop_ind] = -1
                    continue

            confidence = crop_info[batch_data.CONFIDENCE]
            shape = image_batch[int(crop_info[batch_data.FRAME_ID])].frame.shape
            alpr_check_score = self.alpr_frame_check(crop_info, confidence, shape)

            scores[crop_ind] = -1
            if alpr_check_score > 0:
                scores[crop_ind] = confidence * alpr_check_score
        return scores

    def update_score(self, attr, key, scores):
        if key in attr and attr[key] is not None:
            scores.append(attr[key])
        else:
            scores.append(0)

    def add_to_low_prio_set(self, obj_id, metadata):
        can_add = False
        if obj_id not in self.processed or len(self.processed[obj_id]) == 0:
            can_add = True
        else:
            attr = self.processed[obj_id][-1].attributes
            has_plate = True if "pscore" in attr and attr["pscore"] >= self.post_score_success_th else False
            has_mmc = True if "mscore" in attr and attr["mscore"] >= self.post_score_success_th else False
            if metadata["plate"]:
                # if this crop is in plate ROI, and we don't have a plate, add it
                if not has_plate:
                    can_add = True
            elif metadata["mmc"]:
                # if this crop is in mmc ROI, and we don't have a mmc and plate, add it
                if not has_plate and not has_mmc:
                    can_add = True
        if can_add:
            self.low_prio_set.add(obj_id)

    def calculate_post_score(self, obj_id: int, attr_data: AttrData) -> float:
        if (
            attr_data.attributes is None
            or "scores" not in attr_data.attributes
            or len(attr_data.attributes["scores"]) == 0
        ):
            return 0

        scores = []

        plate_score = 0
        if "pscore" in attr_data.attributes and attr_data.attributes["pscore"] is not None:
            plate_score = attr_data.attributes["pscore"]

        def add_plate_score(plate_score):
            # add plate score from now or from processed if we have a filter for plate and plate is good
            if plate_score == 0 and obj_id in self.processed and len(self.processed[obj_id]):
                attr = self.processed[obj_id][-1].attributes
                if "pscore" in attr and attr["pscore"] is not None:
                    plate_score = attr["pscore"]
            if plate_score > 0:
                scores.append((plate_score, self.args.post_score_weights["plate"]))

        if attr_data.metadata["plate"]:
            # car is in the ROI of plate
            scores.append((plate_score, self.args.post_score_weights["plate"]))
        elif attr_data.metadata["mmc"]:
            # car is in the ROI of mmc
            if "mscore" in attr_data.attributes and attr_data.attributes["mscore"] is not None:
                mmc_score = attr_data.attributes["mscore"]
                scores.append((mmc_score, self.args.post_score_weights["mmc"]))
            if self.filter.plate:
                add_plate_score(plate_score)

        if len(scores) == 0:
            return 0

        total_scores = 0
        total_weights = 0
        for score, weight in scores:
            total_scores += score * weight
            total_weights += weight

        if total_weights == 0:
            return 0  # not suppose to happen, but just in case

        return total_scores / total_weights

    def is_center_in_bbox(self, bb1, bb2):
        center_bbox = [None, None, None, None]
        qx = (bb1[2] - bb1[0]) / 4
        center_bbox[0] = bb1[0] + qx
        center_bbox[2] = bb1[2] - qx
        qy = (bb1[3] - bb1[1]) / 4
        center_bbox[1] = bb1[1] + qy
        center_bbox[3] = bb1[3] - qy

        return (
            center_bbox[0] <= (bb2[2] + bb2[0]) / 2 <= center_bbox[2]
            and center_bbox[1] <= (bb2[1] + bb2[3]) / 2 <= center_bbox[3]
        )

    def prepare_batch(self, batch: List[WorkItem]):
        def get_image_area(image_in):
            return image_in.shape[0] * image_in.shape[1]

        ratio_threshold = self.collage_ratio_thresh
        max_in_collage = 1 if self.is_container_id else self.analyzer_batch_limit
        # Sort images by area in ascending order
        sorted_batch_arr = sorted(enumerate(batch), key=lambda x: get_image_area(x[1].image))
        sorted_to_orig_idx = {j: i for j, (i, _) in enumerate(sorted_batch_arr)}
        sorted_to_obj_id = {j: ns.id for j, (i, ns) in enumerate(sorted_batch_arr)}

        sorted_batch = [work_item for _, work_item in sorted_batch_arr]

        uniq_ids = set(sorted_to_obj_id.values())
        if len(uniq_ids) == 1:
            # all images are from the same object, no need to split
            ratio_threshold = 100

        # Split images into multiple arrays based on area ratio
        blurred_set = set()
        sub_batches = []
        current_batches = []
        metrics = {}
        for i, b in enumerate(sorted_batch):
            image = b.image
            orig_idx = sorted_to_orig_idx[i]
            bluriness = motion_score(image)
            area = get_image_area(image)
            location = batch[orig_idx].extra.location
            metrics[orig_idx] = {"area": area, "bluriness": bluriness, "location": location}

            if bluriness <= self.args.min_blur_th:
                blurred_set.add(orig_idx)
                continue
            if len(current_batches) == 0:
                current_batches = [b]
                max_area = area
                min_area = area

            # we support up to 6 images in a collage
            elif len(current_batches) == max_in_collage or (
                (area > max_area and area / max_area > ratio_threshold)
                or (area < min_area and min_area / area > ratio_threshold)
            ):
                sub_batches.append(current_batches)
                current_batches = [b]
                max_area = area
                min_area = area
            else:
                current_batches.append(b)
                max_area = max(max_area, area)
                min_area = min(min_area, area)

        if len(current_batches) > 0:
            sub_batches.append(current_batches)

        return sub_batches, sorted_to_orig_idx, sorted_to_obj_id, blurred_set, metrics

    def _run_model(self, batch: List[WorkItem]) -> List[Any]:
        results = [None] * len(batch)
        res_per_id = {}

        sub_batches, sorted_to_orig_idx, sorted_to_obj_id, blurred_set, metrics = self.prepare_batch(batch)

        processed_count = 0
        use_image = set()
        zoom_images = {}
        for sub_batch in sub_batches:
            images = []
            obj_ids = []
            for ind, work_item in enumerate(sub_batch):
                obj_ids.append(work_item.id)
                images.append(work_item.image)

            collage, clg_bboxes = create_collage(images)
            api_res = self.run_api(collage, obj_ids)

            # to avoid using the same result more than once
            processed_api_res = {"alpr": [], "containerId": []}
            for idx, clg_box in enumerate(clg_bboxes):
                batch_idx = sorted_to_orig_idx[processed_count + idx]
                obj_id = sorted_to_obj_id[processed_count + idx]
                info = {}
                if "containerId" in api_res:
                    info = self.process_container_res(api_res, processed_api_res, clg_box)
                if "alpr" in api_res:
                    info_alpr = self.process_alpr_res(api_res, clg_box, processed_api_res)
                    if "scores" in info_alpr:
                        info_alpr["scores"].update(info.get("scores", {}))
                    info.update(info_alpr)

                if obj_id in res_per_id:
                    res_per_id[obj_id], did_update = self.merge_results(res_per_id[obj_id], info)
                elif obj_id in self.processed:
                    res_per_id[obj_id], did_update = self.merge_results(self.processed[obj_id][-1].attributes, info)
                else:
                    res_per_id[obj_id], did_update = self.merge_results({}, info)

                improved_plate = did_update.get("plate", 0)
                improved_mmc = (
                    (1 if did_update.get("make", 0) else 0)
                    + (1 if did_update.get("model", 0) else 0)
                    + (1 if did_update.get("colors", 0) else 0)
                )
                if improved_plate or improved_mmc == 3 or (batch[batch_idx].extra.metadata["mmc"] and improved_mmc):
                    use_image.add(batch_idx)
                if improved_plate:
                    zoom_images[batch_idx] = self.crop_zoom_function(collage, info["plate_box"])
                if self.is_container_id:
                    if batch_idx not in use_image:
                        improved_cont = sum(
                            [1 if did_update.get(key, 0) else 0 for key in container_label_map.values()]
                        )
                        if improved_cont:
                            use_image.add(batch_idx)
                            if "containerId_box" in info:
                                zoom_images[batch_idx] = self.crop_zoom_function(collage, info["containerId_box"])

            processed_count += len(sub_batch)

        for batch_idx, work_item in enumerate(batch):

            if batch_idx in blurred_set:
                results[batch_idx] = {}
                results[batch_idx]["use_image"] = True
                continue
            obj_id = work_item.id
            if obj_id not in res_per_id:
                results[batch_idx] = {}
                results[batch_idx]["use_image"] = False
                continue
            results[batch_idx] = copy.deepcopy(res_per_id[obj_id])
            results[batch_idx]["use_image"] = batch_idx in use_image
            if batch_idx in use_image:
                results[batch_idx]["metrics"] = metrics[batch_idx]
            if batch_idx in zoom_images:
                results[batch_idx]["zoom_image"] = zoom_images.pop(batch_idx)

        for attr in results:
            if attr is not None:
                self.update_stats(attr)

        return results

    # def update_filters(self, enabled, filters):
    #     super().update_filters(enabled, filters)
    #     self.is_lpr = self.args.alprTokenEnable and bool(self.filter.plate or self.filter.mmc)
    #     self.is_container_id = self.args.containerTokenEnable and bool(self.filter.containerId)
    #     if  self.is_container_id:
    #         self.analyzer_batch_limit = self.container_batch_limit
    #         self.is_lpr = False
    #         self.supported_subclasses = self.containerId_classes
    #     else:
    #         self.analyzer_batch_limit = self.alpr_batch_limit
    #         self.supported_subclasses = self.alpr_classes
    #         self.is_container_id = False

    def process_alpr_res(self, api_res, clg_box, processed_api_res):

        # empty return message means failure
        if not api_res["alpr"]:
            return {}

        # no results means no car
        if "results" not in api_res["alpr"] or not api_res["alpr"]["results"]:
            return failure_alpr()

        # parse and check validity
        result_ok = False
        info = {}
        if len(api_res["alpr"]["results"]):
            # find the first result which its bbox is inside the clg_bbox
            for api_idx, recognition in enumerate(api_res["alpr"]["results"]):
                if api_idx in processed_api_res["alpr"]:
                    # this result was already used
                    continue
                keyword = "agent" if "agent" in recognition else "vehicle"
                if keyword in recognition and recognition[keyword] is not None:
                    api_box = self.create_bbox_from_api(recognition[keyword]["box"])

                    # check if the result's bbox is inside the clg_bbox
                    if self.is_center_in_bbox(clg_box, api_box):
                        # mark the result as used
                        processed_api_res["alpr"].append(api_idx)
                        info = self.get_vehicle_res(api_res, api_idx, clg_box, keyword=keyword)
                        if "plate" in info:
                            info["plate_box"] = self.create_bbox_from_api(recognition["plate"]["box"])
                        result_ok = True
                        break
        if not result_ok:
            info = failure_alpr()

        return info

    def update_container_info(self, result, info):
        if not info:
            info = default_containerId()
        # Labels to extract
        label = container_label_map.get(result["object"]["label"], None)
        current_score = result["object"]["score"]
        if label and current_score > info["scores"][label]:
            info[label] = [result["texts"][0]["value"]]
            info["scores"][label] = result["texts"][0]["score"]
            # info["globalType"] = ["success"]
        return info

    def process_container_res(self, api_res, processed_api_res, clg_box):
        # empty return message means failure
        if not api_res["containerId"]:
            return {}

        # no results means no car
        if "results" not in api_res["containerId"] or not api_res["containerId"]["results"]:
            return failure_containerId()

        # parse and check validity
        info = {}
        bboxes = []

        if len(api_res["containerId"]["results"]):
            # find the first result which its bbox is inside the clg_bbox
            for api_idx, recognition in enumerate(api_res["containerId"]["results"]):
                keyword = "object"
                if keyword in recognition and recognition[keyword] is not None:
                    bboxes.append(self.create_bbox_from_api(recognition[keyword]["value"]))
                    processed_api_res["containerId"].append(api_idx)
                    info = self.update_container_info(recognition, info)
        info["globalType"] = ["success"]
        if all(score == 0 for score in info["scores"].values()):
            info = failure_containerId()
            info["globalType"] = ["failure"]
        elif bboxes:
            wh = np.array(clg_box[2:] + clg_box[2:])
            bboxes = np.array(bboxes)
            bbox = np.array(np.min(bboxes[:, :2], axis=0).tolist() + np.max(bboxes[:, 2:], axis=0).tolist())
            info["containerId_box"] = (bbox * wh).astype(int).tolist()
        return info

    def create_bbox_from_api(self, api_res):
        api_box = [None] * 4
        api_box[0] = api_res["xmin"]
        api_box[1] = api_res["ymin"]
        api_box[2] = api_res["xmax"]
        api_box[3] = api_res["ymax"]
        return api_box

    def run_api(self, vehicle, obj_ids):

        # join the values together with underscores
        tmp_image = Image.fromarray(vehicle)
        jpeg_bytes = io.BytesIO()
        tmp_image.save(jpeg_bytes, format="JPEG", quality=self.args.jpegQuality)
        jpeg_bytes = jpeg_bytes.getvalue()
        api_res = {}
        if self.is_lpr:
            api_res["alpr"] = dict(
                recognition_api(
                    camera_id=self.instance_id,
                    jpeg_bytes=jpeg_bytes,
                    alprCloudURL=self.alprCloudURL,
                    containerCloudURL=self.args.containerCloudURL,
                    mmc=self.args.mmc,
                    regions=self.args.regions,
                    api_key=self.args.alprtoken,
                    sdk_url=self.args.sdk_url,
                    config={"detection_mode": "vehicle", "mode": "fast"},
                    timestamp=None,
                    stats=self.stats,
                )
            )

        if self.is_container_id:
            api_res["containerId"] = dict(
                recognition_api(
                    camera_id=self.instance_id,
                    jpeg_bytes=jpeg_bytes,
                    alprCloudURL=self.alprCloudURL,
                    containerCloudURL=self.args.containerCloudURL,
                    mmc=self.args.mmc,
                    regions=[],
                    api_key=self.args.containertoken,
                    sdk_url="container-api",
                    config={"detection_mode": "vehicle", "mode": "fast"},
                    timestamp=None,
                    stats=self.stats,
                )
            )
        return api_res

    def update_stats(self, attr):
        plate_en = len(self.filter.plate)
        mmc_en = len(self.filter.mmc)

        plate_hit = plate_en and ("plate" in attr)
        mmc_hit = mmc_en and (("make" in attr) or ("model" in attr) or ("colors" in attr))

        if plate_hit or mmc_hit:
            self.stats["lpr"]["hits"] += 1
        else:
            self.stats["lpr"]["miss"] += 1

    def merge_results(self, history, update):
        result = copy.deepcopy(history)
        did_update = {}  # for each field name, did we update it?
        if update:
            if ("scores" not in result) or (result["scores"] is None):
                result["scores"] = {}

            # globalType
            if not ("globalType" in history):
                result["globalType"] = update["globalType"]
            elif "success" in update["globalType"]:
                result["globalType"] = ["success"]
            else:
                result["globalType"] = history["globalType"]

            for field_name in self.fields:
                updt, value, score = field_update(history, update, field_name)
                did_update[field_name] = updt
                if updt:
                    result[field_name] = value
                    result["scores"][field_name] = score

            # add plate score as pscore
            if "scores" in result and "plate" in result["scores"]:
                result["pscore"] = result["scores"]["plate"]

            # add mmc score as mscore
            if "mmc" in update["scores"] or "mmc" in result["scores"]:
                up = update["scores"]["mmc"] if "mmc" in update["scores"] else 0
                his = result["scores"]["mmc"] if "mmc" in result["scores"] else 0
                result["mscore"] = max(up, his)

            if "latency" in update or "latency" in history:
                result["latency"] = [update.get("latency", [0])[0] + history.get("latency", [0])[0]]

        return result, did_update

    def get_vehicle_res(self, api_res: dict, idx: int, bbox: List[int], keyword="agent") -> dict:
        info = {}
        info["scores"] = {}
        info["latency"] = [0]
        # alpr test
        logger.debug(f'ALPR Results: {api_res["alpr"]["results"][idx]}')
        result = api_res["alpr"]["results"][idx]
        if not (result["plate"] is None):
            if self.plate_check(result, bbox):
                info["plate"] = [result["plate"]["props"]["plate"][0]["value"]]
                info["scores"]["plate"] = result["plate"]["props"]["plate"][0]["score"]
                info["region"] = [result["plate"]["props"]["region"][0]["value"].lower()]
                info["scores"]["region"] = result["plate"]["props"]["region"][0]["score"]

        if keyword in result and not (result[keyword] is None):
            if (
                "type" in result[keyword]
                and not (result[keyword]["type"] is None)
                and (result[keyword]["score"] > self.args.confidenceLimit["mmc"])
            ):
                info["type"] = [result[keyword]["type"].lower()]
                info["scores"]["type"] = result[keyword]["score"]
            else:
                info["type"] = ALPR_UNKNOWN
                info["scores"]["type"] = 0

            make_model_score = 0
            color_score = 0
            if "props" in result[keyword] and not (result[keyword]["props"] is None):
                if "make_model" in result[keyword]["props"] and (
                    result[keyword]["props"]["make_model"][0]["score"] > self.args.confidenceLimit["mmc"]
                ):
                    info["make"] = [result[keyword]["props"]["make_model"][0]["make"].lower()]
                    info["model"] = [result[keyword]["props"]["make_model"][0]["model"].lower()]

                    make_model_score = result[keyword]["props"]["make_model"][0]["score"]
                    info["scores"]["make"] = make_model_score
                    info["scores"]["model"] = make_model_score

                if "color" in result[keyword]["props"] and (
                    result[keyword]["props"]["color"][0]["score"] > self.args.confidenceLimit["mmc"]
                ):

                    color = result[keyword]["props"]["color"][0]["value"].lower()
                    # no silver in lumix UI
                    if color == "silver":
                        color = "grey"
                    info["colors"] = [color]
                    color_score = result[keyword]["props"]["color"][0]["score"]
                    info["scores"]["colors"] = color_score

                if "orientation" in result[keyword]["props"] and (
                    result[keyword]["props"]["orientation"][0]["score"] > self.args.confidenceLimit["mmc"]
                ):
                    info["orientation"] = [result[keyword]["props"]["orientation"][0]["value"].lower()]
                    info["scores"]["orientation"] = result[keyword]["props"]["orientation"][0]["score"]

                mmc_score = 0
                thresh = self.args.mmc_success_thresh
                make = 1 if make_model_score >= thresh["make"] else (make_model_score / thresh["make"])
                model = 1 if make_model_score >= thresh["model"] else (make_model_score / thresh["model"])
                colors = 1 if color_score >= thresh["colors"] else (color_score / thresh["colors"])
                mmc_score = (make + model + colors) / 3
                info["scores"]["mmc"] = mmc_score

        if "processing_time" in api_res["alpr"]:
            info["latency"][0] += api_res["alpr"]["processing_time"]
        info["globalType"] = ["success"]

        return info

    def plate_check(self, results, bbox):
        if results["plate"] is None:
            return False

        if results["plate"]["score"] < self.args.confidenceLimit["plate"]:
            return False

        if results["plate"]["props"]["plate"][0]["score"] < self.args.confidenceLimit["plate"]:
            return False

        box = dict(results["plate"]["box"])
        result = (
            (box["xmin"] > bbox[0])
            and (box["ymin"] > bbox[1])
            and (box["xmax"] < (bbox[2] - 2))
            and (box["ymax"] < (bbox[3] - 2))
        )
        return result


def failure_alpr():
    info = {}
    info["scores"] = {}
    info["latency"] = [0]
    info["type"] = ALPR_UNKNOWN
    info["make"] = ALPR_UNKNOWN
    info["model"] = ALPR_UNKNOWN
    info["colors"] = ALPR_UNKNOWN
    info["orientation"] = ALPR_UNKNOWN
    info["scores"]["type"] = 0
    info["scores"]["make"] = 0
    info["scores"]["model"] = 0
    info["scores"]["colors"] = 0
    info["scores"]["orientation"] = 0
    info["globalType"] = ["failure"]
    return info


def default_containerId():
    info = {}
    info["scores"] = {}
    info["latency"] = [0]
    info["sizeCode"] = ALPR_UNKNOWN
    info["ownerCode"] = ALPR_UNKNOWN
    info["serialNumber"] = ALPR_UNKNOWN
    info["scores"]["sizeCode"] = 0
    info["scores"]["ownerCode"] = 0
    info["scores"]["serialNumber"] = 0
    return info


def failure_containerId():
    info = default_containerId()
    info["globalType"] = ["failure"]
    return info


def recognition_api(
    jpeg_bytes,
    alprCloudURL,
    containerCloudURL,
    regions=[],
    api_key=None,
    sdk_url=None,
    config={},
    camera_id=None,
    timestamp=None,
    mmc=None,
    stats=None,
):
    fp = io.BytesIO(jpeg_bytes)
    fp.seek(0)

    global _session
    data = dict(regions=regions, config=json.dumps(config))
    if camera_id:
        data["camera_id"] = camera_id
    if mmc:
        data["mmc"] = mmc
    if timestamp:
        data["timestamp"] = timestamp
    response = None
    t0 = time.time()
    try:
        if sdk_url:
            if stats is not None:
                stats["lpr"]["api_calls"] += 1
            if "container-api" in sdk_url:
                response = requests.post(
                    containerCloudURL,
                    files=dict(image=("example.jpg", fp, "image/jpeg")),
                    headers={
                        "Authorization": "Token " + api_key,
                    },
                    timeout=TIMEOUT_SECONDS,
                    data=data,
                )
            else:
                response = requests.post(
                    sdk_url + "/v1/plate-reader/", files=dict(upload=fp), data=data, timeout=TIMEOUT_SECONDS
                )
        else:
            if not _session:
                _session = requests.Session()
                _session.headers.update({"Authorization": "Token " + api_key})
            fp.seek(0)
            if stats is not None:
                stats["lpr"]["api_calls"] += 1
            response = _session.post(alprCloudURL, files=dict(upload=fp), data=data, timeout=TIMEOUT_SECONDS)
            if response.status_code == 429:  # Max calls per second reached
                if stats is not None:
                    stats["lpr"]["api_errors"] += 1
                logger.error("ALPR: Max calls per second reached ")

        if response is None:
            return {}
        if response.status_code < 200 or response.status_code > 300:
            if stats is not None:
                stats["lpr"]["api_errors"] += 1
            logger.error(f"ALPR: url: {alprCloudURL} response: {response.text}")

            return {}
        return response.json(object_pairs_hook=OrderedDict)
    except Exception as e:
        log_exception(logger, "ALPR Connection error", e)
        if stats is not None:
            stats["lpr"]["api_errors"] += 1
        logger.warning(f"Time for unsuccessful ALPR attempt: {time.time()- t0}")
        return {}


def get_alpr_data(crop):
    global _config
    if _config is not None and _config.alprTokenEnable:
        encode_param = [(cv2.IMWRITE_JPEG_QUALITY), 80]
        jpeg_bytes = cv2.imencode(".jpg", crop, encode_param)[1].tobytes()

        api_res = dict(
            recognition_api(
                camera_id="violentDetection",
                jpeg_bytes=jpeg_bytes,
                alprCloudURL=_config.alprCloudURL,
                containerCloudURL=_config.containerCloudURL,
                mmc=True,
                regions=[],
                api_key=_config.alprtoken,
                sdk_url=None,  # TODO: On permise
                config={"detection_mode": "vehicle", "mode": "fast"},
                timestamp=None,
            )
        )
        if api_res and "results" in api_res and len(api_res["results"]):
            return parse_alpr_res(api_res)
    return {}


def parse_alpr_res(api_res):
    info = {}
    if not (api_res["results"][0]["plate"] is None):
        info["plate"] = api_res["results"][0]["plate"]["props"]["plate"][0]["value"]
        info["region"] = api_res["results"][0]["plate"]["props"]["region"][0]["value"].lower()
    keyword = "agent" if "agent" in api_res["results"][0] else "vehicle"
    if keyword in api_res["results"][0] and not (api_res["results"][0][keyword] is None):

        if "type" in api_res["results"][0][keyword]:
            info["type"] = api_res["results"][0][keyword]["type"].lower()
        else:
            info["type"] = "unknown"

        if "props" in api_res["results"][0][keyword] and not (api_res["results"][0][keyword]["props"] is None):
            if "make_model" in api_res["results"][0][keyword]["props"]:
                info["make"] = api_res["results"][0][keyword]["props"]["make_model"][0]["make"].lower()
                info["model"] = api_res["results"][0][keyword]["props"]["make_model"][0]["model"].lower()

            if "color" in api_res["results"][0][keyword]["props"]:

                color = api_res["results"][0][keyword]["props"]["color"][0]["value"].lower()
                # no silver in lumix UI
                if color == "silver":
                    color = "grey"
                info["colors"] = color
        return info


def get_container_label(label):
    if "Size and Type Codes" in label:
        return "sizeCode"
    if "Serial Number" in label:
        return "serialNumber"
    if "Owner Code and Category Identifier" in label:
        return "ownerCode"
    return None


def field_update(history, update, field):
    # Skip if the field doesnt exist on update
    if not (field in update):
        return False, None, None
    else:
        # Update history if the field doesnt exist on history
        if not (field in history):
            return True, update[field], update["scores"][field]
        else:
            # Dont update if score is not improved
            if update["scores"][field] > history["scores"][field]:
                return True, update[field], update["scores"][field]
            else:
                return False, None, None


def get_total_width(images):
    if len(images) == 0:
        return 0
    return sum([image.shape[1] for image in images])


def get_max_height(images):
    if len(images) == 0:
        return 0
    return max([image.shape[0] for image in images])


def generate_bbox(col_start, row_start, shape):
    bbox = [None] * 4
    bbox[0] = col_start
    bbox[1] = row_start
    bbox[2] = col_start + shape[1]
    bbox[3] = row_start + shape[0]
    return bbox


# create collage of images (of different sizes) in a naive way:
# if there are less than 3 images, just put them in a row
# if there are more than 3 images, arrang them in two rows
# it is assumed input array has no more than 6 images
# TODO: make it more intelligent
def create_collage(images):
    # change to two rows
    row_a = images
    row_b = []
    if len(images) == 3:
        row_a = [images[0], images[1]]
        row_b = [images[2]]
    elif len(images) == 4:
        row_a = [images[0], images[1]]
        row_b = [images[2], images[3]]
    elif len(images) == 5:
        row_a = [images[0], images[1], images[2]]
        row_b = [images[3], images[4]]
    elif len(images) == 6:
        row_a = [images[0], images[1], images[2]]
        row_b = [images[3], images[4], images[5]]

    total_width = max(get_total_width(row) for row in [row_a, row_b])
    row_a_height = get_max_height(row_a)
    row_b_height = get_max_height(row_b)
    total_height = row_a_height + row_b_height

    result = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    bboxes = []
    col = 0
    for i, image in enumerate(row_a):
        result[0 : image.shape[0], col : col + image.shape[1]] = image
        bboxes.append(generate_bbox(col, 0, image.shape))
        col += image.shape[1]  # move to the next column

    row = get_max_height(row_a)
    col = 0
    for i, image in enumerate(row_b):
        result[row : row + image.shape[0], col : col + image.shape[1]] = image
        bboxes.append(generate_bbox(col, row_a_height, image.shape))
        col += image.shape[1]  # move to the next column

    return result, bboxes
