from typing import List
import numpy as np

from general.analyzer_general import logger
from general.clip_encoder import ClipDispatcher
from general.core import AnalyticImage
from .alerts_utils import UNKNOWN_STATE, NEGATIVE_STATE, get_zones_bbox
from .classification import ClassificationBaseAlert



class ClipAlert(ClassificationBaseAlert):
    clip_threshold = 0.5
    zone_enabled = True
    is_crop_mode = False
    num_of_crops_required = 4  # bulk mode for clip
    encodings_matrix: np.array
    separation_gap = 0
    allow_crop_mode = True
    force_clip = True

    def __init__(self, alert_dict: dict, context):
        super(ClipAlert, self).__init__(alert_dict, context)
        zones = self.selectedCamera.get("zones", {})
        self.bbox = get_zones_bbox(zones) if self.zone_enabled else None
        self.set_flow_values()
        self.encodings_matrix = np.array([s.encodings for s in self.object_states])
        self.clip_state_ids = [s.id for s in self.object_states]
        if self.encodings_matrix.size == 0:
            logger.error("No encodings provided for the clip alert")
            raise ValueError("No encodings provided for the clip alert")

        clip_dispatcher = self.context.get_clip_dispatcher()
        clip_enabled = False
        if clip_dispatcher.enabled:
            if self.is_crop_mode:
                clip_enabled = clip_dispatcher.crops_enabled
                if not clip_enabled and self.force_clip:
                    clip_dispatcher.set_crops_enabled(True)
                    clip_enabled = True
            else:
                clip_enabled = clip_dispatcher.thumbs_enabled
                if not clip_enabled and self.force_clip:
                    clip_dispatcher.set_thumbs_enabled(True)
                    clip_enabled = True

        if not clip_enabled:
            logger.error("Clip dispatcher not enabled, cant activate the alert")
            raise ValueError("Clip dispatcher not enabled, please enable it in the config")
        self.clip_dispatcher = clip_dispatcher
        self.extra_fields = {"extra_fields": {"query": self.special_filter}}

        # for trigger alerts on full frame
        if not self.allow_crop_mode:
            self.bbox = None



    def set_flow_values(self):
        super().set_flow_values()
        self.clip_threshold = self.formValue.get("clip_threshold", 0.2)
        self.separation_gap = self.formValue.get("separation_gap", 0.05)

    def parse_object_data_multi_state(self, obj_data: np.array) -> int:
        scores = self.encodings_matrix @ obj_data
        sidxs = np.argsort(scores)
        if scores[sidxs[-1]] > self.clip_threshold and scores[sidxs[-1]] > scores[sidxs[-2]] + self.separation_gap:
            return self.clip_state_ids[sidxs[-1]]
        return UNKNOWN_STATE

    def parse_object_data_single_state(self, obj_data: np.array) -> int:
        scores = self.encodings_matrix @ obj_data
        if scores.max() > self.clip_threshold:
            return self.clip_state_ids[0]
        return NEGATIVE_STATE


    def extract_object_data_from_images(self, images: List[AnalyticImage]):
        timestamps = [img.timestamp for img in images]

        # first get the encodings from thumbnails or crops
        if self.is_crop_mode:
            self.get_crops(images, timestamps)
            if len(self.crops) >= self.num_of_crops_required:
                encodings = self.clip_dispatcher.encode_object_crops(self.crops)
                for i, enc in enumerate(encodings):
                    self.object_data_buffer.append(enc)
                    self.object_data_ts_buffer.append(self.crops_ts[i])
        else:
            new_encs = [d for d in self.clip_dispatcher.last_descriptors if d[1] > self.last_checked_image]
            if len(new_encs) > 0:
                for enc, ts in sorted(new_encs, key=lambda x: x[1]):
                    self.object_data_buffer.append(enc)
                    self.object_data_ts_buffer.append(ts)
                self.last_checked_image = self.object_data_ts_buffer[-1]


    # def gen_validation_crop_from_crops(self, matches: np.array):
    #     if self.validation_crop_num > 1:
    #         crop_shape = np.array(self.crops[-1].shape[:2])
    #         collage_shape = np.array(crop_shape) * np.array(self.validation_grid)
    #         collage = np.zeros(collage_shape.tolist() + [3], dtype=np.uint8)
    #         for i, pidx in enumerate(self.validator_poses):
    #             if pidx >= len(self.validator_poses):
    #                 break
    #             x, y = divmod(i, self.validation_grid[1])
    #             collage[
    #                 x * crop_shape[0] : (x + 1) * crop_shape[0], y * crop_shape[1] : (y + 1) * crop_shape[1]
    #             ] = self.crops[pidx]
    #         validation_im = collage
    #     else:
    #         mx_match = np.argmax(matches)
    #         validation_im = self.crops[mx_match]
    #     return downscale_to_max_dim(validation_im, VALIDATION_FULL_IMAGE_SIZE)
    #
    #
    # def generate_validation_image(self, alert_valid_idx, images):
    #     timestamps = [img.timestamp for img in images]
    #     if self.is_crop_mode:
    #         vds = self.gen_validation_crop_from_crops(alert_valid_idx)
    #     else:
    #         ts_to_compare = np.array(timestamps)
    #         compare_to = np.array(self.object_data_ts_buffer)[np.array(alert_valid_idx)]
    #         min_distances = np.min(np.abs(ts_to_compare[:, None] - compare_to[None, :]), axis=1)
    #         image_idx = np.argmin(min_distances)
    #         vds = downscale_to_max_dim(images[image_idx].frame, VALIDATION_FULL_IMAGE_SIZE)
    #     return vds