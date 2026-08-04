# ## obsolete  - already replaced with new infrastructure, but kept here for reference
#
#
# from typing import List, Dict
# import bisect
# import math
# import cv2
#
# import numpy as np
#
# from general.core import AlertRouting, hex_to_desc, divide_n_into_s_parts_inclusive, BatchDataResolver
# from general.img_utils import crop_image, downscale_to_max_dim
# from .base_alerts import BaseAlert, AlertCandidate, duration_unit_to_sec
# import json
# from general import proj
# from general.img_utils import crop_image_by_bbox_and_ar
#
# VALIDATION_FULL_IMAGE_SIZE = 1024
#
#
# class HistoryMaxScores:
#     def __init__(self, max_len: int):
#         self.max_len = max_len
#         self.scores: Dict[int, float] = {}
#         self.timestamps = []
#
#     def add_score(self, ts, score):
#         if ts in self.scores:
#             self.scores[ts] = max(self.scores[ts], score)
#         else:
#             bisect.insort(self.timestamps, ts)
#             self.scores[ts] = score
#             if len(self.timestamps) > self.max_len:
#                 self.scores.pop(self.timestamps.pop(0), None)
#
#     def get_scores_from_ts(self, from_ts: int):
#         return [self.scores[ts] for ts in self.timestamps if ts > from_ts]
#
#     def get_last_scores(self, n: int = 1):
#         return [self.scores[ts] for ts in self.timestamps[-n:]]
#
#     def get_last_time_over_threshold(self, threshold: float):
#         for ts in self.timestamps[::-1]:
#             if self.scores[ts] > threshold:
#                 return ts
#         return -1
#
#     @property
#     def last_ts(self):
#         return self.timestamps[-1] if self.timestamps else 0
#
#     def __len__(self):
#         return len(self.timestamps)
#
#     def clear(self):
#         self.scores.clear()
#         self.timestamps.clear()
#
#
# class ClipAlert(BaseAlert):
#     default_routing = AlertRouting.NO_ROUTING  # ROUTE_VCC_DEFAULT_TRUE
#     clip_threshold = 0.5
#     zone_enabled = True
#     latency = 3
#     encoding: np.array
#     period_ms = 1000
#     match_percentile = 75
#     validation_crop_num = -1
#     is_crop_mode = False
#     thumbnail_th_factor = 0.95
#     clip_ar = 1.0
#     thumb_latency_frames = -1
#     num_of_crops_required = -1
#
#     def __init__(self, alert_dict: dict, context):
#         super(ClipAlert, self).__init__(alert_dict, context)
#         zones = self.selectedCamera.get("zones", {})
#         bbox = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
#         bbox_ok = False
#         self.bbox = None
#
#         if self.zone_enabled:
#             for k, v in zones.items():
#                 selection: List[Dict] = v.get("selection", {})
#                 for node in selection:
#                     xy = np.array([node.get("x", 0), node.get("y", 0)])
#                     bbox[0] = np.minimum(bbox[0], xy)
#                     bbox[1] = np.maximum(bbox[1], xy)
#                     bbox_ok = True
#             if bbox_ok:
#                 self.bbox = bbox.reshape(-1)
#         self.set_flow_values()
#         self.is_active = False
#
#         self.clip_dispatcher = self.context.get_clip_dispatcher()
#         if not self.clip_dispatcher.enabled:
#             raise ValueError("Clip dispatcher not enabled")
#         self.clip_ar = self.clip_dispatcher.aspect_ratio
#         self.extra_fields = {"extra_fields": {"query": self.special_filter}}
#         self.thumbnail_clip_threshold = self.clip_threshold * self.thumbnail_th_factor  # hysteresis
#         self.last_checked_thumbnail = 0
#         self.is_crop_mode = self.is_crop_mode or self.bbox is not None
#         self.use_thumbnail_trigger = self.is_crop_mode
#
#         self.crops = []
#         self.crops_ts = []
#         if self.num_of_crops_required < 0:
#             self.num_of_crops_required = self.latency
#         self.last_crop_ts = 0
#         if self.thumb_latency_frames < 0:
#             self.thumb_latency_frames = self.latency * 2 + 1
#         self.thumbnail_latency = self.thumb_latency_frames * self.period_ms
#         self.scores = HistoryMaxScores(self.thumb_latency_frames)
#         self.last_triggered_ts = -100000
#         if self.validation_crop_num < 0:
#             self.validation_crop_num = self.latency
#         if self.validation_crop_num > 1:
#             self.validator_poses = divide_n_into_s_parts_inclusive(self.latency, self.validation_crop_num)
#             srt = math.ceil(math.sqrt(self.validation_crop_num))
#
#             self.validation_grid = (math.ceil(self.validation_crop_num / srt), srt)
#         self.crops_scores = HistoryMaxScores(self.latency)
#
#     def set_flow_values(self):
#         if self.formValue:
#             period = self.formValue.get("period", -1)
#             if period > 0:
#                 units = self.formValue.get("periodUnit", 0)
#                 self.period_ms = period * 1000 * duration_unit_to_sec[units]
#             states = []
#             self.special_filter = self.formValue.get("query")
#             self.clip_threshold = self.formValue.get("clip_threshold", 0.2)
#             self.encoding = hex_to_desc(self.formValue.get("encodedQuery"))
#
#     def _process_thumbnails_matches(self) -> bool:
#         new_encs = [d for d in self.clip_dispatcher.last_descriptors if d[1] > self.last_checked_thumbnail]
#         if len(new_encs) > 0:
#             thumbnail_enc, thumbnail_timestamps = zip(*new_encs)
#             matches = np.array(thumbnail_enc) @ self.encoding.T
#             for ts, m in zip(thumbnail_timestamps, matches):
#                 self.scores.add_score(ts, np.max(m))
#             self.last_checked_thumbnail = np.max(thumbnail_timestamps)
#             return True
#         return False
#
#     def get_crops(self, images, batch_data, timestamps):
#         for tidx, ts in enumerate(timestamps):
#             if ts > self.last_crop_ts + self.period_ms:
#                 b_data = batch_data.query(BatchDataResolver.INDEX, tidx)
#                 crop = crop_image(images[tidx].frame, b_data)
#                 self.crops.append(crop)
#                 self.crops_ts.append(ts)
#                 self.last_crop_ts = ts
#
#     def gen_validation_crop_from_crops(self, matches: np.array):
#         if self.validation_crop_num > 1:
#             crop_shape = np.array(self.crops[-1].shape[:2])
#             collage_shape = np.array(crop_shape) * np.array(self.validation_grid)
#             collage = np.zeros(collage_shape.tolist() + [3], dtype=np.uint8)
#             for i, pidx in enumerate(self.validator_poses):
#                 if pidx >= len(self.validator_poses):
#                     break
#                 x, y = divmod(i, self.validation_grid[1])
#                 collage[
#                     x * crop_shape[0] : (x + 1) * crop_shape[0], y * crop_shape[1] : (y + 1) * crop_shape[1]
#                 ] = self.crops[pidx]
#             validation_im = collage
#         else:
#             mx_match = np.argmax(matches)
#             validation_im = self.crops[mx_match]
#         return downscale_to_max_dim(validation_im, VALIDATION_FULL_IMAGE_SIZE)
#
#     def is_active_batch(self, images, motion_data, batch_data) -> List[AlertCandidate]:
#         alert_candidates = []
#         timestamps = [img.timestamp for img in images]
#
#         is_update = self._process_thumbnails_matches()
#         # check dispatcher thumbnails
#         if is_update:
#             if self.is_active:
#                 _score = np.percentile(self.scores.get_last_scores(self.latency), 100 - self.match_percentile)
#                 if _score < self.thumbnail_clip_threshold:
#                     self.is_active = False
#             elif self.use_thumbnail_trigger:
#                 last_ts = self.scores.get_last_time_over_threshold(self.thumbnail_clip_threshold)
#                 if last_ts > 0:
#                     self.last_triggered_ts = last_ts
#
#             print(f"Thumbnails time: {timestamps[0]} last scores: {self.scores.get_last_scores(self.latency)}")
#
#         thumbnail_trigger = (
#             not self.use_thumbnail_trigger or timestamps[0] - self.last_triggered_ts < self.thumbnail_latency
#         )
#
#         if (
#             not self.is_active
#             and (thumbnail_trigger or len(self.crops) > 0)
#             and self.condition_trigger(images, motion_data, batch_data)
#         ):
#             # full image mode - use clip encoding to trigger
#             if not self.is_crop_mode and len(self.scores) >= self.latency:
#                 _score = np.percentile(self.scores.get_last_scores(self.latency), self.match_percentile)
#                 if _score > self.clip_threshold:
#                     last_ts = self.scores.last_ts
#                     if self.scores.last_ts in timestamps:
#                         idx = timestamps.index(last_ts)
#                     else:
#                         idx = np.argmin(np.abs(np.array(timestamps) - last_ts))
#                     validation_crop = downscale_to_max_dim(images[idx].frame, VALIDATION_FULL_IMAGE_SIZE)
#                     alert_candidates.append(
#                         self.build_alert_candidate(
#                             images[idx].timestamp, validation_images=[validation_crop], extra=self.extra_fields
#                         )
#                     )
#                     self.is_active = True
#             # crop mode
#             elif self.is_crop_mode:
#                 self.get_crops(images, batch_data, timestamps)
#
#                 if len(self.crops) >= self.num_of_crops_required:
#                     encodings = self.clip_dispatcher.encode_object_crops(self.crops_for_encoding)
#                     matches = np.max(encodings @ self.encoding.T, axis=1)
#                     for idx, m in enumerate(matches):
#                         self.crops_scores.add_score(self.crops_ts[idx], m)
#                         last_scores = self.crops_scores.get_last_scores(self.latency)
#                         if len(last_scores) < self.latency:
#                             continue
#                         _score = np.percentile(last_scores, self.match_percentile)
#                         print(f"crops time: {self.crops_ts[idx]} scores: {last_scores}")
#                         if _score > self.clip_threshold and not self.is_active:
#                             validation_crop = self.gen_validation_crop_from_crops(matches)
#                             alert_candidates.append(
#                                 self.build_alert_candidate(
#                                     timestamps[-1], validation_images=[validation_crop], extra=self.extra_fields
#                                 )
#                             )
#                             self.is_active = True
#                             self.crops_scores.clear()
#                     self.crops.clear()
#                     self.crops_ts.clear()
#         return alert_candidates
#
#     @property
#     def crops_for_encoding(self):
#         return self.crops
#
#     def condition_trigger(self, images, motion_data, batch_data):
#         return True
#
#
#
# class ClipFightingAlert(ClipAlert):
#     default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE  # ROUTE_VCC_DEFAULT_TRUE
#     clip_threshold = 0.205
#     min_clip_threshold = 0.16
#     is_crop_mode = True
#     zone_enabled = False
#     period_ms = 500
#     match_percentile = 100
#     validation_crop_num = 6
#     thumbnail_th_factor = 0.82
#     latency = 3
#     thumb_latency_frames = 7
#     num_of_crops_required = 6
#     min_overlap = 0
#
#     def __init__(self, alert_dict: dict, context):
#         super(ClipFightingAlert, self).__init__(alert_dict, context)
#         with open(proj.info_path("clip", "violence_prompts.json"), "r") as f:
#             self.encoding_dict = json.load(f)
#         prompt_keys = list(self.encoding_dict.keys())
#         self.encoding = np.array([self.encoding_dict[p] for p in prompt_keys])
#         self.crops_size = [self.routing_crop_size, int(self.routing_crop_size * self.clip_ar)]
#
#     def set_flow_values(self):
#         self.sensitivity = self.formValue.get("sensitivity", 0.5)
#         if self.sensitivity != 0.5:
#             delta = self.clip_threshold - self.min_clip_threshold
#             factor = (1.0 - np.clip(self.sensitivity, 0, 1)) / 0.5
#             self.clip_threshold = self.min_clip_threshold + delta * factor
#
#     def get_crops(self, images, batch_data, timestamps):
#         counts = self._person_count_per_idx(batch_data)
#         for tidx, ts in enumerate(timestamps):
#             if ts > self.last_crop_ts + self.period_ms and counts[tidx] > 1:
#                 b_data = batch_data.query(BatchDataResolver.FRAME_ID, tidx)
#                 crop = self._crop_image_persons_bbox(images[tidx].frame, b_data)
#                 self.crops.append(crop)
#                 self.crops_ts.append(ts)
#                 self.last_crop_ts = ts
#
#     def _crop_image_persons_bbox(self, img, img_data: np.array = None):
#         person_data = img_data[img_data[:, BatchDataResolver.CLASS] == self.person_value]
#         person_crops = person_data[:, BatchDataResolver.POS]
#         mins = np.min(person_crops[:, :2], axis=0)
#         maxs = np.max(person_crops[:, 2:], axis=0)
#         bbox = np.concatenate([mins, maxs]) * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
#         return cv2.resize(crop_image_by_bbox_and_ar(img, bbox, self.clip_ar), self.crops_size)
#
#     def _person_count_per_idx(self, batch_data: BatchDataResolver):
#         arr = batch_data.query(BatchDataResolver.CLASS, self.person_value, [BatchDataResolver.FRAME_ID, BatchDataResolver.INDEX]).astype(int)
#         valid_overlaps = np.max(batch_data.all_overlaps[np.ix_(arr[:, 1], arr[:, 1])], axis=1) >= self.min_overlap
#         arr = arr[valid_overlaps, 0]
#         return np.bincount(arr, minlength=batch_data.batch_size)
#
#     def condition_trigger(self, images, motion_data, batch_data):
#         # require at least to person in a frame
#         return any(self._person_count_per_idx(batch_data) > 1)
#
#     @property
#     def crops_for_encoding(self):
#         return self.crops[::2]  # take every second crop
