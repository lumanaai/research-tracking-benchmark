## THIS ALERT HAS BEEN DEPRECATED. FightingAlert in alerts/violence.py should be used instead.

# import json
# from copy import copy
# from typing import List

# import cv2
# import numpy as np

# from alerts.alerts_utils import ObjectState, ObjectInfo
# from alerts.base_alerts import AlertCandidate
# from alerts.clip_trigger import ClipAlert
# from general import proj
# from general.core import AlertRouting, BDR, AlertInfo
# from level1.violence.violence_detector import ViolenceDetector

# VALIDATION_SINGLE_CROP_SIZE = 512
# VALIDATION_FULL_IMAGE_SIZE = 1024


# class FightingAlert(ClipAlert):
#     default_routing = AlertRouting.ROUTE_VCC_DEFAULT_TRUE  # ROUTE_VCC_DEFAULT_TRUE
#     vd_threshold = -1
#     vd_marginal_threshold = -2
#     clip_threshold = 0.09421
#     allow_crop_mode = False
#     violence_detection_ts = -10000000
#     violence_timout = 60000  # 1 minute
#     force_clip = True
#     conf_to_latency_conversion = {0: 1, 1: 1, 2: 1}

#     def __init__(self, alert_dict: dict, context):
#         super(FightingAlert, self).__init__(alert_dict, context)
#         self.in_triggered_state = False
#         self.violent_ent_ids = set()
#         self.last_violent_ents = set()
#         analytic_config = self.context.get_config()
#         vd_config = analytic_config.get("l1_models", {}).get("attributes", {}).get("violence", {})
#         self.violence_detector = ViolenceDetector(vd_config)
#         self.sequence_len = self.violence_detector.sequence_length
#         buffer_size = self.violence_detector.inference_buffer_size
#         self.channels = buffer_size[1]
#         self.mem = np.zeros(buffer_size, dtype=self.violence_detector.data_type)
#         self.frame_idx = 0
#         det_res = self.context.context.detector_resolution  # default [384, 640]
#         if det_res[0] > det_res[1]:
#             crop_sz = [VALIDATION_SINGLE_CROP_SIZE, det_res[1] * VALIDATION_SINGLE_CROP_SIZE // det_res[0]]
#         else:
#             crop_sz = [det_res[0] * VALIDATION_SINGLE_CROP_SIZE // det_res[1], VALIDATION_SINGLE_CROP_SIZE]

#         mosaic_sz = [VALIDATION_FULL_IMAGE_SIZE // d for d in crop_sz]
#         self.validation_crop_size = crop_sz
#         validation_num_samples = mosaic_sz[0] * mosaic_sz[1]
#         self.validation_mem = np.zeros((mosaic_sz[0] * crop_sz[0], mosaic_sz[1] * crop_sz[1], 3), dtype=np.uint8)
#         self.validation_mosaic = mosaic_sz

#         start_frame = self.sequence_len // validation_num_samples
#         end_frame = self.sequence_len * (validation_num_samples - 1) // validation_num_samples
#         self.validation_idxs = np.linspace(start_frame, end_frame, num=validation_num_samples, dtype=int).tolist()

#     def set_flow_values(self):
#         self.duration = 0
#         states = []
#         with open(proj.info_path("clip", "violence_prompts.json"), "r") as f:
#             encoding_dict = json.load(f)
#             prompt_keys = list(encoding_dict.keys())
#             encoding = np.squeeze(np.array([encoding_dict[p] for p in prompt_keys]))
#             states.append(ObjectState(1, "violence", encoding))
#         self.object_info = ObjectInfo("violenceAlert","violence", None, states)

#     def _person_count_per_idx(self, batch_data: BDR):
#         person_count_per_frame = batch_data.query(BDR.CLASS, self.person_value, BDR.FRAME_ID).astype(int)
#         return np.bincount(person_count_per_frame, minlength=batch_data.batch_size)

#     def is_active_batch(self, images, motion_data, batch_data) -> List[AlertCandidate]:
#         last_ts = images[-1].timestamp
#         alert_candidates = []

#         # dont check anything if we are in cooldown, no person in the scene or if the violent entities are still in the
#         # scene to avoid sending multiple alerts

#         early_exit = False
#         if last_ts - self.violence_detection_ts < self.violence_timout:
#             early_exit = True
#         elif np.all(self._person_count_per_idx(batch_data) < 2):  # only one person in the scene, no violence
#             early_exit = True
#         else:
#             batch_ids = set(batch_data.unique(BDR.ID, BDR.CLASS, self.person_value).astype(int).tolist())
#             if len(self.last_violent_ents.intersection(batch_ids)) > 0:
#                 early_exit = True

#         if early_exit:
#             self.reset_trigger(last_ts)
#             return alert_candidates

#         # first check clip trigger
#         if not self.in_triggered_state:
#             clip_candidate = super(FightingAlert, self).is_active_batch(images, motion_data, batch_data)
#             if clip_candidate:
#                 self.restart_sequence_collection()

#         if self.in_triggered_state:
#             self.violent_ent_ids.update(
#                 set(batch_data.unique(BDR.ID, BDR.CLASS, self.person_value).astype(int).tolist()) - {-1}
#             )
#             for image in images:
#                 t, c = divmod(self.frame_idx, self.channels)
#                 self.violence_detector.transform_single_image(image.processed, self.mem[t, c])
#                 self._add_to_mosaic(image.processed, self.frame_idx)
#                 self.frame_idx += 1
#                 if self.frame_idx == self.sequence_len:
#                     violence = self.violence_detector.forward_on_sequence([self.mem])
#                     if violence > self.vd_threshold:  # hit - raise alert
#                         val_image = self.validation_mem.copy()
#                         cand = self.build_alert_candidate(image.timestamp, crops=[image.processed], validation_images=[val_image])
#                         if self.activate_ddata and self.internal_alert:
#                             cand = self.add_debug_data(cand, violence)
#                         alert_candidates.append(cand)
#                         self.violence_detection_ts = image.timestamp
#                         self.reset_trigger(image.timestamp)
#                         self.last_violent_ents = copy(self.violent_ent_ids)
#                         break
#                     elif violence > self.vd_marginal_threshold:  # marginal hit - retry
#                         self.restart_sequence_collection()
#                     else:  # no hit - reset
#                         self.reset_trigger(image.timestamp)
#                         break
#         return alert_candidates
    
#     def add_debug_data(self, candidate: AlertCandidate, violence: float) -> AlertCandidate:
#         candidate.extra = candidate.extra or {}
#         # add debug data for internal alerts
#         debug_data = {
#             "alert_id": candidate.alert_id,
#             "timestamp": candidate.timestamp,
#             "input_images": self.mem,
#             "crops": candidate.crops,
#             "validation_images": candidate.validation_images,
#             "violence_score": violence
#         }
#         candidate.extra["debug_data"] = self.serialize_debug_data(debug_data)
#         return candidate

#     def restart_sequence_collection(self):
#         self.in_triggered_state = True
#         self.frame_idx = 0
#         self.violent_ent_ids.clear()
#         self.last_violent_ents.clear()


#     def reset_trigger(self, ts: int):
#         self.in_triggered_state = False
#         self.frame_idx = 0
#         self._clear_buffers(ts)
#         self.clear_state()

#     def _add_to_mosaic(self, img, frame_idx):
#         if frame_idx in self.validation_idxs:
#             t, c = divmod(self.validation_idxs.index(frame_idx), self.validation_mosaic[1])
#             h, w = self.validation_crop_size
#             y0 = t * h
#             y1 = (t + 1) * h
#             x0 = c * w
#             x1 = (c + 1) * w
#             self.validation_mem[y0:y1, x0:x1] = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

#     def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
#         alert_info.alertMessage = "Violence detected"