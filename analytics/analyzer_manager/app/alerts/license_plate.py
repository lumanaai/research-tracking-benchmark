from dataclasses import dataclass
from typing import List, Dict, Set
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious  # noqa

from general.img_utils import crop_image
from alerts.base_alerts import BaseAlert, AlertCandidate, ObjectAlert
from general.core import BatchDataResolver, AlertRouting, AnalyticImage, MotionData, BDR
from general.analyzer_general import logger, GLOBAL_TYPE_KEY, GLOBAL_FAILURE
from level1.license_plate.lp_attributes import LPAttributesModel
from level1.advanced import batch_data_convert_parts_to_object


@dataclass
class LPRecord:
    tracker_id: int = -1
    timestamp: int = -1
    bbox: np.array = None
    class_name: str = None

    def export_dict(self):
        return {
            "tracker_id": self.tracker_id,
            "timestamp": self.timestamp,
            "bbox": self.bbox.tolist(),
        }


class LicensePlateAlert(ObjectAlert):
    type_name = "plate_nationality"
    form_value: str = "licensePlateNationality"
    default_routing = AlertRouting.NO_ROUTING
    min_lp_vehicle_overlap: float = 0.5  # minimum overlap between lp and vehicle to consider them matched
    iou_reclassify_th: float = 0.2  # threshold for re-classifying the same tracker id
    crop_margin: float = [0.0, 0.0]  # margin for cropping license plates
    enable_hist_clean: bool = False  # enable history cleaning
    lp_types: List[str]

    def __init__(self, alert_dict: dict, context):
        super(LicensePlateAlert, self).__init__(alert_dict, context)
        self.set_flow_values()
        self.lp_subclass = self.context.get_class_handler().plate_subclass_value
        self.vehicle_class = self.context.get_class_handler().vehicle_value
        self.classifier = LPAttributesModel({})
        # history to avoid sending alerts for the same car
        self.alert_history: Set[int] = set()
        self.non_interest_history: Dict[int, LPRecord] = {}
        self.l1_required_type = {self.context.get_class_handler().vehicle_value: None}


    def set_flow_values(self):
        super().set_flow_values()
        self.lp_types = self.formValue.get(self.form_value)

    def on_entities_removed(self, ent_ids: List[int]):
        for ent_id in ent_ids:
            self.alert_history.discard(ent_id)
            self.non_interest_history.pop(ent_id, None)

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BDR
    ) -> List[AlertCandidate]:
        alert_candidates = []
        # [x,y,x,y, tracker_id, cls, subclass, conf, det_id, age, conflict, frame_id, timestamp, [location, center, center location,  area, index]]
        # check if there are license plates detections
        lp_detections = batch_data.query(BatchDataResolver.SUBCLASS, self.lp_subclass)
        vehicle_detections = batch_data.query(BatchDataResolver.CLASS, self.vehicle_class)
        if len(lp_detections) and len(vehicle_detections):
            # match license plates to vehicles (replacing bbox) and keeping tracker id
            paired_batch_data = batch_data_convert_parts_to_object(
                batch_data, self.lp_subclass, self.vehicle_class, self.min_lp_vehicle_overlap
            )
            vars_data = self.apply_roi_filter(paired_batch_data.data[paired_batch_data.data[:, BDR.ID] >= 0, :])
            crops = []
            detections = []
            for det in vars_data:
                track_id = int(det[BDR.ID])
                if track_id in self.non_interest_history:
                    # validate static plates
                    im_h, im_w = images[-1].frame.shape[:2]
                    iou = bbox_ious(
                        det[BDR.POS][None, :] * np.array([im_w, im_h, im_w, im_h]),
                        np.array(self.non_interest_history[track_id].bbox)[None, :]
                        * np.array([im_w, im_h, im_w, im_h]),
                    )[0][0]
                    if iou > self.iou_reclassify_th:
                        continue  # plate is static
                    # plate changed its position
                    self.non_interest_history.pop(track_id)

                if track_id not in self.alert_history and track_id not in self.non_interest_history:
                    frame_id = int(det[BDR.FRAME_ID])
                    crop = crop_image(images[frame_id].frame, det[BDR.POS], self.crop_margin, bgr_map=False)
                    if crop is not None:
                        crops.append(crop)
                        det_inst = LPRecord(
                            tracker_id=track_id,
                            timestamp=det[BDR.TIMESTAMP],
                            bbox=det[BDR.POS],
                        )
                        detections.append(det_inst)

            res = self.classifier.forward_on_crop_list(crops)

            for i, det_inst in enumerate(detections):
                if res[i][GLOBAL_TYPE_KEY] == GLOBAL_FAILURE:
                    self.non_interest_history[det_inst.tracker_id] = det_inst
                    logger.info(f"Classifier didn't recognize lp detection at {det_inst.timestamp}")
                    continue
                inter = self.classifier.interpreter.interpret_scores(res[i]["scores"], 0)
                type_value, conf = inter[self.form_value][0].value, inter[self.form_value][0].confidence
                # update record
                det_inst.class_name = type_value
                # check if lp type is of interest
                if type_value in self.lp_types and det_inst.tracker_id not in self.alert_history:
                    self.alert_history.add(det_inst.tracker_id)
                    candidate = self.build_alert_candidate(
                        det_inst.timestamp, det_inst.tracker_id, extra=det_inst.export_dict(), crops=[crops[i]]
                    )
                    alert_candidates.append(candidate)
                    logger.info(
                        f"New lp of interest was detected at {det_inst.timestamp} with tracker id {det_inst.tracker_id} and {conf.name.lower()} confidence"
                    )
                else:
                    # save non-interesting lp detection to avoid repeated classifications
                    self.non_interest_history[det_inst.tracker_id] = det_inst
                    logger.info(f"Classifier didn't recognize lp of interest at {det_inst.timestamp}")

        return alert_candidates
