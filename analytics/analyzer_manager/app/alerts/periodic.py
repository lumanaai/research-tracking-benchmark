from typing import List, Dict

import numpy as np
import cv2

from general.core import AlertRouting, AlertInfo, AnalyticImage, MotionData, BatchDataResolver
from general.img_utils import crop_image, downscale_to_max_dim
from .base_alerts import BaseAlert, AlertCandidate, duration_unit_to_sec


FULL_IMAGE_SIZE = 1280
OF_MAX_DIM = 320
MAX_SENSITIVITY = 20


class SnapshotAlert(BaseAlert):
    default_routing = AlertRouting.NO_ROUTING
    period_ms = -1
    disable_period = -1

    def __init__(self, alert_dict: dict, context):
        super(SnapshotAlert, self).__init__(alert_dict, context)
        zones = self.selectedCamera.get("zones", {})
        bbox = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
        bbox_ok = False
        for k, v in zones.items():
            selection: List[Dict] = v.get("selection", {})
            for node in selection:
                xy = np.array([node.get("x", 0), node.get("y", 0)])
                bbox[0] = np.minimum(bbox[0], xy)
                bbox[1] = np.maximum(bbox[1], xy)
                bbox_ok = True
        if bbox_ok:
            self.bbox = bbox.reshape(-1)
        else:
            self.bbox = None
        self.set_flow_values()
        self.last_alert_ts = 0

    def set_flow_values(self):
        if self.formValue:
            period = self.formValue.get("period", -1)
            if period < 0:
                raise ValueError("Period not set")
            units = self.formValue.get("periodUnit", 0)
            self.period_ms = period * 1000 * duration_unit_to_sec[units]

    def is_active_batch(
        self, images: List[AnalyticImage], motion_data: MotionData, batch_data: BatchDataResolver
    ) -> List[AlertCandidate]:

        alert_candidates = []
        timestamps = [image.timestamp for image in images]
        for ts in timestamps:
            if ts - self.last_alert_ts > self.period_ms:
                candidate = self.build_alert_candidate(ts)
                if self.bbox is not None:
                    crop = crop_image(images[-1].frame, self.bbox)
                    candidate.crops = [crop]
                alert_candidates.append(candidate)
                self.last_alert_ts = ts
        return alert_candidates


class PeriodicAlert(SnapshotAlert):
    default_routing = AlertRouting.ROUTE_VCC_DEFAULT_FALSE
    of_threshold = 0

    def __init__(self, alert_dict: dict, context):
        super(PeriodicAlert, self).__init__(alert_dict, context)
        self.last_crop = np.array([])
        self.disable_ts = np.inf
        self.motion_filter = True
        analytic_config = self.context.get_config()
        if "alertConfig" in analytic_config:
            if "periodic" in analytic_config["alertConfig"]:
                cfg = analytic_config["alertConfig"]["periodic"]
                self.motion_filter = cfg.get("motion_filter", True)
        self.extra_fields = {"extra_fields": {"query": self.special_filter}}

    def set_flow_values(self):
        super().set_flow_values()
        if self.formValue:
            self.special_filter = self.formValue.get("query", -1)
            disable = self.formValue.get("disablePeriod", 0)
            disable_units = self.formValue.get("disablePeriodUnit", 0)
            self.disable_period = disable * 1000 * duration_unit_to_sec[disable_units]

            max_sensitivity = MAX_SENSITIVITY * (OF_MAX_DIM / FULL_IMAGE_SIZE if self.bbox is None else 1)
            self.sensitivity = self.apply_from_dict("sensitivity", self.formValue, 0.5)
            self.of_threshold = self.sensitivity * max_sensitivity

    def is_active_batch(self, images, motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []
        ts = images[-1].timestamp
        if self.disable_ts < ts:
            self.enabled = False
            return alert_candidates

        if ts - self.last_alert_ts > self.period_ms:
            if self.bbox is None:
                crop = downscale_to_max_dim(images[-1].frame, FULL_IMAGE_SIZE)
            else:
                crop = crop_image(images[-1].frame, self.bbox)
            gray = cv2.cvtColor(downscale_to_max_dim(crop, OF_MAX_DIM), cv2.COLOR_RGB2GRAY)

            self.last_alert_ts = ts
            has_motion = True
            if self.motion_filter:
                if self.sensitivity < 0.99:  # using motion to skip events
                    if gray.shape == self.last_crop.shape:
                        flow = cv2.calcOpticalFlowFarneback(gray, self.last_crop, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        # Convert flow to polar coordinates (magnitude and angle)
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        p = np.percentile(magnitude, 97)
                        if p < self.of_threshold:
                            has_motion = False
            if has_motion:
                alert_candidates.append(
                    self.build_alert_candidate(ts, validation_images=[crop], extra=self.extra_fields)
                )
                self.last_crop = gray
            if self.disable_period > 0 and not np.isfinite(self.disable_ts):
                self.disable_ts = ts + self.disable_period
        return alert_candidates

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = super(PeriodicAlert, self).build_alert_info(candidate)
        alert_info.specialFilter = self.special_filter
        return alert_info
