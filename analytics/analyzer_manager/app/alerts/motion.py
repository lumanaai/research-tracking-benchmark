from typing import Dict, List

import numpy as np
from scipy.signal import convolve2d

from general.core import AnalyticImage, MotionData
from .alerts import BaseAlert
from .base_alerts import AlertCandidate


class MotionAlert(BaseAlert):
    type_name = "motionAlerts"
    default_sensitivity = 0.75
    object_based_alert = False

    def __init__(self, init_dict: Dict, context):
        super(MotionAlert, self).__init__(init_dict, context)
        self.set_flow_values()
        if self.sensitivity is None:
            self.sensitivity = MotionAlert.default_sensitivity
        self.threshold = (1 - self.sensitivity) * 100
        self.alertZoomThumbnail = False  # bypass zoom alerts for now
        self.blockout = max(self.blockout, 1000)
        self.kernel = np.ones((3, 3), np.uint8)
        self.extra_fields = {"extra_fields": {"sensitivity": self.sensitivity}}
        self.alert_message = f"Motion sensitivity is above {int(self.sensitivity * 100)}%"
        self.roi = self.roi.T if self.roi is not None else None

    def set_flow_values(self):
        if self.formValue:
            sensitivity = self.apply_from_dict("sensitivity", self.formValue, MotionAlert.default_sensitivity * 100)
            self.sensitivity = sensitivity / 100
            self.positive_alert = True

    def is_active_batch(self, images: List[AnalyticImage], motion_data: MotionData, batch_data) -> List[AlertCandidate]:
        active_alerts = []

        max_val = motion_data.max > self.threshold if self.positive_alert else motion_data.max < self.threshold
        if max_val:
            if self.positive_alert:
                data = motion_data.unified > self.threshold
            else:
                data = motion_data.unified < self.threshold

            if self.roi is not None:
                data = data & self.roi
            conv_result = convolve2d(data, self.kernel, mode="same")
            if np.any(conv_result > 2):
                active_alerts.append(self.build_alert_candidate(images[-1].timestamp, extra=self.extra_fields))
        return active_alerts


