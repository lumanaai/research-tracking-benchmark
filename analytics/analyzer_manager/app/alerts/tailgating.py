from typing import Dict

from general.core import BatchDataResolver
from general.img_utils import crop_image
from .base_alerts import duration_unit_to_sec, duration_unit_to_str
from .linecrossing import LineCrossingAlert


class TailgatingAlert(LineCrossingAlert):
    type_name = "tailgating"
    single_alert_object_filter = False
    ambient_motion_filter = False

    def __init__(self, alert_dict: Dict, context):
        super(TailgatingAlert, self).__init__(alert_dict, context)
        self.line_cross_hysteresis_period = 500
        self.lastTailgateTs = 0
        self.lineCrossed = {-1: True}
        self.last_line_crossed = {}
        keys_to_init = []
        line = self.lines[0]
        if line.d == 0:
            keys_to_init.extend([-1, 1])
        else:
            keys_to_init.append(line.d)
        for key in keys_to_init:
            self.last_line_crossed[key] = {"ent_id": 0, "timestamp": -1, "crop": None}
        units = 0
        count  = self.count
        if self.formValue:
            count = self.apply_from_dict("duration", self.formValue, 1)
            units = self.apply_from_dict("durationUnit", self.formValue, 0)
            self.count = count* duration_unit_to_sec[units]
        self.alertZoomThumbnail = True

        self.alert_message = "Tailgating of {}" + f" is below {count} {duration_unit_to_str[units]}"

    def on_line_crossed(self, ent_id, id_data, crossing, images, line_idx = 0):
        candidate = None
        last_crossed_ts = self.last_line_crossed[crossing]["timestamp"]
        last_crossed_ent_id = self.last_line_crossed[crossing]["ent_id"]
        tailgating_gap = round((id_data[BatchDataResolver.TIMESTAMP] - last_crossed_ts) / 1000, 2)
        crossed_image = images[int(id_data[BatchDataResolver.FRAME_ID])].frame
        new_crop = crop_image(crossed_image, id_data[BatchDataResolver.POS])
        if ent_id != last_crossed_ent_id and tailgating_gap <= self.count:
            # no need to crop in the base class, as we crop here to have all the entities
            candidate = super().on_line_crossed(ent_id, id_data, crossing, images=None)
        if candidate:
            candidate.crops += [self.last_line_crossed[crossing]["crop"], new_crop]
        # either way, update the internal state
        self.alerted_ids[ent_id] = {"timestamp": id_data[BatchDataResolver.TIMESTAMP]}
        self.last_line_crossed[crossing]["timestamp"] = id_data[BatchDataResolver.TIMESTAMP]
        self.last_line_crossed[crossing]["ent_id"] = ent_id
        self.last_line_crossed[crossing]["crop"] = new_crop
        return candidate
