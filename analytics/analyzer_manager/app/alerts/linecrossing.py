from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional

import numpy as np

from general import apply_from_dict
from general.analyzer_general import doIntersect, Point, bbxSide, logger
from general.core import BatchDataResolver, AnalyticImage, AlertCategory, AlertInfo, TrackingType, EntityClassification
from .base_alerts import ObjectAlert, AlertCandidate


def position_left(pos: np.ndarray):
    return np.vstack((pos[:, 0], np.average(pos[:, (1, 3)], axis=1))).T


def position_right(pos: np.ndarray):
    return np.vstack((pos[:, 2], np.average(pos[:, (1, 3)], axis=1))).T


def position_top(pos: np.ndarray):
    return np.vstack((np.average(pos[:, (0, 2)], axis=1), pos[:, 1])).T


def position_bottom(pos: np.ndarray):
    return np.vstack((np.average(pos[:, (0, 2)], axis=1), pos[:, 3])).T


def position_center(pos: np.ndarray):
    return np.vstack((np.average(pos[:, (0, 2)], axis=1), np.average(pos[:, (1, 3)], axis=1))).T


def position_to_reference_point_function(position) -> Callable:
    if position == "bottom":
        return position_bottom
    elif position == "top":
        return position_top
    elif position == "right":
        return position_right
    elif position == "left":
        return position_left
    else:  # center
        return position_center


@dataclass
class Line:
    p1: Point
    p2: Point
    d: int
    position: str
    position_converter: Callable


class LineCrossingAlert(ObjectAlert):
    type_name = "lineCrossing"
    line_cross_hysteresis_period = 1000
    optional_line = False
    single_alert_object_filter = True
    ambient_motion_filter = True

    def __init__(self, alert_dict: Dict, context, multiline: bool = False):
        super(LineCrossingAlert, self).__init__(alert_dict, context)
        # cfg step
        analytic_config = self.context.get_config()
        self.position_cfg = "center" if self.context.is_location_center else "bottom"
        if "alertConfig" in analytic_config:
            if "lineCrossing" in analytic_config["alertConfig"]:
                cfg = analytic_config["alertConfig"]["lineCrossing"]
                min_blockout = cfg.get("minBlockout", 0)
                self.blockout = max(min_blockout, self.blockout)
                self.position_cfg = cfg.get("position", self.position_cfg)
                self.line_cross_hysteresis_period = cfg.get("hysteresis", self.line_cross_hysteresis_period)
                self.single_alert_object_filter = cfg.get("singleAlertPerObject", self.single_alert_object_filter)
        self.lines = []
        self.set_flow_values()

        self.id_location = {}
        self.alerted_ids = {}

        self._read_lines(alert_dict)

        self.lineCrossed = {}
        if self.category == AlertCategory.Safety.value:
            self.alert_message = "Trespassing of {}"
        else:
            self.alert_message = "{} cross a line"
        if self.category == AlertCategory.Integrations.value:
            self.internal_alert = 2

    def generate_line(self, line_config):
        p1 = Point(line_config["p1"]["x"], line_config["p1"]["y"])
        p2 = Point(line_config["p2"]["x"], line_config["p2"]["y"])
        d: int = apply_from_dict("d", line_config, 0)
        if d == 2:
            d = -1

        if self.position_cfg is None:
            position = bbxSide(p1, p2, d)
        else:
            position = self.position_cfg

        return Line(p1, p2, d, position, position_to_reference_point_function(position))

    def on_line_crossed(self, ent_id, id_data, crossing, images, line_idx=0):
        candidate = self.build_alert_candidate(
            id_data[BatchDataResolver.TIMESTAMP],
            ent_id,
            id_data,
            {"crossing": crossing, "line": line_idx},
            images=images,
        )
        self.alerted_ids[ent_id] = {"timestamp": candidate.timestamp}
        return candidate

    def get_line(self, ent_id, idx=0):
        return self.lines[idx]

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []
        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            return alert_candidates
        alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).astype(int).tolist()
        for ent_id in alert_ids:

            id_data = vars_data[vars_data[:, BatchDataResolver.ID] == ent_id, :]

            if self.line_cross_hysteresis_period and ent_id in self.alerted_ids:
                blockout_period = self.alerted_ids[ent_id]["timestamp"] + self.line_cross_hysteresis_period
                id_data = id_data[id_data[:, BatchDataResolver.TIMESTAMP] > blockout_period, :]

            id_pos = id_data[:, BatchDataResolver.POS]
            pt_offset = 0
            if ent_id in self.id_location:
                id_pos = np.vstack((self.id_location[ent_id], id_pos))
                pt_offset = 1  # pos array index to data array
            self.id_location[ent_id] = id_pos[-1]

            if len(id_pos) < 2:
                continue
            for idx, line_t in enumerate(self.lines):
                line = self.get_line(ent_id, idx)
                if line is None:
                    continue
                ref_points_np = line.position_converter(id_pos)
                ref_points = [Point(p[0], p[1]) for p in ref_points_np]
                pt = 0
                crossing = False
                while pt < len(ref_points) - 1:
                    crossing = doIntersect(
                        line.p1,
                        line.p2,
                        ref_points[pt],
                        ref_points[pt + 1],
                        line.d,
                    )
                    if crossing:
                        candidate = self.on_line_crossed(ent_id, id_data[pt + 1 - pt_offset], crossing, images)
                        if candidate is not None:
                            alert_candidates.append(candidate)
                            break
                        else:
                            # update next line and repeat
                            curr_pos = line.position
                            if len(self.lines) > 1 and self.alerted_ids[ent_id]["line"] < len(self.lines):
                                line = self.get_line(ent_id)
                                if line.position != curr_pos:
                                    ref_points_np = line.position_converter(id_pos)
                                    ref_points = [Point(p[0], p[1]) for p in ref_points_np]
                                continue
                    pt += 1
                if crossing:
                    break

        return alert_candidates

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)
        ent_id = candidate.ent_ids[0] if len(candidate.ent_ids) > 0 else -1
        if self.single_alert_object_filter and ent_id >= 0 and ent_id in self.alerted_ids:
            self.alerted_ids[ent_id]["is_alerted"] = True
        if candidate.extra:
            alert_info.alertData = {"value": candidate.extra.get("crossing", 0)}
        return alert_info

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
            if ent in self.alerted_ids:
                del self.alerted_ids[ent]
            if ent in self.id_location:
                del self.id_location[ent]

    def generate_alert_message(self, candidate: AlertCandidate, alert_info: AlertInfo):
        object_name = self.context.get_class_handler().object_int_to_str(alert_info.object_id)
        alert_info.alertMessage = self.alert_message.format(object_name).capitalize()

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        ent_id = candidate.ent_ids[0] if len(candidate.ent_ids) > 0 else -1
        if self.single_alert_object_filter and ent_id >= 0 and ent_id in self.alerted_ids:
            # if we already alerted for this object, skip it
            alerted = self.alerted_ids[ent_id].get("is_alerted", False)
            if alerted:
                return False
        return super(LineCrossingAlert, self).check_alert_candidate(candidate, force)

    def _read_lines(self, alert_dict):
        # set the lines data:
        line_config = self.selectedCamera.get("lineCrossing", {})
        if line_config:
            if "lines" in self.selectedCamera["lineCrossing"]:
                self.lines = []
                for line_config in self.selectedCamera["lineCrossing"]["lines"]:
                    self.lines.append(self.generate_line(line_config))
            else:
                self.lines = [self.generate_line(self.selectedCamera["lineCrossing"])]

        elif "lineCrossing" in alert_dict and alert_dict["lineCrossing"] is not None:
            self.lines = [self.generate_line(alert_dict["lineCrossing"])]
        elif self.optional_line:
            self.lines = []
        else:
            logger.error("incompatible configuration - missing line crossing data")
            self.enabled = False


class CountingAlert(LineCrossingAlert):
    line_cross_hysteresis_period = 0
    single_alert_object_filter = False
    ambient_motion_filter = False
    disable_crops = True  # no need for object thumbnail

    def __init__(self, alert_dict: Dict, context):
        super(CountingAlert, self).__init__(alert_dict, context)
        self.blockout: float = 0  # dont blockout
        self.flow = TrackingType.Counting.value
        analytic_config = self.context.get_config()

        # set all the lines to undirected
        self.entrance_direction = []
        for line in self.lines:
            self.entrance_direction.append(line.d)
            line.d = 0
        self.ent_crossing_data = {}
        self.awaiting_candidates = []
        if self.object_ids is None or len(self.object_ids) == 0:
            self.object_ids = self.context.get_class_handler().object_ids
            for obj_id in self.object_ids:
                self.l1_required_type[obj_id] = None
        self.position_cfg = "center"
        if "alertConfig" in analytic_config:
            if "Counting" in analytic_config["alertConfig"]:
                cfg = analytic_config["alertConfig"]["Counting"]
                self.position_cfg = cfg.get("position", self.position_cfg)
                self.disable_crops = cfg.get("disable_crops", self.disable_crops)

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = super().is_active_batch(images, motion_data, batch_data)
        for candidate in alert_candidates:
            line_idx = candidate.extra.get("line", 0)
            d = candidate.extra.get("crossing")
            direction = "in" if d == self.entrance_direction[line_idx] else "out"
            for ent_id in candidate.ent_ids:
                if self.ent_crossing_data.get(ent_id) is None:
                    self.ent_crossing_data[ent_id] = {
                        "in": [],
                        "out": [],
                        "crop_in": [],
                        "crop_out": [],
                        "idBase": 0,
                        "trackerClass": EntityClassification.UNKNOWN,
                        "objectId": -1,
                    }
                    ent_data = self.entity_db.get_entity(ent_id)
                    if ent_data is not None:
                        self.ent_crossing_data[ent_id]["idBase"] = ent_data.first_seen
                        self.ent_crossing_data[ent_id]["trackerClass"] = ent_data.tracker_type
                        self.ent_crossing_data[ent_id]["objectId"] = ent_data.object_id

                self.ent_crossing_data[ent_id][direction].append(candidate.timestamp)
                self.ent_crossing_data[ent_id][f"crop_{direction}"] = candidate.crops
        if self.awaiting_candidates:
            awaiting_candidates = copy(self.awaiting_candidates)
            self.awaiting_candidates.clear()
            return awaiting_candidates
        return []

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            if ent in self.ent_crossing_data:

                self._generate_candidate(ent)
                del self.ent_crossing_data[ent]
        super().on_entities_removed(ent_ids)

    def _generate_candidate(self, ent):
        in_count = len(self.ent_crossing_data[ent]["in"])
        out_count = len(self.ent_crossing_data[ent]["out"])
        if in_count > out_count:
            count = 1
            crop = self.ent_crossing_data[ent]["crop_in"]
            ts = self.ent_crossing_data[ent]["in"][0]
        elif out_count > in_count:
            count = -1
            crop = self.ent_crossing_data[ent]["crop_out"]
            ts = self.ent_crossing_data[ent]["out"][-1]

        else:
            logger.info(f"Count alert: entity {ent} has equal in and out counts")
            return
        ent_data = self.entity_db.get_entity(ent)
        if ent_data is not None:
            extra = {
                "idBase": ent_data.first_seen,
                "trackerClass": ent_data.tracker_type,
                "objectId": ent_data.object_id,
            }
        else:
            extra = {}
        extra["crossing"] = count
        if self.disable_crops:
            crop = []
        candidate = self.build_alert_candidate(ts, ent, None, extra, crops=crop)
        logger.debug(f"new counting alert with direction {count} for entity {ent}")
        self.awaiting_candidates.append(candidate)

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        return True

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)

        # pro caution against issues where the entity is already deleted from the entity db
        if alert_info.idBase < 0 or alert_info.trackerClass < 0:
            alert_info.idBase = candidate.extra.get("idBase", alert_info.idBase)
            alert_info.trackerClass = candidate.extra.get("trackerClass", alert_info.trackerClass)
            alert_info.object_id = candidate.extra.get("objectId", alert_info.object_id)
        return alert_info
