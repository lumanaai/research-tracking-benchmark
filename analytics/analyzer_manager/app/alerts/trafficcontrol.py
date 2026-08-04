from typing import Dict

from general.analyzer_general import Point, bbxSide
from general.core import BatchDataResolver, AlertCategory, SafetyType
from .base_alerts import AlertCandidate, duration_unit_to_sec, duration_unit_to_str
from .linecrossing import LineCrossingAlert, position_to_reference_point_function, Line


class TrafficControlAlert(LineCrossingAlert):
    type_name = "trafficControl"
    single_alert_object_filter = False
    ambient_motion_filter = False

    def __init__(self, alert_dict: Dict, context):
        super(TrafficControlAlert, self).__init__(alert_dict, context, multiline=True)
        self.line_cross_hysteresis_period = 0
        analytic_config = self.context.get_config()
        self.lines = []
        self.timeLimit = False
        units = 0
        duration = 0
        if (
            "trafficControl" in alert_dict
            and alert_dict["trafficControl"] is not None
            and (len(alert_dict["trafficControl"]["lines"]) > 1)
        ):

            for line in alert_dict["trafficControl"]["lines"]:
                d = 0
                if "d" in line and line["d"] is not None:
                    # cloud is using 0,1,2
                    if line["d"] == 2:
                        d = -1
                    else:
                        d = line["d"]
                p1 = Point(line["p1"]["x"], line["p1"]["y"])
                p2 = Point(line["p2"]["x"], line["p2"]["y"])

                position = "center" if self.context.is_location_center else "bottom"

                if "alertConfig" in analytic_config:
                    if "trafficControl" in analytic_config["alertConfig"]:
                        if "minBlockout" in analytic_config["alertConfig"]["trafficControl"]:
                            self.blockout = max(
                                self.blockout, analytic_config["alertConfig"]["trafficControl"]["minBlockout"]
                            )
                        if "position" in analytic_config["alertConfig"]["trafficControl"]:
                            position = analytic_config["alertConfig"]["trafficControl"]["position"]
                        
                        if "dynamicPosition" in analytic_config["alertConfig"]["trafficControl"]:
                            position = bbxSide(p1, p2, d)

                self.lines.append(
                    Line(
                        p1=Point(line["p1"]["x"], line["p1"]["y"]),
                        p2=Point(line["p2"]["x"], line["p2"]["y"]),
                        d=d,
                        position=position,
                        position_converter=position_to_reference_point_function(position),
                    )
                )

            if "distance" in alert_dict["trafficControl"]:
                if alert_dict["trafficControl"]["distance"] is not None:
                    self.distance = alert_dict["trafficControl"]["distance"]
                if alert_dict["trafficControl"]["distanceUnits"] is not None:
                    self.distanceUnits = alert_dict["trafficControl"]["distanceUnits"]
            else:
                self.distance = 0
                self.distanceUnits = "meter"

            if "timeLimit" in alert_dict["trafficControl"] and alert_dict["trafficControl"]["timeLimit"] is not None:
                self.timeLimit = True
                self.duration = alert_dict["trafficControl"]["timeLimit"]["count"]
                self.positive_time_limit = alert_dict["trafficControl"]["timeLimit"]["direction"] == "above"
            else:
                self.timeLimit = False
                self.duration = 0
                self.positive_time_limit = True
        elif not self.legacy:
            traffic_dict = self.apply_from_dict("trafficControl", self.selectedCamera, {})
            lines = self.apply_from_dict("lines", traffic_dict, [])

            for line in lines:
                d = 0
                if "d" in line and line["d"] is not None:
                    # cloud is using 0,1,2
                    if line["d"] == 2:
                        d = -1
                    else:
                        d = line["d"]
                p1 = Point(line["p1"]["x"], line["p1"]["y"])
                p2 = Point(line["p2"]["x"], line["p2"]["y"])

                if "alertConfig" in analytic_config:
                    if "trafficControl" in analytic_config["alertConfig"]:
                        if "minBlockout" in analytic_config["alertConfig"]["trafficControl"]:
                            self.blockout = max(
                                self.blockout, analytic_config["alertConfig"]["trafficControl"]["minBlockout"]
                            )
                        if "position" in analytic_config["alertConfig"]["trafficControl"]:
                            self.position_cfg = analytic_config["alertConfig"]["trafficControl"]["position"]

                self.lines.append(
                    Line(
                        p1=Point(line["p1"]["x"], line["p1"]["y"]),
                        p2=Point(line["p2"]["x"], line["p2"]["y"]),
                        d=d,
                        position=self.position_cfg,
                        position_converter=position_to_reference_point_function(self.position_cfg),
                    )
                )

            self.distance = 0
            self.distanceUnits = "meter"

            if "distance" in traffic_dict:
                if traffic_dict["distance"] is not None:
                    self.distance = traffic_dict["distance"]
                if traffic_dict["distanceUnits"] is not None:
                    self.distanceUnits = traffic_dict["distanceUnits"]

            self.is_speed_control = (
                alert_dict["selectedFlow"]["category"] == AlertCategory.Safety.value
                and alert_dict["selectedFlow"]["flowType"] == SafetyType.SpeedLimit.value
            )
            if self.is_speed_control:
                self.timeLimit = True
                duration = self.apply_from_dict("duration", self.formValue, 0)
                units = self.apply_from_dict("durationUnit", self.formValue, 0)
                self.duration =  duration * duration_unit_to_sec[units] * 1000
                self.positive_time_limit = False

        if self.category == AlertCategory.Safety.value:
            self.alert_message = "{} are following a path" + f" in less than {duration} {duration_unit_to_str[units]}"
        else:
            self.alert_message = "{} complete a path"

    def _read_lines(self, alert_dict):
        pass # done in constructor - need to ignore what is done in line crossing alert


    def on_line_crossed(self, ent_id, id_data, crossing, images, line_idx = 0):
        candidate = None
        if ent_id not in self.alerted_ids:
            self.alerted_ids[ent_id] = {"line": 0, "timestamp": id_data[BatchDataResolver.TIMESTAMP]}
        self.alerted_ids[ent_id]["line"] += 1
        if self.alerted_ids[ent_id]["line"] == len(self.lines):
            alert_period = id_data[BatchDataResolver.TIMESTAMP] - self.alerted_ids[ent_id]["timestamp"]
            if self.timeLimit:
                # we have time limit test
                time_valid = (self.positive_time_limit and alert_period > self.duration) or (
                    not self.positive_time_limit and alert_period < self.duration
                )
            else:
                time_valid = True
            if time_valid:
                extra = {"crossing": crossing, "extra_fields": {"duration": alert_period}}
                candidate = self.build_alert_candidate(
                    id_data[BatchDataResolver.TIMESTAMP], ent_id, id_data, extra, images=images
                )
        return candidate

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)
        alert_period = candidate.extra["extra_fields"]["duration"]
        speed = round(self.distance * 1000 / alert_period, 2) if alert_period > 0 else 0

        # if not self.timeLimit:
        #     alert_info.alertMessage += f" speed: {speed} {self.distanceUnits}/seconds".capitalize()
        # else:
        #     alert_info.alertMessage += f" passed in {alert_period / 1000} seconds".capitalize()
        alert_info.alertData = speed
        return alert_info

    def get_line(self, ent_id, idx=0):
        # should only work on first iteration since it must be a sequence, unlike multi line
        if idx > 0:
            return None
        if ent_id in self.alerted_ids:
            if self.alerted_ids[ent_id]["line"] < len(self.lines):
                return self.lines[self.alerted_ids[ent_id]["line"]]
            else:
                return None
        else:
            # first line
            return self.lines[0]
