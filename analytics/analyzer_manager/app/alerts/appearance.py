from typing import Dict, List, Optional
import numpy as np

from general.core import BatchDataResolver, AnalyticImage
from .base_alerts import ObjectAlert, AlertCandidate


class AppearanceAlert(ObjectAlert):
    type_name = "appearance"
    hysteresis_threshold_ms = 100000
    motion_filter = True
    history_filter = True

    def __init__(self, alert_dict: Dict, context):
        super(AppearanceAlert, self).__init__(alert_dict, context)
        self.alert_on_tracked_only = True
        self.alerted_ids = {}
        analytic_config = self.context.get_config()
        if "alertConfig" in analytic_config:
            self.hysteresis_threshold_ms = (
                analytic_config["alertConfig"]
                .get("objectAppearance", {})
                .get("hysteresis", self.hysteresis_threshold_ms)
            )
        self.max_hold = 30000  # leave time to overcome blacklist\filter issues

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []

        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            return alert_candidates
        alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()

        ids_to_report = []
        for id_ in alert_ids:
            batch_ts = vars_data[vars_data[:, BatchDataResolver.ID] == id_, BatchDataResolver.TIMESTAMP]
            last_seen = self.alerted_ids.get(id_, 0)
            # this cause re-appearance when connection lost on static object
            # if batch_ts[-1] >= last_seen + self.hysteresis_threshold_ms:
            if last_seen == 0:
                ids_to_report.append(id_)
            self.alerted_ids[id_] = batch_ts[-1].astype(int)

        # update alert info in case of active
        for obj_active in ids_to_report:
            idx = np.where(vars_data[:, BatchDataResolver.ID] == obj_active)[0][0]
            ts = vars_data[idx, BatchDataResolver.TIMESTAMP]
            candidate = self.build_alert_candidate(ts, obj_active, vars_data[idx, :], images=images)
            alert_candidates.append(candidate)
        return alert_candidates

    def out_of_schedule_maintenance(self, images, motion_data, batch_data):
        # this should avoid appearance right out of schedule break
        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) > 0:
            alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()
            for _id in alert_ids:
                self.alerted_ids[_id] = images[-1].timestamp

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
            self.alerted_ids.pop(ent, None)
