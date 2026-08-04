from typing import Dict, List

import numpy as np

from general import apply_from_dict
from general.core import BatchDataResolver, BDR, AnalyticImage
from .base_alerts import ObjectAlert, AlertCandidate


class ZoneAlert(ObjectAlert):
    type_name: str = "zoneTrespassing"
    alert_message: str = "Zone Trespassing Detected"
    history_filter = True

    def __init__(self, alert_dict: Dict, context):
        super(ZoneAlert, self).__init__(alert_dict, context)
        self.alert_on_tracked_only = True
        self.alerted_ids = {}
        # analytic_config = self.context.get_config()
        self.max_hold = 30000  # leave time to overcome blacklist\filter issues
        self.zones = apply_from_dict("zones", self.selectedCamera, None)
        # map entity id -> last timestamp seen in green zone
        self.green_history: Dict[int, int] = {}
        # build dict keyed by zone color -> list of marked indices (merged per color)
        assert "markedIdx" in self.selectedCamera and "markedIdxTrespassing" in self.selectedCamera, "Both markedIdx and markedIdxTrespassing must be defined"
        # Be advised: on previous alerts coomon markedIdx field aggregated all zones, here they should be separated by color: multiple greens aggregated into 'markedIdx' and single red gets into 'markedIdxTrespassing',
        # potential BUG could araise if backend aggregates both greens and red into the common 'markedIdx' only (as it was up untill now)
        self.marked_idx_dict = {}
        self.marked_idx_dict['green'] = set(self.selectedCamera.get("markedIdx"))
        self.marked_idx_dict['red'] = set(self.selectedCamera.get("markedIdxTrespassing"))
        
    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        # detections := [x,y,x,y,id,cls, subclass, conf, det_id, age, conflict, frame_id, timestamp, [location, center, center location,  area, index]]
        alert_candidates = []
        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            return alert_candidates
        ids_to_report = []
        # check green zone
        if len(vars_data[0]) > 0:
            self.green_history.update({int(v[BDR.ID]): int(v[BDR.TIMESTAMP]) for v in vars_data[0]})
        # check red zone and alert only if red timestamp > last green timestamp
        if len(vars_data[1]) > 0:
            curr_r_dict = {int(v[BDR.ID]): int(v[BDR.TIMESTAMP]) for v in vars_data[1]}
            for id_, last_red_ts in curr_r_dict.items():
                if id_ in self.green_history:
                    last_green_ts = self.green_history.get(id_)
                    # only alert if the red event occurs after being in green
                    if last_green_ts is not None and last_red_ts > last_green_ts:
                        last_seen = self.alerted_ids.get(id_, 0)
                        # prevent duplicate reporting on static objects unless connection loss resets
                        if last_seen == 0:
                            ids_to_report.append(id_)
                        self.alerted_ids[id_] = last_red_ts
        # update alert info in case of active
        for obj_active in ids_to_report:
            idx = np.where(vars_data[1][:, BDR.ID] == obj_active)[0][0]
            ts = vars_data[1][idx, BDR.TIMESTAMP]
            candidate = self.build_alert_candidate(ts, obj_active, vars_data[1][idx, :], images=images)
            alert_candidates.append(candidate)
        return alert_candidates
    
    def apply_roi_filter(self, vars_data: np.array, position=BatchDataResolver.LOCATION) -> np.array:
        if len(vars_data) == 0:
            empty = np.empty((0, vars_data.shape[1] if vars_data.ndim == 2 else 0))
            return empty, empty
        locations = vars_data[:, position].astype(int)
        green_mask = np.isin(locations, list(self.marked_idx_dict['green']))
        red_mask = np.isin(locations, list(self.marked_idx_dict['red']))
        return vars_data[green_mask], vars_data[red_mask]


    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
            self.green_history.pop(ent, None)
            self.alerted_ids.pop(ent, None)
