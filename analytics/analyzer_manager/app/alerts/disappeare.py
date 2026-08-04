from typing import Dict, List
import numpy as np

from general.core import BatchDataResolver, AnalyticImage
from .base_alerts import ObjectAlert, AlertCandidate


class DisappeareAlert(ObjectAlert):
    type_name = "disappear"

    def __init__(self, alert_dict: Dict, context):
        super(DisappeareAlert, self).__init__(alert_dict, context)
        self.alert_on_tracked_only = True

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []

        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            alert_ids = []
        else:
            alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()

        missing_ids = self.active_ids.difference(alert_ids)
        newly_missing = missing_ids.difference(self.alerted_ids)
        self.alerted_ids.update(newly_missing)
        self.active_ids.update(alert_ids)

        # update alert info in case of active
        for obj_active in newly_missing:
            known_dets = batch_data.query(BatchDataResolver.ID, obj_active)
            timestamp = known_dets[-1, BatchDataResolver.TIMESTAMP] if len(known_dets) else images[0].timestamp
            candidate = self.build_alert_candidate(timestamp, obj_active, known_dets, extra={})
            alert_candidates.append(candidate)
        return alert_candidates
