import enum
from copy import copy
from typing import Dict, Any, Optional, List, Set

import numpy as np
from general import apply_from_dict
from general.analyzer_general import roi_gen, SUPPORTED_FILTERS, ROI_SHAPE, logger
from general.core import AlertsAction, BatchDataResolver, DashboardsAction
from general.entity_db import ActiveEntityData


class VariableType(enum.IntEnum):
    BASE = -1
    OBJECT = 0
    EVENT = 1


class VariableBase:
    variable_type = VariableType.BASE.value
    is_object_based: bool = False
    count: int
    next_sync: int

    def __init__(self, variable_dict: Dict, context):
        self.variable_id: str = apply_from_dict("_id", variable_dict, "")
        self.period_min = max(1, round(apply_from_dict("periodInMinutes", variable_dict, 5)))
        self.period = self.period_min * 60000  # min to mSec
        self.is_realtime = apply_from_dict("realTime", variable_dict, False)
        self.count = 0
        self.next_sync = 0
        self.need_rt_update = False

    def update_stat(self, frame_count):
        self.count += 1
        if self.is_realtime:
            self.need_rt_update = True

    def on_entities_update(self, ent_ids: List[str]):
        pass

    def on_entities_removed(self, ent_ids: List[str]):
        pass

    def sync(self, curr_timestamp) -> Optional[Dict]:
        if self.need_rt_update or curr_timestamp > self.next_sync:
            if curr_timestamp > self.next_sync:
                report = self._export_stats(rt_update=False)
                self.calc_next_sync(curr_timestamp)
                self._reset_stat()
            else:
                report = self._export_stats(rt_update=True)
            self.need_rt_update = False
            return report
        return None

    def calc_next_sync(self, curr_timestamp):
        self.next_sync = np.ceil(curr_timestamp / self.period) * self.period

    def _reset_stat(self):
        self.count = 0

    def _export_stats(self, rt_update: bool = False) -> Dict:
        stats = {
            "variableId": self.variable_id,
            "variableType": self.variable_type,
            "periodInMinutes": self.period_min,
            "realTime": rt_update,
            "count": self.count,
        }
        return stats


class EventVariable(VariableBase):
    variable_type = VariableType.EVENT.value
    is_object_based: bool = False

    def __init__(self, variable_dict: Dict, context):
        super(EventVariable, self).__init__(variable_dict, context)


class ObjectVariable(VariableBase):
    variable_type = VariableType.OBJECT.value
    is_object_based: bool = True

    stat_max: int
    stat_min: int
    stat_mean: float
    stat_last: int
    stat_denom: int

    def __init__(self, variable_dict: Dict, context):
        super(ObjectVariable, self).__init__(variable_dict, context)
        self.active_ids = set()
        self.context = context
        self.unseen_ids = set()

        configuration_dict = variable_dict["configuration"]
        self.object = apply_from_dict("object", configuration_dict, -1)

        self.marked_idx = apply_from_dict("markedIdx", configuration_dict, None)
        self.roiFilter: bool = self.marked_idx is not None and len(self.marked_idx) > 0
        self.roi: Any = None if not self.roiFilter else roi_gen(self.marked_idx)

        m_filters = apply_from_dict("filters", configuration_dict, {})
        self.filters = {}
        for key in m_filters.keys():
            if (
                key in SUPPORTED_FILTERS
                and m_filters[key] is not None
                and isinstance(m_filters[key], list)
                and len(m_filters[key]) > 0
            ):
                self.filters[key] = set(m_filters[key])
        default_analyzer = context.l1_manager.default_filters_analyzer().get(self.object, None)
        self.l1_analyzer = default_analyzer if self.filters else None

        # check if filters are set
        self.filter_en = len(self.filters) > 0
        self.traffic_path = []
        self.heatmap = np.zeros(ROI_SHAPE[0] * ROI_SHAPE[1])
        self._reset_stat()

    def _reset_stat(self):
        self.count = 0
        self.stat_max = 0
        self.stat_min = 10000000
        self.stat_mean = 0
        self.stat_last = 0
        self.stat_denom = 0
        self.heatmap = self.heatmap * 0

    def on_entities_update(self, ent_ids: List[int]):
        for ent in ent_ids:
            ent_data: ActiveEntityData = self.context.entity_db.get_entity(ent)
            is_match = ent_data.object_id == self.object

            if is_match:
                if self.l1_analyzer and ent_data.has_l1(self.l1_analyzer):
                    is_match = ent_data.match_filters(
                        [self.object],
                        {self.object: self.filters},
                        {self.object: len(self.filters) == 0},
                        strict_filters={self.object: True},
                    )
                    is_match &= not ent_data.is_blacklisted
            if is_match:
                # if it is the first time we encounter the ent
                if ent not in self.active_ids:
                    self.unseen_ids.add(ent)
                self.active_ids.add(ent)
            else:
                self.active_ids.discard(ent)

    def update_statistics(self, batch_data: BatchDataResolver, batch_size: int):
        batch_ids = batch_data.unique(batch_data.ID, batch_data.CLASS, self.object).astype(int)
        intersect = set(batch_ids).intersection(self.active_ids)
        if len(intersect) > 0:
            if self.roiFilter:
                vars_data = batch_data.advance_and_query(
                    [(BatchDataResolver.ID, list(intersect)), (BatchDataResolver.LOCATION, self.marked_idx)]
                )
            else:
                vars_data = batch_data.query(BatchDataResolver.ID, list(intersect))
            new_ids = set(vars_data[:, BatchDataResolver.ID].astype(int)).intersection(self.unseen_ids)
            if len(new_ids) > 0:
                self.count += len(new_ids)
                self.unseen_ids -= set(new_ids)
                if self.is_realtime:
                    self.need_rt_update = True
            counts = np.bincount(vars_data[:, BatchDataResolver.FRAME_ID].astype(int), minlength=batch_size).tolist()
            locations = vars_data[:, BatchDataResolver.LOCATION].astype(int)
            for loc in locations:
                self.heatmap[loc] += 1
        else:
            counts = [0] * batch_size

        self.stat_max = np.max([self.stat_max] + counts)
        self.stat_min = np.min([self.stat_min] + counts)
        self.stat_mean = (self.stat_mean * self.stat_denom + np.sum(counts)) / (self.stat_denom + batch_size)
        self.stat_last = counts[-1]
        self.stat_denom += batch_size

    def on_entities_removed(self, ent_ids: List[str]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
            self.unseen_ids.discard(ent)

    def _export_stats(self, rt_update: bool = False) -> Dict:
        variable = super(ObjectVariable, self)._export_stats()
        stats = {"avg": self.stat_mean, "min": self.stat_min, "max": self.stat_max, "latest": self.stat_last}
        variable["stat"] = stats
        if not (self.is_realtime and rt_update):
            variable["heatmap"] = self.heatmap.tolist()
            variable["trafficPath"] = self.traffic_path

        return variable


class DashboardManagerDB:
    _variables: List[VariableBase]
    _analytic_config: Dict
    last_update: int

    def __init__(self, context, init_msg):
        self._event_variables: Dict[str, EventVariable] = {}
        self._obj_variable: List[ObjectVariable] = []
        self._analytic_config = context.config
        self.class_handler = context.class_handler
        self.alert_manager = context.alert_manager
        self.l1_manager = context.l1_manager
        self.entity_db = context.entity_db
        self.entity_db.on_entities_update += self.on_entities_update
        self.entity_db.on_entities_removed += self.on_entities_removed
        self.entity_db.on_entities_purged += self.on_entities_purged
        self.last_update = 0
        self.l1_required_ents: Dict[int, Set] = {}
        if "variables" in init_msg:
            for variable_dict in init_msg["variables"]:
                self._add_variable(variable_dict)

    def _add_variable(self, variable_dict):
        if (
            "configuration" in variable_dict
            and variable_dict["configuration"] is not None
            and len(variable_dict["configuration"]) > 0
        ):
            variable = ObjectVariable(variable_dict, self)
            self._obj_variable.append(variable)
        elif "event" in variable_dict and variable_dict["event"] is not None:
            # first generate alert
            alert = variable_dict["event"]
            alert["action"] = AlertsAction.add.value
            alert["_id"] = variable_dict["_id"]
            alert["enabled"] = True
            alert["variableAlert"] = True
            self.alert_manager.parse_alerts_msg({"alert": alert})

            # then generate variable
            variable = EventVariable(variable_dict, self)
            self._event_variables[variable_dict["_id"]] = variable
        else:
            raise TypeError

    def _remove_variable(self, variable_dict):
        variable_id = variable_dict["_id"]
        obj_ids = [v.variable_id for v in self._obj_variable]
        if variable_id in obj_ids:
            del self._obj_variable[obj_ids.index(variable_id)]
        elif variable_id in self._event_variables:
            alert = {"action": AlertsAction.remove.value, "_id": variable_id}
            self.alert_manager.parse_alerts_msg({"alert": alert})
            self._event_variables.pop(variable_id)

    def parse_dashboard_msg(self, dashboard_msg):
        if "variable" in dashboard_msg:
            variable_dict = dashboard_msg["variable"]
            action = DashboardsAction(variable_dict["action"])
            if action is DashboardsAction.add:
                self._add_variable(variable_dict)
            elif action is DashboardsAction.remove:
                self._remove_variable(variable_dict)
            elif action is DashboardsAction.update:
                self._remove_variable(variable_dict)
                self._add_variable(variable_dict)

    def on_entities_removed(self, ent_ids):
        for variable in self._obj_variable:
            variable.on_entities_removed(ent_ids)

    def on_entities_purged(self,ent_ids):
        self.on_entities_removed(ent_ids)
        for e_id in ent_ids:
            self.l1_required_ents.pop(e_id, None)

    def on_entities_update(self, ent_ids: List[int]):
        # called every time entity metadata is changed (new or update)
        for variable in self._obj_variable:
            variable.on_entities_update(ent_ids)
            for e in variable.unseen_ids:
                if e in ent_ids and variable.l1_analyzer is not None:
                    self.l1_required_ents[e] = self.l1_required_ents.get(e, set())
                    self.l1_required_ents[e].add(variable.l1_analyzer)

    def process_batch_data(self, batch_data: BatchDataResolver, alerts_res, timestamps: List[int]) -> List[dict]:
        batch_size = len(timestamps)
        reports = []
        for variable in self._obj_variable:
            variable.update_statistics(batch_data, batch_size)
            var_report = variable.sync(timestamps[-1])
            if var_report is not None:
                reports.append(var_report)

        for event_info in alerts_res:
            alert_id = event_info.id
            if alert_id in self._event_variables:
                self._event_variables[alert_id].update_stat(event_info)
            else:
                logger.error(f"unidentified event id {alert_id} - variable not found")

        for variable in self._event_variables.values():
            var_report = variable.sync(timestamps[-1])
            if var_report is not None:
                reports.append(var_report)

        self.last_update = timestamps[-1]
        return reports

    def pop_questionable_ents(self) -> Dict[int, Set]:
        l1_required = copy(self.l1_required_ents)
        self.l1_required_ents.clear()
        return l1_required
