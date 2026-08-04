from typing import Dict, List, Tuple
import numpy as np

from general.analyzer_general import logger
from general.core import AnalyticImage, AlertInfo
from alerts.base_alerts import BaseAlert
from .base_alerts import BaseAlert, AlertCandidate, duration_unit_to_sec, duration_unit_to_str


class EmptyShelfAlert(BaseAlert):
    type_name: str = "empty_shelf"
    alert_message: str = "Empty Shelf Detected"
    occupancy_th: float = 0.5
    filled_shelf_th: float = 0.85                       # threshold to reset alert when shelf is filled again
    emptyshelf_per_ms = 30 * 1000                       # check-up period: 30 seconds in milliseconds
    emptyshelf_last_per: int = -1
    alerts_status: Dict[int, Tuple[bool, int]] = {}     # keeps track of alert status and last triggered timestamp
    flexibility: int = 5000                             # allowed delay in expert response in milliseconds
    immediate: bool = False                             # whether to request immediate expert response

    def __init__(self, alert_dict: Dict, context):
        super(EmptyShelfAlert, self).__init__(alert_dict, context)
        entity_db = context.get_entity_db()
        self.verbose = self.context.get_config().get("debug", {}).get("verbose", False)
        self.state_shelves_manager = entity_db.state_shelves_manager
        if not self.state_shelves_manager.enabled:
            self.enable = False
            raise ValueError("State shelves manager is not enabled")
        self.shelves_id = []
        self.set_flow_values()
        if not self.shelves_id:
            raise ValueError("No shelves were defined for empty shelf alert.")
        # parse user's ROIs
        self.num_alert_shelves = len(self.shelves_id)                                       # subset of shelves to monitor
        self.alerts_status = {i: (False, 0) for i in range(self.num_alert_shelves)}
        self.alert_ignore_last_per = [0] * self.num_alert_shelves                           # last ignored timestamp for each shelf
        self.orig_emptyshelf_per_ms = self.emptyshelf_per_ms
        self.shelves_occupancy_history = {id: [] for id in self.shelves_id}                 # history of occupancy for each shelf
        self.valid_alerts = True
        # sync sampling parameters with state shelves manager
        self.state_shelves_manager.emptyshelf_per_ms = min(self.emptyshelf_per_ms, self.state_shelves_manager.emptyshelf_per_ms)
        self.state_shelves_manager.orig_emptyshelf_per_ms = min(self.orig_emptyshelf_per_ms, self.state_shelves_manager.orig_emptyshelf_per_ms)
        self.state_shelves_manager.flexibility = min(self.flexibility, self.state_shelves_manager.flexibility)
        self.state_shelves_manager.immediate = True if self.state_shelves_manager.immediate else self.immediate

    def set_flow_values(self):
        if self.formValue:
            self.occupancy_th = self.formValue.get("threshold", self.occupancy_th)
            if self.occupancy_th > 1.0:
                self.occupancy_th /= 100.0  # convert from percentage to fraction
            self.emptyshelf_per_ms = self.formValue.get("emptyshelf_per_ms", self.emptyshelf_per_ms)
            requested_shelves = self.formValue.get("shelves", [])
            available_shelves = set(self.state_shelves_manager.poly_objs_dict.keys())
            self.shelves_id = [sid for sid in requested_shelves if sid in available_shelves]
            if len(self.shelves_id) != len(requested_shelves):
                raise ValueError("Some requested shelves for an alert were not found in the state shelves manager.")

    def crop_alerted_polygon(self, shelf_id: int, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        minx, miny, maxx, maxy = self.state_shelves_manager.poly_objs_dict[shelf_id].polygon.bounds  # float
        x0 = max(0, int(np.floor(minx * w)))
        y0 = max(0, int(np.floor(miny * h)))
        x1 = min(w, int(np.ceil(maxx * w)))
        y1 = min(h, int(np.ceil(maxy * h)))
        if x1 <= x0 or y1 <= y0:
            return img[0:0, 0:0, :]  # empty crop
        return img[y0:y1, x0:x1, :]  # RGB crop

    def _determine_alert(self, curr_timestamp):
        """
        Determine if a new alert should be raised based on the current shelf occupancy.
        If alert already raised before, do not raise a new one.
        Reset alert status only when shelf is filled again.
        """
        alerts_out = [False] * self.num_alert_shelves
        for i, shelf_id in enumerate(self.shelves_id):
            poly = self.state_shelves_manager.poly_objs_dict[shelf_id]
            if self.alerts_status[i][0] and poly.occupancy > self.filled_shelf_th and not poly.is_occl:
                # reset alert status if shelf is filled again and not occluded
                self.alerts_status[i] = (False, curr_timestamp)
            alerts_out[i] = poly.occupancy < self.occupancy_th
            if alerts_out[i] and not self.alerts_status[i][0]:
                # new alert
                self.alerts_status[i] = (alerts_out[i], curr_timestamp)
            elif alerts_out[i] and self.alerts_status[i][0]:
                # already alerted, cancel repeated alert
                alerts_out[i] = False
        return alerts_out

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alerts = [False] * self.num_alert_shelves
        alert_candidates = []
        # create output alerts
        alerts = self._determine_alert(images[-1].timestamp)
        for p_idx, alrt in enumerate(alerts):
            if alrt:
                poly_crop = self.crop_alerted_polygon(self.shelves_id[p_idx], images[-1].frame)     # be advised: cropped frame isn't the same as the one the decision is based on
                candidate = self.build_alert_candidate(
                    images[-1].timestamp, extra=self.state_shelves_manager.poly_objs_dict[self.shelves_id[p_idx]].export_dict(), crops=[poly_crop]
                )
                alert_candidates.append(candidate)
        if self.verbose:    # for debug only
            if type(self) != EmptyShelfCounterAlert:
                print(f"----- {type(self)} -----")
                print("##### Shelf Occupancy #####")
                print(f"Timestamp: {images[-1].timestamp} ({round((images[-1].timestamp / 1000), 1)} seconds)")
                for i, shelf_id in enumerate(self.shelves_id):
                    poly = self.state_shelves_manager.poly_objs_dict[shelf_id]
                    print(f"Shelf {shelf_id}: occupancy {poly.occupancy}, alert status {self.alerts_status[i]}")
                print(f"##### Output Alerts: {alerts} ######")

        earliest_req_ts = self.state_shelves_manager.expert_requests_queue[0] if self.state_shelves_manager.expert_requests_queue else images[-1].timestamp
        if images[-1].timestamp - earliest_req_ts > 3 * self.state_shelves_manager.emptyshelf_per_ms:
            self.valid_alerts = False
            self.context.validate_alert(self.event_id, is_valid=False, reason="Expert failed to respond in a timely manner.")
        else:
            if not self.valid_alerts:
                self.valid_alerts = True
                self.context.validate_alert(self.event_id, is_valid=True)
        return alert_candidates

    def build_alert_info(self, candidate: AlertCandidate) -> AlertInfo:
        alert_info = self.build_alert_info_dc(candidate)
        alert_info.crops += candidate.crops
        alert_info.validation_images += candidate.validation_images
        extra_fields = candidate.extra if candidate.extra else {}
        alert_info.extra.update(extra_fields)
        alert_info.alertData = candidate.extra["occupancy"]
        if self.special_filter is not None:
            alert_info.specialFilter = self.special_filter
        alert_info.alertMessage = self.alert_message
        return alert_info
    
    
class EmptyShelfCounterAlert(EmptyShelfAlert):
    alert_message: str = "Shelf Occupancy Status Report"
    occupancy_th = 1.1  # always trigger
    report_frequency_ms: int = 30 * 1000  # report every 20 seconds shelf occupancy status if below threshold
    alert_ignore_last_per: List[int]  # last ignored timestamp for each shelf

    def set_flow_values(self):
        self.shelves_id = self.formValue.get("shelves")
        self.report_frequency_ms = self.formValue.get("report_frequency_ms", self.report_frequency_ms)
        self.emptyshelf_per_ms = min(self.emptyshelf_per_ms, self.report_frequency_ms)

    def _determine_alert(self, curr_timestamp):
        """
        Determine if a new alert should be raised based on the current shelf occupancy.
        If alert already raised before, do not raise a new one unless ignoring period has passed.
        In the EmptyShelfCounterAlert class alerts are raised to report occupancy.
        """
        alerts_out = [False] * self.num_alert_shelves
        for i, shelf_id in enumerate(self.shelves_id):
            alerts_out[i] = self.state_shelves_manager.poly_objs_dict[shelf_id].occupancy < self.occupancy_th
            curr_per = (curr_timestamp - self.alerts_status[i][1]) // self.report_frequency_ms
            if alerts_out[i] and not self.alerts_status[i][0]:
                # new alert
                self.alerts_status[i] = (True, curr_timestamp)
            elif curr_per > self.alert_ignore_last_per[i]:
                self.alert_ignore_last_per[i] = curr_per
                self.alerts_status[i] = (alerts_out[i], curr_timestamp)
            elif alerts_out[i] and self.alerts_status[i][0]:
                # cancel repeated alert if ignoring period has not passed
                alerts_out[i] = False
        return alerts_out


class EmptyShelfDropAlert(EmptyShelfAlert):
    emptyshelf_per_ms: int = 4 * 1000  # check-up period: 4 sec in milliseconds
    duration: int = 10 * 1000
    occupancy_th: float = 0.25  # occupancy drop threshold
    flexibility: int = 2000
    immediate: bool = True  # request immediate expert response

    def set_flow_values(self):
        super().set_flow_values()
        duration = self.formValue.get("duration", 0)
        units = self.apply_from_dict("durationUnit", self.formValue, 0)
        self.occupancy_th = self.apply_from_dict("threshold", self.formValue, 0)
        self.occupancy_th /= 100.0  # convert from percentage to fraction
        self.duration = int(duration * duration_unit_to_sec[units] * 1000)  # convert to milliseconds
        self.emptyshelf_per_ms = min(self.emptyshelf_per_ms, self.duration // 2)
        self.flexibility = min(self.flexibility, self.emptyshelf_per_ms // 2)
        self.alert_message = f"shelf occupancy had dropped by {int(self.occupancy_th * 100)}% in less than {duration} {duration_unit_to_str[units]}"

    def _determine_alert(self, curr_timestamp):
        """
        Determine if a new alert should be raised based on the current shelf occupancy.
        If alert already raised before, do not raise a new one.
        Raise alerts only if occupancy drops by a certain threshold within the specified duration.
        Notice: A sample isn't taken into account if the polygon is effectively occluded.
        """
        alerts_out = [False] * self.num_alert_shelves
        for i, shelf_id in enumerate(self.shelves_id):
            poly = self.state_shelves_manager.poly_objs_dict[shelf_id]
            self.shelves_occupancy_history[shelf_id].extend(poly.last_itr_occupancies)
            # prune old history entries
            while self.shelves_occupancy_history[shelf_id] and curr_timestamp - self.shelves_occupancy_history[shelf_id][0][0] > self.duration:
                self.shelves_occupancy_history[shelf_id].pop(0)
            ## check for occupancy drop
            if len(self.shelves_occupancy_history[shelf_id]) > 0:
                # check delta occupancy
                occupancies_vec = [occ for _, occ in self.shelves_occupancy_history[shelf_id]]
                max_idx = occupancies_vec.index(max(occupancies_vec))
                post_max_values = occupancies_vec[max_idx + 1:]
                if post_max_values:
                    delta_occupancy = round(occupancies_vec[max_idx] - min(post_max_values), 3)
                else:
                    delta_occupancy = 0.0  # no samples after max, can't compute drop
                # check drop condition
                if delta_occupancy >= self.occupancy_th:
                    if not self.alerts_status[i][0]:  # new alert triggered only once
                        self.alerts_status[i] = (True, curr_timestamp)
                        alerts_out[i] = True
                        self.shelves_occupancy_history[shelf_id] = [self.shelves_occupancy_history[shelf_id][-1]]   # keep only last sample and prevent multiple alerts for a continuous drop
                        logger.info(
                            f"Shelf {i} occupancy dropped by {delta_occupancy:.2f} within {self.duration} ms, triggering alert."
                        )
                        # print(        # for debug only
                        #     f"Shelf {i} occupancy dropped by {delta_occupancy:.2f} within {self.duration} ms, triggering alert."
                        # )
                elif self.alerts_status[i][0]:
                    # reset alert status if occupancy gap returns to normal
                    self.alerts_status[i] = (False, curr_timestamp)

        return alerts_out