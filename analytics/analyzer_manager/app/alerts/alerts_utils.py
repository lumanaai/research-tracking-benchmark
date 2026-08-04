from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

from alerts.base_alerts import duration_unit_to_sec

IGNORED_STATE = 0
NEGATIVE_STATE = -1
UNKNOWN_STATE = -2
SPECIAL_STATES = {IGNORED_STATE, UNKNOWN_STATE}

@dataclass
class ObjectState:
    id: int
    description: str
    encodings: Optional[np.array]
    alert_on: bool = True


@dataclass
class ObjectInfo:
    id: str
    name: str
    coordinates: np.array
    states: List[ObjectState]

    state_map: Dict[int, ObjectState] = None

    def __post_init__(self):
        self.state_map = {}
        for state in self.states:
            if state.id != IGNORED_STATE:
                self.state_map[state.id] = state

    def get_state(self, state_id: int) -> ObjectState:
        return self.state_map[state_id]


def read_object_info_from_form(form: Dict[str, Any]) -> ObjectInfo:
    obj_dict = form.get("objectInfo")
    states = []
    for i, state in enumerate(obj_dict.get("states", [])):
        id_ = IGNORED_STATE if state.get("ignore", False) else i + 1
        enc = state.get("encodedQuery", None)
        if enc is not None:
            enc = np.frombuffer(bytes.fromhex(enc), dtype=np.float32)
        states.append(ObjectState(id_, state.get("query"), enc, state.get("alertOn", True)))

    return ObjectInfo(obj_dict.get("id"), obj_dict.get("name"), np.array(obj_dict.get("position")), states)

def read_transition_duration_from_form(form: Dict[str, Any]) -> Dict:
    td_list = form.get("transitionDuration", [])
    trans_dict = {}
    for d in td_list:
        units = int(d.get("durationUnit"))
        duration = max(1000, int(d.get("duration")) * duration_unit_to_sec[units] * 1000)
        from_state = int(d.get("from"))
        to_state = int(d.get("to"))
        if from_state not in trans_dict:
            trans_dict[from_state] = {}
        trans_dict[from_state][to_state] = duration
    return trans_dict

def get_zones_bbox(zones: Dict[str, Any]) -> Optional[np.array]:
    bbox_ok = False
    bbox = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
    for k, v in zones.items():
        selection: List[Dict] = v.get("selection", {})
        for node in selection:
            xy = np.array([node.get("x", 0), node.get("y", 0)])
            bbox[0] = np.minimum(bbox[0], xy)
            bbox[1] = np.maximum(bbox[1], xy)
            bbox_ok = True
    if bbox_ok:
        return bbox.reshape(-1)
    return None


class StateMachine(object):
    state: int
    last_triggered_ts: int
    last_state_change: int
    current_state_alerted: bool
    last_triggered_state: Optional[int]
    object_info: ObjectInfo
    transitions_duration: Dict[int, Dict[int, int]]
    last_known_state: int = UNKNOWN_STATE

    def __init__(self, object_info: ObjectInfo, latency, duration, transitions = None):
        self.object_info = object_info
        self.latency = latency
        self.duration = duration
        self.num_states = len(object_info.states)
        self.alert_on = {s.id: s.alert_on for s in self.object_info.states if s.id != IGNORED_STATE}
        self.is_single_state = self.num_states == 1
        self.valid_count_th = np.ceil(self.latency / 2)
        if transitions is None:
            self.transitions_duration = {}
        else:
            self.transitions_duration = transitions
        self.states = deque(maxlen=self.latency)
        self.clear(reset=True)

    def add_measurement(self, state: int, timestamp: int) -> bool:
        # ignored state are not going into the buffer
        if state == IGNORED_STATE:
            return False

        is_alert = False
        denoised_state = self.denoise_state(state)
        if denoised_state not in SPECIAL_STATES and denoised_state != self.state:
            self.last_known_state = self.state
            self.last_state_change = timestamp
            self.state = denoised_state
            self.current_state_alerted = False
        if self.state not in SPECIAL_STATES and not self.current_state_alerted:
            dur = self.get_duration()
            is_criteria_met = timestamp - self.last_state_change >= dur

            if is_criteria_met:
                self.current_state_alerted = True
                if self.last_triggered_state is None:
                    self.last_triggered_state = self.state

                elif self.is_single_state or self.last_triggered_state != self.state:
                    self.last_triggered_ts = timestamp
                    self.last_triggered_state = self.state
                    if self.state != NEGATIVE_STATE:
                        is_alert = self.alert_on.get(self.state, False)
        return is_alert

    def denoise_state(self, state):
        self.states.append(state)
        denoised = UNKNOWN_STATE
        counts = Counter(self.states)
        max_state, max_val = max(counts.items(), key=lambda x: x[1])
        if max_val >= self.valid_count_th:
            denoised = max_state
        return denoised

    def clear(self, reset: bool = False):
        self.states.clear()
        self.state = UNKNOWN_STATE
        if reset:
            self.last_triggered_ts = -100000
            self.last_state_change = -100000
            self.current_state_alerted = False
            self.last_triggered_state = None

    def get_duration(self) -> int:
        return self.transitions_duration.get(self.last_known_state, {}).get(self.state, self.duration)

    # def check_criteria_multi_state(self, ts: int) -> bool:
    #     is_criteria_met = False
    #     counts = Counter(self.states)
    #     max_state, max_val = max(counts.items(), key=lambda x: x[1])
    #     if max_state >= 0 and max_val >= self.valid_count_th:
    #         if max_state == self.state:
    #             is_criteria_met = (
    #                     self.states[-1] == max_state
    #                     and self.alert_on[max_state]
    #                     and ts - self.last_triggered_ts >= self.duration
    #             )
    #         else:
    #             self.state = max_state
    #
    #     return is_criteria_met
    #
    # def check_criteria_single_state(self, ts: int) -> bool:
    #     count = sum(1 for s in self.states if s == self.clip_state_ids[0])
    #     state = self.clip_state_ids[0] if count >= self.valid_count_th else IGNORED_STATE
    #     if state != self.state:
    #         self.last_state_change = ts
    #         self.state = state
    #     if state > IGNORED_STATE:
    #         return ts - self.last_state_change >= self.duration
    #     return False

from typing import Dict, Tuple, Optional


class WLSimilarity:
    """
    OCR-aware string similarity metric.
    Combines:
    1) Weighted Levenshtein edit-distance
    2) Position-aware character class consistency
    Final score in [0, 1], where higher means more similar.
    """
    def __init__(
        self,
        confusion_costs: Optional[Dict[Tuple[str, str], float]] = None,    # holds pairwise confusion costs
        insertion_cost: float = 1.0,                                    # cost of inserting a character
        deletion_cost: float = 1.0,                                     # cost of deleting a character
        default_substitution_cost: float = 1.0,                         # cost of substituting a character
        class_match_score: float = 0.5,
        confusion_match_score: float = 0.75,
        exact_match_score: float = 1.0,
        edit_weight: float = 0.7,                                       # weight of edit-distance in final score
        class_weight: float = 0.3,                                      # weight of position-class score in final score
    ):
        # Default OCR confusions if not provided
        self.confusion_costs = confusion_costs or {
            ('0', 'O'): 0.25, ('O', '0'): 0.25,
            ('1', 'I'): 0.25, ('I', '1'): 0.25,
            ('1', 'l'): 0.25, ('l', '1'): 0.25,
            ('5', 'S'): 0.30, ('S', '5'): 0.30,
            ('2', 'Z'): 0.30, ('Z', '2'): 0.30,
            ('8', 'B'): 0.30, ('B', '8'): 0.30,
        }
        self.insertion_cost = insertion_cost
        self.deletion_cost = deletion_cost
        self.default_substitution_cost = default_substitution_cost
        self.class_match_score = class_match_score
        self.confusion_match_score = confusion_match_score
        self.exact_match_score = exact_match_score
        self.edit_weight = edit_weight
        self.class_weight = class_weight

    @staticmethod
    def _char_class(c: str) -> str:
        if c.isdigit():
            return "digit"
        if c.isalpha():
            return "alpha"
        return "other"

    def weighted_levenshtein(self, a: str, b: str) -> float:
        n, m = len(a), len(b)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i * self.deletion_cost
        for j in range(m + 1):
            dp[0][j] = j * self.insertion_cost
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                ca, cb = a[i - 1], b[j - 1]
                if ca == cb:
                    sub_cost = 0.0
                else:
                    sub_cost = self.confusion_costs.get(
                        (ca, cb), self.default_substitution_cost
                    )
                dp[i][j] = min(
                    dp[i - 1][j] + self.deletion_cost,
                    dp[i][j - 1] + self.insertion_cost,
                    dp[i - 1][j - 1] + sub_cost
                )
        return dp[n][m]

    def levenshtein_similarity(self, a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        dist = self.weighted_levenshtein(a, b)
        return 1.0 - dist / max(len(a), len(b))

    def position_class_score(self, query: str, pred: str) -> float:
        L = min(len(query), len(pred))
        score = 0.0
        for i in range(L):
            g, p = query[i], pred[i]
            if g == p:
                score += self.exact_match_score
            elif (g, p) in self.confusion_costs:
                score += self.confusion_match_score
            elif self._char_class(g) == self._char_class(p):
                score += self.class_match_score
        return score / max(len(query), len(pred))

    def similarity(self, query: str, pred: str) -> float:
        s_edit = self.levenshtein_similarity(query, pred)
        s_class = self.position_class_score(query, pred)
        return self.edit_weight * s_edit + self.class_weight * s_class

