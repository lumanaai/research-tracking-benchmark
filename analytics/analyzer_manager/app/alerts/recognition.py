from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np

from alerts.face_db_utils import FaceEntity, parse_face_list_items, load_face_groups_from_db
from general.core import (
    BatchDataResolver,
    AttrProperty,
    AnalyticImage,
    AdvanceAnalyzerType,
    AlertsConfidence,
)
from general.entity import ActiveEntityData
from level1.face.face_utils import FaceConfidence, conf_order_map, conf2cosine_th, attr_conf_to_face_conf
from .alerts_utils import WLSimilarity
from .base_alerts import AlertCandidate
from .linecrossing import LineCrossingAlert


@dataclass
class RecognitionHistory:
    data: Any
    has_l1: bool
    extra: Any = None
    checked: bool = False


class RecognitionAlert(LineCrossingAlert):
    type_name = "recognition"
    recognition = ""  # name of list item in the form
    form_list_str = ""  # how thew list is presented in the form
    optional_line = True
    recognition_field = ""
    single_alert_object_filter = False
    ambient_motion_filter = False
    groups_field = ""
    groups: List = None
    list_subjects: List = None

    def __init__(self, alert_dict: Dict, context):
        self.list_subjects = []
        super(RecognitionAlert, self).__init__(alert_dict, context)
        self.green_list_enabled: bool = False
        self.red_list_enabled: bool = False
        self.unknown_subject_alert: bool = False
        self.green_list: List = []
        self.red_list: List = []
        self.max_hold = 5000
        self.last_checked_ts = 0
        self.activated_ents: Dict[int, AlertCandidate] = {}
        self.ent_recognition_history: Dict[int, RecognitionHistory] = {}
        analytic_config = self.context.get_config()

        self.required_l1 = True
        class_handler = self.context.get_class_handler()
        if self.type_name == "lpr":
            object_id = class_handler.object_str_to_int(class_handler.VEHICLE)
            object_str = class_handler.VEHICLE
            rec_config = analytic_config.get("alertConfig", {}).get("recognition", {})
            use_external_apis = analytic_config.get("l1_models", {}).get("use_external_apis", False)
            use_external_lpr = rec_config.get("useALPR", True) or use_external_apis
            if use_external_lpr:
                adv_analyzer = AdvanceAnalyzerType.ALPR.value
            else:
                adv_analyzer = AdvanceAnalyzerType.LPREC.value
            self.recognition_analyzer = adv_analyzer
            self.l1_required_type[object_id] = self.recognition_analyzer
        elif self.type_name == "face":
            class_handler = self.context.get_class_handler()
            object_id = class_handler.object_str_to_int(class_handler.PERSON)
            object_str = class_handler.PERSON
            self.recognition_analyzer = AdvanceAnalyzerType.FACE.value
            self.l1_required_type[object_id] = self.recognition_analyzer
        elif self.type_name == "containerId":
            if class_handler.container_value < 0:
                object_id = class_handler.object_str_to_int(class_handler.VEHICLE)
                object_str = class_handler.VEHICLE
            else:
                object_id = class_handler.object_str_to_int(class_handler.CONTAINER)
                object_str = class_handler.CONTAINER
            self.recognition_analyzer = AdvanceAnalyzerType.CONTAINER.value
            self.l1_required_type[object_id] = self.recognition_analyzer

        else:
            raise NotImplementedError
        # new alert format

        if "configuration" in alert_dict and "filters" in alert_dict["configuration"]:
            filters = alert_dict["configuration"]["filters"]

            # lpr lists
            if "greenList" in filters and filters["greenList"] is not None and len(filters["greenList"]):
                self.green_list_enabled = True
                for subjects in filters["greenList"].split(","):
                    self.green_list.append(subjects.strip().lower())

            if "redList" in filters and filters["redList"] is not None and len(filters["redList"]):
                self.red_list_enabled = True
                for subjects in filters["redList"].split(","):
                    self.red_list.append(subjects.strip().lower())

            if "unrecognized" in filters and filters["unrecognized"] is not None:
                self.unknown_subject_alert = filters["unrecognized"]

        selected_flow = self.apply_from_dict("selectedFlow", alert_dict, None)
        if selected_flow is not None:

            # vehicle object isn't set
            self.objects = [object_str]
            self.object_ids = [object_id]
            self.filters = {object_id: {}}
            self.filtersDisabled = {object_id: True}
            self.strict_filters = {object_id: True}

            # determine if its green or red list
            self.red_list_enabled = self.apply_from_dict(
                "appears", self.apply_from_dict(self.form_list_str, self.formValue, {}), []
            )
            self.green_list_enabled = not self.red_list_enabled

            self.group_subjects = []
            self.list_subjects = self.parse_list_item()
            self.groups = self.parse_groups()
            self.on_db_update()
            if not self.is_db_valid:
                raise RuntimeError(f"Recognition alert {self.event_id} is not valid due to DB issue")

            self.unknown_subject_alert = self.apply_from_dict(
                "unrecognized", self.apply_from_dict(self.form_list_str, self.formValue, {}), True
            )
        self.subject_check = self.unknown_subject_alert or self.red_list_enabled or self.green_list_enabled

        # set the mode of the alert between appearance and line crossing
        self.is_linecrossing = len(self.lines) > 0
        if self.is_linecrossing:
            self.is_active_internal = self.is_active_linecross
        else:
            self.is_active_internal = self.is_active_appearance
            self.alerted_ids = set()
        if self.red_list_enabled:
            self.alert_message = "appears in the list"
        else:
            self.alert_message = "doesn't appear in the list"

    def on_db_update(self):
        subjects = self.group_subjects + self.list_subjects
        if self.red_list_enabled:
            self.red_list = subjects
        else:
            self.green_list = subjects

    def parse_list_item(self) -> List:
        elements = self.apply_from_dict("list", self.apply_from_dict(self.form_list_str, self.formValue, {}), [])
        subjects = []
        for element in elements:
            subjects.append(element[self.recognition].lower())
        return subjects

    def on_entities_removed(self, ent_ids: List[int]):
        if self.is_linecrossing:
            super().on_entities_removed(ent_ids)
        else:
            for ent in ent_ids:
                self.active_ids.discard(ent)
                self.alerted_ids.discard(ent)

    def on_entities_purged(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.ent_recognition_history.pop(ent, None)
            self.activated_ents.pop(ent, None)

    def on_entities_update(self, ent_ids: List[int]):
        for ent in ent_ids:
            ent_data: ActiveEntityData = self.entity_db.get_entity(ent)
            if ent_data.object_id in self.object_ids:
                self.ent_recognition_history[ent] = self.get_recognition_data_from_ent(ent_data)

    def get_recognition_data_from_ent(self, ent_data: ActiveEntityData) -> RecognitionHistory:
        has_l1 = ent_data.has_l1(self.recognition_analyzer)
        rec_data = ent_data.attributes.get(self.recognition_field, None)
        return RecognitionHistory(rec_data, has_l1)

    def is_match(
        self, record: RecognitionHistory, candidate: AlertCandidate = None, force: bool = False
    ) -> Optional[bool]:
        if record.data is None or not record.has_l1:
            return self.unknown_subject_alert if force else None
        value = record.data[0].value
        is_red = value in self.red_list and self.red_list_enabled
        is_not_green = value not in self.green_list and self.green_list_enabled
        return is_not_green or is_red

    def alert_message_from_ent(self, ent_data, candidate: AlertCandidate) -> str:
        return ""

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        candidates = self.is_active_internal(images, motion_data, batch_data)
        # register the alert
        for cand in candidates:
            ent_id = cand.ent_ids[0]
            self.activated_ents[ent_id] = cand
        return candidates

    def is_active_appearance(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidate = []
        # find active ids
        vars_data = self.get_batch_candidates_data(batch_data)
        self.last_checked_ts = images[-1].timestamp
        if len(vars_data) > 0:
            alert_ids = np.unique(vars_data[:, BatchDataResolver.ID]).astype(int).tolist()
            alert_ids = set(alert_ids).difference(self.alerted_ids)
        else:
            return alert_candidate

        for ent_id in alert_ids:
            ent_data: ActiveEntityData = self.entity_db.get_entity(ent_id)
            # avoid blacklisted entities until they get out of blacklist status
            if ent_data is not None and ent_data.is_blacklisted:
                continue
            idx = np.where(vars_data[:, BatchDataResolver.ID] == ent_id)[0][0]
            candidate = self.build_alert_candidate(
                vars_data[idx, BatchDataResolver.TIMESTAMP], ent_id, vars_data[idx, :], images=images, extra={}
            )
            alert_candidate.append(candidate)
            self.alerted_ids.add(ent_id)
        return alert_candidate

    def is_active_linecross(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        self.last_checked_ts = images[-1].timestamp
        return super().is_active_batch(images, motion_data, batch_data)

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        ent_id = candidate.ent_ids[0]
        ent_data: ActiveEntityData = self.entity_db.get_entity(ent_id)

        # handle case where the entity already purged or max hold is out - make final decision
        if ent_data is None or ent_data.last_seen < self.last_checked_ts - self.max_hold:
            hist_record = self.ent_recognition_history.pop(ent_id, None)
            if hist_record is not None and hist_record.has_l1:
                return self.green_list_enabled
            else:
                return self.unknown_subject_alert

        hist_record = self.ent_recognition_history.get(ent_id, None)

        # if the record wasn't checked yet, check it
        if hist_record is not None and hist_record.has_l1 and not hist_record.checked:
            is_match = self.is_match(hist_record, candidate, force)
            hist_record.checked = True
            if is_match is not None:
                return is_match

        # reaching here means that either we didnt managed to match
        if force and self.is_linecrossing:
            return self.unknown_subject_alert
        return None

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)

        ent_id = candidate.ent_ids[0]
        ent_data: ActiveEntityData = self.entity_db.get_entity(ent_id)

        alert_info.alertMessage = self.alert_message_from_ent(ent_data, candidate)
        return alert_info

    def parse_groups(self) -> List:
        return self.formValue.get(self.groups_field, [])


class LPRAlert(RecognitionAlert):
    type_name = "lpr"
    recognition = "plate"
    form_list_str = "plates"
    recognition_field = "plate"
    groups_field = "vehicleGroups"
    high_conf_sim_th = 0.95
    medium_conf_sim_th = 0.75
    conf_to_no_match_th: Dict[AlertsConfidence, float] = {
        AlertsConfidence.high.value: 0.95,
        AlertsConfidence.medium.value: 0.9,
        AlertsConfidence.low.value: 0.8,
    }

    def __init__(self, alert_dict: Dict, context):
        super().__init__(alert_dict, context)

        analytic_config = self.context.get_config()
        lpr_config = analytic_config.get("alertConfig", {}).get("lpr", {})
        self.max_hold = lpr_config.get("maxHold", self.max_hold)
        if "high_conf_sim_th" in lpr_config:
            self.high_conf_sim_th = lpr_config.get("highSimTh", self.high_conf_sim_th)
        elif "Asia/Jerusalem" == alert_dict.get("timezone", ""):
            self.high_conf_sim_th = 0.99
        self.medium_conf_sim_th = lpr_config.get("medSimTh", self.medium_conf_sim_th)

        self.lp_similarity = WLSimilarity()
        self.no_match_decision_th = self.conf_to_no_match_th.get(self.alert_confidence, self.high_conf_sim_th)

    def alert_message_from_ent(self, ent_data, candidate: AlertCandidate) -> str:
        if ent_data is None:
            return f"Unknown car {self.alert_message}".capitalize()
        description = ent_data.attributes

        def prop_list_to_str_list(lst: List[AttrProperty]) -> List[str]:
            return [item.value for item in lst]

        if "colors" in description:
            alert_message = ",".join(item for item in prop_list_to_str_list(description["colors"])).capitalize()
            alert_message += " "
        else:
            alert_message = ""

        if "make" in description:
            alert_message += ",".join(item for item in prop_list_to_str_list(description["make"])).title()

            if "model" in description:
                model = ",".join(item for item in prop_list_to_str_list(description["model"])).capitalize()
                alert_message += f" {model}"
        else:
            alert_message += "unknown car make"

        if "plate" not in description:
            alert_message += f" car plate wasn't detected"
        else:
            plate_id = description["plate"][0].value.upper()
            # region = description["region"][0].value.upper()
            alert_message += f" Plate: {plate_id}"  #  from region: {region}"
        # if self.is_linecrossing:
        #     alert_message += " crossed the line"
        return f"{alert_message} {self.alert_message}".capitalize()

    def on_db_update(self):
        self.group_subjects = []
        prev_valid_db = self.is_db_valid
        if self.groups:  # requires db update
            self.is_db_valid = False
            vehicle_db = self.context.get_analytics_db().get("vehicles", {})
            group_db = self.context.get_analytics_db().get("vehicle_groups", [])
            for g in self.groups:
                gid = g.get("id", -1)
                group_data = [gd for gd in group_db if gd.id == gid]
                if len(group_data) != 1:
                    self.context.validate_alert(
                        self.event_id, False, f"Group ID {gid} not found in vehicle_groups database"
                    )
                    return
                group_vehicles = group_data[0].vehicleIds
                if not group_vehicles:
                    self.context.validate_alert(self.event_id, False, f"Group ID {gid} has no vehicles")
                    return
                for p in group_vehicles:
                    if p in vehicle_db:
                        self.group_subjects.append(vehicle_db[p].plate)
                    else:
                        self.context.validate_alert(
                            self.event_id, False, f"Vehicle ID {p} not found in vehicles database"
                        )
                        return
        self.is_db_valid = True
        super().on_db_update()
        if not prev_valid_db:  # alert wasn't valid due to DB issue and now it is
            self.context.validate_alert(self.event_id, True)

    def is_match(
        self, record: RecognitionHistory, candidate: AlertCandidate = None, force: bool = False
    ) -> Optional[bool]:
        if record.data is None or not record.has_l1:
            return self.unknown_subject_alert if force else None

        ptr_lst = []
        flag = None
        if self.red_list_enabled:
            ptr_lst = self.red_list
            flag = False
        elif self.green_list_enabled:
            ptr_lst = self.green_list
            flag = True

        value = record.data[0].value
        for query in ptr_lst:
            sim_score = self.lp_similarity.similarity(query, value)
            if sim_score >= self.high_conf_sim_th:
                return not flag
            elif sim_score >= self.medium_conf_sim_th:
                if not force:
                    flag = None
                elif sim_score > self.no_match_decision_th:  # lp is close enough, force to accept
                    flag = not flag
        return flag


class ContainerIdAlert(RecognitionAlert):
    type_name = "containerId"
    recognition = "serial"
    form_list_str = "containers"
    recognition_field = "serialNumber"

    def __init__(self, alert_dict: Dict, context):
        super().__init__(alert_dict, context)

        analytic_config = self.context.get_config()
        if "alertConfig" in analytic_config:
            if "lpr" in analytic_config["alertConfig"]:
                if "maxHold" in analytic_config["alertConfig"]["lpr"]:
                    self.max_hold = analytic_config["alertConfig"]["lpr"]["maxHold"]
        class_handler = self.context.get_class_handler()
        if class_handler.container_value < 0:
            self.supported_classes = [class_handler.class_str_to_int("truck")]
        else:
            self.supported_classes = class_handler.get_object_classes(class_handler.container_value)
        self.supported_classes_set = set(self.supported_classes)

    def get_batch_candidates_data(self, batch_data: BatchDataResolver) -> np.ndarray:
        candidates = (
            batch_data.unique(batch_data.ID, batch_data.SUBCLASS, list(self.supported_classes)).astype(int).tolist()
        )
        if self.alert_on_tracked_only and -1 in candidates:
            candidates.remove(-1)

        # candidates = list(self.get_batch_candidates(batch_data))
        vars_data = np.asarray([])
        if len(candidates) > 0:
            vars_data = batch_data.query(BatchDataResolver.ID, candidates)
            vars_data = self.apply_roi_filter(vars_data, BatchDataResolver.CENTER_LOCATION)
        return vars_data

    def check_alert_candidate(self, candidate: AlertCandidate, force: bool = False) -> Optional[bool]:
        ent_data = self.entity_db.get_entity(candidate.ent_ids[0])
        if ent_data is not None and ent_data.class_id not in self.supported_classes_set:
            return False
        else:
            return super().check_alert_candidate(candidate, force)

    def alert_message_from_ent(self, ent_data, candidate: AlertCandidate) -> str:
        container_serial_prefix = "Unknown Container serial number "
        if ent_data is not None:
            description = ent_data.attributes
            if "serialNumber" in description:
                container_serial_prefix = "Container with serial number " + description["serialNumber"][0].value
        return f"{container_serial_prefix} {self.alert_message}".capitalize()


def _match_scores_red(face_id, id_matrix, match_th, person_list):
    matches = []
    max_score = 0
    if id_matrix.size > 0:
        match_scores = face_id @ id_matrix.T
        sort_idx = np.argsort(match_scores)[::-1]
        matches = [person_list[idx] for idx in sort_idx if match_scores[idx] >= match_th]
        max_score = match_scores[sort_idx[0]]
    return matches, max_score


class FaceAlert(RecognitionAlert):
    type_name = "face"
    recognition = "face"
    form_list_str = "people"
    subject_mapping: Dict[str, np.array]
    min_conf: FaceConfidence = FaceConfidence.MEDIUM
    filter_driver = False
    unmatched_threshold = 0.2
    groups_field = "personGroups"
    recognition_field = "faceIdHex"
    recognition_fields = ["faceIdHex", "faceIdHexV2"]
    confidence_factors = {0: -0.1, 1: -0.05, 2: 0}
    match_th_v2 = 0.35
    max_hold = 120000  # unlimited while no face is available
    max_hold_green = 10000  # for green list, need faster response

    def __init__(self, alert_dict: Dict, context):
        self.subject_mapping = {}
        self.person_v1 = []
        self.person_v2 = []
        self.id_matrix = np.array([])
        self.id_matrix_v2 = np.array([])

        analytic_config = context.get_config()
        if "alertConfig" in analytic_config:
            face_config = analytic_config["alertConfig"].get("face", {})
            self.max_hold = face_config.get("maxHold", self.max_hold)
            self.max_hold_green = face_config.get("maxHold", self.max_hold_green)
            self.confidence_factors.update(face_config.get("confidenceFactors", {}))

        super().__init__(alert_dict, context)
        confidence = self.settings.get("confidence", 2)  # default is high confidence
        self.confidence_factor = self.confidence_factors.get(confidence, 0)

    def match_descriptor_to_subject(self, query: np.ndarray, data: FaceEntity):
        return

    def is_match(
        self, record: RecognitionHistory, candidate: AlertCandidate = None, force: bool = False
    ) -> Optional[bool]:

        if record.data is None or not record.has_l1:
            return None  # self.unknown_subject_alert is checked in higher level

        matches = []
        face_id = record.data.get(self.recognition_fields[0]).data
        face_id_v2 = record.data.get(self.recognition_fields[1]).data
        face_conf = record.extra
        conf_ok = conf_order_map[face_conf] > conf_order_map[self.min_conf]
        match_th = conf2cosine_th[face_conf] + self.confidence_factor
        match_th_v2 = 0.35
        is_match = False
        if self.red_list_enabled:
            matches_v1, max_score_v1 = _match_scores_red(face_id, self.id_matrix, match_th, self.person_v1)
            matches_v2, max_score_v2 = _match_scores_red(face_id_v2, self.id_matrix_v2, match_th_v2, self.person_v2)
            matches = matches_v2 + matches_v1
            is_match = len(matches) > 0
            max_score = max_score_v2 if len(matches_v2) else max(max_score_v1, max_score_v2)
            if not is_match and not conf_ok and max_score > self.unmatched_threshold and not force:
                is_match = None
        elif self.green_list_enabled:
            max_score_v1 = np.max(face_id @ self.id_matrix.T) if self.person_v1 else -1
            max_score_v2 = np.max(face_id_v2 @ self.id_matrix_v2.T) if self.person_v2 else -1
            is_found_v1 = match_th < max_score_v1
            is_found_v2 = match_th_v2 < max_score_v2
            is_match = not is_found_v1 and not is_found_v2
            max_score = max(max_score_v1, max_score_v2)
            if is_match and not conf_ok and max_score > self.unmatched_threshold and not force:
                is_match = None

        if len(matches):
            if candidate.extra is None:
                candidate.extra = {}
            candidate.extra["matches"] = [(subj.id, subj.name) for subj in matches]
        return is_match

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)
        person_data = alert_info.extra.get("entity_data", {})
        if "person_name" in candidate.extra:
            person_data["person_name"] = [candidate.extra.get("person_name")]
            alert_info.extra["entity_data"] = person_data
        return alert_info

    def alert_message_from_ent(self, ent_data, candidate: AlertCandidate) -> str:
        person_desc = "Unrecognized Person"

        if ent_data is not None:
            description = ent_data.attributes
            matches = candidate.extra.get("matches", [])

            if "faceIdHex" in description and len(matches):
                best_match = matches[0]
                # if self.is_linecrossing:
                #    return f"Person {best_match[1]} with ID {best_match[0]} crossed the line"
                # return f"Found Person {best_match[1]} with ID {best_match[0]}"
                person_desc = f"Person {best_match[1]}"
                candidate.extra["person_name"] = best_match[1]

        return f"{person_desc} {self.alert_message}".capitalize()

    def parse_list_item(self) -> List:
        elements = self.apply_from_dict("list", self.apply_from_dict(self.form_list_str, self.formValue, {}), [])
        return parse_face_list_items(elements)

    def on_db_update(self):
        self.group_subjects = []
        prev_valid_db = self.is_db_valid
        if self.groups:  # requires db update
            self.is_db_valid = False
            person_db = self.context.get_analytics_db().get("persons", {})
            group_db = self.context.get_analytics_db().get("persons_groups", [])

            def on_error(msg):
                self.context.validate_alert(self.event_id, False, msg)

            group_entities = load_face_groups_from_db(self.groups, person_db, group_db, on_error)
            if group_entities is None:
                return
            self.group_subjects = group_entities
        self.is_db_valid = True
        super().on_db_update()
        list_of_persons = []
        if self.red_list_enabled:
            list_of_persons = self.red_list
        elif self.green_list_enabled:
            list_of_persons = self.green_list
            self.max_hold = self.max_hold_green

        if len(list_of_persons) == 0:
            self.context.validate_alert(self.event_id, False, "No persons found in the selected list")
            return

        self.person_v1 = [subject for subject in list_of_persons if subject.faceid_version == 1]
        self.person_v2 = [subject for subject in list_of_persons if subject.faceid_version == 2]

        self.id_matrix = np.array([subject.descriptor for subject in self.person_v1])
        self.id_matrix_v2 = np.array([subject.descriptor for subject in self.person_v2])

        if not prev_valid_db:  # alert wasn't valid due to DB issue and now it is
            self.context.validate_alert(self.event_id, True)

    def get_recognition_data_from_ent(self, ent_data: ActiveEntityData):
        has_l1 = ent_data.has_l1(self.recognition_analyzer)
        rec_data = {k: ent_data.attributes.get(k) for k in self.recognition_fields if k in ent_data.attributes}
        face_conf = ent_data.attributes.get("faceConfidence", FaceConfidence.NONE)
        if has_l1 and conf_order_map[face_conf] < conf_order_map[self.min_conf]:
            face_metrics = ent_data.attributes.get("faceMetrics", [])
            face_cls = [attr_conf_to_face_conf[m.confidence] for m in face_metrics if m.value == "faceClassification"]
            if face_cls:
                face_conf = face_cls[0]
        has_l1 &= conf_order_map[face_conf] >= conf_order_map[self.min_conf]
        return RecognitionHistory(rec_data, has_l1, face_conf)
