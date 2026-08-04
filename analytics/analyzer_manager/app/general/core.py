import collections
import csv
import enum
import glob
import heapq
import logging
import logging.handlers
import math
import os
import struct
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field, fields
from typing import List, Any, Tuple, Dict, Union, Optional, Callable, Set

import numpy as np

from general import apply_from_dict
from .analyzer_general import ROI_SHAPE, logger, GLOBAL_FAILURE, GLOBAL_TYPE_KEY, MULTIPLE_ENUM, ANALYTIC_ALERT_TYPE
from .proj import info_path

AREA_FACTOR = ROI_SHAPE[0] * ROI_SHAPE[1]


class BatchDataResolver(object):
    POS = [0, 1, 2, 3]
    ID = 4
    CLASS = 5
    SUBCLASS = 6
    CONFIDENCE = 7
    DETECTOR_ID = 8
    AGE = 9
    CONFLICT = 10
    FRAME_ID = 11
    TIMESTAMP = 12
    LOCATION = 13
    CENTER = [14, 15]
    CENTER_LOCATION = 16
    AREA = 17
    INDEX = 18

    def __init__(
        self,
        tacker_results,
        image_timestamps,
        det_resolution,
        full_resolution,
        n_objects,
        is_location_center: bool = False,
    ):
        # [x,y,x,y,id,cls, subclass, conf, det_id, age, conflict, frame_id, timestamp, [location, center, center location,  area, index]]
        result_agg = np.empty((0, 13))
        im_size = np.array(det_resolution[::-1])
        for i in range(len(image_timestamps)):
            if len(tacker_results[i]) == 0:
                continue
            sample_len = tacker_results[i].shape[0]
            aggregated = np.c_[tacker_results[i], np.ones(sample_len) * i, np.ones(sample_len) * image_timestamps[i]]
            result_agg = np.vstack((result_agg, aggregated))
        result_agg[:, self.POS] = result_agg[:, self.POS] / np.array([*im_size, *im_size])
        center_x = (result_agg[:, self.POS[0]] + result_agg[:, self.POS[2]]) * ROI_SHAPE[0] / 2
        center_y = (result_agg[:, self.POS[1]] + result_agg[:, self.POS[3]]) * ROI_SHAPE[1] / 2
        center_locations = np.ravel_multi_index((center_y.astype(int), center_x.astype(int)), ROI_SHAPE, order="C")
        if is_location_center:
            locations = center_locations
        else:
            lower_y = np.minimum(ROI_SHAPE[1] - 1, (result_agg[:, self.POS[3]] * ROI_SHAPE[1]).astype(int))
            locations = np.ravel_multi_index((lower_y, center_x.astype(int)), ROI_SHAPE, order="C")
        area = (
            (result_agg[:, self.POS[2]] - result_agg[:, self.POS[0]])
            * (result_agg[:, self.POS[3]] - result_agg[:, self.POS[1]])
            * AREA_FACTOR
        )
        index = np.arange(result_agg.shape[0])
        result_agg = np.c_[result_agg, locations, center_x, center_y, center_locations, area, index]
        self.data = result_agg
        self.full_resolution = full_resolution
        self._overlaps = None
        self._all_overlaps = None
        batch_size = len(image_timestamps)
        self.object_count = np.bincount(
            (result_agg[:, self.FRAME_ID] * n_objects + result_agg[:, self.CLASS]).astype(int),
            minlength=batch_size * n_objects,
        ).reshape(batch_size, n_objects)
        self._batch_size = batch_size

    def __getitem__(self, item):
        return self.data[item, :]

    def __len__(self):
        return self.data.shape[0]

    def query(self, criteria, value, retrieve=None):
        if isinstance(value, list):
            indices = np.where(np.isin(self.data[:, criteria], value) > 0)
        else:
            indices = np.where(self.data[:, criteria] == value)
        result = self.data[indices]
        if retrieve is not None:
            retrieve = self._normalize_retrieve(retrieve)
            result = result[:, retrieve]
        return result

    def unique(self, criteria, condition=None, values=None):
        if condition is None:
            all_results = self.data[:, criteria]
        else:
            all_results = self.query(condition, values, criteria)
        return np.unique(all_results)

    def count_objects(self, zero_appearance):
        appearance = zero_appearance.copy()
        ids = set(self.data[:, self.ID].astype(int).tolist())
        for ent_id in ids:
            if ent_id < 0:
                continue
            id_data = self.query(self.ID, ent_id)
            unique, counts = np.unique(id_data[:, BatchDataResolver.CLASS], return_counts=True)
            id_class = unique[np.argmax(counts)]
            appearance[id_class] += 1
        return appearance

    def advance_and_query(self, criteria_value_list: List[tuple], retrieve=None):
        indices = set(np.arange(self.data.shape[0]))
        for tup in criteria_value_list:
            criteria, value = tup
            if isinstance(value, collections.abc.Iterable):
                curr_indices = np.where(np.isin(self.data[:, criteria], value) > 0)
            else:
                curr_indices = np.where(self.data[:, criteria] == value)
            # do intersection
            indices = indices & set(np.asarray(curr_indices[0]).flatten())
        indices = list(indices)
        result = self.data[indices]
        if retrieve is not None:
            retrieve = self._normalize_retrieve(retrieve)
            result = result[:, retrieve]
        return result

    def all(self, retrieve):
        if retrieve is not None:
            retrieve = self._normalize_retrieve(retrieve)
            result = self.data[:, retrieve]
        else:
            result = self.data
        return result

    def _calculate_overlap_ratios(self):
        # Calculate the area of the input boxes
        all_boxes = self.data[:, BatchDataResolver.POS]

        # Expand boxes to compare each box against all others
        relevant_data = self.data[self.data[:, BatchDataResolver.CLASS] < 3, :]
        frames, inds = np.unique(relevant_data[:, BatchDataResolver.FRAME_ID], return_index=True)
        inds = np.append(inds, len(relevant_data))
        n = len(self.data)
        self._overlaps = np.zeros(n)
        self._all_overlaps = np.zeros((n, n))
        for i, frame in enumerate(frames):
            idxs = set(relevant_data[inds[i] : inds[i + 1], BatchDataResolver.INDEX].astype(int))
            fr_idxs = self.data[self.data[:, BatchDataResolver.FRAME_ID] == frame, BatchDataResolver.INDEX].astype(int)
            irrelevant_idxs = [i for i, idx in enumerate(fr_idxs) if idx not in idxs]
            if len(fr_idxs) == 0:
                continue
            boxes = all_boxes[fr_idxs]
            boxes_expanded = np.expand_dims(boxes, axis=1)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            # Calculate the intersection coordinates
            x_min = np.maximum(boxes_expanded[:, :, 0], boxes[:, 0])
            y_min = np.maximum(boxes_expanded[:, :, 1], boxes[:, 1])
            x_max = np.minimum(boxes_expanded[:, :, 2], boxes[:, 2])
            y_max = np.minimum(boxes_expanded[:, :, 3], boxes[:, 3])

            # Calculate the intersection area
            intersection_area = np.clip(x_max - x_min, 0, None) * np.clip(y_max - y_min, 0, None)

            # Calculate the overlap ratio for each box against all others
            overlap_ratios = intersection_area / np.expand_dims(areas, axis=1)
            np.fill_diagonal(overlap_ratios, 0)
            overlap_ratios[:, irrelevant_idxs] = 0
            max_overlap = np.max(overlap_ratios, axis=1)
            self._overlaps[fr_idxs] = max_overlap
            self._all_overlaps[np.ix_(fr_idxs, fr_idxs)] = overlap_ratios

    @property
    def overlaps(self):
        if self._overlaps is None:
            self._calculate_overlap_ratios()
        return self._overlaps

    @property
    def all_overlaps(self):
        if self._overlaps is None:
            self._calculate_overlap_ratios()
        return self._all_overlaps

    @property
    def image_size(self):
        return self.full_resolution

    @property
    def batch_size(self):
        return self._batch_size

    @staticmethod
    def _normalize_retrieve(retrieve):
        if not isinstance(retrieve, int):

            def flatten(list_of_lists):
                if len(list_of_lists) == 0:
                    return list_of_lists
                if isinstance(list_of_lists[0], list):
                    return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
                return list_of_lists[:1] + flatten(list_of_lists[1:])

            retrieve = flatten(retrieve)
        return retrieve


BDR = BatchDataResolver  # alias


class BatchDataFactory:
    def __init__(self, context):
        self.context = context

    def create(self, tacker_results, image_timestamps) -> BatchDataResolver:
        return BatchDataResolver(
            tacker_results,
            image_timestamps,
            self.context.detector_resolution,
            self.context.resolution,
            self.context.class_handler.n_objects,
            self.context.is_location_center,
        )


class Event(object):
    def __init__(self):
        self._eventhandlers = []

    def __iadd__(self, handler):
        self._eventhandlers.append(handler)
        return self

    def __isub__(self, handler):
        self._eventhandlers.remove(handler)
        return self

    def __call__(self, *args, **kwargs):
        for eventhandler in self._eventhandlers:
            eventhandler(*args, **kwargs)


class DescriptorVector(object):
    data: np.ndarray

    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        if isinstance(other, DescriptorVector):
            return np.all(self.data == other.data)
        return False

    def descriptor_to_hex(self):
        return self.data.astype(np.float32).tobytes().hex()

    def descriptor_to_hex_struct(self):
        vector_str = "".join([str(struct.pack("f", elem)) for elem in self.data])
        return vector_str.encode("utf-8").hex()


@dataclass
class AnalyticImage(object):
    frame: Any
    timestamp: int
    frame_number: int
    processed: Any = None


class BaseConfig(dict):
    def __init__(self, args_dict: dict = None):
        super().__init__()
        self.__dict__ = self
        if args_dict is not None:
            for key in args_dict.keys():
                self[key] = args_dict[key]


class BaseWeightsConfig(BaseConfig):
    weights: str = ""  # path for builtin weight file
    unique_weights: str = ""  # path for specific external weight file
    device: str = "cuda"  # device to load weights to
    half: bool = True  # use fp 16
    is_trt: bool = False

    def init_weight(self, resource_dir, default_ext: str = ".pt"):
        default_weights = os.path.join(resource_dir, self.weights)
        if self.unique_weights:
            self.weights = self.unique_weights
        else:  # default weights
            weights = default_weights
            engine_weights_name = self.weights.split(".")[0] + "*.engine"
            engine_files = sorted(glob.glob(os.path.join(resource_dir, engine_weights_name)), key=os.path.getmtime)
            if len(engine_files) > 0:
                weights = engine_files[-1]
            self.weights = weights

        if not os.path.exists(self.weights):
            logger.error(f"Could not find weights file {self.weights}, reverting to default weights {default_weights}")
            self.weights = default_weights
        logger.info(f"using weights file {self.weights}")
        self.is_trt = self.weights.endswith(".engine")


class ClassHandler(object):
    PERSON = "person"
    VEHICLE = "vehicle"
    PET = "pet"
    WEAPON = "weapon"
    HAZARD = "hazard"
    EMPTY_SHELF = "empty_shelf"
    FACE = "face"
    SHOPPING_CART = "shoppingcart"
    BODY_PART = "bodypart"
    PLATE = "plate"
    CONTAINER = "container"

    def __init__(self, metadata_path: str = None):
        self._weapon = None
        if metadata_path is None:
            # default metadata file
            metadata_path = info_path("detection", "32cls.csv")  # latest

        with open(metadata_path, "rt") as f:
            table = csv.DictReader(f)
            metadata: List[Dict] = []
            for row in table:
                metadata.append(row)

        indices = np.array([int(class_data["index"]) for class_data in metadata])
        mapping_dst = np.array([int(class_data["object_id"]) for class_data in metadata])
        u_obj_inds = np.unique(mapping_dst, return_index=True)[1]
        mapping = np.zeros_like(indices)
        mapping[indices] = mapping_dst
        self._mapping = mapping
        self.classes = [metadata[i]["class"] for i in indices]
        self.objects = [metadata[i]["object"] for i in u_obj_inds]

        self.object_names = [metadata[i]["object"] for i in indices]
        self.confidence_thresholds = {
            "day": np.zeros_like(indices, dtype=float) - 1,
            "night": np.zeros_like(indices, dtype=float) - 1,
        }
        self.n_classes = len(self.classes)
        for i in indices:
            self.confidence_thresholds["day"][i] = float(metadata[i].get("day_threshold", -1))
            self.confidence_thresholds["night"][i] = float(metadata[i].get("night_threshold", -1))
        obj_mapping = {}
        reverse_object_mapping = {}
        for idx in u_obj_inds:
            reverse_object_mapping[mapping_dst[idx]] = metadata[idx]["object"]
            obj_mapping[metadata[idx]["object"]] = mapping_dst[idx]
        self._object_mapping = obj_mapping
        self._reverse_object_mapping = reverse_object_mapping
        self._objects_ids = [int(ind) for ind in list(reverse_object_mapping.keys()) if ind >= 0]
        self.n_objects = np.max(self._objects_ids) + 1

        def get_from_mapping(key: str):
            if key in obj_mapping:
                return int(obj_mapping[key])
            else:
                return -1

        # set known objects
        self._vehicle = get_from_mapping(ClassHandler.VEHICLE)
        self._person = get_from_mapping(ClassHandler.PERSON)
        self._pet = get_from_mapping(ClassHandler.PET)
        self._weapon = get_from_mapping(ClassHandler.WEAPON)
        self._hazard = get_from_mapping(ClassHandler.HAZARD)
        self._face = self.class_str_to_int(ClassHandler.FACE) if ClassHandler.FACE in self.classes else -1
        self._plate = self.class_str_to_int(ClassHandler.PLATE) if ClassHandler.PLATE in self.classes else -1
        self._shoppingcart = get_from_mapping(ClassHandler.SHOPPING_CART)
        self._container = get_from_mapping(ClassHandler.CONTAINER)
        self._empty_shelf = get_from_mapping(ClassHandler.EMPTY_SHELF)

        # vehicle type mapping
        vehicle_type_map_str = {}
        if "car" in self.classes:
            vehicle_type_map_str["car"] = EntityClassification.CAR
        if "bicycle" in self.classes:
            vehicle_type_map_str["bicycle"] = EntityClassification.BICYCLE
        if "motorcycle" in self.classes:
            vehicle_type_map_str["motorcycle"] = EntityClassification.MOTORCYCLE
        if "truck" in self.classes:
            vehicle_type_map_str["truck"] = EntityClassification.TRUCK
        if "bus" in self.classes:
            vehicle_type_map_str["bus"] = EntityClassification.BUS
        if "forklift" in self.classes:
            vehicle_type_map_str["forklift"] = EntityClassification.FORKLIFT
        if "boat" in self.classes:
            vehicle_type_map_str["boat"] = EntityClassification.BOAT
        self.vehicle_types = {self.classes.index(k): v for k, v in vehicle_type_map_str.items()}
        self.vehicle_string_types = vehicle_type_map_str

        object_types_map = {}
        if self._person > 0:
            object_types_map[self._person] = EntityClassification.UNKNOWN_GENDER
        if self._pet > 0:
            object_types_map[self._pet] = EntityClassification.ANIMAL
        if self._weapon > 0:
            object_types_map[self._weapon] = EntityClassification.WEAPON
        if self._hazard > 0:
            object_types_map[self._hazard] = EntityClassification.HAZARD
        if self._shoppingcart > 0:
            object_types_map[self._shoppingcart] = EntityClassification.SHOPPING_CART
        if self._container > 0:
            object_types_map[self._container] = EntityClassification.CONTAINER
        self.object_types = object_types_map

        # Map group to target category code
        self.entity_class_to_cloud_type_mapping = {
            EntityClassification.UNKNOWN: -1,
            EntityClassification.MALE: self.person_value,
            EntityClassification.FEMALE: self.person_value,
            EntityClassification.CHILD: self.person_value,
            EntityClassification.UNKNOWN_GENDER: self.person_value,
            EntityClassification.CAR: self.vehicle_value,
            EntityClassification.MOTORCYCLE: self.vehicle_value,
            EntityClassification.BICYCLE: self.vehicle_value,
            EntityClassification.BUS: self.vehicle_value,
            EntityClassification.TRUCK: self.vehicle_value,
            EntityClassification.FORKLIFT: self.vehicle_value,
            EntityClassification.BOAT: self.vehicle_value,
            EntityClassification.ANIMAL: self.pet_value,
            EntityClassification.WEAPON: self.weapon_value,
            EntityClassification.HAZARD: self.hazard_value,
            EntityClassification.SHOPPING_CART: self._shoppingcart,
            EntityClassification.CONTAINER: self.container_value,
        }.copy()

        default_entity_class_map = {}
        if self.person_value >= 0:
            default_entity_class_map[self.person_value] = EntityClassification.UNKNOWN_GENDER
        if self.vehicle_value >= 0:
            default_entity_class_map[self.vehicle_value] = EntityClassification.CAR
        if self.pet_value >= 0:
            default_entity_class_map[self.pet_value] = EntityClassification.ANIMAL
        if self.weapon_value >= 0:
            default_entity_class_map[self.weapon_value] = EntityClassification.WEAPON
        if self.hazard_value >= 0:
            default_entity_class_map[self.hazard_value] = EntityClassification.HAZARD
        if self._shoppingcart >= 0:
            default_entity_class_map[self._shoppingcart] = EntityClassification.SHOPPING_CART
        if self.container_value >= 0:
            default_entity_class_map[self.container_value] = EntityClassification.CONTAINER
        self.object_to_default_entity_class = default_entity_class_map

    def update_custom_object(self, msg):
        if "customObject" in msg:
            object_id = msg["customObject"]["objectId"]
            object_name = msg["customObject"]["objectName"]
            if object_id not in self.objects:
                self.objects.append(object_id)
                self.object_names.append(object_name)
                self._object_mapping[object_name] = object_id
                self._reverse_object_mapping[object_id] = object_name

    def is_int_an_object(self, obj: int) -> bool:
        return obj in self._reverse_object_mapping

    def class_str_to_int(self, class_str: str) -> int:
        return int(self.classes.index(class_str))

    def object_str_to_int(self, obj: str) -> int:
        return int(self._object_mapping.get(obj, -1))

    def object_int_to_str(self, obj: int) -> str:
        return self._reverse_object_mapping.get(obj, "unknown")

    def get_object_name(self, obj) -> str:
        if isinstance(obj, int):
            return self.object_int_to_str(obj)
        return obj

    def get_object_from_class(self, class_) -> int:
        if isinstance(class_, str):
            class_ = self.class_str_to_int(class_)
        return int(self._mapping[class_])

    def get_class_data(self, index: int) -> Tuple[str, str]:
        return self.classes[index], self.object_names[index]

    def is_a(self, class_ind, object_value) -> bool:
        return self._mapping[class_ind] == object_value

    def union(self, classes: List, objects: List):
        all_classes = classes
        if len(objects) > 0:
            if isinstance(objects[0], str):
                objects_inds = [i for i in range(self.n_classes) if self.object_names[i] in objects]
            else:
                objects_inds = [i for i in range(self.n_classes) if self._mapping[i] in objects]
            all_classes += objects_inds
        return np.unique(all_classes)

    def intersection(self, classes: List, objects: List):
        all_classes = []
        for class_ind in classes:
            if self._mapping[class_ind] in objects:
                all_classes.append(class_ind)

        return all_classes

    def filter_objects(self, classes: List, objects: List):
        all_classes = []
        for obj in objects:
            obj_classes = np.argwhere(self._mapping == obj).flatten()
            intersection = np.intersect1d(obj_classes, classes)
            if len(intersection) > 0:
                all_classes += intersection.tolist()
            else:
                all_classes += obj_classes.tolist()
        return all_classes

    def filter_out_objects(self, objects_to_filter: List) -> List[int]:
        obj_set = set(objects_to_filter)
        return [i for i, obj in enumerate(self._mapping) if obj not in obj_set and obj >= 0]

    @property
    def person_value(self):
        return self._person

    @property
    def vehicle_value(self):
        return self._vehicle

    @property
    def pet_value(self):
        return self._pet

    @property
    def weapon_value(self):
        return self._weapon

    @property
    def hazard_value(self):
        return self._hazard

    @property
    def container_value(self):
        return self._container

    @property
    def face_subclass_value(self):
        return self._face

    @property
    def plate_subclass_value(self):
        return self._plate

    @property
    def num_classes(self):
        return self.n_classes

    @property
    def classes_names(self):
        return self.classes

    @property
    def objects_names(self):
        return self.object_names

    @property
    def class_to_object_mapping(self):
        return self._mapping

    @property
    def object_ids(self):
        return self._objects_ids

    def get_object_classes(self, object_id) -> List[int]:
        return np.argwhere(self._mapping == object_id).flatten().tolist()


class CommProtocol(int, enum.Enum):
    HTTP = 0
    UDP = 1
    TCP = 2
    BOLD = 3
    CEF = 4


class HttpMethods(int, enum.Enum):
    POST = 0
    GET = 1
    PUT = 2
    DELETE = 3
    PATCH = 4
    OPTIONS = 5


class AlertStatus(int, enum.Enum):
    DISABLED = 0
    IDLE = 1
    ACTIVE = 2
    BLOCKEDOUT = 3


# Alert category / type enums + the ``alert_mapping`` LUT live in a standalone
# module so lightweight consumers (e.g. ``local_vcc``) can import them without
# pulling in the rest of ``general.core``. Re-exported here for backwards
# compatibility with existing ``from .core import AlertType`` style imports.
from .alert_types import (  # noqa: F401,E402
    EVENT_TYPE_FACTOR,
    AlertCategory,
    SafetyType,
    IdentificationType,
    TrackingType,
    StatusType,
    CustomizedCapabilitiesType,
    IntegrationsAlertType,
    RetailAlertType,
    ObjectStateAlertType,
    ProtectedGearWearOptions,
    ProtectedGearType,
    AlertType,
    alert_mapping,
    event_type_to_alert_type,
    split_event_type,
)


class SpecialClasses(enum.Enum):
    SHOPPING_CART = "shoppingcart"
    FORKLIFT = "forklift"


class AlertsAction(enum.Enum):
    add = 0
    remove = 1
    update = 2


class AnalyticMode(enum.Enum):
    enabled = 0
    thumbnails_only = 1
    disabled = 2


class DashboardsAction(enum.Enum):
    add = 0
    remove = 1
    update = 2


class DashboardsType(str, enum.Enum):
    shelves = "shelves"
    counting = "counting"


class ThumbnailType(enum.IntEnum):
    Thumbnail = 0
    Alert = 1
    Training = 2
    Crop = 3
    Snapshot = 4


class AlertsConfidence(enum.Enum):
    low = 0
    medium = 1
    high = 2


class AlertRouting(enum.Enum):
    """Routing of the alert"""

    NO_ROUTING = 0
    ROUTE_VCC_DEFAULT_TRUE = 1
    ROUTE_VCC_DEFAULT_FALSE = 2
    ROUTE_VCC_INTERNAL = 3
    MANUAL_OLD = 4
    MANUAL = 10
    MANUAL_ROUTE_VCC_DEFAULT_TRUE = 11
    MANUAL_ROUTE_VCC_DEFAULT_FALSE = 12
    MANUAL_ROUTE_VCC_INTERNAL = 13


class ActionType(enum.Enum):
    NOTIFY = 0
    SEND_MESSAGE = 1
    GPIO = 2
    SPEAKER = 3


class EntityClassification(enum.IntEnum):
    """general gross classification for entities"""

    UNKNOWN = 0
    MALE = 1
    FEMALE = 2
    CHILD = 4
    UNKNOWN_GENDER = 8
    CAR = 128
    MOTORCYCLE = 256
    BICYCLE = 512
    BUS = 1024
    TRUCK = 2048
    FORKLIFT = 4096
    BOAT = 8192
    ANIMAL = 32768
    WEAPON = 131072
    HAZARD = 524288
    SHOPPING_CART = 2097152
    CONTAINER = 8388608


def bold_builder(location, camera, event, bold_config):
    url = "https://sdksupport.boldgroup.solutions/integration/Lumix/signal"

    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
    }
    data = {
        "csid": bold_config["csid"],
        "accountId": bold_config["accountId"],
        "event": event,
        "area": location,
        "zone": camera,
        "url": bold_config["url"],
    }
    return Namespace(protocol=CommProtocol.BOLD.value, url=url, header=headers, body=data)


class RawTCPHandler(logging.handlers.SocketHandler):
    def makePickle(self, record):
        return (self.format(record) + "\n").encode("utf-8")


def cef_builder(value, location, camera, event):

    # Create a cef_logger
    cef_logger = logging.getLogger("CEFLogger")
    cef_logger.setLevel(logging.INFO)

    address_split = value["address"].split(":")
    syslog_host = address_split[0]
    if len(address_split) > 1:
        syslog_port = int(address_split[1].split("/")[0])
    else:
        syslog_port = 514

    if int(value["protocol"]) == CommProtocol.TCP.value:
        handler = RawTCPHandler(syslog_host, syslog_port)
    else:
        handler = logging.handlers.SysLogHandler(address=(syslog_host, syslog_port))
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    cef_logger.handlers = []
    cef_logger.propagate = False
    cef_logger.addHandler(handler)

    data_header = "CEF:0|Lumana|Alert|1.0|" + event + "|"
    data_footer = "|location=" + location + " camera=" + camera
    return Namespace(
        protocol=CommProtocol.CEF.value, cef_logger=cef_logger, data_header=data_header, data_footer=data_footer
    )


def msg_builder(value):
    url: str = value["address"]
    msg = apply_from_dict("msg", value, "")
    body = apply_from_dict("body", value, {})
    if int(value["protocol"]) == CommProtocol.HTTP.value:
        return Namespace(protocol=CommProtocol.HTTP.value, url=url, method=value["method"], body=body)
    else:
        try:
            trafficController = apply_from_dict("trafficController", value, False)
            address_split = value["address"].split(":")
            address = address_split[0]
            if len(address_split) > 1:
                port = int(address_split[1].split("/")[0])
                if len(address_split[1].split("/")) > 1:
                    stream = address_split[1].split("/")[1]
                else:
                    stream = ""
                # TODO: support stream
            else:
                port = 443

            return Namespace(
                protocol=int(value["protocol"]),
                address=address,
                port=port,
                msg=msg,
                body=body,
                trafficController=trafficController,
            )
        except Exception as e:
            logger.error("wrong configuration of communication protocol, attempting to convert to http")

        # if we got here the protocol configuration is incorrect
        if url.startswith("http"):
            return Namespace(protocol=CommProtocol.HTTP.value, url=url, method=value["method"], body=body)

    # if everything fails return none so it could be ignored
    return None


@dataclass
class Metadata:
    thumbnails: List = field(default_factory=lambda: [])
    info: Dict = field(default_factory=lambda: None)
    training: List = field(default_factory=lambda: [])
    snapshots: List = field(default_factory=lambda: [])
    trafficMsgs: List = field(default_factory=lambda: [])


@dataclass
class AlertInfo:
    eventId: str  # event_id
    detectionType: int  # AlertType.value
    type: int = MULTIPLE_ENUM
    alertType: int = ANALYTIC_ALERT_TYPE
    timestamp: int = -1
    idIndex: Optional[int] = -1
    idBase: Optional[int] = -1
    thumbnails: List = field(default_factory=lambda: [])
    snapshots: List = field(default_factory=lambda: [])
    validationThumbnails: List = field(default_factory=lambda: [])
    mainThumbnail: Optional[str] = None
    mainThumbnailTimestamp: int = -1
    alertMessage: Optional[str] = ""
    debugData: Optional[str] = ""
    alertData: Optional = None
    priority: int = -1
    flowType: int = -1
    category: int = -1
    confidence: int = 5
    perimeter: Optional[Dict] = None
    internal: int = 0
    routing: int = 0
    trackerClass: int = -1
    hasLocalActions: bool = False
    accountId: Optional[str] = None
    partitionId: Optional[str] = None
    policyCode: Optional[int] = None
    storeLocalThumbnail: bool = (False,)
    extra: Optional[Dict] = field(default_factory=lambda: {})
    specialFilter: Optional[str] = None
    showOnAlertPage: int = 1
    showOnWalls: int = 1
    clearEvent: Optional[bool] = None
    priorityNumber: Optional[int] = None

    _actions: Dict[str, List] = field(default_factory=lambda: {})
    _variableAlert: bool = False
    _crops: List = field(default_factory=lambda: [])
    _validation_images: List = field(default_factory=lambda: [])

    @property
    def object_id(self):
        return self.type if self.type != MULTIPLE_ENUM else None

    @object_id.setter
    def object_id(self, value: Optional[int]):
        if value is not None:
            self.type = value

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}

    @property
    def crops(self):
        return self._crops

    @crops.setter
    def crops(self, value: List):
        self._crops = value

    @property
    def validation_images(self):
        return self._validation_images

    @validation_images.setter
    def validation_images(self, value: List):
        self._validation_images = value


class VccServiceType(enum.Enum):
    """VCC service type"""

    VCC_SERVICE_CUSTOM = 0
    VCC_SERVICE_SAFETY = 1
    VCC_SERVICE_SPECIAL_FILTER = 2

    @staticmethod
    def service_from_alert_info(alert_info: AlertInfo):
        if alert_info.category == AlertCategory.Safety.value:
            if alert_info.flowType in [SafetyType.Weapon.value, SafetyType.Fire.value, SafetyType.Fall.value]:
                return VccServiceType.VCC_SERVICE_SAFETY

        if alert_info.specialFilter is not None:
            return VccServiceType.VCC_SERVICE_SPECIAL_FILTER
        return VccServiceType.VCC_SERVICE_CUSTOM


@dataclass
class ThumbnailInfo:
    thumbnail: str
    timestamp: int
    frame_number: int
    normalizedTimestamp: Optional[int] = None
    annotations: List = field(default_factory=lambda: [])
    thumType: int = ThumbnailType.Thumbnail.value


class MotionData:
    per_frame_motion: List[Optional[np.array]]
    per_frame_index: List[int]
    unified: np.ndarray = None
    _index: Optional[int] = None
    _max: Optional[int] = None

    def __init__(self, motion_per_frame, motion_index, motion_max):
        self.per_frame_motion = motion_per_frame
        self.per_frame_index = motion_index
        self._index = np.max(motion_index)
        self._max = np.max(motion_max)
        if self._max > 0:
            motion_lst = [mv for mv in motion_per_frame if mv is not None]
            self.unified = np.max(np.stack(motion_lst, axis=2), axis=2)
        else:
            self.unified = np.zeros(ROI_SHAPE, dtype=np.uint8)

    @property
    def index(self) -> int:
        return self._index

    @property
    def max(self) -> int:
        return self._max


class MotionExtractorType(str, enum.Enum):
    CV2 = "cv2"
    IM_DIFF = "im-diff"


class AdvanceAnalyzerType(str, enum.Enum):
    BASE = "base"
    ALPR = "alpr"
    HUMAN_PARSING = "human_parsing"
    CLOTHES = "clothes"
    WEAPONS = "weapons_classification"
    FACE = "face"
    VEHICLE = "vehicle"
    LPREC = "license-plate"
    CONTAINER = "container"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


advance_analyzer_priority = {
    AdvanceAnalyzerType.BASE: 0,
    AdvanceAnalyzerType.CLOTHES: 1,
    AdvanceAnalyzerType.FACE: 2,
    AdvanceAnalyzerType.HUMAN_PARSING: 3,
    AdvanceAnalyzerType.VEHICLE: 4,
    AdvanceAnalyzerType.LPREC: 5,
    AdvanceAnalyzerType.CONTAINER: 6,
    AdvanceAnalyzerType.ALPR: 7,
    AdvanceAnalyzerType.WEAPONS: 8,
}


class AdvanceAnalyticResults(int, enum.Enum):
    NO_VALID_CANDIDATE = -2
    FAILURE = -1
    NOT_GOOD = 0
    SUCCESS = 1


class AdvanceAnalyticPriority(int, enum.Enum):
    LOW = 0
    REQUIRED = 1
    HIGH = 2
    IMMEDIATE = 3


class AttrConfidence(int, enum.Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class AttrProperty:
    export_behavior_legacy = False
    value: str
    score: float
    confidence: AttrConfidence

    def __init__(self, value: str, score: float, confidence: AttrConfidence, priority: int = 0):

        self.value = value
        self.score = score
        self.confidence = confidence
        self.priority = priority

    def __str__(self):
        return f"{self.value}: {self.score} with conf {self.confidence} and priority {self.priority}"

    def __repr__(self):
        return f"AttrProperty(value={self.value!r}, score={self.score!r}, confidence={self.confidence!r})"

    def export(self):
        if AttrProperty.export_behavior_legacy:
            return self.value
        else:
            return {"value": self.value, "score": self.score, "confidence": self.confidence.value}

    def __eq__(self, other):
        if isinstance(other, AttrProperty):
            return self.value == other.value and self.score == other.score and self.confidence == other.confidence
        return False


def is_l1_not_failure(attr_dict: Dict[str, AttrProperty]) -> bool:
    if GLOBAL_TYPE_KEY in attr_dict:
        return attr_dict[GLOBAL_TYPE_KEY].value != GLOBAL_FAILURE
    return True


@dataclass
class WorkItem:
    id: Union[int, str]
    image: np.array
    extra: Any = None


def atleast_2d(arr: np.ndarray) -> np.ndarray:
    if len(arr.shape) < 2:
        arr = arr[:, np.newaxis]
    return arr


class PostInitMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__()
        return instance


class BaseInfer(metaclass=PostInitMeta):
    is_dynamic: bool = False
    batch_size: int
    data_type: Any
    output_shapes: Dict[str, List]
    im_size: List[int]
    concat_func: Callable

    def __post_init__(self):
        self.concat_func = lambda x: x

    def infer(self, data: List[np.array]):
        n_input = len(data)
        iterations = math.ceil(len(data) / self.batch_size)
        reminder = n_input % self.batch_size
        if reminder > 0 and not self.is_dynamic:
            data = np.array(data)
            data = np.vstack((data, np.zeros((self.batch_size - reminder,) + data.shape[1:], dtype=self.data_type)))
        outputs = {}

        if self.is_dynamic:
            for output in self.output_shapes:
                outputs[output] = atleast_2d(np.empty(([0] + self.output_shapes[output][1:]), dtype=self.data_type))
            for i in range(iterations):
                stack = np.stack(data[i * self.batch_size : (i + 1) * self.batch_size], axis=0).astype(self.data_type)
                results = self.run_engine(stack)
                for output in self.output_shapes:
                    if self.is_dynamic:
                        outputs[output] = np.vstack((outputs[output], atleast_2d(self.get_results(results, output))))
            for output in self.output_shapes:
                outputs[output] = outputs[output][0:n_input]
        else:
            for output in self.output_shapes:
                outputs[output] = np.empty(([0] + self.output_shapes[output][:]), dtype=self.data_type)
            for i in range(iterations):
                stack = np.stack(data[i * self.batch_size : (i + 1) * self.batch_size], axis=0).astype(self.data_type)
                results = self.run_engine(stack)
                for output in self.output_shapes:
                    outputs[output] = np.vstack((outputs[output], self.get_results(results, output)[np.newaxis, :]))
            for output in self.output_shapes:
                outputs[output] = self.concat_func(outputs[output])[0:n_input]

        output_results = list(outputs.values())
        return output_results[0] if len(output_results) == 1 else tuple(output_results)

    def run_engine(self, crop_stack: np.ndarray):
        pass

    def get_results(self, results, output_name: str) -> np.ndarray:
        pass

    @property
    def is_half(self):
        return self.data_type == np.float16


@dataclass
class EndpointInfo:
    model_name: str
    model_version: int = 1
    url: str = "0.0.0.0:8001"
    timeout: Optional[float] = None
    verbose: bool = False
    protocol: str = "grpc"


class EndpointFactory(object):
    _instance = None
    _url: str
    _protocol: str

    def __new__(cls, init_msg: dict = None):
        if cls._instance is None:
            cls._instance = super(EndpointFactory, cls).__new__(cls)
            if init_msg is None:
                # read from config file
                from .proj import load_config

                app_config = load_config()
                init_msg = app_config.get("analytics", {}).get("inferenceServerUri", {})
            url = init_msg.get("uri", None)
            protocol = init_msg.get("protocol", None)
            cls._instance._url = url
            cls._instance._protocol = protocol
        return cls._instance

    def create(self, model_name: str) -> EndpointInfo:
        endpoint = EndpointInfo(model_name)
        endpoint.protocol = self._protocol
        endpoint.url = self._instance._url
        return endpoint


# default encoder for sending jsons
def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, DescriptorVector):
        return obj.descriptor_to_hex()
    elif isinstance(obj, AttrProperty):
        return obj.export()


def bbox_to_location(bbox_tlbr: np.ndarray) -> List[int]:
    bboxes = atleast_2d(bbox_tlbr)
    center_x = (bboxes[:, 0] + bboxes[:, 2]) * ROI_SHAPE[0] / 2
    lower_y = np.minimum(ROI_SHAPE[1] - 1, (bboxes[:, 3] * ROI_SHAPE[1]).astype(int))
    return np.ravel_multi_index((lower_y, center_x.astype(int)), ROI_SHAPE, order="C").astype(int).tolist()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=0):
    e_x = np.exp(x)  # Subtracting np.max(x) for numerical stability
    return e_x / e_x.sum(axis=axis, keepdims=True)


def calc_max_overlap(boxes: np.ndarray) -> np.ndarray:
    boxes_expanded = np.expand_dims(boxes, axis=1)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate the intersection coordinates
    x_min = np.maximum(boxes_expanded[:, :, 0], boxes[:, 0])
    y_min = np.maximum(boxes_expanded[:, :, 1], boxes[:, 1])
    x_max = np.minimum(boxes_expanded[:, :, 2], boxes[:, 2])
    y_max = np.minimum(boxes_expanded[:, :, 3], boxes[:, 3])

    # Calculate the intersection area
    intersection_area = np.clip(x_max - x_min, 0, None) * np.clip(y_max - y_min, 0, None)

    # Calculate the overlap ratio for each box against all others
    overlap_ratios = intersection_area / np.expand_dims(areas, axis=1)
    np.fill_diagonal(overlap_ratios, 0)
    return overlap_ratios


def calculate_overlap_matrix(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:

    # Split into separate components for readability
    left_a, top_a, right_a, bottom_a = group_a[:, 0], group_a[:, 1], group_a[:, 2], group_a[:, 3]
    left_b, top_b, right_b, bottom_b = group_b[:, 0], group_b[:, 1], group_b[:, 2], group_b[:, 3]

    # Calculate intersection coordinates for all pairs
    inter_top = np.maximum(top_a[:, None], top_b)  # Shape (n_a, n_b)
    inter_left = np.maximum(left_a[:, None], left_b)  # Shape (n_a, n_b)
    inter_bottom = np.minimum(bottom_a[:, None], bottom_b)  # Shape (n_a, n_b)
    inter_right = np.minimum(right_a[:, None], right_b)  # Shape (n_a, n_b)

    # Compute intersection width, height, and area
    inter_width = np.maximum(0, inter_right - inter_left)  # Shape (n_a, n_b)
    inter_height = np.maximum(0, inter_bottom - inter_top)  # Shape (n_a, n_b)
    intersection_area = inter_width * inter_height  # Shape (n_a, n_b)

    # Compute area of each bounding box in group_a and group_b
    area_a = (bottom_a - top_a) * (right_a - left_a)  # Shape (n_a,)

    # Calculate overlap percentage as (intersection / area)
    return intersection_area / area_a.reshape(-1, 1)


def hex_to_desc(hex_string: str):
    return np.frombuffer(bytes.fromhex(hex_string), dtype=np.float32)


def divide_n_into_s_parts_inclusive(N, s):

    # Adjust for the inclusion of 0 and N by dividing the space between them
    interval_count = s - 1  # Number of intervals between 0 and N
    part_size = N // interval_count
    remainder = N % interval_count

    # Create the parts list
    parts = [0]  # Start with 0
    current_value = -1
    for i in range(1, interval_count + 1):
        current_value += part_size + (1 if i <= remainder else 0)
        parts.append(current_value)

    return parts


def medoid(array: np.ndarray) -> int:
    """
    Computes the medoid of a set of points.

    Args:
        array (np.ndarray): A 2D array where each row represents a point.

    Returns:
        int: index of medoid
    """
    # Compute pairwise distances (Euclidean distance)
    distances = np.linalg.norm(array[:, np.newaxis] - array, axis=2)

    # Sum distances for each point
    total_distances = distances.sum(axis=1)

    # Find the index of the medoid (minimum total distance)

    return np.argmin(total_distances)


class BoundedPriorityQueue:
    def __init__(self, maxsize: int):
        self.maxsize: int = maxsize
        self._queue = []
        self._counter: int = 0  # unique sequence count

    def push_pop(self, item: Any, priority: float) -> Tuple[bool, Any]:
        # Use negative priority for max-heap, counter for tie-breaker
        self._counter += 1
        poped_item = None
        is_pushed = True
        entry = (-priority, self._counter, item)
        if len(self._queue) < self.maxsize:
            heapq.heappush(self._queue, entry)
        else:
            if entry > self._queue[0]:
                _, _, poped_item = heapq.heappushpop(self._queue, entry)
            else:
                is_pushed = False
        return is_pushed, poped_item

    def pop(self):
        if self._queue:
            _, _, item = heapq.heappop(self._queue)
            return item
        return None

    def __len__(self):
        return len(self._queue)

    def __iter__(self):
        return (item for _, _, item in sorted(self._queue, reverse=True))


class MemoryHandler:
    def __init__(self, num_buffers, num_crops, crop_size: List[int] = None, dtype=np.uint8):
        self._num_buffers = num_buffers
        self._num_crops = num_crops
        self._data = np.empty([num_buffers, num_crops, *crop_size], dtype=dtype)
        self._free = set(list(range(num_buffers)))
        self._crop_index = np.zeros(num_buffers, dtype=int)

    def acquire(self) -> int:
        if len(self._free) == 0:
            return -1  # no free buffers
        idx = self._free.pop()
        return idx

    def acquire_multiple(self, count: int) -> Optional[List[int]]:
        idxs = []
        if len(self._free) < count:
            return None  # not enough free buffers
        for _ in range(count):
            idxs.append(self._free.pop())
        return idxs

    def release(self, idx: int):
        self._free.add(idx)
        self._crop_index[idx] = 0

    def release_multiple(self, idxs: List[int]):
        for idx in idxs:
            self.release(idx)

    def next(self, idx: int) -> np.ndarray:
        crop_idx = self._crop_index[idx] % self._num_crops
        array = self._data[idx, crop_idx]
        self._crop_index[idx] += 1
        return array

    def extend_num_buffers(self, new_max: int):
        if new_max <= self._num_buffers:
            return
        new_data = np.empty([new_max, self._num_crops, *self._data.shape[2:]], dtype=self._data.dtype)
        new_data[: self._num_buffers] = self._data
        self._data = new_data
        for idx in range(self._num_buffers, new_max):
            self._free.add(idx)

        crop_index = np.zeros(new_max, dtype=int)
        crop_index[: self._num_buffers] = self._crop_index
        self._crop_index = crop_index

        self._num_buffers = new_max


class ObjectManager(ABC):

    @abstractmethod
    def track(self, batch_data: BDR, image_batch: List[AnalyticImage], motion_data: MotionData):
        pass

    @abstractmethod
    def sync_and_cleanup(self, base: int) -> Dict:
        pass

    @property
    @abstractmethod
    def enabled(self) -> bool:
        return False


class ZoneRegion:
    zone_region_id: str
    version: int
    id: str
    polygons: List[List[List[float]]]
    roi: Set[int]

    def __init__(self, init_dict: Dict):
        self.zone_region_id = init_dict["zoneRegionId"]
        self.id = init_dict["_id"]
        data = init_dict["data"]
        self.polygons = []
        self.roi = set()
        if not data.get("fullObjectView", False):
            zones = data.get("zones", [])
            polygon = []
            for z in zones:
                for pt in z["selection"]:
                    polygon.append([pt["x"], pt["y"]])
                self.polygons.append(polygon)
            self.roi.update(set(data.get("markedIdx", [])))
