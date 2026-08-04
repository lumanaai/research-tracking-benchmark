import json
import sqlite3
from dataclasses import dataclass
from typing import List, Type, TypeVar, Dict
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from general.analyzer_general import logger
from general.cloud_converter import cloud_TrackerTypes_converter
from general.core import ClassHandler


class PersonRepresentative(BaseModel):
    id: int
    personId: int
    faceIdBin: bytes
    orgId: str
    orgIdHash: int
    faceConfidence: int
    manualUpload: int = 0
    faceIdVersion: Optional[int] = 1


class Person(BaseModel):
    personId: int
    orgId: str
    orgIdHash: int
    bestImage: Optional[str] = None
    bestTracker: Optional[str] = None
    lastSeen: Optional[str] = None
    lastUpdate: Optional[str] = None
    name: Optional[str] = None
    isBlocked: int = 0
    consistencyMode: int = 0
    isHidden: int = 0

    @property
    def image_path(self) -> Optional[str]:
        if self.bestImage:
            return f"{self.orgId}/{self.bestImage}"
        return None


class CustomObject(BaseModel):
    id: int
    orgId: str = ""
    orgIdHash: int = -1
    name: str
    type: int
    bestImage: str
    clipImageThr: Optional[float] = None
    clipTextThr: Optional[float] = None
    reidThr: Optional[float] = None
    blipThr: Optional[float] = None
    prompt: str
    promptEncHex: Optional[bytes] = None
    lastUpdate: Optional[int] = None

    @property
    def prompt_descriptor(self) -> Optional[np.array]:
        return np.frombuffer(self.promptEncHex, dtype=np.float32) if self.promptEncHex else None

    @property
    def customObjectId(self) -> int:
        return self.id

    @property
    def is_reid_enabled(self) -> bool:
        return self.reidThr is not None and self.reidThr > 0

    @property
    def is_clip_enabled(self) -> bool:
        return self.clipImageThr is not None and self.clipImageThr > 0


class CustomObjectRepresentative(BaseModel):
    id: int
    customObjectId: int
    orgId: str = ""
    orgIdHash: int = -1
    reIdVector: bytes
    clipsVector: bytes
    reidVersion: int
    isPositive: int


class PersonGroup(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    id: str = Field(alias="_id")
    name: str
    personIds: List[int]
    createdByOrgId: str
    createdAt: int
    updatedAt: int

    @field_validator("personIds", mode="before")
    @classmethod
    def parse_person_ids(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class VehicleGroup(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    id: str = Field(alias="_id")
    name: str
    vehicleIds: List[int]
    createdByOrgId: str
    createdAt: int
    updatedAt: int

    @property
    def group_id(self) -> str:
        return self.id

    @field_validator("vehicleIds", mode="before")
    @classmethod
    def parse_vehicle_ids(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class Vehicle(BaseModel):
    id: int
    plate: str
    owner: Optional[str] = ""
    make: Optional[str] = ""
    color: Optional[str] = ""
    orgId: str
    orgIdHash: int


class ShelfData(BaseModel):
    id: int
    name: str
    cameraId: str
    location: str
    createdAt: int
    updatedAt: int
    tags: Optional[str] = None

    @property
    def location_dict(self) -> Optional[Dict]:
        if self.location:
            return json.loads(self.location)
        return None


class CustomObjectsState(BaseModel):
    id: int
    sessionId: str
    orgId: str
    orgIdHash: int
    cameraId: str
    cameraIdHash: int
    states: str  # JSON field stored as string
    ROI: str  # JSON field stored as string
    perStateThresholds: str  # JSON field stored as string
    perStateBestImages: str  # JSON field stored as string
    statusCode: int
    createdAt: int
    updatedAt: int
    templateId: Optional[int] = 0
    name: str

    @property
    def states_dict(self) -> Optional[Dict]:
        if self.states:
            return json.loads(self.states)
        return None

    @property
    def roi_dict(self) -> Optional[Dict]:
        if self.ROI:
            return json.loads(self.ROI)
        return None

    @property
    def per_state_thresholds_dict(self) -> Optional[Dict]:
        if self.perStateThresholds:
            return json.loads(self.perStateThresholds)
        return None

    @property
    def per_state_best_images_dict(self) -> Optional[Dict]:
        if self.perStateBestImages:
            return json.loads(self.perStateBestImages)
        return None


class CustomObjectsStateRepresentative(BaseModel):
    id: int
    customObjectsStateId: int
    orgId: str
    orgIdHash: int
    edgeId: str
    cameraId: str
    thumbnail: str
    thumType: str
    state: int
    embedding: bytes
    createdAt: int
    updatedAt: int

    @property
    def embedding_np(self) -> np.ndarray:
        return np.frombuffer(self.embedding, dtype=np.float32)


@dataclass
class CustomObjectData:
    object_type: int
    custom_object: CustomObject
    reid_pos_mean: np.ndarray
    reid_neg: np.ndarray
    clip_pos_mean: np.ndarray
    clip_neg: np.ndarray


@dataclass
class PersonData:
    person: Person
    pos_reps: List[PersonRepresentative]
    neg_reps: List[PersonRepresentative]


PERSON_TABLE = "persons"
PERSON_REPRESENTATIVES_TABLE = "persons_representatives"
PERSON_REPRESENTATIVES_NEGATIVE_TABLE = "persons_representatives_negative"
CUSTOM_OBJECTS_TABLE = "custom_objects"
CUSTOM_OBJECTS_REPRESENTATIVES_TABLE = "custom_objects_representatives"
SHELVES_TABLE = "shelves"
CUSTOM_OBJECTS_STATE_REPRESENTATIVE = "custom_objects_state_representative"
CUSTOM_OBJECTS_STATE = "custom_objects_state"
PERSON_GROUPS_TABLE = "person_groups"
VEHICLE_GROUPS_TABLE = "vehicle_groups"
VEHICLES_TABLE = "vehicles"

SQLITE_KNOW_TABLES: Dict[str, Type[BaseModel]] = {
    PERSON_TABLE: Person,
    PERSON_REPRESENTATIVES_TABLE: PersonRepresentative,
    PERSON_REPRESENTATIVES_NEGATIVE_TABLE: PersonRepresentative,
    CUSTOM_OBJECTS_TABLE: CustomObject,
    CUSTOM_OBJECTS_REPRESENTATIVES_TABLE: CustomObjectRepresentative,
    SHELVES_TABLE: ShelfData,
    CUSTOM_OBJECTS_STATE_REPRESENTATIVE: CustomObjectsStateRepresentative,
    CUSTOM_OBJECTS_STATE: CustomObjectsState,
    PERSON_GROUPS_TABLE: PersonGroup,
    VEHICLE_GROUPS_TABLE: VehicleGroup,
    VEHICLES_TABLE: Vehicle,
}

T = TypeVar("T", bound=BaseModel)


class SQLiteReader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def fetch_all(self, query: str, model: Type[T], params: tuple = ()) -> List[T]:
        cur = self.conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        return [model(**dict(row)) for row in rows]

    def fetch_one(self, query: str, model: Type[T], params: tuple = ()) -> Optional[T]:
        cur = self.conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        return model(**dict(row)) if row else None

    def close(self):
        self.conn.close()


def generate_unified_table(
    all_tables: Dict[str, Type[BaseModel]], class_handler: ClassHandler, camera_id, person_db_enabled: bool = False
) -> Dict:
    # rearrange custom object data
    custom_obj_tbl = all_tables.get(CUSTOM_OBJECTS_TABLE, [])
    custom_obj_reps = all_tables.get(CUSTOM_OBJECTS_REPRESENTATIVES_TABLE, [])
    custom_object_data = []

    def get_mean_descriptor(descriptor_mat: np.ndarray) -> np.ndarray:
        mean_vec = np.mean(descriptor_mat, axis=0)
        return mean_vec / np.linalg.norm(mean_vec)

    reverse_converter = {v: class_handler.object_str_to_int(k) for k, v in cloud_TrackerTypes_converter.items()}
    for obj in custom_obj_tbl:
        obj_type = reverse_converter.get(obj.type)  # tracker type id. needs conversion to object type

        pos_reps = [rep for rep in custom_obj_reps if rep.customObjectId == obj.customObjectId and rep.isPositive == 1]
        neg_reps = [rep for rep in custom_obj_reps if rep.customObjectId == obj.customObjectId and rep.isPositive == 0]
        if not pos_reps and not neg_reps:
            logger.error(f"No representatives found for custom object id {obj.customObjectId}")
            continue
        pos_reid_mat = np.array([np.frombuffer(rep.reIdVector, dtype=np.float32) for rep in pos_reps])
        neg_reid_mat = np.array([np.frombuffer(rep.reIdVector, dtype=np.float32) for rep in neg_reps])
        pos_clip_vec = np.array([np.frombuffer(rep.clipsVector, dtype=np.float32) for rep in pos_reps])
        neg_clip_vec = np.array([np.frombuffer(rep.clipsVector, dtype=np.float32) for rep in neg_reps])
        object_data = CustomObjectData(
            object_type=obj_type,
            custom_object=obj,
            reid_pos_mean=get_mean_descriptor(pos_reid_mat) if pos_reid_mat.size > 0 else np.array([]),
            reid_neg=neg_reid_mat,
            clip_pos_mean=get_mean_descriptor(pos_clip_vec) if pos_clip_vec.size > 0 else np.array([]),
            clip_neg=neg_clip_vec,
        )
        custom_object_data.append(object_data)

    person_data = {}
    vehicles_data = {}
    groups_data = []
    vehicle_groups_data = []
    if person_db_enabled:
        person_tbl = all_tables.get(PERSON_TABLE, [])
        person_reps = all_tables.get(PERSON_REPRESENTATIVES_TABLE, [])
        person_neg = all_tables.get(PERSON_REPRESENTATIVES_NEGATIVE_TABLE, [])
        for person in person_tbl:
            pos_reps = [rep for rep in person_reps if rep.personId == person.personId]
            neg_reps = [rep for rep in person_neg if rep.personId == person.personId]
            data = PersonData(
                person=person,
                pos_reps=pos_reps,
                neg_reps=neg_reps,
            )
            person_data[person.personId] = data
        groups_data = all_tables.get(PERSON_GROUPS_TABLE, [])
        vehicle_groups_data = all_tables.get(VEHICLE_GROUPS_TABLE, [])
        vehicles_data = {v.id: v for v in all_tables.get(VEHICLES_TABLE, [])}

    shelf_data = []
    obj_state = {}

    if SHELVES_TABLE in all_tables and all_tables[SHELVES_TABLE]:
        shelf_data = [s for s in all_tables[SHELVES_TABLE] if s.cameraId == camera_id]

    if CUSTOM_OBJECTS_STATE in all_tables and all_tables[CUSTOM_OBJECTS_STATE]:
        obj_state["metadata"] = [s for s in all_tables[CUSTOM_OBJECTS_STATE] if s.cameraId == camera_id]
        ids = set([s.id for s in obj_state["metadata"]])
        obj_state["representatives"] = [
            s
            for s in all_tables[CUSTOM_OBJECTS_STATE_REPRESENTATIVE]
            if s.customObjectsStateId in ids and not np.all(s.embedding_np == 0)
        ]

    db_out = {
        "persons": person_data,
        "vehicles": vehicles_data,
        "custom_objects": custom_object_data,
        "shelves": shelf_data,
        "states": obj_state,
        "persons_groups": groups_data,
        "vehicle_groups": vehicle_groups_data,
    }
    # remove empty tables for aggregations
    for k in list(db_out.keys()):
        if not db_out[k]:
            db_out.pop(k)
    return db_out
