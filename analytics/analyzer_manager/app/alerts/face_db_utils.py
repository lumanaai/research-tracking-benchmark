from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import numpy as np

from general.core import hex_to_desc


@dataclass
class FaceEntity:
    """Unified face entity representation used across face alerts."""

    id: int
    name: str
    descriptor: np.ndarray
    bestImage: Optional[str] = None
    faceid_version: int = 0  # 0 means auto-detect from descriptor length

    def __post_init__(self):
        if self.faceid_version == 0:
            self.faceid_version = 2 if len(self.descriptor) == 512 else 1


@dataclass
class FaceDBResult:
    """Result of loading faces from form list and/or DB groups."""

    entities_v1: List[FaceEntity] = field(default_factory=list)
    entities_v2: List[FaceEntity] = field(default_factory=list)
    matrix_v1: np.ndarray = field(default_factory=lambda: np.array([]))
    matrix_v2: np.ndarray = field(default_factory=lambda: np.array([]))
    id_to_name: Dict[int, str] = field(default_factory=dict)
    id_to_best_image: Dict[int, str] = field(default_factory=dict)

    @property
    def all_entities(self) -> List[FaceEntity]:
        return self.entities_v1 + self.entities_v2


def parse_face_list_items(elements: List[dict]) -> List[FaceEntity]:
    """
    Parse face list items from form value into FaceEntity objects.
    Supports multiple formats: zoomImageHex, representativesHex, representatives.
    """
    subjects = []
    for element in elements:
        person_id = element.get("personId", element.get("id"))
        name = element.get("name", "")
        best_image = element.get("bestImage")

        if "zoomImageHex" in element:
            subjects.append(FaceEntity(person_id, name, hex_to_desc(element["zoomImageHex"]), best_image))
        elif "representativesHex" in element:
            for rep in element["representativesHex"]:
                subjects.append(FaceEntity(person_id, name, hex_to_desc(rep), best_image))
        elif "representatives" in element:
            faceid_version = element.get("faceIdVersion", 0)
            for rep in element["representatives"]:
                subjects.append(
                    FaceEntity(
                        id=person_id,
                        name=name,
                        descriptor=hex_to_desc(rep),
                        bestImage=best_image,
                        faceid_version=faceid_version,
                    )
                )
        else:
            raise ValueError(f"Face Alert: unknown list element structure: {element}")
    return subjects


def load_face_groups_from_db(
    groups: List[dict],
    person_db: Dict,
    group_db: List,
    error_callback: Optional[Callable[[str], None]] = None,
) -> Optional[List[FaceEntity]]:
    """
    Load face entities from person groups in the DB.

    Args:
        groups: List of group dicts from form value (each has 'groupId' or 'id' key).
        person_db: Dict mapping personId -> PersonData (with .pos_reps and .person).
        group_db: List of PersonGroup objects (with .id and .personIds).
        error_callback: Optional callback for reporting errors. Called with error message.
            If provided and an error occurs, returns None to signal failure.

    Returns:
        List of FaceEntity objects, or None if an error occurred.
    """
    subjects = []
    existing_persons = set()

    for g in groups:
        gid = g.get("groupId", g.get("id"))
        group_data = [gd for gd in group_db if gd.id == gid]
        if len(group_data) != 1:
            if error_callback:
                error_callback(f"Group ID {gid} not found in persons_groups database")
            return None
        group_persons = group_data[0].personIds
        if not group_persons:
            if error_callback:
                error_callback(f"Group ID {gid} has no persons")
            return None
        for p in group_persons:
            if p in existing_persons:
                continue
            if p in person_db:
                person_data = person_db[p]
                if len(person_data.pos_reps) == 0:
                    if error_callback:
                        error_callback(f"Person ID {p} has no representatives")
                    return None
                for rep in person_data.pos_reps:
                    subjects.append(
                        FaceEntity(
                            p,
                            person_data.person.name,
                            np.frombuffer(rep.faceIdBin, dtype=np.float32),
                            person_data.person.image_path,
                            rep.faceIdVersion,
                        )
                    )
                existing_persons.add(p)
            else:
                if error_callback:
                    error_callback(f"Person ID {p} not found in persons database")
                return None
    return subjects


def build_face_db_result(entities: List[FaceEntity]) -> FaceDBResult:
    """
    Split face entities by version and build matrices and lookup dicts.
    """
    result = FaceDBResult()
    result.entities_v1 = [e for e in entities if e.faceid_version == 1]
    result.entities_v2 = [e for e in entities if e.faceid_version == 2]
    result.matrix_v1 = np.array([e.descriptor for e in result.entities_v1]) if result.entities_v1 else np.array([])
    result.matrix_v2 = np.array([e.descriptor for e in result.entities_v2]) if result.entities_v2 else np.array([])

    for e in entities:
        result.id_to_name[e.id] = e.name
        result.id_to_best_image[e.id] = e.bestImage
    return result
