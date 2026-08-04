"""
BIC (ISO 6346) container ID classification and validation.

Classifies raw OCR text strings into BIC segment fields (ownerCode, serialNumber, sizeCode)
and validates using the ISO 6346 modulo-11 check digit algorithm.

BIC format: AAAU1234567
  - AAA    = owner/operator code (3 uppercase Latin letters, BIC-registered)
  - U      = equipment category (U / J / Z)
  - 1234567 = serial number + check digit (7 digits, stored together as serialNumber field)
             ISO 6346 splits this as 6-digit serial + 1-digit check, but this implementation
             stores all 7 digits as one serialNumber value and validates via validate_check_digit()

Full regex: ^[A-Z]{3}[UJZ][0-9]{7}$
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BicField(str, Enum):
    OWNER_CODE = "ownerCode"
    SERIAL_NUMBER = "serialNumber"
    SIZE_CODE = "sizeCode"


class BicMatchType(str, Enum):
    FULL_BIC = "full_bic"
    OWNER_CATEGORY = "owner_category"
    SERIAL_CHECK = "serial_check"
    PARTIAL_OWNER = "partial_owner"
    PARTIAL_SERIAL = "partial_serial"
    SIZE_TYPE = "size_type"
    GARBAGE = "garbage"


# ISO 6346 character value mapping
# Letters: A=10, B=12, C=13, D=14, E=15, F=16, G=17, H=18, I=19, J=20, K=21,
#          L=23, M=24, N=25, O=26, P=27, Q=28, R=29, S=30, T=31, U=32, V=34,
#          W=35, X=36, Y=37, Z=38
# (Multiples of 11 are skipped: 11, 22, 33)
_ISO6346_CHAR_VALUES = {}
_val = 10
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _ISO6346_CHAR_VALUES[_c] = _val
    _val += 1
    if _val % 11 == 0:
        _val += 1
for _d in "0123456789":
    _ISO6346_CHAR_VALUES[_d] = int(_d)

# Compiled regex patterns
_RE_FULL_BIC = re.compile(r"^[A-Z]{3}[UJZ]\d{7}$")
_RE_OWNER_CATEGORY = re.compile(r"^[A-Z]{3}[UJZ]$")
_RE_SERIAL_CHECK = re.compile(r"^\d{7}$")
_RE_PARTIAL_OWNER = re.compile(r"^[A-Z]{3}$")
_RE_PARTIAL_SERIAL = re.compile(r"^\d{6}$")
# ISO 6346 size/type code: 2 digits + 1 letter + 1 alphanumeric (e.g. 45G1, 22R1)
_RE_SIZE_TYPE = re.compile(r"^\d{2}[A-Z][A-Z0-9]$")

# Confidence multipliers per match type
_MATCH_CONFIDENCE = {
    BicMatchType.FULL_BIC: 1.0,
    BicMatchType.OWNER_CATEGORY: 1.0,
    BicMatchType.SERIAL_CHECK: 1.0,
    BicMatchType.PARTIAL_OWNER: 0.8,
    BicMatchType.PARTIAL_SERIAL: 0.8,
    BicMatchType.SIZE_TYPE: 0.9,
}

# Precomputed powers of 2 for ISO 6346 check digit (positions 0-9)
_POWERS_OF_2 = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# Check digit validation multipliers
CHECK_DIGIT_PASS_MULTIPLIER = 1.0
CHECK_DIGIT_FAIL_MULTIPLIER = 0.7


@dataclass
class BicClassification:
    """Result of classifying a single OCR string."""
    match_type: BicMatchType
    fields: List[Tuple[BicField, str]]  # list of (field, value) pairs extracted
    confidence_multiplier: float = 1.0
    check_digit_valid: Optional[bool] = None  # None if not applicable


def canonicalize_text(raw: str) -> str:
    """Canonicalize OCR text: uppercase, strip non-alnum, collapse whitespace."""
    upper = raw.upper()
    cleaned = re.sub(r"[^A-Z0-9]", "", upper)
    return cleaned


def compute_check_digit(code_10: str) -> int:
    """
    Compute ISO 6346 check digit from the first 10 characters of a BIC code.

    Args:
        code_10: First 10 characters (3 letters + category + 6 digits).

    Returns:
        Expected check digit (0-9), or -1 if input is invalid.
    """
    if len(code_10) != 10:
        return -1
    total = 0
    for i, ch in enumerate(code_10):
        val = _ISO6346_CHAR_VALUES.get(ch)
        if val is None:
            return -1
        total += val * _POWERS_OF_2[i]
    return (total % 11) % 10


def validate_check_digit(full_bic: str) -> bool:
    """
    Validate a full 11-character BIC code's check digit.

    Returns True if the check digit (11th character) matches the computed value.
    """
    if len(full_bic) != 11:
        return False
    expected = compute_check_digit(full_bic[:10])
    if expected < 0:
        return False
    try:
        actual = int(full_bic[10])
    except ValueError:
        return False
    return actual == expected


def classify_text(normalized: str) -> BicClassification:
    """
    Classify a normalized OCR string into BIC segment field(s).

    Returns a BicClassification with match_type, extracted fields,
    confidence multiplier, and check digit validation result.
    """
    if not normalized:
        return BicClassification(
            match_type=BicMatchType.GARBAGE,
            fields=[],
            confidence_multiplier=0.0,
        )

    # Full BIC: AAAU1234567 (11 chars)
    if _RE_FULL_BIC.match(normalized):
        owner_code = normalized[:4]
        serial_number = normalized[4:]
        check_valid = validate_check_digit(normalized)
        multiplier = CHECK_DIGIT_PASS_MULTIPLIER if check_valid else CHECK_DIGIT_FAIL_MULTIPLIER
        return BicClassification(
            match_type=BicMatchType.FULL_BIC,
            fields=[
                (BicField.OWNER_CODE, owner_code),
                (BicField.SERIAL_NUMBER, serial_number),
            ],
            confidence_multiplier=multiplier,
            check_digit_valid=check_valid,
        )

    # Owner + category: AAAU (4 chars)
    if _RE_OWNER_CATEGORY.match(normalized):
        return BicClassification(
            match_type=BicMatchType.OWNER_CATEGORY,
            fields=[(BicField.OWNER_CODE, normalized)],
            confidence_multiplier=_MATCH_CONFIDENCE[BicMatchType.OWNER_CATEGORY],
        )

    # Serial + check digit: 7 digits
    if _RE_SERIAL_CHECK.match(normalized):
        return BicClassification(
            match_type=BicMatchType.SERIAL_CHECK,
            fields=[(BicField.SERIAL_NUMBER, normalized)],
            confidence_multiplier=_MATCH_CONFIDENCE[BicMatchType.SERIAL_CHECK],
        )

    # Partial owner: 3 letters
    if _RE_PARTIAL_OWNER.match(normalized):
        return BicClassification(
            match_type=BicMatchType.PARTIAL_OWNER,
            fields=[(BicField.OWNER_CODE, normalized)],
            confidence_multiplier=_MATCH_CONFIDENCE[BicMatchType.PARTIAL_OWNER],
        )

    # Partial serial: 6 digits
    if _RE_PARTIAL_SERIAL.match(normalized):
        return BicClassification(
            match_type=BicMatchType.PARTIAL_SERIAL,
            fields=[(BicField.SERIAL_NUMBER, normalized)],
            confidence_multiplier=_MATCH_CONFIDENCE[BicMatchType.PARTIAL_SERIAL],
        )

    # Size/type code: 4 chars like 45G1, 22R1
    if _RE_SIZE_TYPE.match(normalized):
        return BicClassification(
            match_type=BicMatchType.SIZE_TYPE,
            fields=[(BicField.SIZE_CODE, normalized)],
            confidence_multiplier=_MATCH_CONFIDENCE[BicMatchType.SIZE_TYPE],
        )

    # No match — garbage
    return BicClassification(
        match_type=BicMatchType.GARBAGE,
        fields=[],
        confidence_multiplier=0.0,
    )


@dataclass
class FieldObservation:
    """A single observation of a BIC field value from one crop."""
    value: str
    score: float  # OCR confidence × match multiplier × check digit multiplier
    bbox: Optional[List[float]] = None
    timestamp: int = 0
    field_type: BicField = BicField.SERIAL_NUMBER
    radius_factor: float = 2.0  # multiplier for spatial clustering radius (default 2.0)

    @property
    def centroid(self) -> Optional[Tuple[float, float]]:
        return _bbox_centroid(self.bbox)

    @property
    def radius(self) -> float:
        """radius_factor * max(crop_width, crop_height) as spatial clustering radius."""
        if not self.bbox or len(self.bbox) < 4:
            return 100.0  # default fallback
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return self.radius_factor * max(w, h)


def _bbox_centroid(bbox: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) centroid from [x1, y1, x2, y2] bbox, or None."""
    if not bbox or len(bbox) < 4:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _centroid_distance(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> float:
    """Euclidean distance between two centroids. Returns inf if either is None."""
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass
class EntityFieldAccumulator:
    """Accumulates BIC field observations for a single tracked container entity.

    New design: flat segment pool with radius-based spatial clustering at emission time.
    Full BIC segments are emitted immediately and their serials are blocked from re-emission.
    Partial segments accumulate and are clustered by proximity.
    """
    segment_pool: List[FieldObservation] = field(default_factory=list)
    blocked_serials: Dict[str, float] = field(default_factory=dict)  # serial_value -> best_score
    frame_count: int = 0
    last_observation_ts: int = 0

    def add_observation(self, obs: FieldObservation):
        """Add a segment observation to the pool."""
        self.segment_pool.append(obs)
        self._update_ts(obs)

    def is_serial_blocked(self, serial_value: str, similarity_fn, threshold: float = 0.95) -> bool:
        """Check if a serial is already blocked (ISO-validated and emitted)."""
        for blocked in self.blocked_serials:
            if similarity_fn(serial_value, blocked) >= threshold:
                return True
        return False

    def block_serial(self, serial_value: str, score: float):
        """Mark a serial as blocked (ISO-validated full BIC emitted)."""
        self.blocked_serials[serial_value] = score

    def increment_frame(self):
        self.frame_count += 1

    def get_segments_by_field(self, field_type: BicField) -> List[FieldObservation]:
        """Get all observations for a specific field type."""
        return [obs for obs in self.segment_pool if obs.field_type == field_type]

    def cluster_by_radius(self, similarity_fn, similarity_th: float = 0.95) -> List[Dict[BicField, FieldObservation]]:
        """Cluster segments by spatial proximity. Each cluster represents one container.

        Algorithm:
        1. Anchor on serials (or owners if no serials exist)
        2. For each anchor, find closest completing segments within radius
        3. Each segment is assigned to at most one cluster (nearest anchor wins)

        Returns list of dicts mapping BicField -> best FieldObservation for that cluster.
        """
        serials = self.get_segments_by_field(BicField.SERIAL_NUMBER)
        owners = self.get_segments_by_field(BicField.OWNER_CODE)
        sizes = self.get_segments_by_field(BicField.SIZE_CODE)

        # Deduplicate anchors by similarity (merge similar serials, keep best score)
        anchor_groups = self._deduplicate_anchors(serials, similarity_fn, similarity_th)

        if not anchor_groups and owners:
            # No serials — anchor on owners instead
            anchor_groups = self._deduplicate_anchors(owners, similarity_fn, similarity_th)
            return self._cluster_from_owner_anchors(anchor_groups, sizes)

        if not anchor_groups:
            return []

        # Build clusters: each anchor serial gets closest owner + size within radius
        clusters = []
        used_owners = set()
        used_sizes = set()

        for anchor in anchor_groups:
            cluster: Dict[BicField, FieldObservation] = {BicField.SERIAL_NUMBER: anchor}
            anchor_centroid = anchor.centroid
            anchor_radius = anchor.radius

            # Find closest owner within radius
            best_owner = self._find_closest(
                anchor_centroid, anchor_radius, owners, used_owners
            )
            if best_owner is not None:
                cluster[BicField.OWNER_CODE] = best_owner
                used_owners.add(id(best_owner))

            # Find closest size within radius
            best_size = self._find_closest(
                anchor_centroid, anchor_radius, sizes, used_sizes
            )
            if best_size is not None:
                cluster[BicField.SIZE_CODE] = best_size
                used_sizes.add(id(best_size))

            clusters.append(cluster)

        return clusters

    def _deduplicate_anchors(
        self, observations: List[FieldObservation], similarity_fn, threshold: float
    ) -> List[FieldObservation]:
        """Group similar observations, return the best-scoring representative from each group."""
        if not observations:
            return []

        groups: List[List[FieldObservation]] = []
        for obs in observations:
            placed = False
            for group in groups:
                if similarity_fn(obs.value, group[0].value) >= threshold:
                    group.append(obs)
                    placed = True
                    break
            if not placed:
                groups.append([obs])

        # Return the highest-scoring observation from each group
        return [max(group, key=lambda o: o.score) for group in groups]

    def _find_closest(
        self,
        anchor_centroid: Optional[Tuple[float, float]],
        anchor_radius: float,
        candidates: List[FieldObservation],
        used: set,
    ) -> Optional[FieldObservation]:
        """Find the closest candidate within radius that hasn't been used."""
        if anchor_centroid is None:
            # No spatial info — take best scoring unused candidate
            best = None
            for c in candidates:
                if id(c) not in used:
                    if best is None or c.score > best.score:
                        best = c
            return best

        best = None
        best_dist = float("inf")
        for c in candidates:
            if id(c) in used:
                continue
            c_centroid = c.centroid
            dist = _centroid_distance(anchor_centroid, c_centroid)
            if dist <= anchor_radius and dist < best_dist:
                best = c
                best_dist = dist
        return best

    def _cluster_from_owner_anchors(
        self, owner_anchors: List[FieldObservation], sizes: List[FieldObservation]
    ) -> List[Dict[BicField, FieldObservation]]:
        """When no serials exist, cluster using owners as anchors with sizes."""
        clusters = []
        used_sizes = set()
        for anchor in owner_anchors:
            cluster: Dict[BicField, FieldObservation] = {BicField.OWNER_CODE: anchor}
            best_size = self._find_closest(
                anchor.centroid, anchor.radius, sizes, used_sizes
            )
            if best_size is not None:
                cluster[BicField.SIZE_CODE] = best_size
                used_sizes.add(id(best_size))
            clusters.append(cluster)
        return clusters

    def _update_ts(self, obs: FieldObservation):
        self.last_observation_ts = max(self.last_observation_ts, obs.timestamp)
