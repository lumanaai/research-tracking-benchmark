"""Alert category / type enums and the ``alert_mapping`` LUT.

Kept in a standalone module (no numpy / cv2 imports) so lightweight consumers
(e.g. ``local_vcc``) can import it without pulling in the rest of
``general.core`` / ``general.analyzer_general``.

``general.core`` re-exports every symbol from this module for backwards
compatibility.

Event-type encoding used by the edge stack:

    event_type_id = category * EVENT_TYPE_FACTOR + flow

where ``category`` is an :class:`AlertCategory` value and ``flow`` is the
sub-enum value (``SafetyType`` for Safety, ``TrackingType`` for Tracking, ...).
:func:`event_type_to_alert_type` decodes an event-type id back to an
:class:`AlertType`.
"""

import enum
from typing import Optional, Tuple

EVENT_TYPE_FACTOR = 1_000_000


class AlertCategory(enum.Enum):
    Safety = 0
    Identification = 1
    Tracking = 2
    Status = 3
    CustomizedCapabilities = 4
    Integrations = 5
    Retail = 6
    ObjectState = 7


class SafetyType(enum.Enum):
    Motion = 0
    Tampering = 1
    Proximity = 2
    Weapon = 3
    Trespassing = 4  #
    Tailgating = 5
    ZoneProtection = 6
    SpeedLimit = 7  #
    Door = 8
    Fire = 9
    Fall = 10
    Fighting = 11
    FaceAppearance = 12
    WeaponHeld = 13
    ZoneTrespassing = 14


class IdentificationType(enum.Enum):
    FaceDetection = 0
    LPR = 1
    ContainerId = 2


class TrackingType(enum.Enum):
    Apperance = 0
    Disappearing = 1
    Loitering = 2
    LineCrossing = 3
    Occupancy = 4
    TrafficControl = 5
    MissingObject = 6
    Absense = 7
    Counting = 8
    Snapshot = 9
    Phone = 10
    RegionCounting = 90


class StatusType(enum.Enum):
    pass


class CustomizedCapabilitiesType(enum.Enum):
    ProtectiveGear = 0
    PersonalSafety = 1
    PeriodicText = 2
    Gloves = 3
    Hands = 4
    DistOnLane = 5
    LicensePlateC = 6
    Classification = 99


class IntegrationsAlertType(enum.Enum):
    access_control_tailgate = 1
    developer = 3
    validation = 4


class RetailAlertType(enum.Enum):
    ShelfOccupancy = 0
    ShelfDrop = 1
    EmptyShelfCounter = 100


class ObjectStateAlertType(enum.Enum):
    StateObjectChange = 0
    StateObjectTransition = 1
    StateObjectDuration = 2


class ProtectedGearWearOptions(enum.Enum):
    not_wearing = 0
    wearing = 1


class ProtectedGearType(enum.Enum):
    hard_hat = 0
    safety_vest = 1


class AlertType(enum.Enum):
    unknown = -1
    appearance = 0
    disappeare = 1
    loitering = 2
    lineCrossing = 3
    lpr = 4
    faceDetection = 5
    videoTampering = 6
    motionScore = 7
    occupancy = 8
    movement = 9
    trafficControl = 10
    tailgating = 11
    proximity = 12
    weapon = 13
    doors = 14
    fire = 15
    fall = 16
    periodicText = 17
    ppe = 18
    clip = 19
    fighting = 20
    gloves = 21
    missingObject = 22
    containerId = 23
    absence = 24
    counting = 25
    zoneProtection = 26
    developer = 27
    classification = 28
    faceAppearance = 29
    hands = 30
    snapshot = 31
    emptyShelf = 32
    lpc = 33
    emptyShelfCounter = 34
    emptyShelfDrop = 35
    zoneTrespassing = 36
    validationAlert = 37
    brandishingWeapon = 38
    stateObjectTransition = 39
    stateObjectChange = 40
    stateObjectDuration = 41
    phone = 42
    distOnLane = 43
    regionCount = 44


alert_mapping = {
    AlertCategory.Safety.value: {
        SafetyType.Motion.value: AlertType.motionScore.value,
        SafetyType.Tampering.value: AlertType.videoTampering.value,
        SafetyType.Proximity.value: AlertType.proximity.value,
        SafetyType.Weapon.value: AlertType.weapon.value,
        SafetyType.Trespassing.value: AlertType.lineCrossing.value,
        SafetyType.Tailgating.value: AlertType.tailgating.value,
        SafetyType.ZoneProtection.value: AlertType.zoneProtection.value,
        SafetyType.SpeedLimit.value: AlertType.trafficControl.value,
        SafetyType.Door.value: AlertType.doors.value,
        SafetyType.Fire.value: AlertType.fire.value,
        SafetyType.Fall.value: AlertType.fall.value,
        SafetyType.Fighting.value: AlertType.fighting.value,
        SafetyType.FaceAppearance.value: AlertType.faceAppearance.value,
        SafetyType.WeaponHeld.value: AlertType.brandishingWeapon.value,
        SafetyType.ZoneTrespassing.value: AlertType.zoneTrespassing.value,
    },
    AlertCategory.Identification.value: {
        IdentificationType.FaceDetection.value: AlertType.faceDetection.value,
        IdentificationType.LPR.value: AlertType.lpr.value,
        IdentificationType.ContainerId.value: AlertType.containerId.value,
    },
    AlertCategory.Tracking.value: {
        TrackingType.Apperance.value: AlertType.appearance.value,
        TrackingType.Disappearing.value: AlertType.disappeare.value,
        TrackingType.Loitering.value: AlertType.loitering.value,
        TrackingType.LineCrossing.value: AlertType.lineCrossing.value,
        TrackingType.Occupancy.value: AlertType.occupancy.value,
        TrackingType.TrafficControl.value: AlertType.trafficControl.value,
        TrackingType.MissingObject.value: AlertType.missingObject.value,
        TrackingType.Absense.value: AlertType.absence.value,
        TrackingType.Counting.value: AlertType.counting.value,
        TrackingType.Snapshot.value: AlertType.snapshot.value,
        TrackingType.Phone.value: AlertType.phone.value,
        TrackingType.RegionCounting.value: AlertType.regionCount.value,
    },
    AlertCategory.CustomizedCapabilities.value: {
        CustomizedCapabilitiesType.ProtectiveGear.value: AlertType.ppe.value,
        CustomizedCapabilitiesType.PersonalSafety.value: AlertType.proximity.value,
        CustomizedCapabilitiesType.PeriodicText.value: AlertType.periodicText.value,
        CustomizedCapabilitiesType.Gloves.value: AlertType.gloves.value,
        CustomizedCapabilitiesType.Hands.value: AlertType.hands.value,
        CustomizedCapabilitiesType.Classification.value: AlertType.classification.value,
        CustomizedCapabilitiesType.LicensePlateC.value: AlertType.lpc.value,
        CustomizedCapabilitiesType.DistOnLane.value: AlertType.distOnLane.value,
    },
    AlertCategory.Integrations.value: {
        IntegrationsAlertType.access_control_tailgate.value: AlertType.lineCrossing.value,
        IntegrationsAlertType.developer.value: AlertType.developer.value,
        IntegrationsAlertType.validation.value: AlertType.validationAlert.value,
    },
    AlertCategory.Retail.value: {
        RetailAlertType.ShelfOccupancy.value: AlertType.emptyShelf.value,
        RetailAlertType.EmptyShelfCounter.value: AlertType.emptyShelfCounter.value,
        RetailAlertType.ShelfDrop.value: AlertType.emptyShelfDrop.value,
    },
    AlertCategory.ObjectState.value: {
        ObjectStateAlertType.StateObjectChange.value: AlertType.stateObjectChange.value,
        ObjectStateAlertType.StateObjectTransition.value: AlertType.stateObjectTransition.value,
        ObjectStateAlertType.StateObjectDuration.value: AlertType.stateObjectDuration.value,
    },
}


def split_event_type(event_type_id: int) -> Tuple[int, int]:
    """Return ``(category, flow)`` extracted from an ``event_type_id``."""
    category = event_type_id // EVENT_TYPE_FACTOR
    flow = event_type_id % EVENT_TYPE_FACTOR
    return category, flow


def event_type_to_alert_type(event_type_id: int) -> Optional[AlertType]:
    """Resolve ``event_type_id`` (``category * EVENT_TYPE_FACTOR + flow``) to an AlertType.

    Returns ``None`` when the category or flow is not present in
    :data:`alert_mapping`.
    """
    category, flow = split_event_type(event_type_id)
    per_category = alert_mapping.get(category)
    if per_category is None:
        return None
    value = per_category.get(flow)
    if value is None:
        return None
    try:
        return AlertType(value)
    except ValueError:
        return None
