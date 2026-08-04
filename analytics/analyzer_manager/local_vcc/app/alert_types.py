"""local_vcc alert-type helpers.

Re-exports the enums / mapping from
``analyzer_manager/app/general/alert_types.py`` (the single source of truth)
and dynamically builds :class:`VccAlertType`, the enum used as keys in
``app.vcc_common.prompt_map``.

To expose a new AlertType to the prompt map, add an entry to
:data:`_VCC_ALERT_ALIASES` and a matching entry to
``app.vcc_common.prompt_map``.
"""

import enum
from typing import Optional

from general.alert_types import (  # noqa: F401  (re-exported)
    EVENT_TYPE_FACTOR,
    AlertCategory,
    AlertType,
    CustomizedCapabilitiesType,
    IdentificationType,
    IntegrationsAlertType,
    ObjectStateAlertType,
    RetailAlertType,
    SafetyType,
    StatusType,
    TrackingType,
    alert_mapping,
    event_type_to_alert_type,
    split_event_type,
)


_VCC_ALERT_ALIASES = {
    "WEAPON": AlertType.weapon,
    "BRANDISHING_WEAPON": AlertType.brandishingWeapon,
    "FIRE": AlertType.fire,
    "FALL": AlertType.fall,
    "PROTECTIVEGEAR": AlertType.ppe,
    "PERIODICTEXT": AlertType.periodicText,
    "VIOLENCE": AlertType.fighting,
}

VccAlertType = enum.IntEnum(
    "VccAlertType",
    {name: at.value for name, at in _VCC_ALERT_ALIASES.items()},
)


def alert_type_to_vcc(alert_type: AlertType) -> Optional[VccAlertType]:
    """Return the ``VccAlertType`` corresponding to ``alert_type`` if any."""
    try:
        return VccAlertType(alert_type.value)
    except ValueError:
        return None
