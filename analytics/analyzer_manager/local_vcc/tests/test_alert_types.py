"""Tests for event-type decoding and dynamic VccAlertType."""
from app.alert_types import (
    EVENT_TYPE_FACTOR,
    AlertCategory,
    AlertType,
    CustomizedCapabilitiesType,
    SafetyType,
    VccAlertType,
    alert_mapping,
    alert_type_to_vcc,
    event_type_to_alert_type,
    split_event_type,
)


def _event_id(category, flow):
    return category.value * EVENT_TYPE_FACTOR + flow.value


class TestEventTypeDecoding:
    def test_split_event_type_roundtrip(self):
        eid = _event_id(AlertCategory.Safety, SafetyType.Weapon)
        cat, flow = split_event_type(eid)
        assert cat == AlertCategory.Safety.value
        assert flow == SafetyType.Weapon.value

    def test_safety_weapon_maps_to_weapon(self):
        eid = _event_id(AlertCategory.Safety, SafetyType.Weapon)
        assert event_type_to_alert_type(eid) is AlertType.weapon

    def test_customized_periodic_text_maps(self):
        eid = _event_id(AlertCategory.CustomizedCapabilities,
                        CustomizedCapabilitiesType.PeriodicText)
        assert event_type_to_alert_type(eid) is AlertType.periodicText

    def test_unknown_category_returns_none(self):
        eid = 999 * EVENT_TYPE_FACTOR + 1
        assert event_type_to_alert_type(eid) is None

    def test_unknown_flow_returns_none(self):
        eid = _event_id(AlertCategory.Safety, SafetyType.Weapon) + 500
        # flow now doesn't exist in Safety mapping
        assert event_type_to_alert_type(eid) is None


class TestVccAlertType:
    def test_members_align_with_prompt_map(self):
        from app.vcc_common import prompt_map
        # every VccAlertType member with an alias should have a prompt
        for member in VccAlertType:
            assert member in prompt_map, f"missing prompt for {member.name}"

    def test_alert_type_to_vcc_known(self):
        assert alert_type_to_vcc(AlertType.weapon) is VccAlertType.WEAPON
        assert alert_type_to_vcc(AlertType.fire) is VccAlertType.FIRE
        assert alert_type_to_vcc(AlertType.periodicText) is VccAlertType.PERIODICTEXT

    def test_alert_type_to_vcc_unknown(self):
        assert alert_type_to_vcc(AlertType.appearance) is None

    def test_alert_mapping_is_populated(self):
        assert AlertCategory.Safety.value in alert_mapping
        assert SafetyType.Weapon.value in alert_mapping[AlertCategory.Safety.value]

