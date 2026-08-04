"""Tests for PromptResolver."""
from app.alert_types import (
    EVENT_TYPE_FACTOR,
    AlertCategory,
    CustomizedCapabilitiesType,
    SafetyType,
)
from app.prompts import DEFAULT_TEMPLATE, PromptResolver
from app.vcc_common import BrandishingWeaponData, CoTStructuredOutput, prompt_map
from app.alert_types import VccAlertType


def _eid(cat, flow):
    return cat.value * EVENT_TYPE_FACTOR + flow.value


class TestPromptResolver:
    def setup_method(self):
        self.resolver = PromptResolver()

    def test_returns_lut_prompt_for_known_alert(self):
        eid = _eid(AlertCategory.Safety, SafetyType.Weapon)
        prompt = self.resolver.resolve(eid, None)
        assert prompt == prompt_map[VccAlertType.WEAPON]

    def test_periodic_text_requires_filter_prompt(self):
        eid = _eid(AlertCategory.CustomizedCapabilities,
                   CustomizedCapabilitiesType.PeriodicText)
        assert self.resolver.resolve(eid, None) is None

    def test_periodic_text_substitutes_question(self):
        eid = _eid(AlertCategory.CustomizedCapabilities,
                   CustomizedCapabilitiesType.PeriodicText)
        prompt = self.resolver.resolve(eid, "is the door open")
        assert prompt is not None
        assert "is the door open" in prompt

    def test_unknown_event_type_falls_back_to_default_template(self):
        prompt = self.resolver.resolve(999 * EVENT_TYPE_FACTOR + 1, "a red truck")
        assert prompt == DEFAULT_TEMPLATE.replace("{q}", "a red truck")

    def test_unknown_event_type_without_filter_returns_none(self):
        assert self.resolver.resolve(999 * EVENT_TYPE_FACTOR + 1, None) is None

    def test_custom_default_template(self):
        resolver = PromptResolver(default_template="ANSWER YES/NO: {q}?")
        prompt = resolver.resolve(999 * EVENT_TYPE_FACTOR + 1, "smoke")
        assert prompt == "ANSWER YES/NO: smoke?"

    def test_resolve_schema_defaults_to_cot_for_known_alert(self):
        eid = _eid(AlertCategory.Safety, SafetyType.Weapon)
        assert self.resolver.resolve_schema(eid) is CoTStructuredOutput

    def test_resolve_schema_defaults_to_cot_for_unknown_event_type(self):
        assert self.resolver.resolve_schema(999 * EVENT_TYPE_FACTOR + 1) is CoTStructuredOutput

    def test_resolve_schema_uses_override_for_brandishing_weapon(self, monkeypatch):
        import app.prompts as prompts_mod

        monkeypatch.setattr(
            prompts_mod, "alert_type_to_vcc", lambda _at: VccAlertType.BRANDISHING_WEAPON
        )
        assert self.resolver.resolve_schema(0) is BrandishingWeaponData

