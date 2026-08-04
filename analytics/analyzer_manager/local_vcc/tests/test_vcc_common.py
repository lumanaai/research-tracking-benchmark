"""Tests for the structured-output schemas/LUT in app.vcc_common."""
from app.alert_types import VccAlertType
from app.vcc_common import (
    BrandishingWeaponData,
    CoTStructuredOutput,
    default_structured_output,
    structured_output_map,
)


def test_default_structured_output_is_cot():
    assert default_structured_output is CoTStructuredOutput


def test_structured_output_map_has_brandishing_weapon_override():
    assert structured_output_map[VccAlertType.BRANDISHING_WEAPON] is BrandishingWeaponData


def test_cot_structured_output_final_answer_is_plain_field():
    out = CoTStructuredOutput(description="a scene", final_answer=True)
    assert out.final_answer is True


def test_brandishing_weapon_final_answer_requires_both_fields():
    both_true = BrandishingWeaponData(description="x", visible_firearm=True, firearm_in_hands=True)
    assert both_true.final_answer is True

    visible_only = BrandishingWeaponData(description="x", visible_firearm=True, firearm_in_hands=False)
    assert visible_only.final_answer is False

    in_hands_only = BrandishingWeaponData(description="x", visible_firearm=False, firearm_in_hands=True)
    assert in_hands_only.final_answer is False

    neither = BrandishingWeaponData(description="x", visible_firearm=False, firearm_in_hands=False)
    assert neither.final_answer is False
