"""End-to-end tests of AlertVerifier with a fake VlmClient."""
import pytest
from PIL import Image

from app.alert_types import (
    EVENT_TYPE_FACTOR,
    AlertCategory,
    CustomizedCapabilitiesType,
    SafetyType,
)
from app.models import AlertVerificationRequest
from app.prompts import PromptResolver
from app.vcc_common import CoTStructuredOutput
from app.verifier import AlertVerifier


class FakeVlm:
    """Fake VlmClient. ``responses`` holds one entry per expected call:
    a schema instance for a successful structured reply, or ``None`` to
    simulate a VLM/parsing failure."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def query_structured(self, image_url, prompt, schema, low_res=True):
        self.calls.append({
            "image_url": image_url, "prompt": prompt, "schema": schema, "low_res": low_res,
        })
        return self.responses.pop(0)


def _eid(cat, flow):
    return cat.value * EVENT_TYPE_FACTOR + flow.value


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "img.jpg"
    Image.new("RGB", (16, 16), (0, 0, 255)).save(path, format="JPEG")
    return str(path)


@pytest.fixture
def image_path_2(tmp_path):
    path = tmp_path / "img2.jpg"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(path, format="JPEG")
    return str(path)


def _make_verifier(responses):
    return AlertVerifier(FakeVlm(responses), PromptResolver()), None


def test_verified_true_on_yes(image_path):
    fake = FakeVlm([CoTStructuredOutput(description="a person holding a gun", final_answer=True)])
    verifier = AlertVerifier(fake, PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=1,
        cameraId="cam-1",
        eventTypeId=_eid(AlertCategory.Safety, SafetyType.Weapon),
        images=[image_path],
    )
    resp = verifier.verify(req)
    assert resp.verified is True
    assert resp.success is True
    assert resp.alertInstanceId == 1
    assert resp.cameraId == "cam-1"
    assert "a person holding a gun" in resp.message
    assert fake.calls[0]["schema"] is CoTStructuredOutput


def test_short_circuits_on_first_yes(image_path, image_path_2):
    fake = FakeVlm([
        CoTStructuredOutput(description="no", final_answer=False),
        CoTStructuredOutput(description="yes indeed", final_answer=True),
    ])
    verifier = AlertVerifier(fake, PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=2,
        cameraId="cam-2",
        eventTypeId=_eid(AlertCategory.Safety, SafetyType.Weapon),
        images=[image_path, image_path_2],
    )
    resp = verifier.verify(req)
    assert resp.verified is True
    assert resp.cameraId == "cam-2"
    assert len(fake.calls) == 2


def test_verified_false_when_all_no(image_path, image_path_2):
    fake = FakeVlm([
        CoTStructuredOutput(description="no", final_answer=False),
        CoTStructuredOutput(description="nope", final_answer=False),
    ])
    verifier = AlertVerifier(fake, PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=3,
        cameraId="cam-3",
        eventTypeId=_eid(AlertCategory.Safety, SafetyType.Weapon),
        images=[image_path, image_path_2],
    )
    resp = verifier.verify(req)
    assert resp.verified is False
    assert resp.success is True
    assert resp.cameraId == "cam-3"


def test_invalid_structured_response_marks_failure(image_path):
    verifier = AlertVerifier(FakeVlm([None]), PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=4,
        cameraId="cam-4",
        eventTypeId=_eid(AlertCategory.Safety, SafetyType.Weapon),
        images=[image_path],
    )
    resp = verifier.verify(req)
    assert resp.verified is False
    assert resp.success is False
    assert "structured response" in resp.message
    assert resp.cameraId == "cam-4"


def test_no_prompt_returns_failure(image_path):
    verifier = AlertVerifier(FakeVlm([]), PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=5,
        cameraId="cam-5",
        eventTypeId=999 * EVENT_TYPE_FACTOR + 1,  # unknown, no filter prompt
        images=[image_path],
    )
    resp = verifier.verify(req)
    assert resp.verified is False
    assert resp.success is False
    assert resp.cameraId == "cam-5"


def test_no_images_returns_failure():
    verifier = AlertVerifier(FakeVlm([]), PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=6,
        cameraId="cam-6",
        eventTypeId=_eid(AlertCategory.Safety, SafetyType.Weapon),
        images=[],
    )
    resp = verifier.verify(req)
    assert resp.verified is False
    assert resp.success is False
    assert resp.cameraId == "cam-6"


def test_periodic_text_uses_filter_prompt(image_path):
    fake = FakeVlm([CoTStructuredOutput(description="the light is on", final_answer=True)])
    verifier = AlertVerifier(fake, PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=7,
        cameraId="cam-7",
        eventTypeId=_eid(AlertCategory.CustomizedCapabilities,
                         CustomizedCapabilitiesType.PeriodicText),
        filterPrompt="is the light on",
        images=[image_path],
    )
    resp = verifier.verify(req)
    assert resp.verified is True
    assert resp.cameraId == "cam-7"
    assert "is the light on" in fake.calls[0]["prompt"]


def test_brandishing_weapon_requires_visible_and_in_hands(image_path, monkeypatch):
    """BrandishingWeaponData.final_answer is an AND of both sub-fields."""
    import app.prompts as prompts_mod
    from app.alert_types import VccAlertType
    from app.vcc_common import BrandishingWeaponData

    monkeypatch.setattr(prompts_mod, "alert_type_to_vcc", lambda _at: VccAlertType.BRANDISHING_WEAPON)

    fake = FakeVlm([
        BrandishingWeaponData(description="gun on belt", visible_firearm=True, firearm_in_hands=False),
    ])
    verifier = AlertVerifier(fake, PromptResolver())
    req = AlertVerificationRequest(
        alertInstanceId=8,
        cameraId="cam-8",
        eventTypeId=_eid(AlertCategory.Safety, SafetyType.Weapon),
        images=[image_path],
    )
    resp = verifier.verify(req)
    assert fake.calls[0]["schema"] is BrandishingWeaponData
    assert resp.verified is False  # visible but not in hands -> final_answer False
    assert resp.success is True
