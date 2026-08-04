"""VCC prompt LUT.

Keys use :class:`VccAlertType` (built dynamically from
``app.alert_types.AlertType``). To add a new prompt, add a matching alias to
``_VCC_ALERT_ALIASES`` in ``alert_types.py``.
"""

from pydantic import BaseModel

from app.alert_types import VccAlertType


class CoTStructuredOutput(BaseModel):
    description: str
    final_answer: bool


class BrandishingWeaponData(BaseModel):
    description: str
    visible_firearm: bool
    firearm_in_hands: bool

    @property
    def final_answer(self) -> bool:
        return self.visible_firearm and self.firearm_in_hands


default_structured_output = CoTStructuredOutput
structured_output_map = {VccAlertType.BRANDISHING_WEAPON: BrandishingWeaponData}

prompt_map = {
    VccAlertType.WEAPON: """ 
        You are tasked with identifying if the person in the image is carrying a firearm.
        First, describe in steps what the person is holding in his hands. Notice similar objects like phones, keys, tablets, hand drills, bottles, or cans. 
        Also notice what the person is carrying on his belt or over its shoulder.
        The description should be short and concise - no more than 20 words.
        Then answer, is there a visible firearm (such as a gun or similar weapon) in this image? 
        """,
    VccAlertType.BRANDISHING_WEAPON: """ 
        You are tasked with identifying if the person in the image is carrying a firearm in his hands.
        First, describe what the person is holding in his hands. Notice similar objects like phones, keys, tablets, hand drills, bottles, or cans. 
        Also notice what the person is carrying on his belt or over his shoulder.
        The description should be short and concise - no more than 20 words.
        Then answer:
        visible_firearm - Is there a visible firearm (such as a gun or similar weapon) in this image? 
        firearm_in_hands - Is the firearm actively held in the person’s hand(s).
        """,
    VccAlertType.FIRE: """
        You are tasked with determining whether there is fire in the image.

        1. **Image Description**:
        - Describe the image any elements related to fire (e.g., flames, smoke).
        - Explicitly mention features that resemble fire but are not (e.g., lights, reflections, haze, or corrupted pixels).
        - Prioritize evidence of actual fire behavior (e.g., flickering flames, rising smoke) over ambiguous features.
        - The description should be short and concise - no more than 20 words.

        3. **Final Answer**:
        - Based on your analysis, state if there is fire in the image.

        Note: Be cautious of light sources, reflections, and other false-positive indicators. Explain your reasoning clearly to avoid misinterpretation.
        """,
    VccAlertType.FALL: """ 
        These are frames from a video arranged in a collage.
        You are tasked with determining whether the person in the collage is falling.
        First describe what the person is doing in the image, describe elements that support the presence of a fall.
        Describe also elements that can lead to misinterpretation such as occlusions, sitting down, lying down, kneeling, jumping, bending down, leaning forward, etc.
        The description should be short and concise - no more than 20 words.
        Finally answer, Is there person falling? 
        """,
    VccAlertType.PROTECTIVEGEAR: """ 
        You are tasked with determining whether the person in the image wearing protective gear covering his head, such as safety helmet (hardhat).
        First describe the headwear of the person, describe  elements support the presence of safety helmet.
        Describe also elements that can lead to misinterpretation like other types of hats.
        The description should be short and concise - no more than 20 words.
        In the case the the head is not visible, answer that the person is wearing a helmet.
        Finally answer, Is there person wearing safety helmet? 
        """,
    VccAlertType.PERIODICTEXT: """
        You are tasked with answering the following question: {question}.
        First describe the scene, describe elements support positive answer.
        Describe also elements that can lead to misinterpretation.
        The description should be short and concise - no more than 20 words.
        Finally answer, {question}?
        """,
    VccAlertType.VIOLENCE: """
        These are frames from a video arranged in a collage.
        Your task is to determine whether there is physical violence in this video.
        **Definition**: Any physical aggression must be classified as violence, even if it could be playful, or staged. Do not infer intent or context. 
        If physical aggression is visible, it counts as violence.
        First, describe the physical interactions and elements that indicate violence.
        Then, describe any factors that could lead to misinterpretation.
        Keep your description short and concise—no more than 30 words.
        Finally, answer: Is there physical violence?
    """,
}
