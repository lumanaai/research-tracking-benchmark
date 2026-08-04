from typing import Optional, Type

from pydantic import BaseModel

from app.alert_types import (
    VccAlertType,
    alert_type_to_vcc,
    event_type_to_alert_type,
)
from app.logger import logger
from app.vcc_common import default_structured_output, prompt_map, structured_output_map

DEFAULT_TEMPLATE = "does this image contains {q}? Answer yes or no only."


class PromptResolver:
    """Resolve a prompt string for an :class:`AlertVerificationRequest`."""

    def __init__(self, default_template: str = DEFAULT_TEMPLATE):
        self.default_template = default_template

    def resolve(self, event_type_id: int, filter_prompt: Optional[str]) -> Optional[str]:
        alert_type = event_type_to_alert_type(event_type_id)
        vcc_type = alert_type_to_vcc(alert_type) if alert_type is not None else None

        if vcc_type is not None:
            template = prompt_map.get(vcc_type)
            if template is None:
                logger.warning(
                    f"No prompt configured for {vcc_type.name} "
                    f"(eventTypeId={event_type_id})"
                )
            elif vcc_type == VccAlertType.PERIODICTEXT:
                if not filter_prompt:
                    logger.error(
                        f"PERIODICTEXT alert requires filterPrompt (eventTypeId={event_type_id})"
                    )
                    return None
                return template.format(question=filter_prompt.strip())
            else:
                return template

        if filter_prompt:
            return self.default_template.replace("{q}", filter_prompt.strip())

        return None

    def resolve_schema(self, event_type_id: int) -> Type[BaseModel]:
        """Return the structured-output schema the VLM response must match."""
        alert_type = event_type_to_alert_type(event_type_id)
        vcc_type = alert_type_to_vcc(alert_type) if alert_type is not None else None
        return structured_output_map.get(vcc_type, default_structured_output)
