import time

from app.image_utils import load_images
from app.logger import logger
from app.models import AlertVerificationRequest, AlertVerificationResponse
from app.prompts import PromptResolver
from app.vlm_client import VlmClient


class AlertVerifier:
    """Verify an alert by querying the local VLM with the alert's images."""

    def __init__(self, vlm_client: VlmClient, prompt_resolver: PromptResolver):
        self.vlm = vlm_client
        self.prompts = prompt_resolver

    def verify(self, request: AlertVerificationRequest) -> AlertVerificationResponse:
        t0 = time.time()
        verified = False
        success = True
        message = ""
        parsed = None

        prompt = self.prompts.resolve(request.eventTypeId, request.filterPrompt)
        if not prompt:
            message = f"No prompt for eventTypeId {request.eventTypeId} and no filterPrompt provided"
            logger.error(message)
            return AlertVerificationResponse(
                alertInstanceId=request.alertInstanceId,
                cameraId=request.cameraId,
                verified=False,
                success=False,
                message=message,
            )

        image_urls = load_images(request.images or [])
        if not image_urls:
            message = "No images to process"
            logger.error(f"{message} (alertInstanceId={request.alertInstanceId})")
            return AlertVerificationResponse(
                alertInstanceId=request.alertInstanceId,
                cameraId=request.cameraId,
                verified=False,
                success=False,
                message=message,
            )

        schema = self.prompts.resolve_schema(request.eventTypeId)
        logger.info(
            f"Verifying alertInstanceId={request.alertInstanceId} "
            f"eventTypeId={request.eventTypeId} with {len(image_urls)} image(s); "
            f"prompt={prompt!r} schema={schema.__name__}"
        )

        for url in image_urls:
            parsed = self.vlm.query_structured(url, prompt, schema, low_res=False)
            if parsed is None:
                success = False
                message = "VLM did not return a valid structured response"
                logger.warning(
                    f"VLM structured query failed for alertInstanceId={request.alertInstanceId}"
                )
                break
            if parsed.final_answer:
                verified = True
                break

        elapsed = time.time() - t0
        logger.info(
            f"Done alertInstanceId={request.alertInstanceId} verified={verified} "
            f"success={success} elapsed={elapsed:.2f}s response={parsed!r}"
        )
        if success and not message and parsed is not None:
            message = parsed.model_dump_json()
        return AlertVerificationResponse(
            alertInstanceId=request.alertInstanceId,
            cameraId=request.cameraId,
            verified=verified,
            success=success,
            message=message,
        )
