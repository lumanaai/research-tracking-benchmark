import os
from typing import Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from app.logger import logger, log_exception


class VlmClient:
    """Thin wrapper around the local llama.cpp OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: int = 200,
    ):
        base_url = base_url or os.environ.get("LLAMA_SERVER_URL", "http://localhost:7999")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.base_url = base_url
        self.model = model or os.environ.get("LLAMA_MODEL_NAME", "local_vcc")
        if timeout is None:
            timeout = float(os.environ.get("LLAMA_REQUEST_TIMEOUT", "5"))
        self.timeout = timeout
        self.max_tokens = max_tokens
        # max_retries=0: a single attempt per call. Retries are handled (or not)
        # by the caller, which needs a predictable worst-case latency per image
        # rather than the SDK silently retrying up to 3x behind the scenes.
        self.client = OpenAI(base_url=self.base_url, api_key="none", timeout=timeout, max_retries=0)
        logger.info(
            f"VlmClient initialized (base_url={self.base_url}, model={self.model}, "
            f"timeout={self.timeout}s, max_retries=0)"
        )

    def query_structured(
        self,
        image_url: str,
        prompt: str,
        schema: Type[BaseModel],
        low_res: bool = True,
    ) -> Optional[BaseModel]:
        """Send a single image + prompt to the VLM, constrained to ``schema``.

        Returns the parsed ``schema`` instance, or ``None`` on any failure
        (request error, malformed response, schema mismatch) so callers can
        treat that uniformly as a VLM failure.
        """
        detail = "low" if low_res else "high"
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
            ],
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[message],
                max_completion_tokens=self.max_tokens,
                response_format=response_format,
            )
        except Exception as e:
            log_exception(logger, "VLM structured request failed", e)
            return None

        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            logger.error(f"Malformed VLM response: {e}")
            return None

        try:
            return schema.model_validate_json(content)
        except Exception as e:
            logger.error(f"Failed to parse structured VLM response: {e} (content={content!r})")
            return None
