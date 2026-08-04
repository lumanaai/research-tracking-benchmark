import os
import signal

from app.logger import logger
from app.prompts import PromptResolver
from app.verifier import AlertVerifier
from app.vlm_client import VlmClient
from app.worker import VccWorker
from general.proj import load_config


def main():
    host = os.environ.get("RABBIT_HOST", "rabbitmq")
    msg_queue_cfg = load_config().get("msgQueue", {})
    request_queue = msg_queue_cfg.get("vccQueue", "vccVerificationRequestQueue")
    response_queue = msg_queue_cfg.get("vccRspQueue", "vccVerificationResponseQueue")
    request_queue_ttl_ms = int(msg_queue_cfg.get("vccQueueTTL", 60000))
    response_queue_ttl_ms = int(msg_queue_cfg.get("vccRspQueueTTL", 60000))
    max_timeout_seconds = float(os.environ.get("VCC_MAX_TIMEOUT_SECONDS", "20"))

    vlm_client = VlmClient()
    prompt_resolver = PromptResolver()
    verifier = AlertVerifier(vlm_client, prompt_resolver)

    worker = VccWorker(
        verifier=verifier,
        host=host,
        request_queue=request_queue,
        response_queue=response_queue,
        request_queue_ttl_ms=request_queue_ttl_ms,
        response_queue_ttl_ms=response_queue_ttl_ms,
        max_timeout_seconds=max_timeout_seconds,
    )

    def _handle_signal(signum, _frame):
        logger.info("Received signal %s, stopping worker", signum)
        worker.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("local_vcc worker starting")
    worker.run()


if __name__ == "__main__":
    main()
