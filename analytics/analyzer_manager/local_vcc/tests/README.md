# local_vcc tests

Two layers:

## 1. Code-level unit tests (fast, no docker)

Cover the worker's pure-Python behavior with mocked I/O:

| File | What it covers |
| --- | --- |
| `test_alert_types.py`     | `eventTypeId` decoding, `VccAlertType` built dynamically from `alert_mapping`, LUT alignment |
| `test_prompts.py`         | `PromptResolver` — LUT hits, `PERIODICTEXT` templating, `filterPrompt` fallback |
| `test_image_utils.py`     | JPEG/PNG loading, missing/corrupt files skipped, base64 data-URL format |
| `test_vlm_client.py`      | `VlmClient` request shape, response parsing, `"NA "` on failure |
| `test_verifier.py`        | `AlertVerifier` end-to-end with a fake `VlmClient` (yes/no/NA/short-circuit/no-prompt/no-images) |
| `test_worker.py`          | `VccWorker._on_message` publishes a response + ACKs, even on malformed payload / verifier exception |

Run from the repo root:

```bash
pip install pytest pika pydantic pillow openai
pytest analyzer_manager/local_vcc/tests -v \
    --ignore=analyzer_manager/local_vcc/tests/docker
```

## 2. Docker-level integration test

Located under `tests/docker/`. Brings up a full mini-stack via `docker compose`:

* **rabbitmq** — the real message bus
* **stub_vlm** — a tiny OpenAI-compatible HTTP server standing in for `llama-server`
  (no GPU / GGUF needed). Verdict controlled by `STUB_VERDICT` (`yes` / `no` / `na`).
* **local_vcc** — the actual `Dockerfile.local_vcc` image, with the entrypoint
  overridden to skip `llama-server` startup and just run `python3 -m app.main`
  against the stub.

The pytest driver publishes `AlertVerificationRequest` messages to
`vccVerificationRequestQueue` and asserts on the `AlertVerificationResponse`
that comes back on `vccVerificationResponseQueue`.

Requires a working `docker` + `docker compose` on the host. Tests self-skip when
docker isn't available. See `docker/README.md` for the stack layout and a list
of gotchas (in particular: keep request payloads here in sync with required
fields on `app/models.py`).

```bash
pytest analyzer_manager/local_vcc/tests/docker -v -s
```

> **Note**: the worker communicates over **RabbitMQ**, not Redis. The docker
> test uses `rabbitmq:3.13-management` accordingly.

