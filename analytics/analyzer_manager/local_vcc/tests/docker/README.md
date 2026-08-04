# local_vcc docker integration test

`test_docker_integration.py` is a real docker-compose e2e test for the
`local_vcc` worker: it builds the actual `Dockerfile.local_vcc` image, runs it
against a real RabbitMQ broker and a stub VLM server (no GPU / GGUF model
needed), and asserts on the `AlertVerificationResponse` messages that come
back.

## Layout

- `docker-compose.test.yml` — rabbitmq + stub_vlm + local_vcc (entrypoint
  overridden to skip `llama-server`/model loading — see comments in the file).
- `stub_vlm/` — a ~40-line stdlib HTTP server that mimics the
  `POST /v1/chat/completions` shape the real llama-server exposes. Its
  `message.content` is a JSON string matching `CoTStructuredOutput`
  (`{"description": ..., "final_answer": true|false}`, or an HTTP 500 for
  `na`), controlled by the `STUB_VERDICT` env var (`yes` | `no` | `na`), read
  once at process start. It does **not** honor the caller's
  `response_format`/schema — it always emits the `CoTStructuredOutput` shape,
  which is fine as long as every `eventTypeId` used by this test resolves to
  the default schema (none of the current tests exercise `BRANDISHING_WEAPON`,
  the one schema override in `structured_output_map`).
- `test_docker_integration.py` — builds the image, brings the stack up via
  `docker compose`, publishes requests to `vccVerificationRequestQueue`, and
  reads `vccVerificationResponseQueue`.

## Running

```bash
pytest analyzer_manager/local_vcc/tests/docker -v -s
```

Requires `docker` + the `compose` plugin on `PATH`; the whole module
self-skips (`pytestmark`) if the docker daemon isn't reachable. Verified
working end-to-end on 2026-07-15, both before and after the move to
structured (JSON-schema-constrained) VLM output (`4 passed in ~110-130s`);
the image build is cached after the first run so re-runs are much faster.

## Gotchas / things that have bitten this test before

- **The request/response models are the source of truth, not this file.**
  `app/models.py` defines `AlertVerificationRequest` / `AlertVerificationResponse`
  as Pydantic models with no defaults on most fields — e.g. `cameraId` is a
  *required* string. If a new required field is added to those models, every
  request dict built in this test (and in `tests/test_worker.py` /
  `tests/test_verifier.py`) needs the same field, otherwise
  `AlertVerificationRequest.model_validate(...)` raises inside
  `worker.py::_on_message`, which is swallowed and turned into a
  `success=False` response — so the *happy-path* test (`test_verifies_alert_yes`)
  fails with `assert False is True` while the two failure-path tests keep
  passing "for the wrong reason". Grep for `AlertVerificationRequest(` /
  request dict literals across `tests/` whenever `app/models.py` changes.
- `test_malformed_request_publishes_failure` intentionally sends non-JSON
  bytes, so it never reaches model validation — it doesn't need to carry the
  required fields.
- Noisy `pika`/`StreamLostError`/`ConnectionResetError` tracebacks in the
  captured log during teardown are expected: the `compose_stack` fixture's
  `finally` block runs `docker compose down -v` while pika's retry/heartbeat
  machinery is still winding down against the now-dead broker. Harmless as
  long as the test's own assertions pass.
- The `local_vcc` service in `docker-compose.test.yml` overrides the image's
  `ENTRYPOINT` to run `python3 -u -m app.main` directly — it does **not**
  exercise `entrypoint.sh` or real `llama-server`/GGUF loading. There is no
  test in this repo that boots the real 18GB VIA model; this suite only
  proves the RabbitMQ <-> worker <-> OpenAI-compatible-HTTP wiring.
- As of the structured-output change (`app/vlm_client.py::query_structured`,
  `app/vcc_common.py::CoTStructuredOutput`/`structured_output_map`), the
  worker no longer parses free-text "yes"/"no"/"NA" — every VLM reply must be
  a JSON object matching the resolved schema. `stub_vlm/server.py` was updated
  to emit that JSON; if `stub_vlm` and `app/vcc_common.py`'s schemas ever
  drift apart, the symptom looks identical to the `cameraId` bug above:
  `test_verifies_alert_yes` fails with `success=False` because
  `schema.model_validate_json(content)` raises inside `query_structured` and
  gets swallowed into a `None` return.
- As of the reliability rework (`app/worker.py`), message handling is no
  longer synchronous inside `_on_message` for the happy/verifier-exception
  paths: valid requests are enqueued onto a bounded internal queue and
  processed by a background worker thread, which schedules the eventual
  publish + ack back onto the IO thread via
  `connection.add_callback_threadsafe`. This test still passes without any
  changes because it only asserts on the eventual response message, not on
  synchronous timing — but if you're debugging a hang here, check whether the
  background worker thread in the container actually started (see
  `app/worker.py`'s "Reliability under load" section in the top-level
  README) rather than assuming `_on_message` itself is stuck.
