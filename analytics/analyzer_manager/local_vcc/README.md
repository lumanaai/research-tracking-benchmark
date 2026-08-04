# local_vcc

Local Visual Content Check (VCC) service. A single Docker container that runs
`llama.cpp` server (VLM) alongside a Python worker. The worker consumes alert
verification requests from RabbitMQ, sends the alert's images to the local VLM
constrained to a JSON schema, and publishes the verification result.

The bundled model (`models/model.gguf` + `models/mmproj.gguf`) is
**Qwen3-VL-8B-Instruct** (dense, `general.architecture = qwen3vl`,
`qwen3vl.context_length = 262144` native) — confirmed by reading the GGUF
header metadata directly. `setup_dockers.py --local-vcc` syncs both files
from `s3://<WEIGHTS_BUCKET>/cloud/oracle/Qwen3_8B/` (see
`sync_local_vcc_model` in that script) into `models/` under those exact
names, which is what `Dockerfile.local_vcc`'s `LLAMA_MODEL_PATH` /
`LLAMA_MMPROJ_PATH` defaults expect.

## Layout

```
analyzer_manager/
├── Dockerfile.local_vcc         # llama.cpp:server-cuda base + python worker
└── local_vcc/
    ├── entrypoint.sh            # starts llama-server (bg) + python worker (fg)
    ├── requirements.txt
    ├── models/                  # place GGUF + mmproj files here before docker build
    └── app/
        ├── main.py              # process entrypoint
        ├── worker.py            # pika blocking consumer
        ├── verifier.py          # AlertVerifier: prompt+schema -> images -> VLM -> structured verdict
        ├── vlm_client.py        # OpenAI-compatible client for llama-server (JSON-schema constrained)
        ├── prompts.py           # PromptResolver (eventType -> VccAlertType -> prompt + schema)
        ├── vcc_common.py        # prompt_map + structured_output_map keyed by VccAlertType
        ├── alert_types.py       # thin wrapper: re-exports general.alert_types + builds VccAlertType
        ├── image_utils.py       # load image paths -> base64 data URLs
        ├── models.py            # Pydantic request/response models
        └── logger.py            # wraps analyzer_manager/app/general/logger.py
```

Shared modules (single source of truth):

- `analyzer_manager/app/general/alert_types.py` — `AlertCategory`, `AlertType`,
  `alert_mapping`, `event_type_to_alert_type`, …
- `analyzer_manager/app/general/logger.py` — JSON logger with daily rotation.

The image copies both files (and the lightweight `app/general/__init__.py`)
into `/opt/local_vcc/general/` so they resolve as `general.alert_types` and
`general.logger`. Nothing else from `app/general/` is copied — the heavy
`general.core` / `general.analyzer_general` (cv2, numpy, …) are **not**
imported by the worker.

## Message contract

Request queue (`VCC_REQUEST_QUEUE`, default `vccVerificationRequestQueue`):

```jsonc
{
  "alertInstanceId": 123,
  "cameraId": "cam-42",                   // required
  "eventTypeId": 3,                       // category * 1_000_000 + flow
  "filterPrompt": null,                   // required for periodicText / unknown types
  "images": ["/shared/alerts/edge/cam/abc.jpg"],
  "requestStartTimestamp": 1720000000000
}
```

Response queue (`VCC_RESPONSE_QUEUE`, default `vccVerificationResponseQueue`):

```json
{
  "alertInstanceId": 123,
  "cameraId": "cam-42",
  "verified": true,
  "success": true,
  "message": "{\"description\": \"...\", \"final_answer\": true}"
}
```

- `cameraId` is a required field on both the request and response models. A
  request missing it fails Pydantic validation, which the worker reports back
  as `success=false, verified=false` (see `worker.py::_on_message`).

- `verified` — the VLM's structured response's `final_answer` was `True` on
  at least one image (see "Structured output" below).
- `success`  — the VLM returned a response that parsed into the expected
  schema (as opposed to a request error or an unparsable response).
- `message`  — on success, the JSON-serialized structured-output object
  (`model_dump_json()`) for the last-queried image.
- Image paths must be reachable **inside** the container (mount the shared
  alerts / dev/shm volume).

## Prompt resolution

`eventTypeId` is encoded as `category * 1_000_000 + flow` (matching the edge
stack; see `AlertCategory` / `SafetyType` / … in
`analyzer_manager/app/general/core.py`). At request time:

1. `event_type_to_alert_type(eventTypeId)` → `AlertType` via `alert_mapping`
   (mirrored in `app/alert_types.py`).
2. `alert_type_to_vcc(...)` maps the `AlertType` to a `VccAlertType` alias.
   `VccAlertType` is **built dynamically** from `_VCC_ALERT_ALIASES` in
   `alert_types.py` — add an entry there whenever you add a prompt.
3. `prompt_map[vcc_type]` from `app/vcc_common.py` is the prompt.
4. `PERIODICTEXT` prompts are `str.format` templates; the request's
   `filterPrompt` is substituted as `{question}`.
5. If the event type is unknown, `filterPrompt` (if present) is wrapped by the
   default template `does this image contains {q}? Answer yes or no only.`.
6. If neither route yields a prompt, the response is
   `success=false, verified=false`.

Adding a new prompt:

1. Add an alias to `_VCC_ALERT_ALIASES` in `app/alert_types.py`
   (e.g. `"SMOKE": AlertType.<some_alert_type>`).
2. Add the corresponding entry to `prompt_map` in `app/vcc_common.py`.
3. (Optional) Add a schema override to `structured_output_map` if the default
   `description`/`final_answer` shape isn't expressive enough — see below.

## Structured output

`VlmClient.query_structured` no longer parses free-text "yes"/"no"/"NA"
answers. Every VLM call is constrained (via the OpenAI-compatible
`response_format: {"type": "json_schema", ...}` param, which llama.cpp's
server converts into a GBNF grammar) to return JSON matching a Pydantic
schema, resolved per `eventTypeId` by `PromptResolver.resolve_schema`:

- `structured_output_map` (`app/vcc_common.py`) maps a `VccAlertType` to a
  schema class; anything not in the map falls back to
  `default_structured_output` (`CoTStructuredOutput`).
- Every schema must expose a `final_answer: bool` — either as a plain field
  (`CoTStructuredOutput`) or a computed `@property` combining several fields
  (`BrandishingWeaponData.final_answer = visible_firearm and firearm_in_hands`).
  `AlertVerifier.verify` short-circuits (`verified=True`) on the first image
  whose `final_answer` is `True`.
- If a VLM call fails outright, or its response can't be parsed into the
  schema, `query_structured` returns `None` and the verifier reports
  `success=false, verified=false` — there is no more `"NA <reason>"` string
  sentinel.

Adding a schema override:

1. Define a `BaseModel` subclass in `app/vcc_common.py` with a `description`
   field plus whatever booleans/fields the prompt asks the VLM to reason
   through, and a `final_answer` (field or property).
2. Add `VccAlertType.<X>: YourSchema` to `structured_output_map`.

## Reliability under load

`VccWorker` (`app/worker.py`) decouples **consuming from RabbitMQ** from
**running the VLM query**, so a backlog of requests can never silently vanish
and one slow/wedged VLM call can't stall the broker connection:

- An IO thread runs `channel.start_consuming()` with `prefetch_count=0`
  (unlimited) — it keeps pulling messages off `VCC_REQUEST_QUEUE` as fast as
  the broker will send them, parses/validates each one, and drops it onto a
  bounded in-process queue (`_pending`) stamped with the time it was accepted.
  It never blocks on a VLM call itself.
- A single background worker thread (matching llama-server's `--parallel 1`
  — there's only one GPU inference slot, so more worker threads wouldn't add
  throughput) pulls jobs off `_pending` **in order** and runs
  `AlertVerifier.verify` on each.
- **`_pending` is bounded** to `VCC_MAX_TIMEOUT_SECONDS` slots (assuming
  ~1s/job as a rule of thumb for sizing). If it's full when a new message
  arrives, that request is answered immediately with
  `success=false, verified=false` ("internal queue full") instead of being
  queued — it would already be guaranteed to miss its deadline.
- **Every job's queue-wait is checked before it starts processing.** If a
  job has already been sitting in `_pending` for `>= VCC_MAX_TIMEOUT_SECONDS`
  by the time the worker thread reaches it, it gets an immediate
  `"Timed out after Ns waiting to be processed"` response — the VLM is never
  called for it. (This check only happens once, at dequeue time — an alert
  with many images can still overrun `VCC_MAX_TIMEOUT_SECONDS` once its own
  `verify()` call starts, since each image is a separate bounded VLM call.)
- **The VLM call itself has no hidden retries.** `VlmClient` sets
  `max_retries=0` on the OpenAI client (the SDK defaults to 2, which would
  otherwise let one image call silently balloon to ~3x `LLAMA_REQUEST_TIMEOUT`)
  and a short default timeout (`LLAMA_REQUEST_TIMEOUT`, 5s) — a wedged
  llama-server fails one image fast instead of hanging the worker thread.
- **Every accepted request gets exactly one response**, whether it succeeds,
  fails, or times out — publish + ack happen together, scheduled back onto
  the IO thread via `connection.add_callback_threadsafe` (pika's
  `BlockingConnection` isn't thread-safe, so the worker thread can't touch
  the channel directly).

What this does **not** cover: `VCC_QUEUE_TTL` (RabbitMQ's own per-message TTL)
is still in effect as a broker-side backstop. It only matters if the worker
process itself is disconnected/down for a stretch — while the worker is up,
`prefetch_count=0` means messages reach `_on_message` almost immediately after
arrival, so `VCC_MAX_TIMEOUT_SECONDS` (an application-level, always-answered
timeout) is what actually governs behavior under load. There's currently no
dead-letter queue for the (narrower) case of the process being down long
enough for the broker to expire messages before any consumer ever sees them —
ask if you want that closed too.

## Context size / vision-token capacity

`--ctx-size` (env `LLAMA_CTX_SIZE`, default `4096`) and `--image-max-tokens`
(env `LLAMA_IMAGE_MAX_TOKENS`, default `2048`) in `entrypoint.sh` were
empirically verified against the real Qwen3-VL-8B model + mmproj on
2026-07-19 (`tests/test_e2e_vlm_capacity.py`, GPU-gated).

**No `--image-min-tokens`.** It was tried and removed: it only forces small
images to be *upscaled* to hit a token floor, which adds no real detail
(upscaling can't recover information the source pixels don't have) and just
burns extra compute/latency — measured directly: a native 320x240 image used
~82 vision tokens, but forcing it up to a 1024-token floor made it as slow as
a real 1280x720 image for no benefit. Qwen3-VL's dynamic-resolution encoder
is trained across a wide range of native token counts, so small images at
their natural (low) token count are not out-of-distribution for the model.

**`--image-max-tokens` is still required** and is the one doing real work:
`app/image_utils.py` does **not** resize images before sending them to the
VLM, and nothing upstream of `local_vcc` guarantees alert images are small —
a camera still could plausibly be 4K+. llama.cpp's vision encoder here uses
*dynamic* resolution, so without an explicit max cap, vision-token usage
scales directly with input pixel count and a high-res image can blow the
context budget outright (measured empirically with no cap: a 2560x1440 image
alone used ~3800/4096 tokens, and 3840x2160 failed outright with an HTTP 400
`exceed_context_size_error`).

Worst case measured **with the current config** (no min-tokens,
`--image-max-tokens 2048`, `--ctx-size 4096`), across every `prompt_map`
entry x five resolutions from 1280x720 up to 4656x3492 (~16MP), using the
real structured-output contract (`response_format` json_schema,
`max_completion_tokens=200`): **2254/4096 tokens (55%)** — comfortable
headroom (`1920x1080` + `FIRE`). `image-max-tokens=2048` reliably caps vision
tokens to roughly that budget regardless of input resolution — e.g. a
4656x3492 image landed at ~2150-2200 total tokens, same ballpark as
1920x1080.
`tests/test_e2e_vlm_capacity.py::test_high_resolution_image_does_not_exceed_context`
guards this regression going forward.

Bottom line: `ctx-size 4096` is safe **only because `--image-max-tokens` is
also set** — if you ever change `LLAMA_CTX_SIZE` or `LLAMA_IMAGE_MAX_TOKENS`
away from their defaults, or remove the max-tokens cap, re-run the capacity
test before trusting the new values.

## Environment variables

| Var                       | Default                                    | Purpose                          |
|----------------------------|--------------------------------------------|----------------------------------|
| `RABBIT_HOST`              | `rabbitmq`                                 | RabbitMQ broker host             |
| `VCC_REQUEST_QUEUE`        | `vccVerificationRequestQueue`              | Request queue name               |
| `VCC_RESPONSE_QUEUE`       | `vccVerificationResponseQueue`             | Response queue name              |
| `VCC_QUEUE_TTL`            | `60000`                                    | Broker-side queue TTL in ms (backstop only, see above) |
| `VCC_MAX_TIMEOUT_SECONDS`  | `30`                                       | Max time (seconds) a request may wait in `_pending` before it's answered with a timeout, instead of being verified. Also sizes `_pending`'s capacity |
| `LLAMA_SERVER_URL`         | `http://localhost:7999`                    | Local llama.cpp base URL         |
| `LLAMA_SERVER_PORT`        | `7999`                                     | Port llama-server binds to       |
| `LLAMA_REQUEST_TIMEOUT`    | `5`                                        | Per-VLM-call timeout in seconds (no retries — see above) |
| `LLAMA_MODEL_NAME`         | `local_vcc`                                | Model alias used in requests     |
| `LLAMA_MODEL_PATH`         | `/models/model.gguf`                       | GGUF weights                     |
| `LLAMA_MMPROJ_PATH`        | `/models/mmproj.gguf`                      | Multi-modal projector (optional) |
| `LLAMA_CTX_SIZE`           | `4096`                                     | llama-server `--ctx-size` (see "Context size / vision-token capacity" above before changing) |
| `LLAMA_IMAGE_MAX_TOKENS`   | `2048`                                     | llama-server `--image-max-tokens` (see "Context size / vision-token capacity" above before changing) |
| `LLAMA_EXTRA_ARGS`         | *(empty)*                                  | Extra `llama-server` flags       |
| `LLAMA_HEALTH_TIMEOUT`     | `30`                                        | Seconds to wait for health       |
| `LOG_LEVEL`                | `INFO`                                     | Python logger level              |

## Build & run

The build **must** be invoked with the `analyzer_manager/` directory as the
build context (not `local_vcc/`) so the image can reuse the shared modules
from `app/general/` (single source of truth for `AlertType` / `alert_mapping`
and the JSON logger).

```bash
# 1. Place your GGUF + mmproj files under analyzer_manager/local_vcc/models/
# 2. Build (from repo root):
docker build \
  -f analyzer_manager/Dockerfile.local_vcc \
  -t lumixai/local_vcc \
  analyzer_manager/

# 3. Run:
docker run --rm --runtime=nvidia \
  -e RABBIT_HOST=rabbitmq \
  -v $HOME/assets:/assets:ro \
  -v $HOME/logs:/usr/src/app/logs \
  --network=host \
  lumixai/local_vcc
```

The container:
1. Launches `llama-server` on `LLAMA_SERVER_PORT` with the configured GGUF (+ mmproj).
2. Polls `/health` until it reports `"ok"` (or `LLAMA_HEALTH_TIMEOUT` seconds).
3. Starts the Python worker, which consumes/publishes on RabbitMQ.
