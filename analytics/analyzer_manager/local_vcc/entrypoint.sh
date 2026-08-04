#!/bin/bash
# ============================================================
# local_vcc entrypoint:
#   1. Launch llama-server (VLM) in the background
#   2. Wait for the /health endpoint to report "ok"
#   3. Exec the Python RabbitMQ worker in the foreground
# ============================================================
set -e

LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-7999}"
LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://localhost:${LLAMA_SERVER_PORT}}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-/models/model.gguf}"
LLAMA_MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-/models/mmproj.gguf}"
LLAMA_MODEL_NAME="${LLAMA_MODEL_NAME:-local_vcc}"
LLAMA_HEALTH_TIMEOUT="${LLAMA_HEALTH_TIMEOUT:-100}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-4096}"
LLAMA_IMAGE_MAX_TOKENS="${LLAMA_IMAGE_MAX_TOKENS:-2048}"

MMPROJ_ARG=""
if [ -f "${LLAMA_MMPROJ_PATH}" ]; then
    MMPROJ_ARG="--mmproj ${LLAMA_MMPROJ_PATH}"
fi

echo "[local_vcc] Starting llama-server on port ${LLAMA_SERVER_PORT} with model ${LLAMA_MODEL_PATH}"
# shellcheck disable=SC2086
/app/llama-server \
    --host 0.0.0.0 \
    --port "${LLAMA_SERVER_PORT}" \
    --model "${LLAMA_MODEL_PATH}" \
    --alias "${LLAMA_MODEL_NAME}" \
    --n-gpu-layers -1 \
    --jinja \
    --top-p 0.8 \
    --top-k 20 \
    --temp 0.2 \
    --min-p 0.0 \
    --flash-attn on \
    --presence-penalty 1.5 \
    --ctx-size "${LLAMA_CTX_SIZE}" \
    --threads 4 \
    --threads-batch 4 \
    --batch-size 4096 \
    --ubatch-size 512 \
    --cache-ram 0 \
    --metrics \
    --split-mode none \
    --main-gpu 0 \
    --parallel 1 \
    --image-max-tokens "${LLAMA_IMAGE_MAX_TOKENS}" \
    -lv 1 \
    ${MMPROJ_ARG} \
    ${LLAMA_EXTRA_ARGS} > /var/log/llama-server.log 2>&1 &

LLAMA_PID=$!
trap 'echo "[local_vcc] shutting down"; kill -TERM ${LLAMA_PID} 2>/dev/null || true; wait ${LLAMA_PID} 2>/dev/null || true' EXIT INT TERM

echo "[local_vcc] Waiting for llama-server to become healthy (${LLAMA_HEALTH_TIMEOUT}s max)"
for i in $(seq 1 "${LLAMA_HEALTH_TIMEOUT}"); do
    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
        echo "[local_vcc] llama-server exited before becoming healthy"
        echo "[local_vcc] --- llama-server log (last 50 lines) ---"
        tail -n 50 /var/log/llama-server.log 2>/dev/null || true
        exit 1
    fi
    status=$(curl -fsS "${LLAMA_SERVER_URL}/health" 2>/dev/null || true)
    if echo "${status}" | grep -q '"ok"'; then
        echo "[local_vcc] llama-server is healthy"
        break
    fi
    sleep 1
done

echo "[local_vcc] Starting RabbitMQ worker"
exec python3 -u -m app.main

