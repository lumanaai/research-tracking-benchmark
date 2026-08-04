#!/bin/bash
uvicorn offline_main:app \
    --host 0.0.0.0 \
    --port 5055 \
    --workers 1 \
    --log-level debug \
    --log-config ./log_config.yaml \
    --no-access-log
