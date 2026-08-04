#!/bin/bash
echo "building base docker and uploading to docker hub"

# Determine GPU architecture via nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
if echo "$GPU_NAME" | grep -qi "blackwell\|B200\|B100\|GB200\|B300"; then
    TAG="0.1.0-x86_blackwell"
else
    TAG="0.1.0-x86"
fi

echo "Detected GPU: $GPU_NAME -> using tag: $TAG"
docker build -f DockerfileMlBase.cpu -t lumixai/analytic_base:${TAG} . && docker push lumixai/analytic_base:${TAG}

