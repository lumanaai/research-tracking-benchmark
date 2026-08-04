#!/usr/bin/env bash
set -euo pipefail

# --------------------- USER SETTINGS ---------------------

# Your tracking command bits
EXP_FILE="exps/example/mot/yolox_x_mix_det.py"
CKPT="pretrained/bytetrack_x_mot17.pth.tar"
BATCH=1
DEVICES=1
EXPERIMENT_NAME="yolox_x_mix_det"            # must match track.py --experiment-name
OUTPUTS_ROOT="YOLOX_outputs"                 # where your code writes runs

# Configs directory (all *.json inside will be run)
CONFIGS_DIR="./configs"

# ---------------------------------------------------------

TRACK_SCRIPT="tools/track.py"
EXP_DIR="${OUTPUTS_ROOT}/${EXPERIMENT_NAME}"

# loop over all JSON configs and run track.py for each
shopt -s nullglob
configs=("${CONFIGS_DIR}"/*.json)

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configs found in ${CONFIGS_DIR}"
  exit 1
fi

for cfg in "${configs[@]}"; do
  echo "==============================="
  echo "Running config: $(basename "$cfg")"
  echo "==============================="

  python "$TRACK_SCRIPT" \
    -f "$EXP_FILE" \
    -c "$CKPT" \
    -b "$BATCH" \
    -d "$DEVICES" \
    --fp16 --fuse \
    --config "$cfg"

  # get current run's results dir (from latest_run.txt)
  RESULTS_DIR="$(python tools/get_latest_run.py "$EXP_DIR")" || {
    echo "Failed to resolve latest results dir for $(basename "$cfg")"
    continue
  }
  echo "Current results: $RESULTS_DIR"
done
