#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------- USER SETTINGS ---------------------

# Your tracking command bits
EXP_FILE="exps/example/mot/yolox_x_mix_det.py"
CKPT="pretrained/bytetrack_x_mot17.pth.tar"
BATCH=1
DEVICES=1
EXPERIMENT_NAME="yolox_x_mix_det"            # must match track.py --experiment-name
OUTPUTS_ROOT="YOLOX_outputs"                 # where your code writes runs

TRACKEVAL_ROOT="${SCRIPT_DIR}/TrackEval"

BENCHMARK="MOT15"            # Can be changed to other mot benchs.
USE_PARALLEL="False"         # "True" or "False"

# ---------------------------------------------------------

TRACK_SCRIPT="tools/track.py"
HOTA_PREP="${TRACKEVAL_ROOT}/hotaPreparation.py"
RUN_MOT="${TRACKEVAL_ROOT}/scripts/run_mot_challenge.py"
path_folder="${TRACKEVAL_ROOT}/data/"
EXP_DIR="${OUTPUTS_ROOT}/${EXPERIMENT_NAME}"


GT_ROOT="${SCRIPT_DIR}/gt/MOT17" # -> Change to correct ground truth dir
RESULTS_DIR="${SCRIPT_DIR}/YOLOX_outputs/yolox_x_mix_det/run002/track_results" # -> Change to output dir


echo "Current results: $RESULTS_DIR"

  for res_file in "$RESULTS_DIR"/*; do
      # Skip if not a file
      [ -f "$res_file" ] || continue

      # Extract filename without extension
      base="$(basename "$res_file")"       # e.g., MOT17-14-SDP.txt
      seq_name="${base%.*}"                # strip .txt → MOT17-14-SDP

      # Build GT folder path
      gt_folder="${GT_ROOT}/${seq_name}"
      
      gt_file="${GT_ROOT}/${seq_name}/gt.txt"   # adjust if your GT is .../gt/gt.txt
      if [[ ! -f "$gt_file" ]]; then
        echo "No GT found for $seq_name at $gt_file, skipping."
        continue
      fi

      echo "Found GT for $seq_name at $gt_file"

      # Run HOTA prep FROM the results folder; pass only the filename
      pushd "$RESULTS_DIR" >/dev/null

      echo "$HOTA_PREP -d $base -g $gt_file"
      python "$HOTA_PREP" -d "$base" -g "$gt_file" -p "$path_folder" || {
        echo "HOTA_PREP failed for $seq_name"
        popd >/dev/null
        continue
      }

      # Run MOT script (this one usually doesn’t care about CWD)
      log_file="$RESULTS_DIR/${base}.log"

      python "$RUN_MOT" \
          --USE_PARALLEL "$USE_PARALLEL" \
          --METRICS CLEAR \
          --BENCHMARK "$BENCHMARK" \
          >"$log_file" 2>&1 || {
              echo "RUN_MOT failed for $seq_name (see $log_file)"
              popd >/dev/null
              continue
          }
  done

