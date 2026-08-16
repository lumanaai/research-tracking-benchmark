#!/usr/bin/env bash
# Run every wired tracker on every benchmark (full default split) with GT dets.
# Usage (from project root):
#   ./scripts/batch/run_all_gt_trackers.sh
#   BENCHMARKS="fasttracker_bench ua_detrac" TRACKERS="ocsort" ./scripts/batch/run_all_gt_trackers.sh
#   FPS=10 ./scripts/batch/run_all_gt_trackers.sh
#   DRY_RUN=1 ./scripts/batch/run_all_gt_trackers.sh
#
# With FPS set, findings are recorded under the same detector_id but a new run_id
# (…_fpsN_…) and a separate comparison file is written:
#   results/comparisons/<detector_id>_fpsN.md
# Full-rate tables like results/comparisons/gt_vehicles.md are left untouched.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON" >&2
  exit 1
fi

# Default: all benchmarks × motion trackers (skip traffictrack stub).
BENCHMARKS=(${BENCHMARKS:-fasttracker_bench ua_detrac trafficmot cityflow lumana_benchmark})
TRACKERS=(${TRACKERS:-fasttracker ocsort hybridsort analytics_bytetrack analytics_bytetrack_plus botsort})
DETECTOR="${DETECTOR:-gt}"
# Used when DETECTOR=yolov8 (e.g. cuda:0). Ignored for gt/existing.
DEVICE="${DEVICE:-}"
# Optional YOLO weights override (default: mot_pipeline DEFAULT_YOLO_WEIGHTS).
WEIGHTS="${WEIGHTS:-}"
# Optional target tracking FPS (subsamples cached dets; omit for full frame rate).
FPS="${FPS:-}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/results/comparisons}"
DRY_RUN="${DRY_RUN:-0}"
# If 1, abort on first failed job; otherwise continue and report at the end.
FAIL_FAST="${FAIL_FAST:-0}"

resolve_detector_id() {
  WEIGHTS="$WEIGHTS" DETECTOR="$DETECTOR" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from mot_pipeline.paths import DEFAULT_YOLO_WEIGHTS
from mot_pipeline.registry import get_detector

det = os.environ["DETECTOR"]
kwargs = {}
if det == "yolov8":
    w = os.environ.get("WEIGHTS") or ""
    kwargs["weights"] = Path(w) if w else DEFAULT_YOLO_WEIGHTS
elif det == "existing":
    pass
elif det == "gt":
    kwargs["benchmark"] = "fasttracker_bench"
print(get_detector(det, **kwargs).detector_id())
PY
}

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/media/7TBSSD/data/tracking/experiments/_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_all_${DETECTOR}_${TS}.log"

ok=0
fail=0
skip=0
declare -a FAILED_JOBS=()

fps_tag=""
if [[ -n "$FPS" ]]; then
  fps_tag="_fps${FPS/./p}"
fi

DETECTOR_ID="$(resolve_detector_id)"
COMPARE_OUT=""
if [[ -n "$FPS" ]]; then
  COMPARE_OUT="${COMPARE_DIR}/${DETECTOR_ID}${fps_tag}.md"
fi

echo "=== all trackers × all benchmarks (detector=$DETECTOR) ===" | tee "$LOG"
echo "root=$ROOT" | tee -a "$LOG"
echo "python=$PYTHON" | tee -a "$LOG"
echo "benchmarks: ${BENCHMARKS[*]}" | tee -a "$LOG"
echo "trackers:   ${TRACKERS[*]}" | tee -a "$LOG"
echo "detector_id:$DETECTOR_ID" | tee -a "$LOG"
if [[ -n "$FPS" ]]; then
  echo "fps:        $FPS (track subsample)" | tee -a "$LOG"
  echo "compare_out:$COMPARE_OUT" | tee -a "$LOG"
fi
if [[ -n "$WEIGHTS" ]]; then
  echo "weights:    $WEIGHTS" | tee -a "$LOG"
fi
if [[ -n "$DEVICE" ]]; then
  echo "device:     $DEVICE" | tee -a "$LOG"
fi
echo "log:        $LOG" | tee -a "$LOG"
echo | tee -a "$LOG"

# Ensure MOT layouts exist once per benchmark (full default split).
for bench in "${BENCHMARKS[@]}"; do
  echo "[convert] $bench" | tee -a "$LOG"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN: $PYTHON -m mot_pipeline.run convert --benchmark $bench" | tee -a "$LOG"
    continue
  fi
  if ! "$PYTHON" -m mot_pipeline.run convert --benchmark "$bench" >>"$LOG" 2>&1; then
    echo "  FAILED convert $bench" | tee -a "$LOG"
    FAILED_JOBS+=("convert:$bench")
    fail=$((fail + 1))
    if [[ "$FAIL_FAST" == "1" ]]; then
      echo "FAIL_FAST=1 — stopping." | tee -a "$LOG"
      exit 1
    fi
  fi
done
echo | tee -a "$LOG"

for bench in "${BENCHMARKS[@]}"; do
  for tracker in "${TRACKERS[@]}"; do
    run_id="${bench}_${tracker}_${DETECTOR}${fps_tag}_${TS}"
    echo "[run] bench=$bench tracker=$tracker detector=$DETECTOR run_id=$run_id" | tee -a "$LOG"

    if [[ "$tracker" == "traffictrack" ]]; then
      echo "  SKIP stub tracker traffictrack" | tee -a "$LOG"
      skip=$((skip + 1))
      continue
    fi

    cmd=(
      "$PYTHON" -m mot_pipeline.run all
      --benchmark "$bench"
      --tracker "$tracker"
      --detector "$DETECTOR"
      --run-id "$run_id"
    )
    if [[ -n "$DEVICE" ]]; then
      cmd+=(--device "$DEVICE")
    fi
    if [[ -n "$WEIGHTS" ]]; then
      cmd+=(--weights "$WEIGHTS")
    fi
    if [[ -n "$FPS" ]]; then
      cmd+=(--fps "$FPS")
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY_RUN: ${cmd[*]}" | tee -a "$LOG"
      continue
    fi

    if "${cmd[@]}" >>"$LOG" 2>&1; then
      echo "  OK $run_id" | tee -a "$LOG"
      ok=$((ok + 1))
    else
      echo "  FAILED $run_id (see $LOG)" | tee -a "$LOG"
      FAILED_JOBS+=("$run_id")
      fail=$((fail + 1))
      if [[ "$FAIL_FAST" == "1" ]]; then
        echo "FAIL_FAST=1 — stopping." | tee -a "$LOG"
        exit 1
      fi
    fi
  done
done

echo | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
echo "ok=$ok fail=$fail skip=$skip" | tee -a "$LOG"
if ((${#FAILED_JOBS[@]})); then
  echo "failed jobs:" | tee -a "$LOG"
  for j in "${FAILED_JOBS[@]}"; do
    echo "  - $j" | tee -a "$LOG"
  done
fi
echo "findings: /media/7TBSSD/data/tracking/experiments/_findings/<bench>/<tracker>/" | tee -a "$LOG"
echo "log: $LOG" | tee -a "$LOG"

# Write an FPS-tagged comparison table; never overwrite the full-rate md.
if [[ -n "$FPS" && "$ok" -gt 0 && "$DRY_RUN" != "1" ]]; then
  echo | tee -a "$LOG"
  echo "[compare] --detector-id $DETECTOR_ID --target-fps $FPS → $COMPARE_OUT" | tee -a "$LOG"
  mkdir -p "$COMPARE_DIR"
  if "$PYTHON" scripts/analysis/compare_findings.py \
      --detector-id "$DETECTOR_ID" \
      --target-fps "$FPS" \
      --format both \
      --out "$COMPARE_OUT" >>"$LOG" 2>&1; then
    echo "  OK wrote $COMPARE_OUT (+ .csv)" | tee -a "$LOG"
  else
    echo "  FAILED compare_findings (see $LOG)" | tee -a "$LOG"
    fail=$((fail + 1))
  fi
fi

exit $((fail > 0 ? 1 : 0))
