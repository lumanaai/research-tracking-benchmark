#!/usr/bin/env bash
# Parallel detect → track → eval across GPUs / CPU workers.
#
# YOLO detect is the GPU bottleneck; trackers are CPU-only once dets are cached.
# So we:
#   1) convert (sequential; usually cached)
#   2) detect each benchmark on its own GPU (parallel)
#   3) track+eval all bench×tracker jobs in a CPU worker pool
#
# Usage (from project root):
#   DETECTOR=yolov8 GPUS="0 1 2 3" ./scripts/batch/run_all_parallel.sh
#   DETECTOR=yolov8 GPUS="0 1 2 3 4 5 6 7" SHARD_SEQS=1 ./scripts/batch/run_all_parallel.sh
#   TRACK_JOBS=8 DETECTOR=yolov8 GPUS="0 1 2 3" ./scripts/batch/run_all_parallel.sh
#   FPS=10 DETECTOR=yolov8 GPUS="0 1 2 3" ./scripts/batch/run_all_parallel.sh
#   DRY_RUN=1 DETECTOR=yolov8 GPUS="0 1 2 3" ./scripts/batch/run_all_parallel.sh
#
# With FPS set, writes a separate comparison file (does not overwrite full-rate tables):
#   results/comparisons/<detector_id>_fpsN.md
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON" >&2
  exit 1
fi

BENCHMARKS=(${BENCHMARKS:-fasttracker_bench ua_detrac trafficmot cityflow})
TRACKERS=(${TRACKERS:-fasttracker ocsort hybridsort analytics_bytetrack})
DETECTOR="${DETECTOR:-yolov8}"
GPUS=(${GPUS:-0 1 2 3})
WEIGHTS="${WEIGHTS:-}"
# Optional target tracking FPS (subsamples cached dets; omit for full frame rate).
FPS="${FPS:-}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/results/comparisons}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
# If 1, shard sequences of each benchmark across all GPUS (better for big benches).
SHARD_SEQS="${SHARD_SEQS:-0}"
# Max concurrent track+eval jobs (CPU). Default = #GPUs * 2.
TRACK_JOBS="${TRACK_JOBS:-$(( ${#GPUS[@]} * 2 ))}"
DRY_RUN="${DRY_RUN:-0}"
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
LOG="$LOG_DIR/run_all_parallel_${DETECTOR}_${TS}.log"
JOB_DIR="$LOG_DIR/jobs_${DETECTOR}_${TS}"
STATUS_DIR="$JOB_DIR/status"
mkdir -p "$JOB_DIR" "$STATUS_DIR"

fps_tag=""
if [[ -n "$FPS" ]]; then
  fps_tag="_fps${FPS/./p}"
fi

DETECTOR_ID="$(resolve_detector_id)"
COMPARE_OUT=""
if [[ -n "$FPS" ]]; then
  COMPARE_OUT="${COMPARE_DIR}/${DETECTOR_ID}${fps_tag}.md"
fi

log() { echo "$*" | tee -a "$LOG"; }

wait_pool() {
  local max="$1"
  while (( $(jobs -rp | wc -l) >= max )); do
    wait -n 2>/dev/null || wait
  done
}

# Run cmd in background-friendly way; write status file for parent tally.
# Usage: spawn_job <label> <max_parallel> -- <cmd...>
spawn_job() {
  local label="$1"
  local max="$2"
  shift 2
  [[ "${1:-}" == "--" ]] && shift

  local safe
  safe="$(echo "$label" | tr '/: ' '___')"
  local jlog="$JOB_DIR/${safe}.log"
  local st="$STATUS_DIR/${safe}.txt"

  wait_pool "$max"
  log "[start] $label"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "  DRY_RUN: $*"
    echo ok >"$st"
    return 0
  fi
  (
    if "$@" >"$jlog" 2>&1; then
      echo ok >"$st"
      echo "[ok]    $label" >>"$LOG"
      echo "[ok]    $label"
    else
      echo fail >"$st"
      echo "[FAIL]  $label  (see $jlog)" >>"$LOG"
      echo "[FAIL]  $label  (see $jlog)"
      if [[ "$FAIL_FAST" == "1" ]]; then
        echo "FAIL_FAST=1" >"$STATUS_DIR/_abort"
      fi
    fi
  ) &
}

list_seqs() {
  local bench="$1"
  "$PYTHON" - <<PY
from mot_pipeline.registry import get_benchmark
b = get_benchmark("$bench")
split = b.default_split()
b.ensure_mot(split, force=False)
print("\n".join(p.name for p in b.sequence_dirs(split)))
PY
}

tally() {
  local ok=0 fail=0
  local -a failed=()
  for f in "$STATUS_DIR"/*.txt; do
    [[ -e "$f" ]] || continue
    [[ "$(basename "$f")" == _abort ]] && continue
    local name
    name="$(basename "$f" .txt)"
    if [[ "$(cat "$f")" == ok ]]; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
      failed+=("$name")
    fi
  done
  log ""
  log "=== done ==="
  log "ok=$ok fail=$fail"
  if ((${#failed[@]})); then
    log "failed jobs:"
    for j in "${failed[@]}"; do
      log "  - $j"
    done
  fi
  log "findings: /media/7TBSSD/data/tracking/experiments/_findings/<bench>/<tracker>/"
  log "log: $LOG"
  log "job logs: $JOB_DIR"
  return "$fail"
}

log "=== parallel sweep (detector=$DETECTOR) ==="
log "root=$ROOT"
log "python=$PYTHON"
log "benchmarks: ${BENCHMARKS[*]}"
log "trackers:   ${TRACKERS[*]}"
log "detector_id:$DETECTOR_ID"
log "gpus:       ${GPUS[*]}"
log "shard_seqs: $SHARD_SEQS"
log "track_jobs: $TRACK_JOBS"
if [[ -n "$FPS" ]]; then
  log "fps:        $FPS (track subsample)"
  log "compare_out:$COMPARE_OUT"
fi
[[ -n "$WEIGHTS" ]] && log "weights:    $WEIGHTS"
log "log:        $LOG"
log "job logs:   $JOB_DIR"
log ""

# ---- convert ----
for bench in "${BENCHMARKS[@]}"; do
  spawn_job "convert_${bench}" 2 -- \
    "$PYTHON" -m mot_pipeline.run convert --benchmark "$bench"
done
wait || true
if [[ -f "$STATUS_DIR/_abort" ]]; then exit 1; fi
log ""

# ---- detect (GPU parallel) ----
log "=== detect phase ==="
n_gpu=${#GPUS[@]}

if [[ "$DETECTOR" == "gt" || "$DETECTOR" == "existing" ]]; then
  for bench in "${BENCHMARKS[@]}"; do
    spawn_job "detect_${bench}" "$n_gpu" -- \
      "$PYTHON" -m mot_pipeline.run detect --benchmark "$bench" --detector "$DETECTOR"
  done
elif [[ "$SHARD_SEQS" == "1" ]]; then
  for bench in "${BENCHMARKS[@]}"; do
    mapfile -t all_seqs < <(list_seqs "$bench")
    n_seq=${#all_seqs[@]}
    if (( n_seq == 0 )); then
      log "[WARN] no sequences for $bench"
      continue
    fi
    for ((gi=0; gi<n_gpu; gi++)); do
      seqs_chunk=()
      for ((si=gi; si<n_seq; si+=n_gpu)); do
        seqs_chunk+=("${all_seqs[$si]}")
      done
      ((${#seqs_chunk[@]})) || continue
      gpu="${GPUS[$gi]}"
      device="${DEVICE_PREFIX}:${gpu}"
      cmd=(
        "$PYTHON" -m mot_pipeline.run detect
        --benchmark "$bench"
        --detector "$DETECTOR"
        --device "$device"
        --sequences "${seqs_chunk[@]}"
      )
      [[ -n "$WEIGHTS" ]] && cmd+=(--weights "$WEIGHTS")
      spawn_job "detect_${bench}_gpu${gpu}" "$n_gpu" -- "${cmd[@]}"
    done
  done
else
  # One GPU per benchmark (round-robin if more benches than GPUs).
  for i in "${!BENCHMARKS[@]}"; do
    bench="${BENCHMARKS[$i]}"
    gpu="${GPUS[$((i % n_gpu))]}"
    device="${DEVICE_PREFIX}:${gpu}"
    cmd=(
      "$PYTHON" -m mot_pipeline.run detect
      --benchmark "$bench"
      --detector "$DETECTOR"
      --device "$device"
    )
    [[ -n "$WEIGHTS" ]] && cmd+=(--weights "$WEIGHTS")
    spawn_job "detect_${bench}_gpu${gpu}" "$n_gpu" -- "${cmd[@]}"
  done
fi

wait || true
if [[ -f "$STATUS_DIR/_abort" ]]; then exit 1; fi
log ""

# ---- track + eval (CPU parallel; dets cached) ----
log "=== track+eval phase (jobs≤$TRACK_JOBS) ==="
for bench in "${BENCHMARKS[@]}"; do
  for tracker in "${TRACKERS[@]}"; do
    if [[ "$tracker" == "traffictrack" ]]; then
      log "[skip] stub tracker traffictrack"
      continue
    fi
    run_id="${bench}_${tracker}_${DETECTOR}${fps_tag}_${TS}"
    cmd=(
      "$PYTHON" -m mot_pipeline.run all
      --benchmark "$bench"
      --tracker "$tracker"
      --detector "$DETECTOR"
      --run-id "$run_id"
    )
    if [[ "$DETECTOR" == "yolov8" || "$DETECTOR" == "yolox" ]]; then
      # Only used on cache miss; prefer first GPU.
      cmd+=(--device "${DEVICE_PREFIX}:${GPUS[0]}")
    fi
    [[ -n "$WEIGHTS" ]] && cmd+=(--weights "$WEIGHTS")
    [[ -n "$FPS" ]] && cmd+=(--fps "$FPS")
    spawn_job "track_${bench}_${tracker}" "$TRACK_JOBS" -- "${cmd[@]}"
  done
done

wait || true
tally
status=$?

# Write an FPS-tagged comparison table; never overwrite the full-rate md.
# Runs even if some track jobs failed — compare_findings keeps whatever landed.
if [[ -n "$FPS" && "$DRY_RUN" != "1" ]]; then
  log ""
  log "[compare] --detector-id $DETECTOR_ID --target-fps $FPS → $COMPARE_OUT"
  mkdir -p "$COMPARE_DIR"
  if "$PYTHON" scripts/analysis/compare_findings.py \
      --detector-id "$DETECTOR_ID" \
      --target-fps "$FPS" \
      --format both \
      --out "$COMPARE_OUT" >>"$LOG" 2>&1; then
    log "  OK wrote $COMPARE_OUT (+ .csv)"
  else
    log "  FAILED compare_findings (see $LOG)"
    status=1
  fi
fi

exit "$status"
