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
#   DRY_RUN=1 DETECTOR=yolov8 GPUS="0 1 2 3" ./scripts/batch/run_all_parallel.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON" >&2
  exit 1
fi

BENCHMARKS=(${BENCHMARKS:-fasttracker_bench ua_detrac trafficmot cityflow})
TRACKERS=(${TRACKERS:-fasttracker ocsort hybridsort})
DETECTOR="${DETECTOR:-yolov8}"
GPUS=(${GPUS:-0 1 2 3})
WEIGHTS="${WEIGHTS:-}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
# If 1, shard sequences of each benchmark across all GPUS (better for big benches).
SHARD_SEQS="${SHARD_SEQS:-0}"
# Max concurrent track+eval jobs (CPU). Default = #GPUs * 2.
TRACK_JOBS="${TRACK_JOBS:-$(( ${#GPUS[@]} * 2 ))}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/media/7TBSSD/data/tracking/experiments/_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_all_parallel_${DETECTOR}_${TS}.log"
JOB_DIR="$LOG_DIR/jobs_${DETECTOR}_${TS}"
STATUS_DIR="$JOB_DIR/status"
mkdir -p "$JOB_DIR" "$STATUS_DIR"

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
log "gpus:       ${GPUS[*]}"
log "shard_seqs: $SHARD_SEQS"
log "track_jobs: $TRACK_JOBS"
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
    run_id="${bench}_${tracker}_${DETECTOR}_${TS}"
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
    spawn_job "track_${bench}_${tracker}" "$TRACK_JOBS" -- "${cmd[@]}"
  done
done

wait || true
tally
exit $?
