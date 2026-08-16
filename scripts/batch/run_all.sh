#!/usr/bin/env bash
# Unified sweep: all trackers × all benchmarks × {gt, yolov8} × {5, 10, full} FPS.
#
# By default SKIP_DETECT=1: uses the shared detections/ cache only (no YOLO/GT
# re-detect). Track+eval never call `all`, so dets are not touched again.
#
# Usage (from project root):
#   ./scripts/batch/run_all.sh
#   DRY_RUN=1 ./scripts/batch/run_all.sh
#   PARALLEL=0 ./scripts/batch/run_all.sh
#   GPUS="0 1 2 3" TRACK_JOBS=8 FPS_VALUES="5" ./scripts/batch/run_all.sh
#   SKIP_DETECT=0 ./scripts/batch/run_all.sh   # only if you need to (re)build dets
#   DETECTORS="gt" FPS_VALUES="10 full" ./scripts/batch/run_all.sh
#
# Comparison tables:
#   results/comparisons/gt_vehicles.md / _fps5.md / _fps10.md
#   results/comparisons/yolov8m_expert_eff.md / _fps5.md / _fps10.md
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON" >&2
  exit 1
fi

BENCHMARKS=(${BENCHMARKS:-fasttracker_bench ua_detrac trafficmot cityflow lumana_benchmark})
TRACKERS=(${TRACKERS:-fasttracker ocsort hybridsort analytics_bytetrack analytics_bytetrack_plus botsort})
DETECTORS=(${DETECTORS:-gt yolov8})
# "full" = every frame (no --fps). Numeric values subsample cached dets.
FPS_VALUES=(${FPS_VALUES:-5 10 full})

# Default: same checkpoint used to build the shared detections/ cache
# (/media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt).
# Do NOT point at models/… — a different absolute path used to rewrite
# meta.json and force a full YOLO re-detect even when det.txt files exist.
WEIGHTS="${WEIGHTS:-/media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt}"
# 1 = skip detect phase entirely (track from existing detections/ cache).
# Default 1: dets are shared and do not change across FPS sweeps.
SKIP_DETECT="${SKIP_DETECT:-1}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/results/comparisons}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
GPUS=(${GPUS:-0 1 2 3})
# 1 = parallel detect + track pool (default); 0 = sequential like run_all_gt_trackers.
PARALLEL="${PARALLEL:-1}"
SHARD_SEQS="${SHARD_SEQS:-0}"
TRACK_JOBS="${TRACK_JOBS:-$(( ${#GPUS[@]} * 2 ))}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

if [[ " ${DETECTORS[*]} " == *" yolov8 "* || " ${DETECTORS[*]} " == *" yolox "* ]]; then
  if [[ "$SKIP_DETECT" != "1" && ! -f "$WEIGHTS" ]]; then
    echo "ERROR: YOLO weights not found: $WEIGHTS" >&2
    exit 1
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/media/7TBSSD/data/tracking/experiments/_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_all_${TS}.log"
JOB_DIR="$LOG_DIR/jobs_run_all_${TS}"
STATUS_DIR="$JOB_DIR/status"
mkdir -p "$JOB_DIR" "$STATUS_DIR" "$COMPARE_DIR"

ok=0
fail=0
skip=0
declare -a FAILED_JOBS=()

log() { echo "$*" | tee -a "$LOG"; }

wait_pool() {
  local max="$1"
  while (( $(jobs -rp | wc -l) >= max )); do
    wait -n 2>/dev/null || wait
  done
}

spawn_job() {
  local label="$1"
  local max="$2"
  shift 2
  [[ "${1:-}" == "--" ]] && shift

  local safe
  safe="$(echo "$label" | tr '/: ' '___')"
  local jlog="$JOB_DIR/${safe}.log"
  local st="$STATUS_DIR/${safe}.txt"

  if [[ "$PARALLEL" == "1" ]]; then
    wait_pool "$max"
  fi
  log "[start] $label"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "  DRY_RUN: $*"
    echo ok >"$st"
    return 0
  fi

  run_one() {
    if "$@" >"$jlog" 2>&1; then
      echo ok >"$st"
      echo "[ok]    $label" >>"$LOG"
      echo "[ok]    $label"
      return 0
    else
      echo fail >"$st"
      echo "[FAIL]  $label  (see $jlog)" >>"$LOG"
      echo "[FAIL]  $label  (see $jlog)"
      if [[ "$FAIL_FAST" == "1" ]]; then
        echo "FAIL_FAST=1" >"$STATUS_DIR/_abort"
      fi
      return 1
    fi
  }

  if [[ "$PARALLEL" == "1" ]]; then
    ( run_one "$@" ) &
  else
    run_one "$@" || true
  fi
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

resolve_detector_id() {
  local det="$1"
  WEIGHTS="$WEIGHTS" DETECTOR="$det" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from mot_pipeline.registry import get_detector

det = os.environ["DETECTOR"]
kwargs = {}
if det == "yolov8":
    kwargs["weights"] = Path(os.environ["WEIGHTS"])
elif det == "gt":
    kwargs["benchmark"] = "fasttracker_bench"
print(get_detector(det, **kwargs).detector_id())
PY
}

# Short comparison filenames used in results/comparisons/.
compare_stem() {
  local det_id="$1"
  case "$det_id" in
    gt_vehicles) echo "gt_vehicles" ;;
    yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles) echo "yolov8m_expert_eff" ;;
    *) echo "$det_id" ;;
  esac
}

fps_tag_for() {
  local fps="$1"
  if [[ "$fps" == "full" || -z "$fps" ]]; then
    echo ""
  else
    echo "_fps${fps/./p}"
  fi
}

write_compare() {
  local det_id="$1"
  local fps="$2"
  local stem
  stem="$(compare_stem "$det_id")"
  local out tag
  tag="$(fps_tag_for "$fps")"
  out="${COMPARE_DIR}/${stem}${tag}.md"

  local cmd=(
    "$PYTHON" scripts/analysis/compare_findings.py
    --detector-id "$det_id"
    --format both
    --out "$out"
  )
  if [[ "$fps" != "full" && -n "$fps" ]]; then
    cmd+=(--target-fps "$fps")
  fi

  log "[compare] ${cmd[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "  DRY_RUN compare → $out"
    return 0
  fi
  if "${cmd[@]}" >>"$LOG" 2>&1; then
    log "  OK wrote $out (+ .csv)"
  else
    log "  FAILED compare_findings → $out"
    FAILED_JOBS+=("compare:${stem}${tag}")
    fail=$((fail + 1))
  fi
}

# track (using cached dets) then eval — never calls detect.
track_then_eval() {
  local rid="$1"
  shift
  "$@" && "$PYTHON" -m mot_pipeline.run eval --run-id "$rid"
}

tally_status_files() {
  local local_ok=0 local_fail=0
  local -a failed=()
  for f in "$STATUS_DIR"/*.txt; do
    [[ -e "$f" ]] || continue
    [[ "$(basename "$f")" == _abort ]] && continue
    local name
    name="$(basename "$f" .txt)"
    if [[ "$(cat "$f")" == ok ]]; then
      local_ok=$((local_ok + 1))
    else
      local_fail=$((local_fail + 1))
      failed+=("$name")
    fi
  done
  ok=$local_ok
  fail=$((fail + local_fail))
  for j in "${failed[@]}"; do
    FAILED_JOBS+=("$j")
  done
}

log "=== run_all (detectors × FPS × trackers × benchmarks) ==="
log "root=$ROOT"
log "python=$PYTHON"
log "benchmarks: ${BENCHMARKS[*]}"
log "trackers:   ${TRACKERS[*]}"
log "detectors:  ${DETECTORS[*]}"
log "fps:        ${FPS_VALUES[*]}"
log "weights:    $WEIGHTS"
log "skip_detect:$SKIP_DETECT"
log "parallel:   $PARALLEL"
log "gpus:       ${GPUS[*]}"
log "track_jobs: $TRACK_JOBS"
log "shard_seqs: $SHARD_SEQS"
log "compare_dir:$COMPARE_DIR"
log "log:        $LOG"
log "job logs:   $JOB_DIR"
log ""

# ---- convert (once) ----
log "=== convert ==="
for bench in "${BENCHMARKS[@]}"; do
  spawn_job "convert_${bench}" 2 -- \
    "$PYTHON" -m mot_pipeline.run convert --benchmark "$bench"
done
[[ "$PARALLEL" == "1" ]] && wait || true
if [[ -f "$STATUS_DIR/_abort" ]]; then exit 1; fi
log ""

n_gpu=${#GPUS[@]}

for detector in "${DETECTORS[@]}"; do
  if [[ "$detector" == "traffictrack" ]]; then
    continue
  fi
  detector_id="$(resolve_detector_id "$detector")"
  log "=== detector=$detector  detector_id=$detector_id ==="

  # ---- detect once for this detector (reuse cache across FPS) ----
  if [[ "$SKIP_DETECT" == "1" ]]; then
    log "--- detect ($detector) SKIPPED (SKIP_DETECT=1; using detections/ cache) ---"
  else
    log "--- detect ($detector) ---"
    if [[ "$detector" == "gt" || "$detector" == "existing" ]]; then
      for bench in "${BENCHMARKS[@]}"; do
        spawn_job "detect_${detector}_${bench}" "$n_gpu" -- \
          "$PYTHON" -m mot_pipeline.run detect --benchmark "$bench" --detector "$detector"
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
            --detector "$detector"
            --device "$device"
            --weights "$WEIGHTS"
            --sequences "${seqs_chunk[@]}"
          )
          spawn_job "detect_${detector}_${bench}_gpu${gpu}" "$n_gpu" -- "${cmd[@]}"
        done
      done
    else
      for i in "${!BENCHMARKS[@]}"; do
        bench="${BENCHMARKS[$i]}"
        gpu="${GPUS[$((i % n_gpu))]}"
        device="${DEVICE_PREFIX}:${gpu}"
        cmd=(
          "$PYTHON" -m mot_pipeline.run detect
          --benchmark "$bench"
          --detector "$detector"
          --device "$device"
          --weights "$WEIGHTS"
        )
        spawn_job "detect_${detector}_${bench}_gpu${gpu}" "$n_gpu" -- "${cmd[@]}"
      done
    fi
    [[ "$PARALLEL" == "1" ]] && wait || true
    if [[ -f "$STATUS_DIR/_abort" ]]; then exit 1; fi
  fi
  log ""

  # ---- track+eval for each FPS (never re-runs detection) ----
  for fps in "${FPS_VALUES[@]}"; do
    tag="$(fps_tag_for "$fps")"
    fps_label="$fps"
    [[ "$fps" == "full" ]] && fps_label="native/full"
    log "--- track+eval detector=$detector fps=$fps_label ---"

    for bench in "${BENCHMARKS[@]}"; do
      for tracker in "${TRACKERS[@]}"; do
        if [[ "$tracker" == "traffictrack" ]]; then
          log "[skip] stub tracker traffictrack"
          skip=$((skip + 1))
          continue
        fi
        run_id="${bench}_${tracker}_${detector}${tag}_${TS}"
        # track then eval (not `all`) so cached dets are never re-touched.
        track_cmd=(
          "$PYTHON" -m mot_pipeline.run track
          --benchmark "$bench"
          --tracker "$tracker"
          --detector "$detector"
          --run-id "$run_id"
        )
        if [[ "$detector" == "yolov8" || "$detector" == "yolox" ]]; then
          track_cmd+=(--weights "$WEIGHTS")
        fi
        if [[ "$fps" != "full" && -n "$fps" ]]; then
          track_cmd+=(--fps "$fps")
        fi
        spawn_job "track_${detector}_${fps}_${bench}_${tracker}" "$TRACK_JOBS" -- \
          track_then_eval "$run_id" "${track_cmd[@]}"
      done
    done
    [[ "$PARALLEL" == "1" ]] && wait || true
    if [[ -f "$STATUS_DIR/_abort" ]]; then exit 1; fi

    write_compare "$detector_id" "$fps"
    log ""
  done
done

[[ "$PARALLEL" == "1" ]] && wait || true
tally_status_files

log ""
log "=== done ==="
log "ok=$ok fail=$fail skip=$skip"
if ((${#FAILED_JOBS[@]})); then
  log "failed jobs:"
  # unique-ish print
  printf '%s\n' "${FAILED_JOBS[@]}" | sort -u | while read -r j; do
    log "  - $j"
  done
fi
log "findings: /media/7TBSSD/data/tracking/experiments/_findings/<bench>/<tracker>/"
log "comparisons: $COMPARE_DIR"
log "log: $LOG"
log "job logs: $JOB_DIR"

exit $((fail > 0 ? 1 : 0))
