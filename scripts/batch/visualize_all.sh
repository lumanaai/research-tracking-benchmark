#!/usr/bin/env bash
# Visualize latest run_all.sh tracks as synced HTML grids.
#
# Never detects or tracks. Reads experiments/_findings (same latest-row rule
# as compare_findings.py / the published comparison tables) and overlays the
# corresponding experiments/<run_id>/tracks/*.txt on MOT img1 frames.
#
# Usage (from project root):
#
#   Step 1 — generate JPEG cells + HTML (does not start a server):
#     ./scripts/batch/visualize_all.sh
#     BENCHMARKS=lumana_benchmark FPS_VALUES="5" ./scripts/batch/visualize_all.sh
#     N_SEQS=0 BENCHMARKS=lumana_benchmark FPS_VALUES="5" ./scripts/batch/visualize_all.sh  # all sequences
#
#   Step 2 — host those files:
#     ./scripts/batch/serve_visualizations.sh
#
# Viewer: one sequence at a time, all trackers tiled; Dataset + Video dropdowns.
# Default N_SEQS=3 (only that many sequences are rendered). N_SEQS=0 = all with img1/.
# SNIPPET_SEC=0 = full clip. FORCE=1 re-encodes existing cells.
#
# Output (SSD):
#   /media/7TBSSD/data/tracking/visualizations/<detector_stem>/index.html
#   .../<fps5|fps10|full>/<benchmark>/{index.html,manifest.json,cells/}
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON" >&2
  exit 1
fi

BENCHMARKS=(${BENCHMARKS:-fasttracker_bench ua_detrac trafficmot cityflow lumana_benchmark})
TRACKERS=(${TRACKERS:-fasttracker ocsort hybridsort analytics_bytetrack analytics_bytetrack_plus analytics_bytetrack_plus_aug19 botsort})
# Default YOLO only — matches the realistic comparison tables. Override to include gt.
DETECTORS=(${DETECTORS:-yolov8})
FPS_VALUES=(${FPS_VALUES:-5 10 full})

WEIGHTS="${WEIGHTS:-/media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt}"
OUT_ROOT="${OUT_ROOT:-/media/7TBSSD/data/tracking/visualizations}"
N_SEQS="${N_SEQS:-3}"
SNIPPET_SEC="${SNIPPET_SEC:-8}"
START_FRAC="${START_FRAC:-0.2}"
MAX_WIDTH="${MAX_WIDTH:-480}"
CELL_JOBS="${CELL_JOBS:-4}"
PARALLEL="${PARALLEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
FORCE="${FORCE:-0}"
RUN_ID_SUBSTR="${RUN_ID_SUBSTR:-}"
DETECTOR_ID="${DETECTOR_ID:-}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/media/7TBSSD/data/tracking/experiments/_logs}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"
LOG="$LOG_DIR/visualize_all_${TS}.log"
JOB_DIR="$LOG_DIR/jobs_visualize_all_${TS}"
STATUS_DIR="$JOB_DIR/status"
mkdir -p "$JOB_DIR" "$STATUS_DIR"

ok=0
fail=0
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

log "=== visualize_all (latest run_all.sh findings → HTML grids) ==="
log "root=$ROOT"
log "python=$PYTHON"
log "benchmarks: ${BENCHMARKS[*]}"
log "trackers:   ${TRACKERS[*]}"
log "detectors:  ${DETECTORS[*]}"
log "fps:        ${FPS_VALUES[*]}"
log "n_seqs=$N_SEQS snippet_sec=$SNIPPET_SEC start_frac=$START_FRAC max_width=$MAX_WIDTH"
log "out_root=$OUT_ROOT"
log "cell_jobs=$CELL_JOBS parallel=$PARALLEL dry_run=$DRY_RUN force=$FORCE"
log "run_id_substr=${RUN_ID_SUBSTR:-<latest per combo>}"
log "detector_id=${DETECTOR_ID:-<derived from detector/weights>}"
log "log=$LOG"
log ""

n_parallel=${#FPS_VALUES[@]}
(( n_parallel < 1 )) && n_parallel=1

for detector in "${DETECTORS[@]}"; do
  log "=== detector=$detector ==="
  for fps in "${FPS_VALUES[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
      cmd=(
        "$PYTHON" scripts/visualize/visualize_all.py
        --benchmark "$bench"
        --trackers "${TRACKERS[@]}"
        --detector "$detector"
        --weights "$WEIGHTS"
        --fps "$fps"
        --n-seqs "$N_SEQS"
        --snippet-sec "$SNIPPET_SEC"
        --start-frac "$START_FRAC"
        --max-width "$MAX_WIDTH"
        --out-root "$OUT_ROOT"
        --jobs "$CELL_JOBS"
      )
      if [[ -n "$DETECTOR_ID" ]]; then
        cmd+=(--detector-id "$DETECTOR_ID")
      fi
      if [[ -n "$RUN_ID_SUBSTR" ]]; then
        cmd+=(--run-id-substr "$RUN_ID_SUBSTR")
      fi
      if [[ "$FORCE" == "1" ]]; then
        cmd+=(--force)
      fi
      if [[ "$DRY_RUN" == "1" ]]; then
        cmd+=(--dry-run)
      fi
      spawn_job "viz_${detector}_${fps}_${bench}" "$n_parallel" -- "${cmd[@]}"
    done
  done
done

[[ "$PARALLEL" == "1" ]] && wait || true
if [[ -f "$STATUS_DIR/_abort" ]]; then exit 1; fi

if [[ "$DRY_RUN" != "1" ]]; then
  log "[index] refresh parent index.html"
  if "$PYTHON" scripts/visualize/visualize_all.py --refresh-index --out-root "$OUT_ROOT" >>"$LOG" 2>&1; then
    log "  OK"
  else
    log "  FAILED refresh-index"
    fail=$((fail + 1))
    FAILED_JOBS+=("refresh-index")
  fi
fi

tally_status_files

log ""
log "=== done ==="
log "ok=$ok fail=$fail"
if ((${#FAILED_JOBS[@]})); then
  log "failed jobs:"
  printf '%s\n' "${FAILED_JOBS[@]}" | sort -u | while read -r j; do
    log "  - $j"
  done
fi
log "grids: $OUT_ROOT"
log "log: $LOG"
log "job logs: $JOB_DIR"

if [[ "$DRY_RUN" != "1" ]]; then
  log ""
  log "To view: ./scripts/batch/serve_visualizations.sh"
  "$PYTHON" scripts/visualize/visualize_all.py --print-urls --out-root "$OUT_ROOT" --port "${SERVE_PORT:-8765}" | tee -a "$LOG"
fi

exit $((fail > 0 ? 1 : 0))
