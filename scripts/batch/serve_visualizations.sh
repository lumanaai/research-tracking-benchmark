#!/usr/bin/env bash
# Step 2: host already-generated visualization grids (no rendering).
#
# Step 1 (generate JPEG cells + HTML):
#   ./scripts/batch/visualize_all.sh
#
# Step 2 (this script):
#   ./scripts/batch/serve_visualizations.sh
#   PORT=9000 ./scripts/batch/serve_visualizations.sh
#
# Then open e.g.
#   http://127.0.0.1:8765/                          (landing list)
#   http://127.0.0.1:8765/yolov8m_expert_eff/fps5/lumana_benchmark/index.html
#
# Viewer: Dataset dropdown (sibling grids) + Video dropdown (rendered sequences).
#
# Stop: Ctrl+C in this terminal
#   or:  pkill -f 'visualize_all.py --serve-only'
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-/media/7TBSSD/data/tracking/visualizations}"
PORT="${PORT:-8765}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$OUT_ROOT" ]]; then
  echo "ERROR: no visualizations at $OUT_ROOT — run ./scripts/batch/visualize_all.sh first" >&2
  exit 1
fi

echo "Hosting $OUT_ROOT on http://127.0.0.1:${PORT}/"
echo "Stop with Ctrl+C  (or: pkill -f 'visualize_all.py --serve-only')"
exec "$PYTHON" scripts/visualize/visualize_all.py --serve-only --out-root "$OUT_ROOT" --port "$PORT"
