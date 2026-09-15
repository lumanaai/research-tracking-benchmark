---
name: add-mot-benchmark
description: >-
  Wire a new tracking dataset into the MOT pipeline (converter, benchmark
  adapter, class maps, batch sweeps, findings, visualize_all grids). Use when
  adding a benchmark/dataset, converting a new MOT source, or extending
  BENCHMARKS / RAW_DATASETS.
---

# Add MOT benchmark

Wire a new dataset so convert → detect → track → eval works on the shared SSD layout. Keep raw data and heavy artifacts on the SSD; only code/config in the project root.

## Project constants

- Always use `.venv/bin/python` / `.venv/bin/pip`.
- SSD root: `/media/7TBSSD/data/tracking/`.
- Normalized MOT: `mot/<BenchmarkFolder>/<split>/<seq>/{img1,gt/gt.txt,seqinfo.ini}`.
- Det cache: `detections/<bench_id>/<split>/<detector_id>/<seq>/det.txt`.
- Experiments: `experiments/<run_id>/` + `_findings/<bench_id>/<tracker>/<detector_id>/`.

## Checklist

Copy and track progress:

```
- [ ] 1. Raw data on SSD + MOT target layout
- [ ] 2. paths.RAW_DATASETS
- [ ] 3. Converter (or symlink-only)
- [ ] 4. Benchmark class + BENCHMARKS register
- [ ] 5. class_maps policy
- [ ] 6. run.py --benchmark choices (+ optional FastTracker/existing tweaks)
- [ ] 7. Batch BENCHMARKS defaults
- [ ] 8. compare_findings BENCH_ORDER + NATIVE_FPS
- [ ] 9. Viz script + visualize_all wiring + docs
- [ ] 10. Smoke convert + all
```

### 1. Data layout

- Raw: `/media/7TBSSD/data/tracking/<DatasetName>/`.
- MOT out: `/media/7TBSSD/data/tracking/mot/<BenchmarkFolder>/<split>/<seq>/` with:
  - `img1/` frames
  - `gt/gt.txt` — `frame,id,x,y,w,h,conf,class,visibility`
  - `seqinfo.ini` — `frameRate`, `seqLength`, `imWidth`, `imHeight`, `imExt`
- Optional viz sibling: `<DatasetName>_visualizations/`.

Choose CLI id `<bench_id>` (snake_case, e.g. `ua_detrac`).

### 2. Paths

In `mot_pipeline/paths.py`, add:

```python
RAW_DATASETS["<bench_id>"] = SSD_ROOT / "<DatasetName>"
```

### 3. Converter

- If not already MOT: `mot_pipeline/converters/prepare_<bench>.py` with `convert_all(dataset_root, out_root, split=..., sequences=..., force=...)`.
- Use `mot_pipeline/mot_io.py` (`write_seqinfo`, `filter_mot_file`, …).
- Refs: `prepare_ua_detrac.py`, `prepare_trafficmot.py`, `prepare_cityflow.py`.
- Already MOT-native: symlink/ensure inside the benchmark adapter (see `benchmarks/fasttracker_bench.py`) — no separate converter required.

### 4. Benchmark adapter

- Add `mot_pipeline/benchmarks/<bench>.py` subclassing `MotRootBenchmark`:
  - `name`, `mot_folder`, `raw_key`
  - `default_split()`
  - `ensure_mot(...)` → call converter or symlink
- Register in `mot_pipeline/benchmarks/__init__.py` → `BENCHMARKS["<bench_id>"] = ...`.

### 5. Class maps

In `mot_pipeline/class_maps.py`:

- Add a `ClassPolicy` matching **converter GT class ids** (vehicles; motorcycles included by default).
- Register `BENCHMARK_POLICIES["<bench_id>"]`.
- If `analytics_bytetrack` will run: add `AnalyticsClassSpace` + `ANALYTICS_CLASS_SPACES` entry.

### 6. CLI

- `mot_pipeline/run.py`: add `"<bench_id>"` to `--benchmark` choices.
- Only if needed: FastTracker `_default_tracker_config` `by_bench` map; `existing` / class-space auto logic.

### 7. Batch defaults

Append `<bench_id>` to `BENCHMARKS=(${BENCHMARKS:-...})` in:

- `scripts/batch/run_all.sh`
- `scripts/batch/run_all_parallel.sh`
- `scripts/batch/run_all_gt_trackers.sh`
- `scripts/batch/visualize_all.sh`

### 8. Findings tables

In `scripts/analysis/compare_findings.py`:

- `BENCH_ORDER` — display order.
- `NATIVE_FPS` — from `seqinfo.ini` / converter `frameRate` (full-rate runs with no `--fps`).

Optional: `scripts/analysis/compute_benchmark_statistics.py` (`BENCH_META` / class labels).

### 9. Viz + docs

- `scripts/visualize/visualize_<bench>.py` — GT boxes/IDs/trails; default out under `<DatasetName>_visualizations/`.
- Tracker overlay: existing `visualize_mot_results.py` (no change).
- Tracker grids (`visualize_all.sh`): add `<bench_id>` to `DEFAULT_BENCHMARKS` and `PREFERRED_SEQUENCES` in `scripts/visualize/visualize_all.py`, and `DATASET_LABELS` in `scripts/visualize/grid_page.py`. Sequences without `img1/` are skipped.
- `AGENTS.md` — layout table rows + dataset section (structure, convert, viz).
- `README.md` / `important_commands.md` — bench id in tables/commands.

Existing bench ids: [reference.md](reference.md).

### 10. Smoke

```bash
.venv/bin/python -m mot_pipeline.run convert --benchmark <bench_id> --sequences <one_seq>

.venv/bin/python -m mot_pipeline.run all \
  --benchmark <bench_id> --tracker ocsort --detector gt \
  --sequences <one_seq>
```

Confirm MOT seq, `detections/<bench_id>/.../gt_vehicles/`, and `experiments/_findings/<bench_id>/`.
