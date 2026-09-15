---
name: add-mot-tracker
description: >-
  Wire a new motion tracker into the MOT pipeline (adapter, CLI, configs, batch
  sweeps, findings, visualize_all grids). Use when adding a tracker, integrating
  BoT-SORT/OC-SORT-style clones, extending TRACKERS, or registering a tracker stub.
---

# Add MOT tracker

Wire a detection-fed tracker into the full benchmark. Prefer adapters over shelling out to upstream CLIs. Do not edit vendored clones — use a shim when imports are awkward.

## Project constants

- Always use `.venv/bin/python` / `.venv/bin/pip`.
- Data + experiments on SSD: `/media/7TBSSD/data/tracking/` (`mot/`, `detections/`, `experiments/`).
- Findings upsert automatically under `experiments/_findings/<bench>/<tracker>/<detector_id>/`.

## Checklist

Copy and track progress:

```
- [ ] 1. Clone + paths.ROOT
- [ ] 2. Adapter track_sequence
- [ ] 3. Shim (if needed)
- [ ] 4. Register + run.py wiring
- [ ] 5. Config JSON
- [ ] 6. Batch TRACKERS defaults
- [ ] 7. compare_findings TRACKER_ORDER + tracker_meta.py (year, preferred_detector)
- [ ] 8. visualize_all TRACKERS + docs
- [ ] 9. Smoke run
```

After registering a tracker, add a `TRACKER_META` entry in `mot_pipeline/tracker_meta.py` (release year + preferred detector). Re-run `scripts/analysis/bench_tracker_speed.py` so comparison tables get a `ms/frame` reference.

### 1. Clone + path

- Put upstream at project root (plain dir, not submodule), e.g. `BoT-SORT/`.
- If the adapter imports from the clone, add `*_ROOT` in `mot_pipeline/paths.py` (see `OCSORT_ROOT`, `BOTSORT_ROOT`).

### 2. Adapter

Create `mot_pipeline/trackers/<name>.py`:

- Subclass `Tracker` (`mot_pipeline/protocols.py`); set `name = "<name>"`.
- Export `DEFAULT_CFG: dict`.
- Implement `track_sequence(seq_dir, det_path, out_path, config, *, extra=None)`.
- **Contract:** MOT dets in **original image pixels** → MOT track txt `frame,id,x,y,w,h,conf,-1,-1,-1`.
- Reuse `mot_pipeline/trackers/base.py`: `load_mot_dets` via callers, `resolve_tracking_schedule`, `write_mot_tracks`, `dets_to_xyxy_score`, `xyxy_ids_to_frame_result`.
- Reference: `mot_pipeline/trackers/ocsort.py` (simple), `fasttracker.py` (full), `botsort.py` (shimmed).
- Stub only: subclass `StubTracker` (`traffictrack.py`).

### 3. Shim (optional)

If upstream import pulls heavy/broken deps, add `mot_pipeline/trackers/<name>_shim.py` and call it before importing the clone. Patterns: `botsort_shim.py` (stub FastReID + `np.float`), `analytics_shim.py` (stub `general.*`). **Do not patch the clone.**

Details: [reference.md](reference.md).

### 4. Register + CLI

- `mot_pipeline/trackers/__init__.py` → `TRACKERS["<name>"] = Adapter`.
- `mot_pipeline/run.py`:
  - import `DEFAULT_CFG as XX_DEFAULTS`
  - `_default_tracker_config` → path under `configs/trackers/<name>/`
  - `_tracker_defaults` → `dict(XX_DEFAULTS)`
  - `--tracker` `choices=` include `"<name>"`

### 5. Config JSON

Add `mot_pipeline/configs/trackers/<name>/default.json` (or per-benchmark files if needed, like FastTracker).

### 6. Batch defaults

Append `<name>` to `TRACKERS=(${TRACKERS:-...})` in:

- `scripts/batch/run_all.sh`
- `scripts/batch/run_all_parallel.sh`
- `scripts/batch/run_all_gt_trackers.sh`
- `scripts/batch/visualize_all.sh`

Also append to `DEFAULT_TRACKERS` in `scripts/visualize/visualize_all.py`.

Skip stubs explicitly in the loop if they would crash (see `traffictrack`).

### 7. Findings tables

Add to `TRACKER_ORDER` in `scripts/analysis/compare_findings.py`.  
`run_all_compare_findings.sh` needs no change (detector-driven).

### 8. Docs + grids

- `README.md` — trackers table + layout clone dir + license path if applicable.
- `AGENTS.md` — tracker list, defaults, usage snippet.
- `important_commands.md` — tracker list / example commands.
- Deferred features (ReID, etc.) → `TODOs.md`.
- Tracker-grid viewer: keep `TRACKERS` in `visualize_all.sh` in sync (step 6). After a smoke run, `BENCHMARKS=fasttracker_bench FPS_VALUES="5" ./scripts/batch/visualize_all.sh` overlays the new tracker without re-detecting.

### 9. Smoke

```bash
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker <name> --detector gt \
  --sequences task_day_occlusion
```

Confirm tracks under `experiments/<run_id>/tracks/` and a findings row under `_findings/fasttracker_bench/<name>/`.
