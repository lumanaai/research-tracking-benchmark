# Add MOT tracker — reference

## Shim vs plain adapter

| Situation | Pattern | Example |
|-----------|---------|---------|
| Clean `sys.path` + direct import | `_ensure_*_on_path()` in adapter | `ocsort.py`, `hybridsort.py` |
| Heavy/broken deps at import | Separate `<name>_shim.py`; stub `sys.modules` | `botsort_shim.py`, `analytics_shim.py` |
| Code not available yet | `StubTracker` + register + `TODOs.md` | `traffictrack.py` |

## Example adapters

| Tracker | Adapter | Config default |
|---------|---------|----------------|
| FastTracker | `mot_pipeline/trackers/fasttracker.py` | per-bench under `configs/trackers/fasttracker/` |
| OC-SORT | `ocsort.py` | `configs/trackers/ocsort/default.json` |
| HybridSORT | `hybridsort.py` | `configs/trackers/hybridsort/default.json` |
| BoT-SORT (no ReID) | `botsort.py` + `botsort_shim.py` | `configs/trackers/botsort/default.json` |
| Analytics ByteTrack | `analytics_bytetrack.py` + `analytics_shim.py` | `configs/trackers/analytics_bytetrack/benchmark.json` |

## Contract reminders

- Dets: MOT `frame,id,x,y,w,h,conf[,class[,vis]]` in **original** pixels; `id` ignored.
- Out: `frame,id,x,y,w,h,conf,-1,-1,-1` via `write_mot_tracks`.
- Honor `extra["keep_classes"]` / `drop_classes` and `resolve_tracking_schedule` for `--fps` / stride.
- Prefer motion-only first; appearance/ReID stays in `TODOs.md` until weights + crop path are ready.
