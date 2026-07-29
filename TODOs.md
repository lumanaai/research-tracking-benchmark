# TODOs / future work

Deferred items for the MOT pipeline. Do not lose these when wiring new trackers.

## Trackers

- [ ] **Implement TrafficTrack** — paper: [Cai, Lin, Liu 2024](https://link.springer.com/article/10.1007/s00530-024-01407-8). No public code found as of 2026-07. Options when ready:
  - Adapt an author/release zip into `mot_pipeline/trackers/traffictrack.py` (same `track_sequence` contract as FastTracker).
  - Or paper-faithful reimplementation (adaptive KF noise from match cost + confidence; combined motion/appearance distance; trajectory recovery). Needs PDF detail + appearance backbone choice.
- [ ] Keep `traffictrack` registered as a stub until then (`NotImplementedError` with repo/paper hint).

## Appearance / ReID

- [ ] **Add ReID** to trackers that support it:
  - Hybrid-SORT-ReID (`trackers/hybrid_sort_tracker/hybrid_sort_reid.py` + FastReID).
  - Optionally Deep-OC-SORT or other appearance forks of OC-SORT.
- [ ] Store ReID weights on SSD under `/media/7TBSSD/data/tracking/weights/reid/` (not on the NAS).
- [ ] Prefer vehicle-domain ReID when available; pedestrian-trained SBS models may transfer poorly to traffic cams.
- [ ] Measure crop/feature cost per frame (I/O + GPU) vs motion-only FPS.

## Research: gated ReID

- [ ] **Occlusion / confusion-gated ReID** — run appearance matching only when needed, not every frame. Candidates:
  - Low IoU / ambiguous bipartite cost in the motion association stage.
  - Tracks marked lost / occluded (e.g. FastTracker occlusion flags, HybridSORT unmatched tracks).
  - High density or overlapping boxes.
- [ ] Compare constant-ReID vs gated-ReID on FastTracker-Benchmark hard sequences (`task_day_occlusion`, night/tunnel) for IDF1 / AssA / IDSW vs runtime.

## Pipeline polish (optional)

- [ ] Per-benchmark default hyperparameter JSONs for OC-SORT / HybridSORT (today: one shared `default.json`).
- [ ] Optional offline interpolation post-process (OC-SORT / HybridSORT repos ship `tools/interpolation.py`) as an eval ablation flag.
- [ ] Document smoke commands for all trackers × all four benchmarks in `important_commands.md` as they are verified.
