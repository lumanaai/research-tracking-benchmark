from mot_pipeline.trackers.base import StubTracker


class TrafficTrackAdapter(StubTracker):
    name = "traffictrack"
    repo_hint = (
        "TrafficTrack (Cai/Lin/Liu 2024 Multimedia Systems) — no public code yet; "
        "see TODOs.md"
    )
