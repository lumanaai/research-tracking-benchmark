"""Shared helpers for tracker adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mot_pipeline.protocols import Tracker


def load_tracker_config(path: Optional[Path], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(defaults or {})
    if path is not None and Path(path).is_file():
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def write_mot_tracks(
    path: Path,
    results: List[Tuple[int, list, list, list]],
) -> None:
    """Write MOT tracks: frame,id,x,y,w,h,score,-1,-1,-1."""
    fmt = "{frame},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{s:.2f},-1,-1,-1\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame_id, tlwhs, ids, scores in results:
            for tlwh, tid, score in zip(tlwhs, ids, scores):
                if tid < 0:
                    continue
                x, y, w, h = tlwh
                f.write(fmt.format(frame=frame_id, tid=tid, x=x, y=y, w=w, h=h, s=score))


def dets_to_xyxy_score(
    dets: Sequence[Tuple[float, float, float, float, float, int]],
) -> np.ndarray:
    """Convert [(x1,y1,x2,y2,score,cls), ...] to [N,5] float32."""
    if not dets:
        return np.empty((0, 5), dtype=np.float32)
    return np.array([[d[0], d[1], d[2], d[3], d[4]] for d in dets], dtype=np.float32)


def xyxy_ids_to_frame_result(
    frame_id: int,
    tracks: np.ndarray,
    *,
    default_score: float = 1.0,
) -> Tuple[int, list, list, list]:
    """Convert OC-SORT / HybridSORT ``[x1,y1,x2,y2,id]`` rows to MOT frame tuple."""
    tlwhs: list = []
    ids: list = []
    scores: list = []
    if tracks is None or len(tracks) == 0:
        return frame_id, tlwhs, ids, scores
    arr = np.asarray(tracks)
    for row in arr:
        x1, y1, x2, y2 = map(float, row[:4])
        tid = int(row[4])
        score = float(row[5]) if row.shape[0] > 5 else default_score
        tlwhs.append([x1, y1, x2 - x1, y2 - y1])
        ids.append(tid)
        scores.append(score)
    return frame_id, tlwhs, ids, scores


class StubTracker(Tracker):
    """Registered but not yet implemented."""

    name = "stub"
    repo_hint = ""

    def track_sequence(
        self,
        seq_dir: Path,
        det_path: Path,
        out_path: Path,
        config: Dict[str, Any],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError(
            f"Tracker '{self.name}' is not implemented yet. "
            f"Clone {self.repo_hint} (or your fork) and implement "
            f"mot_pipeline.trackers.{self.name}.*.track_sequence with the shared "
            f"contract: MOT dets in original pixels → MOT track txt out. "
            f"See FastTrackerAdapter for a reference."
        )
