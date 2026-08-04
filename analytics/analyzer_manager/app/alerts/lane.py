from typing import Tuple, Optional, Dict, List
from collections import deque
import numpy as np

from general.core import BatchDataResolver, AnalyticImage, ClassHandler
from .base_alerts import ObjectAlert, AlertCandidate
from general.analyzer_general import logger


# ─── Geometry utilities ───────────────────────────────────────────────────────

def line_bbox_intersections(p1: np.ndarray, p2: np.ndarray, bbox: np.ndarray) -> List[np.ndarray]:
    """Find intersection points of the infinite line through p1, p2 with a bbox [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox
    edges = [
        (np.array([x1, y1]), np.array([x2, y1])),  # top
        (np.array([x1, y2]), np.array([x2, y2])),  # bottom
        (np.array([x1, y1]), np.array([x1, y2])),  # left
        (np.array([x2, y1]), np.array([x2, y2])),  # right
    ]
    d = p2 - p1
    intersections = []
    for seg_start, seg_end in edges:
        s = seg_end - seg_start
        denom = d[0] * s[1] - d[1] * s[0]
        if abs(denom) < 1e-10:
            continue
        w = seg_start - p1
        t_seg = (w[0] * d[1] - w[1] * d[0]) / denom
        if -1e-9 <= t_seg <= 1.0 + 1e-9:
            t_line = (w[0] * s[1] - w[1] * s[0]) / denom
            pt = p1 + t_line * d
            if not any(np.linalg.norm(pt - ex) < 1.0 for ex in intersections):
                intersections.append(pt)
    return intersections


def _project_onto_line(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Project a point onto the line (p1->p2) and return the scalar parameter t."""
    d = p2 - p1
    length_sq = np.dot(d, d)
    if length_sq < 1e-10:
        return 0.0
    return np.dot(point - p1, d) / length_sq


def lane_distance_between_bboxes(p1: np.ndarray, p2: np.ndarray,
                                  bbox_a: np.ndarray, bbox_b: np.ndarray) -> Optional[float]:
    """
    Compute the gap distance along the lane line between two bboxes.
    Returns None if either bbox does not intersect the line.
    Returns 0 if bboxes overlap along the line.
    """
    ints_a = line_bbox_intersections(p1, p2, bbox_a)
    ints_b = line_bbox_intersections(p1, p2, bbox_b)
    if len(ints_a) == 0 or len(ints_b) == 0:
        return None

    line_length = np.linalg.norm(p2 - p1)
    ts_a = [_project_onto_line(pt, p1, p2) for pt in ints_a]
    ts_b = [_project_onto_line(pt, p1, p2) for pt in ints_b]

    interval_a = (min(ts_a), max(ts_a))
    interval_b = (min(ts_b), max(ts_b))

    if interval_a[1] < interval_b[0]:
        gap_t = interval_b[0] - interval_a[1]
    elif interval_b[1] < interval_a[0]:
        gap_t = interval_a[0] - interval_b[1]
    else:
        gap_t = 0.0

    return gap_t * line_length


# ─── Lane estimator ──────────────────────────────────────────────────────────

class LaneEstimator:
    """
    Automatically estimates a lane line from accumulated vehicle centroids via PCA.

    PCA handles all orientations (including near-vertical lanes) and uses the
    explained variance ratio (λ₁/(λ₁+λ₂)) as a confidence metric for linearity.
    Recomputation is gated by a dirty flag so it only runs when new points are added.
    """

    def __init__(self, max_len: int = 30, min_num_pts: int = 4,
                 pca_conf_th: float = 0.9, dedup_radius: float = 10.0):
        self.points: deque = deque(maxlen=max_len)
        self.min_num_pts = min_num_pts
        self.pca_conf_th = pca_conf_th
        self.dedup_radius = dedup_radius
        self._lane_p1: Optional[np.ndarray] = None
        self._lane_p2: Optional[np.ndarray] = None
        self._explained_variance_ratio: float = 0.0
        self._dirty: bool = False

    def add_centroid(self, centroid: np.ndarray) -> bool:
        """Add a centroid if not too close to existing points. Returns True if added."""
        # TODO: this fits for only one lane
        for existing in self.points:
            if np.linalg.norm(centroid - existing) < self.dedup_radius:
                return False
        self.points.append(centroid.copy())
        self._dirty = True
        return True

    def add_centroids_from_batch(self, centroids: np.ndarray) -> int:
        """Add multiple centroids. Returns count of newly added points."""
        added = 0
        for c in centroids:
            if self.add_centroid(c):
                added += 1
        return added

    def lane_fitting(self):
        """fit lane via PCA. Only does work if new points were added."""
        if not self._dirty:
            return
        self._dirty = False

        if len(self.points) < self.min_num_pts:
            self._explained_variance_ratio = 0.0
            self._lane_p1 = None
            self._lane_p2 = None
            return

        pts = np.array(self.points)
        mean = pts.mean(axis=0)
        centered = pts - mean

        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_dir = eigenvectors[:, -1]

        lambda1, lambda2 = eigenvalues[-1], eigenvalues[0]
        total_var = lambda1 + lambda2
        self._explained_variance_ratio = lambda1 / total_var if total_var > 1e-10 else 0.0

        projections = centered @ principal_dir
        t_min, t_max = projections.min(), projections.max()
        self._lane_p1 = mean + t_min * principal_dir
        self._lane_p2 = mean + t_max * principal_dir

    @property
    def is_confident(self) -> bool:
        return self._explained_variance_ratio >= self.pca_conf_th and self._lane_p1 is not None

    @property
    def lane(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.is_confident:
            return self._lane_p1, self._lane_p2
        return None

    @property
    def explained_variance_ratio(self) -> float:
        return self._explained_variance_ratio


# ─── Distance helpers ─────────────────────────────────────────────────────────

def calculate_pairwise_min_edge_distances(bboxes: np.ndarray) -> np.ndarray:
    """Calculate minimum edge-midpoint distances for all bbox pairs in one vectorized pass."""
    if len(bboxes) == 0:
        return np.zeros((0, 0), dtype=float)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    midpoints = np.stack(
        [
            np.stack([x_center, y1], axis=1),
            np.stack([x_center, y2], axis=1),
            np.stack([x1, y_center], axis=1),
            np.stack([x2, y_center], axis=1),
        ],
        axis=1,
    )
    diffs = midpoints[:, None, :, None, :] - midpoints[None, :, None, :, :]
    pairwise_distances = np.sqrt(np.sum(diffs ** 2, axis=-1).min(axis=(-1, -2)))
    return pairwise_distances


class LaneAlert(ObjectAlert):
    type_name = "distOnLane"
    default_msg = "Car distance alert"
    alert_message = "Vehicles are too close to each other"
    use_lane_estimation = True
    debounce_count = 2
    enable_proximity = True
    # Rapid approach detection parameters
    enable_rapid_approach = True
    history_window = 5      # number of distance samples to keep per pair
    low_conf_batch_th = 3   # threshold to mark bad config if low confidence persists

    def __init__(self, alert_dict: Dict, context):
        super(LaneAlert, self).__init__(alert_dict, context)

        class_handler: ClassHandler = self.context.get_class_handler()
        self.vehicle_value = class_handler.vehicle_value
        self.object_ids = [self.vehicle_value]
        if self.vehicle_value not in self.l1_required_type:
            self.l1_required_type[self.vehicle_value] = None
        # Read min distance threshold from 2 user-provided points (lineCrossing config)
        # Points are in normalized [0,1] space; we store the normalized distance
        # and convert to pixels at runtime using frame dimensions
        car_dist_config = self.selectedCamera.get("lineCrossing", {})
        if car_dist_config and "p1" in car_dist_config and "p2" in car_dist_config:
            p1 = car_dist_config["p1"]
            p2 = car_dist_config["p2"]
            self.min_distance_dx_norm = p2["x"] - p1["x"]
            self.min_distance_dy_norm = p2["y"] - p1["y"]
        else:
            self.min_distance_dx_norm = None
            self.min_distance_dy_norm = None
        # Defaults before set_flow_values
        self.min_distance_default = 0.078
        analytic_config = self.context.get_config()
        lane_cfg = analytic_config.get("alertConfig", {}).get("distOnLane", {})
        self._enable_proximity_cfg: Optional[bool] = lane_cfg.get("enableProximity", lane_cfg.get("enable_proximity"))
        self._enable_rapid_approach_cfg: Optional[bool] = lane_cfg.get(
            "enableRapidApproach", lane_cfg.get("enable_rapid_approach")
        )
        self.enable_proximity = True if self._enable_proximity_cfg is None else self._enable_proximity_cfg
        self.enable_rapid_approach = (
            True if self._enable_rapid_approach_cfg is None else self._enable_rapid_approach_cfg
        )
        # Track pairs for absolute proximity: {(id_a, id_b): {"count": int, "last_ts": int, "alertSent": bool}}
        self.trackers: Dict[Tuple[int, int], dict] = {}
        # Track distance history per pair for rapid approach detection
        # {(id_a, id_b): deque([(ts, dist), ...], maxlen=history_window)}
        self.distance_history: Dict[Tuple[int, int], deque] = {}
        # Track rapid approach debounce: {(id_a, id_b): {"count": int, "last_ts": int, "alertSent": bool}}
        self.approach_trackers: Dict[Tuple[int, int], dict] = {}
        # Global cooldown: timestamp of the last triggered alert (any pair)
        self._last_alert_ts: int = -int(self.blockout)
        # Cached pixel threshold (computed once on first batch)
        self._distance_th_px: Optional[float] = None
        # Rapid approach rate threshold in px/sec (derived from distance threshold)
        self._approach_rate_threshold: Optional[float] = None
        # Lane estimator for automatic lane-based distance computation
        self.lane_estimator = LaneEstimator(max_len=30, min_num_pts=4, pca_conf_th=0.9, dedup_radius=5.0)
        # Bad configuration detection: consecutive low-confidence batches with full deque
        self.low_conf_batchs = 0
        self._lane_invalidated = False

    def _get_distance_threshold_px(self, im_w: int, im_h: int) -> float:
        """Get the distance threshold in pixels. Cached after first call."""
        if self._distance_th_px is None:
            if self.min_distance_dx_norm is not None:
                dx_px = self.min_distance_dx_norm * im_w
                dy_px = self.min_distance_dy_norm * im_h
                self._distance_th_px = np.sqrt(dx_px ** 2 + dy_px ** 2)
            else:
                self._distance_th_px = self.min_distance_default * im_w
            # Approach rate: closing 1/3 of the distance threshold in 0.6 sec
            self._approach_rate_threshold = self._distance_th_px / 3.0 / 0.6
        return self._distance_th_px

    def on_entities_removed(self, ent_ids: List[int]):
        for ent in ent_ids:
            self.active_ids.discard(ent)
        # Remove trackers containing removed entities
        self.trackers = {k: v for k, v in self.trackers.items() if not any(e in ent_ids for e in k)}
        self.distance_history = {k: v for k, v in self.distance_history.items() if not any(e in ent_ids for e in k)}
        self.approach_trackers = {k: v for k, v in self.approach_trackers.items() if not any(e in ent_ids for e in k)}

    def _check_rapid_approach(self, pair: Tuple[int, int], ts: int, dist: float) -> bool:
        """Record distance sample and return True if rapid approach is detected."""
        if pair not in self.distance_history:
            self.distance_history[pair] = deque(maxlen=self.history_window)
        self.distance_history[pair].append((ts, dist))
        history = self.distance_history[pair]
        if len(history) < 2:
            return False
        dt_ms = history[-1][0] - history[0][0]
        if dt_ms <= 0:
            return False
        # Rate in px/sec (negative means approaching)
        rate = (history[-1][1] - history[0][1]) / dt_ms * 1000
        return rate < -self._approach_rate_threshold

    def is_active_batch(self, images: List[AnalyticImage], motion_data, batch_data) -> List[AlertCandidate]:
        alert_candidates = []
        vars_data = self.get_batch_candidates_data(batch_data)
        if len(vars_data) == 0:
            return alert_candidates

        im_h, im_w = images[-1].frame.shape[:2]
        distance_th = self._get_distance_threshold_px(im_w, im_h)

        # Accumulate centroids for lane estimation
        scale = np.array([im_w, im_h, im_w, im_h])
        all_bboxes_px = vars_data[:, BatchDataResolver.POS] * scale
        use_lane = False
        lane = None
        if self.use_lane_estimation:
            centroids = np.column_stack([
                (all_bboxes_px[:, 0] + all_bboxes_px[:, 2]) / 2,
                (all_bboxes_px[:, 1] + all_bboxes_px[:, 3]) / 2,
            ])
            self.lane_estimator.add_centroids_from_batch(centroids)
            self.lane_estimator.lane_fitting()

            # Bad config detection: full deque but low confidence → count streak
            if (len(self.lane_estimator.points) == self.lane_estimator.points.maxlen
                    and self.lane_estimator.explained_variance_ratio < self.lane_estimator.pca_conf_th):
                self.low_conf_batchs += 1
                if self.low_conf_batchs >= self.low_conf_batch_th and not self._lane_invalidated:
                    self._lane_invalidated = True
                    self.context.validate_alert(
                        self.event_id, is_valid=False,
                        reason=f"Lane estimation failed: explained variance ratio "
                               f"{self.lane_estimator.explained_variance_ratio:.2f} "
                               f"below {self.lane_estimator.pca_conf_th} after "
                               f"{len(self.lane_estimator.points)} samples - camera may not be aligned with lane or lane is not straight enough."
                    )
            else:
                self.low_conf_batchs = 0

            use_lane = self.lane_estimator.is_confident
            lane = self.lane_estimator.lane

        # Process per frame
        for ind, frame in enumerate(images):
            frame_data = vars_data[vars_data[:, BatchDataResolver.FRAME_ID] == ind, :]
            if len(frame_data) < 2:
                continue    # Need at least 2 vehicles to check distance

            ts = frame.timestamp
            bboxes_px = frame_data[:, BatchDataResolver.POS] * scale
            ids = frame_data[:, BatchDataResolver.ID].astype(int)
            pair_rows, pair_cols = np.triu_indices(len(frame_data), k=1)

            # Pre-compute on-lane status when using lane mode
            on_lane = None
            if use_lane:
                lp1, lp2 = lane
                on_lane = [len(line_bbox_intersections(lp1, lp2, bboxes_px[k])) >= 2
                           for k in range(len(frame_data))]
            else:
                pairwise_midpoint_distances = calculate_pairwise_min_edge_distances(bboxes_px)

            # Check all pairs
            for i, j in zip(pair_rows, pair_cols):
                if use_lane:
                    # Skip pairs where either car is off-lane
                    if not on_lane[i] or not on_lane[j]:
                        continue
                    lp1, lp2 = lane
                    dist = lane_distance_between_bboxes(lp1, lp2, bboxes_px[i], bboxes_px[j])
                    if dist is None:
                        continue
                else:
                    # Fallback: midpoint-based distance
                    dist = pairwise_midpoint_distances[i, j]
                pair = tuple(sorted((ids[i], ids[j])))
                # Path 1: Absolute proximity
                if self.enable_proximity and dist < distance_th:
                    if pair in self.trackers:
                        self.trackers[pair]["count"] += 1
                        self.trackers[pair]["last_ts"] = ts
                    else:
                        self.trackers[pair] = {"count": 1, "last_ts": ts, "alertSent": False}
                    self.trackers[pair]["last_dist"] = dist

                # Path 2: Rapid approach detection (only if within 2x threshold)
                if self.enable_rapid_approach and dist < 2 * distance_th and self._check_rapid_approach(pair, ts, dist):
                    if pair in self.approach_trackers:
                        self.approach_trackers[pair]["count"] += 1
                        self.approach_trackers[pair]["last_ts"] = ts
                    else:
                        self.approach_trackers[pair] = {"count": 1, "last_ts": ts, "alertSent": False}
                    self.approach_trackers[pair]["last_dist"] = dist

        # trigger alerts for pairs that passed the debounce threshold (either path)
        if self.enable_proximity:
            self._trigger_alerts(self.trackers, vars_data, images, alert_candidates, "proximity")
        if self.enable_rapid_approach:
            self._trigger_alerts(self.approach_trackers, vars_data, images, alert_candidates, "rapid_approach")

        return alert_candidates

    def _trigger_alerts(self, trackers: dict, vars_data: np.ndarray,
                     images: List[AnalyticImage], alert_candidates: List[AlertCandidate],
                     trigger_type: str):
        """Check trackers and trigger alerts for pairs that passed debounce."""
        for pair, tracker in list(trackers.items()):
            if tracker["alertSent"]:
                continue
            if tracker["count"] >= self.debounce_count:
                # Global cooldown: skip if any alert was triggered recently
                if tracker["last_ts"] - self._last_alert_ts < self.blockout:
                    continue
                # Also skip if the other path already triggered for this pair
                if trigger_type == "rapid_approach" and pair in self.trackers and self.trackers[pair].get("alertSent"):
                    continue
                if trigger_type == "proximity" and pair in self.approach_trackers and self.approach_trackers[pair].get("alertSent"):
                    continue
                # Filter var_data to the violation frame and the pair's entities
                violation_frame_ids = [
                    ind for ind, image in enumerate(images) if image.timestamp == tracker["last_ts"]
                ]
                if not violation_frame_ids:
                    continue
                pair_mask = np.isin(vars_data[:, BatchDataResolver.ID].astype(int), pair)
                frame_mask = np.isin(
                    vars_data[:, BatchDataResolver.FRAME_ID].astype(int), violation_frame_ids
                )
                candidate_var_data = vars_data[pair_mask & frame_mask]
                if len(candidate_var_data) == 0:
                    continue
                candidate = self.build_alert_candidate(
                    tracker["last_ts"],
                    ent_ids=list(pair),
                    var_data=candidate_var_data,
                    images=images,
                )
                logger.info(
                    f"LaneAlert [{trigger_type}] triggered at t={tracker['last_ts']} "
                    f"with distance {tracker.get('last_dist', 0):.1f}px"
                )
                alert_candidates.append(candidate)
                tracker["alertSent"] = True
                self._last_alert_ts = tracker["last_ts"]

    def build_alert_info(self, candidate: AlertCandidate):
        alert_info = super().build_alert_info(candidate)
        alert_info.alertMessage = self.alert_message
        return alert_info