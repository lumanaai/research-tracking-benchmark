"""ByteTrack Plus (Aug19) — crossover / parked / CIoU subclass.

Vendored from tmp/byte_tracker_plus (2).py. Imports analytics basetrack/utils via
shim; base BYTETracker lives in byte_tracker_base.py (hookified).
"""
from __future__ import annotations

import math
import numpy as np
from collections import deque

from tracking.byte_track.basetrack import TrackState
from tracking.byte_track.utils import is_track_outside_image_bounds
from .byte_tracker_base import BYTETracker, STrack as STrackBase
from .matching_ciou import iou_distance as ciou_iou_distance


class STrack(STrackBase):

    def __init__(self, tlwh, score, cls, subclass, det_id, context):
        super().__init__(tlwh, score, cls, subclass, det_id, context)
        self.center_history = deque(maxlen=context.velocity_window_size)
        self._cached_median_speed_sq = None
        self._append_center(self._tlwh)
        # --- Crossover protection state ---
        self._xover_save = None        # (mean, covariance, center_history_copy) before suspicious match
        self._xover_grace_start = 0    # frame_id when grace period began

    def _append_center(self, tlwh):
        cx = float(tlwh[0] + (tlwh[2] / 2.0))
        cy = float(tlwh[1] + (tlwh[3] / 2.0))
        diag_sq = (tlwh[2] * tlwh[2] + tlwh[3] * tlwh[3]) ** 0.5
        self.center_history.append((cx, cy, diag_sq))
        self._cached_median_speed_sq = None

    def get_median_speed_sq(self):
        if self._cached_median_speed_sq is not None:
            return self._cached_median_speed_sq
        hist = self.center_history
        n = len(hist)
        if n < 2:
            self._cached_median_speed_sq = -1.0
            return -1.0
        speeds = sorted(
            ((hist[i][0] - hist[i-1][0]) ** 2 + (hist[i][1] - hist[i-1][1]) ** 2)
            / max(hist[i][2] * hist[i][2], 1.0)
            for i in range(1, n)
        )
        mid = len(speeds) // 2
        self._cached_median_speed_sq = (
            speeds[mid] if len(speeds) % 2 else (speeds[mid - 1] + speeds[mid]) / 2.0
        )
        return self._cached_median_speed_sq

    def save_xover_state(self, frame_id):
        self._xover_save = (
            self.mean.copy(),
            self.covariance.copy(),
            deque(self.center_history, maxlen=self.center_history.maxlen),
        )
        self._xover_grace_start = frame_id

    def revert_xover_state(self):
        if self._xover_save is None:
            return False
        self.mean, self.covariance, self.center_history = self._xover_save
        self._xover_save = None
        self._xover_grace_start = 0
        self._cached_median_speed_sq = None
        # Re-anchor: set Kalman cx,cy to median of reverted history, zero velocity
        if len(self.center_history) >= 2:
            pts = np.asarray([(c[0], c[1]) for c in self.center_history], dtype=float)
            self.mean[0] = float(np.median(pts[:, 0]))   # cx
            self.mean[1] = float(np.median(pts[:, 1]))   # cy
        self.mean[4] = 0.0   # vx
        self.mean[5] = 0.0   # vy
        return True

    def clear_xover_state(self):
        self._xover_save = None
        self._xover_grace_start = 0

    def re_activate(self, new_track, frame_id, new_id=False):
        super().re_activate(new_track, frame_id, new_id)
        self._append_center(new_track.tlwh)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.age += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked

        if self.tracklet_len >= self.n_init:
            if not self.is_activated:
                # Just transitioned from unconfirmed to activated.
                # Unconfirmed tracks skip predict(), so Kalman velocity is unreliable.
                # Bootstrap it from center_history displacement.
                hist = list(self.center_history)
                if len(hist) >= 2:
                    vx = (hist[-1][0] - hist[-2][0])
                    vy = (hist[-1][1] - hist[-2][1])
                    self.mean[4] = vx
                    self.mean[5] = vy
            self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.det_id = new_track.det_id
        self._append_center(new_track.tlwh)


class BYTETrackerPlus(BYTETracker):
    per_class_confidence: np.array
    per_class_confidence_first: np.array

    def __init__(self, args, frame_rate=30, filter_objects=None, context=None):
        super().__init__(args, frame_rate, filter_objects, context)
        self.velocity_window_size = max(0, int(getattr(args, 'velocity_window_size', 8)))
        self.velocity_noise_th = float(getattr(args, 'velocity_noise_th', 1.5))
        self.velocity_cost_vehicle_cls = None
        self.fast_exit_th = float(getattr(args, 'fast_exit_th', 4.0))
        self.norm_dist_th = float(getattr(args, 'norm_dist_th', 0.05))
        self.norm_speed_th = float(getattr(args, 'norm_speed_th', 0.02 ** 2))
        self.xover_grace_frames = int(getattr(args, 'xover_grace_frames', 8))
        self.xover_mahab_th = float(getattr(args, 'xover_mahab_th', 9.21))
        self.parked_max_time_lost = int(
            self.max_time_lost * getattr(args, 'parked_max_lost_multiplier', 10.0)
        )
        self.use_ciou = bool(getattr(args, 'use_ciou', False))
        self._cached_tv_pos = None
        self._cached_lost_stationary = None

    def _make_strack(self, tlwh, s, c, sc, d):
        return STrack(tlwh, s, c, sc, d, self)

    def _iou_distance(self, atracks, btracks):
        return ciou_iou_distance(atracks, btracks, use_ciou=self.use_ciou)

    def _filter_strack_pool(self, strack_pool):
        return [t for t in strack_pool if t.state == TrackState.Tracked
                or not is_track_outside_image_bounds(
                    t.tlbr, self.img_h, self.img_w,
                    tolerance_ratio=0.02 if (t.mean[4]**2 + t.mean[5]**2) > self.fast_exit_th else 0.1
                )]

    def _on_first_match(self, track, det):
        if (self.velocity_cost_vehicle_cls is not None
                and track.cls == self.velocity_cost_vehicle_cls
                and track.tracklet_len >= 10
                and track._xover_save is None):
            if self._is_track_slow(track):
                det_tlwh = det.tlwh
                _det_cx = det_tlwh[0] + det_tlwh[2] / 2
                _det_cy = det_tlwh[1] + det_tlwh[3] / 2
                _innov = math.hypot(_det_cx - track.mean[0], _det_cy - track.mean[1])
                _det_diag = (det_tlwh[2] ** 2 + det_tlwh[3] ** 2) ** 0.5
                if _det_diag > 0 and _innov / _det_diag > self.norm_dist_th:
                    track.save_xover_state(self.frame_id)

    def _on_track_going_lost(self, track):
        if track._xover_save is not None:
            track.revert_xover_state()
            self._snap_to_nearest_detection(
                track, self._all_detections, self._get_tracked_vehicle_positions())
        elif (self.velocity_cost_vehicle_cls is not None
                and track.cls == self.velocity_cost_vehicle_cls
                and track.tracklet_len >= 10):
            _spd = track.mean[4] * track.mean[4] + track.mean[5] * track.mean[5]
            _diag = track.mean[3] * track.mean[3] * (track.mean[2] * track.mean[2] + 1.0)
            _is_stationary = _spd / max(_diag, 1.0) < self.norm_speed_th
            if not _is_stationary:
                med = track.get_median_speed_sq()
                if med >= 0 and len(track.center_history) >= 10 and med < self.norm_speed_th:
                    _is_stationary = True
            if _is_stationary:
                track.mean[4] = 0.0
                track.mean[5] = 0.0
                hist = list(track.center_history)
                if len(hist) >= 10:
                    xs = sorted(h[0] for h in hist)
                    ys = sorted(h[1] for h in hist)
                    mid = len(xs) // 2
                    track.mean[0] = (xs[mid] + xs[mid - 1]) / 2.0 if len(xs) % 2 == 0 else xs[mid]
                    track.mean[1] = (ys[mid] + ys[mid - 1]) / 2.0 if len(ys) % 2 == 0 else ys[mid]

    def _prepare_unconfirmed(self, unconfirmed, dt):
        # Predict unconfirmed tracks forward so fast-moving objects can maintain
        # IoU overlap with their next-frame detection (without predict, their
        # position stays stale at their last observation).
        STrack.multi_predict(unconfirmed, dt)

    def _should_suppress_new_track(self, track, lost_stracks):
        if self.velocity_cost_vehicle_cls is None:
            return False
        stationary_positions = self._get_lost_stationary_positions(lost_stracks)
        if not stationary_positions:
            return False
        _tlwh = track.tlwh
        _det_cx = _tlwh[0] + _tlwh[2] / 2
        _det_cy = _tlwh[1] + _tlwh[3] / 2
        stationary_guard_radius = 0.5 * (_tlwh[2] * _tlwh[2] + _tlwh[3] * _tlwh[3]) ** 0.5
        for _pcx, _pcy in stationary_positions:
            if math.hypot(_det_cx - _pcx, _det_cy - _pcy) < stationary_guard_radius:
                return True
        return False

    def _should_keep_lost_track(self, track):
        if (self.velocity_cost_vehicle_cls is not None
                and track.cls == self.velocity_cost_vehicle_cls
                and track.mean is not None
                and self.frame_id - track.end_frame <= self.parked_max_time_lost):
            return self._is_track_stationary(track)
        return False

    def _post_state_update(self):
        xover_reverts = []
        for track in self.tracked_stracks:
            if track._xover_save is None:
                continue
            frames_elapsed = self.frame_id - track._xover_grace_start
            if frames_elapsed < self.xover_grace_frames:
                continue
            saved_mean, saved_cov = track._xover_save[0], track._xover_save[1]
            expected_cx = saved_mean[0] + saved_mean[4] * frames_elapsed
            expected_cy = saved_mean[1] + saved_mean[5] * frames_elapsed
            dx = track.mean[0] - expected_cx
            dy = track.mean[1] - expected_cy
            # Propagated positional variance: initial + velocity_uncertainty * N^2
            sigma_x2 = saved_cov[0, 0] + frames_elapsed * frames_elapsed * saved_cov[4, 4]
            sigma_y2 = saved_cov[1, 1] + frames_elapsed * frames_elapsed * saved_cov[5, 5]
            # Cap: don't let velocity-uncertainty term dominate for long-tracked stationary cars.
            _h, _w = saved_mean[3], saved_mean[2] * saved_mean[3]
            sigma_x2 = min(sigma_x2, 0.25 * _w * _w)
            sigma_y2 = min(sigma_y2, 0.25 * _h * _h)
            maha_sq = dx ** 2 / max(sigma_x2, 1.0) + dy ** 2 / max(sigma_y2, 1.0)
            if maha_sq > self.xover_mahab_th:
                _hist = list(track._xover_save[2])
                _still_slow = True
                if len(_hist) >= 5:
                    _speeds = sorted(
                        ((_hist[i][0]-_hist[i-1][0])**2 + (_hist[i][1]-_hist[i-1][1])**2)
                        / max(_hist[i][2] * _hist[i][2], 1.0)
                        for i in range(max(1, len(_hist)-5), len(_hist))
                    )
                    _mid = len(_speeds) // 2
                    _recent_spd = (_speeds[_mid] if len(_speeds) % 2
                                   else (_speeds[_mid-1] + _speeds[_mid]) / 2.0)
                    _still_slow = _recent_spd < self.norm_speed_th
                if _still_slow:
                    track.revert_xover_state()
                    track.mark_lost()
                    xover_reverts.append(track)
                else:
                    track.clear_xover_state()
            else:
                track.clear_xover_state()

        if xover_reverts:
            self._cached_tv_pos = None
            _tv_pos = self._get_tracked_vehicle_positions()
            for track in xover_reverts:
                self.tracked_stracks.remove(track)
                self._snap_to_nearest_detection(
                    track, self._all_detections, _tv_pos, skip_self=True)
                self.lost_stracks.append(track)

    def _invalidate_frame_cache(self):
        self._cached_tv_pos = None
        self._cached_lost_stationary = None

    # --- helpers ---

    def _is_track_stationary(self, track):
        _spd = track.mean[4] * track.mean[4] + track.mean[5] * track.mean[5]
        _diag = track.mean[3] * track.mean[3] * (track.mean[2] * track.mean[2] + 1.0)
        if _spd / max(_diag, 1.0) >= self.norm_speed_th:
            return False
        med = track.get_median_speed_sq()
        return med < 0 or med < self.norm_speed_th

    def _get_tracked_vehicle_positions(self):
        if self._cached_tv_pos is not None:
            return self._cached_tv_pos
        if self.velocity_cost_vehicle_cls is None:
            self._cached_tv_pos = []
        else:
            self._cached_tv_pos = [(t.mean[0], t.mean[1]) for t in self.tracked_stracks
                                   if t.cls == self.velocity_cost_vehicle_cls]
        return self._cached_tv_pos

    def _get_lost_stationary_positions(self, lost_stracks):
        if self._cached_lost_stationary is not None:
            return self._cached_lost_stationary
        self._cached_lost_stationary = []
        if self.velocity_cost_vehicle_cls is None:
            return self._cached_lost_stationary
        for lt in (*self.lost_stracks, *lost_stracks):
            if lt.cls != self.velocity_cost_vehicle_cls or lt.mean is None:
                continue
            if self._is_track_stationary(lt):
                self._cached_lost_stationary.append((lt.mean[0], lt.mean[1]))
        return self._cached_lost_stationary

    @staticmethod
    def _snap_to_nearest_detection(track, all_detections, tv_pos, skip_self=False):
        if not all_detections:
            return
        snap_radius = 0.5 * (track.mean[2] * track.mean[2] + track.mean[3] * track.mean[3]) ** 0.5
        tcx, tcy = track.mean[0], track.mean[1]
        best_dist, best_det = snap_radius, None
        for d in all_detections:
            dcx = d.tlwh[0] + d.tlwh[2] / 2
            dcy = d.tlwh[1] + d.tlwh[3] / 2
            dist = math.hypot(dcx - tcx, dcy - tcy)
            if dist < best_dist:
                snap_diag = (d.tlwh[2] * d.tlwh[2] + d.tlwh[3] * d.tlwh[3]) ** 0.5
                if skip_self:
                    occupied = any(
                        math.hypot(tvx - dcx, tvy - dcy) < 0.5 * snap_diag
                        for tvx, tvy in tv_pos
                        if math.hypot(tvx - tcx, tvy - tcy) > 1.0
                    )
                else:
                    occupied = any(
                        math.hypot(tvx - dcx, tvy - dcy) < 0.5 * snap_diag
                        for tvx, tvy in tv_pos
                    )
                if not occupied:
                    best_dist, best_det = dist, d
        if best_det is not None:
            track.mean[0] = best_det.tlwh[0] + best_det.tlwh[2] / 2
            track.mean[1] = best_det.tlwh[1] + best_det.tlwh[3] / 2
            track.mean[2] = best_det.tlwh[2] / max(best_det.tlwh[3], 1.0)
            track.mean[3] = best_det.tlwh[3]
            track.mean[4] = 0.0
            track.mean[5] = 0.0

    def _is_track_slow(self, track, min_hist=5):
        med = track.get_median_speed_sq()
        if med >= 0:
            return med < self.norm_speed_th if len(track.center_history) >= min_hist else True
        kf_spd = track.mean[4] ** 2 + track.mean[5] ** 2
        kf_diag = track.mean[3] ** 2 * (track.mean[2] ** 2 + 1.0)
        return kf_spd / max(kf_diag, 1.0) < self.norm_speed_th

    def set_vehicle_cls(self, vehicle_cls):
        self.velocity_cost_vehicle_cls = vehicle_cls

    def set_image_size(self, img_size):
        self.img_h, self.img_w = img_size


# alias for run_benchmark_predictions.py compatibility
BYTETracker = BYTETrackerPlus
