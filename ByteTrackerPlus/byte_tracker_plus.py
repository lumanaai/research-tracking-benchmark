"""Analytics ByteTrack engine with crossover / parked-vehicle protections.

Vendored from tmp/byte_tracker_plus.py. Imports basetrack/KF/matching from the
analytics clone (via analytics_shim); do not edit analytics/ for this tracker.
"""

from __future__ import annotations

import builtins as _builtins

def print(*args, **kwargs):  # noqa: A001 — mute temporary [DBG*] spam in batch runs
    if args and isinstance(args[0], str) and args[0].startswith("[DBG"):
        return
    _builtins.print(*args, **kwargs)

import math
import numpy as np
from collections import deque

from general.analyzer_general import logger, DEFAULT_DETECTOR_IM_SIZE
from tracking.byte_track.basetrack import BaseTrack, TrackState
from tracking.byte_track.kalman_filter import KalmanFilter
from tracking.byte_track.matching import iou_distance, fuse_score, linear_assignment
from tracking.byte_track.utils import is_track_outside_image_bounds


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, subclass, det_id, context):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.cls = cls
        self.subclass = int(subclass)
        self.det_id = det_id
        self.tracklet_len = 0
        self.age = 0
        self.context = context
        self.n_init = context.n_init
        self.center_history = deque(maxlen=context.velocity_window_size)
        self._append_center(self._tlwh)

        # --- Crossover protection state ---
        self._xover_save = None        # (mean, covariance, center_history_copy) before suspicious match
        self._xover_grace_start = 0    # frame_id when grace period began

    def _append_center(self, tlwh):
        cx = float(tlwh[0] + (tlwh[2] / 2.0))
        cy = float(tlwh[1] + (tlwh[3] / 2.0))
        diag_sq = (tlwh[2] * tlwh[2] + tlwh[3] * tlwh[3]) ** 0.5
        self.center_history.append((cx, cy, diag_sq))

    def save_xover_state(self, frame_id):
        """Save current state before a suspicious match (crossover protection)."""
        self._xover_save = (
            self.mean.copy(),
            self.covariance.copy(),
            deque(self.center_history, maxlen=self.center_history.maxlen),
        )
        self._xover_grace_start = frame_id

    def revert_xover_state(self):
        """Revert to saved pre-crossover state, then re-anchor position to
        the median of the reverted center_history so the track can IoU-match
        its parked detection even if the saved Kalman mean drifted slightly."""
        if self._xover_save is None:
            return False
        self.mean, self.covariance, self.center_history = self._xover_save
        self._xover_save = None
        self._xover_grace_start = 0
        # Re-anchor: set Kalman cx,cy to median of reverted history, zero velocity
        if len(self.center_history) >= 2:
            pts = np.asarray([(c[0], c[1]) for c in self.center_history], dtype=float)
            self.mean[0] = float(np.median(pts[:, 0]))   # cx
            self.mean[1] = float(np.median(pts[:, 1]))   # cy
        self.mean[4] = 0.0   # vx
        self.mean[5] = 0.0   # vy
        return True

    def clear_xover_state(self):
        """Clear saved state (track continued normally)."""
        self._xover_save = None
        self._xover_grace_start = 0

    def next_id(self) -> int:
        return self.context.next_id()

    def predict(self, dt=None):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, dt)

    @staticmethod
    def multi_predict(stracks, dt=None):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, dt)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.age = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        # self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.subclass = new_track.subclass
        self.det_id = new_track.det_id
        self._append_center(new_track.tlwh)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.age += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
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

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    per_class_confidence: np.array
    per_class_confidence_first: np.array

    def __init__(self, args, frame_rate=30, filter_objects=None, context=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        # self.det_thresh = args.BYTETRACK.det_thresh
        # self.track_thresh = args.track_thresh
        # self.first_track_thresh = args.first_track_thresh
        # self.first_track_compensation = args.first_track_compensation
        #
        self.buffer_size = int(frame_rate * args.track_buffer_seconds)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.n_init = args.n_init

        self.match_thresh = args.match_thresh
        self.match_thresh_2 = args.match_thresh_2
        self.match_thresh_3 = args.match_thresh_3

        self.mot20 = args.mot20
        self.velocity_cost_alpha = getattr(args, 'velocity_cost_alpha', 0.0)
        self.velocity_window_size = max(0, int(getattr(args, 'velocity_window_size', 0)))
        self.velocity_noise_th = float(getattr(args, 'velocity_noise_th', 1.5))
        self.velocity_cost_vehicle_cls = None
        self.fast_exit_th = float(getattr(args, 'fast_exit_th', 4.0))  # speed threshold for fast-moving lost tracks (exiting) to use tighter boundary tolerance
        self.norm_dist_th = 0.05                 # normalized distance threshold for crossover protection (distance / bbox diagonal)
        self.norm_speed_th = 0.02 ** 2          # normalized squared speed threshold for crossover protection (speed / bbox diagonal)
        self.xover_grace_frames = 8             # frames to wait before deciding a track was stolen (crossover protection)
        self.xover_mahab_th = 9.21              # chi2(2 DOF, p=0.99) for Mahalanobis distance threshold to revert stolen track (crossover protection)
        self.parked_max_time_lost = int(self.max_time_lost * getattr(args, 'parked_max_lost_multiplier', 10.0))
        self._next_id = 1
        self.filter_objects = {} if filter_objects is None else filter_objects
        self.img_h, self.img_w = DEFAULT_DETECTOR_IM_SIZE

    def next_id(self) -> int:
        _next_id = self._next_id
        self._next_id += 1
        return _next_id

    def set_confidence(self, conf_thresholds, default_value):
        self.per_class_confidence = conf_thresholds
        self.per_class_confidence_first = self.per_class_confidence + self.args.first_track_compensation

    def update_empty(self, dt=None):
        self.frame_id += 1
        for track in self.tracked_stracks:
            track.mark_lost()
        self.lost_stracks += self.tracked_stracks
        self.tracked_stracks = []

        for track in self.lost_stracks:
            outside_image_bounds = is_track_outside_image_bounds(track.tlbr, self.img_h, self.img_w, tolerance_ratio=0)
            if (self.frame_id - track.end_frame > self.max_time_lost) or outside_image_bounds:
                track.mark_removed()

        self.removed_stracks += [track for track in self.lost_stracks if track.state == TrackState.Removed]
        self.lost_stracks = [track for track in self.lost_stracks if track.state == TrackState.Lost]
        STrack.multi_predict(self.lost_stracks + self.tracked_stracks, dt)

        return []

    #    def update(self, output_results, classes, det_ids, dt=None):
    def update(self, output_results, img_info, classes, det_ids, img_size, frame_img, dt=None):

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]

        objects = classes[:, 0]
        subclasses = classes[:, 1]
        cls_track_th = self.per_class_confidence[subclasses.astype(int)]

        remain_inds = scores >= cls_track_th
        inds_low = scores >= 0.0  #   the actual minimum is filtered in the yolo model
        inds_high = scores < cls_track_th

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        objects_keep = objects[remain_inds]
        objects_second = objects[inds_second]
        classes_keep = subclasses[remain_inds]
        classes_second = subclasses[inds_second]

        det_ids_keep = det_ids[remain_inds]
        det_ids_second = det_ids[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, c, sc, d, self)
                for (tlbr, s, c, sc, d) in zip(dets, scores_keep, objects_keep, classes_keep, det_ids_keep)
                if c not in self.filter_objects
            ]
        else:
            detections = []
        all_detections = detections  # save before any filtering for SNAP in Step 6
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, dt)
        # Filter out lost tracks whose predicted position has exited the image.
        # This prevents a new car entering from the same side from stealing the ID of
        # a car that already left the frame.
        strack_pool = [t for t in strack_pool if t.state == TrackState.Tracked
                       or not is_track_outside_image_bounds(
                           t.tlbr, self.img_h, self.img_w,
                           # fast-moving lost tracks (exiting) get a tighter boundary tolerance
                           tolerance_ratio=0.02 if (t.mean[4]**2 + t.mean[5]**2) > self.fast_exit_th else 0.1
                       )]
        dists = iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # Crossover protection: save state when detection is far from prediction
                # Only for stationary/slow vehicles — fast-moving cars naturally have large
                # innovations due to acceleration and should not trigger crossover protection.
                if (self.velocity_cost_vehicle_cls is not None
                    and track.cls == self.velocity_cost_vehicle_cls
                    and track.tracklet_len >= 10
                    and track._xover_save is None):
                    # Check if the track is stationary (low median speed)
                    _hist = list(track.center_history)
                    _is_slow = True
                    if len(_hist) >= 5:
                        _med_spd = float(np.median([
                            ((_hist[i][0]-_hist[i-1][0])**2 + (_hist[i][1]-_hist[i-1][1])**2)
                            / max(_hist[i][2] * _hist[i][2], 1.0)
                            for i in range(1, len(_hist))
                        ]))
                        _is_slow = _med_spd < self.norm_speed_th
                    else:
                        # no history — fall back to Kalman velocity
                        _kf_spd = track.mean[4] * track.mean[4] + track.mean[5] * track.mean[5]
                        _kf_diag = track.mean[3] * track.mean[3] * (track.mean[2] * track.mean[2] + 1.0)
                        _is_slow = _kf_spd / max(_kf_diag, 1.0) < self.norm_speed_th
                    if _is_slow:
                        det_tlwh = det.tlwh
                        _det_cx_s = det_tlwh[0] + det_tlwh[2] / 2
                        _det_cy_s = det_tlwh[1] + det_tlwh[3] / 2
                        _innov = math.hypot(_det_cx_s - track.mean[0], _det_cy_s - track.mean[1])  # distance from predicted center
                        _det_diag = (det_tlwh[2] * det_tlwh[2] + det_tlwh[3] * det_tlwh[3]) ** 0.5  # bbox diagonal of detection
                        _innov_ratio = _innov / _det_diag if _det_diag > 0 else 0.0
                        if _innov_ratio > self.norm_dist_th:
                            track.save_xover_state(self.frame_id)   # suspected crossover - save state before crossover
                if track.track_id in (3, 7):
                    _dcx = det.tlwh[0] + det.tlwh[2]/2
                    _dcy = det.tlwh[1] + det.tlwh[3]/2
                    print(f"[DBG MATCH2] f={self.frame_id} id={track.track_id} pred=({round(track.mean[0])},{round(track.mean[1])}) det=({round(_dcx)},{round(_dcy)})")
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                if self.velocity_cost_vehicle_cls is not None and track.cls == self.velocity_cost_vehicle_cls:
                    _dcx = det.tlwh[0] + det.tlwh[2]/2
                    _dcy = det.tlwh[1] + det.tlwh[3]/2
                    print(f"[DBG REFIND] f={self.frame_id} id={track.track_id} pred=({round(track.mean[0])},{round(track.mean[1])}) det=({round(_dcx)},{round(_dcy)}) lost_frames={self.frame_id - track.end_frame}")
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, c, sc, d, self)
                for (tlbr, s, c, sc, d) in zip(
                    dets_second, scores_second, objects_second, classes_second, det_ids_second
                )
            ]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=self.match_thresh_3)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        _tv_pos = [(t.mean[0], t.mean[1]) for t in self.tracked_stracks
                   if t.cls == self.velocity_cost_vehicle_cls] if self.velocity_cost_vehicle_cls is not None else []
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.track_id in (7, 3, 26, 53):
                print(f"[DBG UNLOST] f={self.frame_id} id={track.track_id} state={track.state} pos=({round(track.mean[0])},{round(track.mean[1])})")
            if not track.state == TrackState.Lost:
                # If a vehicle was being monitored for crossover and goes lost
                # before the grace period ends, revert immediately rather than
                # waiting — the saved state is the correct parked position.
                if track._xover_save is not None:
                    track.revert_xover_state()
                    # Snap to nearest all_detections entry — but skip any detection
                    # already occupied by another tracked vehicle to avoid placing
                    # this lost track on top of it (which would cause remove_duplicate_stracks
                    # to remove the tracked one if this track is older).
                    _best_dist, _best_det = 0.5 * (track.mean[2] * track.mean[2] + track.mean[3] * track.mean[3]) ** 0.5, None
                    for _d in all_detections:
                        _dcx = _d.tlwh[0] + _d.tlwh[2] / 2
                        _dcy = _d.tlwh[1] + _d.tlwh[3] / 2
                        _dist = math.hypot(_dcx - track.mean[0], _dcy - track.mean[1])
                        if _dist < _best_dist:  # filter out detections that are too far away to be a reasonable match
                            _snap_diag = (_d.tlwh[2] * _d.tlwh[2] + _d.tlwh[3] * _d.tlwh[3]) ** 0.5
                            _occupied = any(
                                math.hypot(_tcx - _dcx, _tcy - _dcy) < 0.5 * _snap_diag
                                for _tcx, _tcy in _tv_pos
                            )
                            if not _occupied:
                                _best_dist, _best_det = _dist, _d
                    if _best_det is not None:
                        track.mean[0] = _best_det.tlwh[0] + _best_det.tlwh[2] / 2
                        track.mean[1] = _best_det.tlwh[1] + _best_det.tlwh[3] / 2
                        track.mean[2] = _best_det.tlwh[2] / max(_best_det.tlwh[3], 1.0)
                        track.mean[3] = _best_det.tlwh[3]
                        track.mean[4] = 0.0
                        track.mean[5] = 0.0
                elif (self.velocity_cost_vehicle_cls is not None
                        and track.cls == self.velocity_cost_vehicle_cls
                        and track.tracklet_len >= 10):
                    _spd = track.mean[4] * track.mean[4] + track.mean[5] * track.mean[5]
                    _diag = track.mean[3] * track.mean[3] * (track.mean[2] * track.mean[2] + 1.0)
                    _is_stationary = _spd / max(_diag, 1.0) < self.norm_speed_th
                    if not _is_stationary:
                        # Current speed is above threshold — but check median history
                        # in case the velocity was corrupted by a nearby passing vehicle
                        _hist = list(track.center_history)
                        if len(_hist) >= 10:
                            _median_spd = float(np.median([
                                ((_hist[i][0]-_hist[i-1][0])**2 + (_hist[i][1]-_hist[i-1][1])**2)
                                / max(_hist[i][2] * _hist[i][2], 1.0)
                                for i in range(1, len(_hist))
                            ]))
                            if _median_spd < self.norm_speed_th:
                                _is_stationary = True
                    if _is_stationary:
                        # Vehicle is stationary — zero out velocity and reset position
                        # to historical median to undo any gradual pull from a passing vehicle
                        track.mean[4] = 0.0
                        track.mean[5] = 0.0
                        _hist = list(track.center_history) if not isinstance(track.center_history, list) else track.center_history
                        if len(_hist) >= 10:
                            _med_x = float(np.median([h[0] for h in _hist]))
                            _med_y = float(np.median([h[1] for h in _hist]))
                            track.mean[0] = _med_x
                            track.mean[1] = _med_y
                if self.velocity_cost_vehicle_cls is not None and track.cls == self.velocity_cost_vehicle_cls:
                    _s = (track.mean[4] * track.mean[4] + track.mean[5] * track.mean[5]) ** 0.5
                    _h = track.mean[3] * (track.mean[2] * track.mean[2] + 1.0) ** 0.5
                    print(f"[DBG LOST] f={self.frame_id} id={track.track_id} pos=({round(track.mean[0])},{round(track.mean[1])}) "
                          f"norm_spd={round(_s/max(_h,1),3)} xover={'yes' if track._xover_save is not None else 'no'}")
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        # Predict unconfirmed tracks forward so fast-moving objects can maintain
        # IoU overlap with their next-frame detection (without predict, their
        # position stays stale at their last observation).
        STrack.multi_predict(unconfirmed, dt)
        dists = iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.match_thresh_2)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            if self.velocity_cost_vehicle_cls is not None and track.cls == self.velocity_cost_vehicle_cls:
                print(f"[DBG UNCONF_REMOVED] f={self.frame_id} id={track.track_id} pos=({round(track.mean[0])},{round(track.mean[1])}) tracklet_len={track.tracklet_len}")
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # Guard: don't spawn new IDs near lost stationary vehicle tracks.
        # Those detections belong to the stationary track and will re-associate next frame.
        _lost_stationary_vehicles = []
        if self.velocity_cost_vehicle_cls is not None:
            for lt in (*self.lost_stracks, *lost_stracks):  # include tracks just lost this frame
                if lt.cls == self.velocity_cost_vehicle_cls and lt.mean is not None:
                    _spd = lt.mean[4] * lt.mean[4] + lt.mean[5] * lt.mean[5]
                    _diag = lt.mean[3] * lt.mean[3] * (lt.mean[2] * lt.mean[2] + 1.0)
                    if _spd / max(_diag, 1.0) < self.norm_speed_th:
                        _hist = list(lt.center_history)
                        if len(_hist) < 2 or (
                            float(np.median([
                                ((_hist[i][0]-_hist[i-1][0])**2 + (_hist[i][1]-_hist[i-1][1])**2)
                                / max(_hist[i][2] * _hist[i][2], 1.0)
                                for i in range(1, len(_hist))
                            ])) < self.norm_speed_th
                        ):
                            # we are sure the vehicle is stationary for some time, so add to list of lost stationary vehicles                        
                            _lost_stationary_vehicles.append((lt.mean[0], lt.mean[1], lt.track_id))

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.per_class_confidence_first[track.subclass]:
                continue
            stationary_guard_radius = 0.5 * (track.tlwh[2] * track.tlwh[2] + track.tlwh[3] * track.tlwh[3]) ** 0.5      # px — detections within this radius are reserved for lost stationary vehicles
            # skip if this detection is close to a lost parked vehicle
            _det_cx = track.tlwh[0] + track.tlwh[2] / 2
            _det_cy = track.tlwh[1] + track.tlwh[3] / 2
            _guarded = False
            # TODO: optimise double loop with spatial index or numpy array operations
            for _pcx, _pcy, _pid in _lost_stationary_vehicles:
                if math.hypot(_det_cx - _pcx, _det_cy - _pcy) < stationary_guard_radius:
                    _guarded = True
                    break
            if _guarded:
                if self.velocity_cost_vehicle_cls is not None and track.cls == self.velocity_cost_vehicle_cls:
                    print(f"[DBG GUARD] f={self.frame_id} det pos=({round(_det_cx)},{round(_det_cy)}) BLOCKED by pid={_pid} at ({round(_pcx)},{round(_pcy)}) radius={round(stationary_guard_radius)}")
                continue
            # if isn't in the reserved area, spawn a new tracklet
            if self.velocity_cost_vehicle_cls is not None and track.cls == self.velocity_cost_vehicle_cls:
                _near_tracked = [(t.track_id, round(t.mean[0]), round(t.mean[1]))
                                 for t in tracked_stracks
                                 if t.cls == self.velocity_cost_vehicle_cls
                                 and math.hypot(t.mean[0] - _det_cx, t.mean[1] - _det_cy) < 300]
                _near_lost = [(t.track_id, round(t.mean[0]), round(t.mean[1]))
                              for t in (*self.lost_stracks, *lost_stracks)
                              if t.cls == self.velocity_cost_vehicle_cls
                              and math.hypot(t.mean[0] - _det_cx, t.mean[1] - _det_cy) < 300]
                # Tracked stationary vehicles within guard radius that are NOT protected (the bug)
                _unguarded_stationary = []
                for _t in tracked_stracks:
                    if _t.cls != self.velocity_cost_vehicle_cls or _t.mean is None:
                        continue
                    if math.hypot(_t.mean[0] - _det_cx, _t.mean[1] - _det_cy) < stationary_guard_radius:
                        _gm = _t._xover_save[0] if _t._xover_save is not None else _t.mean
                        _ts = _gm[4] * _gm[4] + _gm[5] * _gm[5]
                        _th = _gm[3] * _gm[3] * (_gm[2] * _gm[2] + 1.0)
                        if _ts / max(_th, 1.0) < self.norm_speed_th:
                            _unguarded_stationary.append((_t.track_id, round(_gm[0]), round(_gm[1])))
                print(f"[DBG SPAWN] f={self.frame_id} new_id={self._next_id} pos=({round(_det_cx)},{round(_det_cy)}) "
                      f"near_tracked={_near_tracked} near_lost={_near_lost} guard={[(round(x),round(y),pid) for x,y,pid in _lost_stationary_vehicles]}"
                      f" UNGUARDED_STATIONARY={_unguarded_stationary}")
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                # Don't expire stationary lost vehicles — they may be temporarily
                # occluded and should recover when visible again.
                # Cap at parked_max_time_lost to handle the case where the car
                # was stationary, then drove away while occluded.
                if (self.velocity_cost_vehicle_cls is not None
                        and track.cls == self.velocity_cost_vehicle_cls
                        and track.mean is not None
                        and self.frame_id - track.end_frame <= self.parked_max_time_lost):
                    _spd = track.mean[4] * track.mean[4] + track.mean[5] * track.mean[5]
                    _diag = track.mean[3] * track.mean[3] * (track.mean[2] * track.mean[2] + 1.0)
                    if _spd / max(_diag, 1.0) < self.norm_speed_th:
                        _hist = list(track.center_history)
                        if len(_hist) < 2 or (
                            float(np.median([
                                ((_hist[i][0]-_hist[i-1][0])**2 + (_hist[i][1]-_hist[i-1][1])**2)
                                / max(_hist[i][2] * _hist[i][2], 1.0)
                                for i in range(1, len(_hist))
                            ])) < self.norm_speed_th
                        ):
                            continue  # keep parked vehicle alive in lost_stracks
                if self.velocity_cost_vehicle_cls is not None and track.cls == self.velocity_cost_vehicle_cls:
                    print(f"[DBG EXPIRED] f={self.frame_id} id={track.track_id} pos=({round(track.mean[0])},{round(track.mean[1])}) "
                          f"time_lost={self.frame_id - track.end_frame}")
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        if self.velocity_cost_vehicle_cls is not None:
            _pre_tr = {t.track_id for t in self.tracked_stracks if t.cls == self.velocity_cost_vehicle_cls}
            _pre_ls = {t.track_id for t in self.lost_stracks if t.cls == self.velocity_cost_vehicle_cls}
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        if self.velocity_cost_vehicle_cls is not None:
            _post_tr = {t.track_id for t in self.tracked_stracks if t.cls == self.velocity_cost_vehicle_cls}
            _post_ls = {t.track_id for t in self.lost_stracks if t.cls == self.velocity_cost_vehicle_cls}
            _gone_tr = _pre_tr - _post_tr
            _gone_ls = _pre_ls - _post_ls
            if _gone_tr or _gone_ls:
                print(f"[DBG DEDUP] f={self.frame_id} removed_from_tracked={_gone_tr} removed_from_lost={_gone_ls}")

        # Crossover protection: revert stolen vehicle tracks based on drift from expected trajectory
        # Threshold is Mahalanobis-based: chi-squared 2-DOF at 99th percentile = 9.21.
        # Positional variance is propagated from the saved covariance using the constant-velocity
        # model: sigma^2(N) = P[pos,pos] + N^2 * P[vel,vel], capturing both the initial
        # positional uncertainty and the uncertainty in the extrapolated velocity.
        xover_reverts = []
        for track in self.tracked_stracks:
            if track._xover_save is None:
                continue
            frames_elapsed = self.frame_id - track._xover_grace_start
            if frames_elapsed < self.xover_grace_frames:
                continue
            # Where should the track be if it continued with its pre-save velocity?
            saved_mean, saved_cov = track._xover_save[0], track._xover_save[1]
            expected_cx = saved_mean[0] + saved_mean[4] * frames_elapsed
            expected_cy = saved_mean[1] + saved_mean[5] * frames_elapsed
            dx = track.mean[0] - expected_cx
            dy = track.mean[1] - expected_cy
            # Propagated positional variance: initial + velocity_uncertainty * N^2
            sigma_x2 = saved_cov[0, 0] + frames_elapsed ** 2 * saved_cov[4, 4]
            sigma_y2 = saved_cov[1, 1] + frames_elapsed ** 2 * saved_cov[5, 5]
            # Cap: don't let velocity-uncertainty term dominate for long-tracked stationary cars.
            _h, _w = saved_mean[3], saved_mean[2] * saved_mean[3]
            sigma_x2 = min(sigma_x2, 0.25 * _w * _w)
            sigma_y2 = min(sigma_y2, 0.25 * _h * _h)
            maha_sq = dx ** 2 / max(sigma_x2, 1.0) + dy ** 2 / max(sigma_y2, 1.0)
            if maha_sq > self.xover_mahab_th:
                # Track diverged from its expected trajectory — but only revert if
                # it was stationary when saved. Use saved history to reflect pre-steal state.
                _hist = list(track._xover_save[2])  # saved history at save time
                _still_slow = True
                if len(_hist) >= 5:
                    _recent_spd = float(np.median([
                        ((_hist[i][0]-_hist[i-1][0])**2 + (_hist[i][1]-_hist[i-1][1])**2)
                        / max(_hist[i][2] * _hist[i][2], 1.0)
                        for i in range(max(1, len(_hist)-5), len(_hist))
                    ]))
                    _still_slow = _recent_spd < self.norm_speed_th
                if _still_slow:
                    if self.velocity_cost_vehicle_cls is not None and track.cls == self.velocity_cost_vehicle_cls:
                        print(f"[DBG XOVER_REVERT] f={self.frame_id} id={track.track_id} pos=({round(track.mean[0])},{round(track.mean[1])}) maha_sq={round(maha_sq,1)}")
                    track.revert_xover_state()
                    track.mark_lost()
                    xover_reverts.append(track)
                else:
                    # Track accelerated — not a crossover. Clear save.
                    track.clear_xover_state()
            else:
                # Track continued along expected trajectory — all good
                track.clear_xover_state()
        _tv_pos = [(t.mean[0], t.mean[1]) for t in self.tracked_stracks
                   if t.cls == self.velocity_cost_vehicle_cls] if self.velocity_cost_vehicle_cls is not None else []
        for track in xover_reverts:
            self.tracked_stracks.remove(track)
            # Snap reverted track to nearest detection in current frame so it
            # can IoU-match in the very next frame (avoids the dead-lock where
            # the reanchored position has zero IoU with the actual detection).
            # Use all_detections (pre-filter) since the parked car's detection
            # may have been claimed by a phantom in Step 2.
            # Snap radius is bbox-normalized (same as Step 3) to prevent
            # snapping onto a different nearby car.
            _snap_radius = 0.5 * (track.mean[2] * track.mean[2] + track.mean[3] * track.mean[3]) ** 0.5
            _best_dist, _best_det = _snap_radius, None
            for _d in all_detections:
                _dcx = _d.tlwh[0] + _d.tlwh[2] / 2
                _dcy = _d.tlwh[1] + _d.tlwh[3] / 2
                _dist = math.hypot(_dcx - track.mean[0], _dcy - track.mean[1])
                if _dist < _best_dist:
                    _snap_diag = (_d.tlwh[2] * _d.tlwh[2] + _d.tlwh[3] * _d.tlwh[3]) ** 0.5
                    _occupied = any(
                        math.hypot(_tcx - _dcx, _tcy - _dcy) < 0.5 * _snap_diag
                        for _tcx, _tcy in _tv_pos
                        if math.hypot(_tcx - track.mean[0], _tcy - track.mean[1]) > 1.0  # skip self
                    )
                    if not _occupied:
                        _best_dist, _best_det = _dist, _d
            if _best_det is not None:
                track.mean[0] = _best_det.tlwh[0] + _best_det.tlwh[2] / 2
                track.mean[1] = _best_det.tlwh[1] + _best_det.tlwh[3] / 2
                track.mean[2] = _best_det.tlwh[2] / max(_best_det.tlwh[3], 1.0)  # aspect
                track.mean[3] = _best_det.tlwh[3]                                  # height
                track.mean[4] = 0.0
                track.mean[5] = 0.0
            self.lost_stracks.append(track)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        _dbg_ids = {8, 45, 114}
        _dbg_tracked = [(t.track_id, round(t.mean[0]), round(t.mean[1]), round(t.mean[4],1), round(t.mean[5],1), t.tracklet_len) for t in output_stracks if t.track_id in _dbg_ids]
        _dbg_lost = [(t.track_id, round(t.mean[0]), round(t.mean[1]), self.frame_id - t.end_frame) for t in self.lost_stracks if t.track_id in _dbg_ids]
        if _dbg_tracked or _dbg_lost:
            print(f"[DBG37 STATE] f={self.frame_id} tracked={_dbg_tracked} lost={_dbg_lost}")
        if self.velocity_cost_vehicle_cls is not None:
            _veh_tracked = [(t.track_id, round(t.mean[0]), round(t.mean[1]), t.tracklet_len) for t in output_stracks if t.cls == self.velocity_cost_vehicle_cls]
            _veh_lost = [(t.track_id, round(t.mean[0]), round(t.mean[1]), self.frame_id - t.end_frame) for t in self.lost_stracks if t.cls == self.velocity_cost_vehicle_cls]
            print(f"[DBG STATE] f={self.frame_id} tracked={_veh_tracked} lost={_veh_lost}")

        #   our
        outputs = []

        for j, t in enumerate(output_stracks):
            #
            track_id = t.track_id
            cls = t.cls
            subclass = t.subclass
            score = t.score
            det_id = t.det_id
            age = t.age
            found = np.where(det_ids == det_id)
            in_conflict = 0
            if len(found) > 0 and len(found[0]) > 0:
                idx = found[0][0]
                x1, y1, x2, y2 = bboxes[idx]
            else:
                # print("reverting to tracker bbox")
                x1, y1, x2, y2 = t.tlbr

            outputs.append(np.array([x1, y1, x2, y2, track_id, cls, subclass, score, det_id, age, in_conflict]))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs

    def cleanup(self):
        self.removed_stracks = [
            strack for strack in self.removed_stracks if strack.end_frame > self.frame_id - self.buffer_size - 1
        ]

    def replace_ids(self, id_to_keep: int, id_to_merge: int):
        strack_pool = joint_stracks(self.tracked_stracks, self.lost_stracks)
        for track in strack_pool:
            if track.track_id == id_to_merge:
                track.track_id = id_to_keep
                return True
        return False

    def set_vehicle_cls(self, vehicle_cls):
        self.velocity_cost_vehicle_cls = vehicle_cls

    def set_image_size(self, img_size):
        self.img_h, self.img_w = img_size
        logger.info(f"Tracker setting image size to {img_size}")


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

