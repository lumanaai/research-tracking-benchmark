"""Hookified BYTETracker base for ByteTrackerPlusAug19.

Copied from analytics byte_track/byte_tracker.py with overridable hooks so the
Aug19 Plus subclass can inject crossover / CIoU / parked-vehicle behavior
without editing analytics/.
"""
from __future__ import annotations

import numpy as np

from general.analyzer_general import logger, DEFAULT_DETECTOR_IM_SIZE
from tracking.byte_track.basetrack import BaseTrack, TrackState
from tracking.byte_track.kalman_filter import KalmanFilter
from tracking.byte_track.matching import iou_distance, fuse_score, linear_assignment
from tracking.byte_track.utils import is_track_outside_image_bounds


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, subclass, det_id, context):
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
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance, dt
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.age = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.subclass = new_track.subclass
        self.det_id = new_track.det_id

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.age += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked

        if self.tracklet_len >= self.n_init:
            self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.det_id = new_track.det_id

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
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
        self.buffer_size = int(frame_rate * args.track_buffer_seconds)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.n_init = args.n_init

        self.match_thresh = args.match_thresh
        self.match_thresh_2 = args.match_thresh_2
        self.match_thresh_3 = args.match_thresh_3

        self.mot20 = args.mot20
        self._next_id = 1
        self.filter_objects = {} if filter_objects is None else filter_objects
        self.img_h, self.img_w = DEFAULT_DETECTOR_IM_SIZE
        self._all_detections = []

    def next_id(self) -> int:
        _next_id = self._next_id
        self._next_id += 1
        return _next_id

    def set_confidence(self, conf_thresholds, default_value):
        self.per_class_confidence = conf_thresholds
        self.per_class_confidence_first = self.per_class_confidence + self.args.first_track_compensation

    # --- hooks (overridden by BYTETrackerPlus) ---

    def _make_strack(self, tlwh, s, c, sc, d):
        return STrack(tlwh, s, c, sc, d, self)

    def _iou_distance(self, atracks, btracks):
        return iou_distance(atracks, btracks)

    def _filter_strack_pool(self, strack_pool):
        return strack_pool

    def _on_first_match(self, track, det):
        return None

    def _on_track_going_lost(self, track):
        return None

    def _prepare_unconfirmed(self, unconfirmed, dt):
        return None

    def _should_suppress_new_track(self, track, lost_stracks):
        return False

    def _should_keep_lost_track(self, track):
        return False

    def _post_state_update(self):
        return None

    def _invalidate_frame_cache(self):
        return None

    def update_empty(self, dt=None):
        self.frame_id += 1
        for track in self.tracked_stracks:
            track.mark_lost()
        self.lost_stracks += self.tracked_stracks
        self.tracked_stracks = []

        for track in self.lost_stracks:
            outside_image_bounds = is_track_outside_image_bounds(
                track.tlbr, self.img_h, self.img_w, tolerance_ratio=0
            )
            if (self.frame_id - track.end_frame > self.max_time_lost) or outside_image_bounds:
                track.mark_removed()

        self.removed_stracks += [track for track in self.lost_stracks if track.state == TrackState.Removed]
        self.lost_stracks = [track for track in self.lost_stracks if track.state == TrackState.Lost]
        STrack.multi_predict(self.lost_stracks + self.tracked_stracks, dt)

        return []

    def update(self, output_results, img_info, classes, det_ids, img_size, frame_img, dt=None):
        self._invalidate_frame_cache()
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
        inds_low = scores >= 0.0
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
            detections = [
                self._make_strack(STrack.tlbr_to_tlwh(tlbr), s, c, sc, d)
                for (tlbr, s, c, sc, d) in zip(
                    dets, scores_keep, objects_keep, classes_keep, det_ids_keep
                )
                if c not in self.filter_objects
            ]
        else:
            detections = []
        self._all_detections = detections

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool, dt)
        strack_pool = self._filter_strack_pool(strack_pool)
        dists = self._iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                self._on_first_match(track, det)
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if len(dets_second) > 0:
            detections_second = [
                self._make_strack(STrack.tlbr_to_tlwh(tlbr), s, c, sc, d)
                for (tlbr, s, c, sc, d) in zip(
                    dets_second, scores_second, objects_second, classes_second, det_ids_second
                )
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        dists = self._iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(
            dists, thresh=self.match_thresh_3
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                self._on_track_going_lost(track)
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        self._prepare_unconfirmed(unconfirmed, dt)
        dists = self._iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, thresh=self.match_thresh_2
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.per_class_confidence_first[track.subclass]:
                continue
            if self._should_suppress_new_track(track, lost_stracks):
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                if self._should_keep_lost_track(track):
                    continue
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        self._post_state_update()

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for j, t in enumerate(output_stracks):
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
                x1, y1, x2, y2 = t.tlbr
            outputs.append(
                np.array([x1, y1, x2, y2, track_id, cls, subclass, score, det_id, age, in_conflict])
            )

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def cleanup(self):
        self.removed_stracks = [
            strack
            for strack in self.removed_stracks
            if strack.end_frame > self.frame_id - self.buffer_size - 1
        ]

    def replace_ids(self, id_to_keep: int, id_to_merge: int):
        strack_pool = joint_stracks(self.tracked_stracks, self.lost_stracks)
        for track in strack_pool:
            if track.track_id == id_to_merge:
                track.track_id = id_to_keep
                return True
        return False

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
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
