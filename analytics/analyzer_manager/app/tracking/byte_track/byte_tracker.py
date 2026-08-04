import numpy as np

from general.analyzer_general import logger, DEFAULT_DETECTOR_IM_SIZE
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .matching import iou_distance, fuse_score, linear_assignment
from .utils import is_track_outside_image_bounds


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
            self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.det_id = new_track.det_id

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

        """ Add newly detected tracklets to tracked_stracks"""
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
        dists = iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
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

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.match_thresh_2)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.per_class_confidence_first[track.subclass]:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
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
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

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
