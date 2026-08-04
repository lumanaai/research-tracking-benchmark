from tracking.byte_track.basetrack import TrackState
from tracking.byte_track.byte_tracker import STrack, BYTETracker, remove_duplicate_stracks, sub_stracks, joint_stracks
from tracking.byte_track.utils import *
from .matching import iou_distance, fuse_score, linear_assignment
from .memory_utils import ReIDMemoryBank
from .reid_models import TrackerPersonReid, TrackerVehicleReid


class ReidSTrack(STrack):
    def __init__(self, tlwh, score, cls, subclass, det_id, context):
        super(ReidSTrack, self).__init__(tlwh, score, cls, subclass, det_id, context)
        self.det_box = self._tlwh.copy()
        self.reid_count = 0  #  serves for counting time since last reid, when zero, reid will run
        self.conflicts = []
        self.use_reid = self.cls in context.conflict_classes
        self.reid_vec = None
        self.reid_run_count = 0

        self.prev_mean, self.prev_covariance = None, None
        self.prev_score = None
        self.prev_det_id = None
        self.prev_det_box = None
        self.max_area = None

    def activate(self, kalman_filter, frame_id):
        super(ReidSTrack, self).activate(kalman_filter, frame_id)
        self.max_area = self.tlwh[2] * self.tlwh[3]

    def re_activate(self, new_track, frame_id, new_id=False):
        self.last_mean, self.last_covariance = self.mean.copy(), self.covariance.copy()
        self.prev_score = self.score
        self.prev_det_id = self.det_id
        self.prev_det_box = self.det_box.copy()
        super(ReidSTrack, self).re_activate(new_track, frame_id, new_id)
        self.det_box = new_track.tlwh
        self.max_area = max(self.max_area, new_track.tlwh[2] * new_track.tlwh[3])

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        super(ReidSTrack, self).update(new_track, frame_id)

        if self.prev_mean is not None:
            self.prev_mean, self.prev_covariance = self.mean.copy(), self.covariance.copy()
            self.prev_score = self.score
            self.prev_det_id = self.det_id
            self.prev_det_box = self.det_box.copy()
            self.prev_max_area = self.max_area
        new_tlwh = new_track.tlwh
        self.det_box = new_tlwh
        self.max_area = max(self.max_area, new_tlwh[2] * new_tlwh[3])

    def unupdate(self):
        if self.prev_mean is not None:
            self.mean, self.covariance = self.prev_mean, self.prev_covariance
            self.score = self.prev_score
            self.det_id = self.prev_det_id
            self.det_box = self.prev_det_box
            self.max_area = self.prev_max_area


class BYTESReidTracker(BYTETracker):
    max_filter_start: Dict[int, np.array]
    max_filtered_subclasses: Set[int]

    def __init__(self, args, frame_rate=30, filter_objects=None, context=None):
        super(BYTESReidTracker, self).__init__(args, frame_rate, filter_objects, context)

        self.prev_track_thresh = args.prev_track_thresh
        self.prev_first_track_thresh = args.prev_first_track_thresh
        self.prev_match_thresh = args.prev_match_thresh
        self.prev_match_thresh_2 = args.prev_match_thresh_2
        self.prev_match_thresh_3 = args.prev_match_thresh_3

        self.ciou_track_thresh = args.ciou_track_thresh
        self.ciou_first_track_thresh = args.ciou_first_track_thresh
        self.ciou_match_thresh = args.ciou_match_thresh
        self.ciou_match_thresh_2 = args.ciou_match_thresh_2
        self.ciou_match_thresh_3 = args.ciou_match_thresh_3

        self.max_time_lost = int(frame_rate * args.track_buffer_seconds)
        self.max_time_removed = int(frame_rate * args.removed_buffer_seconds)
        if self.max_time_removed < self.max_time_lost:
            self.max_time_removed = self.max_time_lost + 1

        self.use_reid_conflict_solver = args.use_reid_conflict_solver
        self.reid_max_batch_size = args.reid_max_batch_size

        self.ciou_switch(args.use_ciou)

        self.img_h, self.img_w = None, None
        self.reid_args = args.reid_args
        class_handler = context.class_handler
        self.reid_models = {}
        if class_handler.person_value >= 0:
            self.reid_models[class_handler.person_value] = TrackerPersonReid(self.reid_args[class_handler.person_value])
        if class_handler.vehicle_value >= 0:
            self.reid_models[class_handler.vehicle_value] = TrackerVehicleReid(
                self.reid_args[class_handler.vehicle_value]
            )

        self.reid_memory_banks_args = args.reid_memory_banks_args
        self.ReIDMemmoryBank = ReIDMemoryBank(self.reid_memory_banks_args, self.reid_models)

        self.reid_freq = int(frame_rate * args.reid_freq_seconds)
        self.conflict_iou_thresh_dict = args.conflict_iou_thresh_dict
        self.conflict_classes = args.conflict_classes
        self.max_reid_cost = args.max_reid_cost
        self.max_ass_cost_thresh_dict = args.max_ass_cost_thresh_dict
        self.maintain_id_cost_disscount = args.maintain_id_cost_disscount

        self.update_max_size_filter()

        self.prepare_visualization = False
        if self.prepare_visualization:
            self.visualiziation_ids_conflicts = None
            self.visualiziation_ids_out_of_conflict = None

        # a set of all used ids
        # self.all_activated_ids = set()

        self.verify_conflict_ids_consistency()

    def update_max_size_filter(self):
        self.max_filter_start, sub_cls = update_size_filter(
            self.args.cls_max_size or {}, self.args.min_max_size_ratio_start
        )
        self.max_filtered_subclasses = self.args.max_size_filtered_subclasses.intersection(sub_cls)

    def ciou_switch(self, use_ciou):
        if use_ciou:
            self.use_ciou = True
            self.track_thresh = self.ciou_track_thresh
            self.first_track_thresh = self.ciou_first_track_thresh
            self.match_thresh = self.ciou_match_thresh
            self.match_thresh_2 = self.ciou_match_thresh_2
            self.match_thresh_3 = self.ciou_match_thresh_3
        else:
            self.use_ciou = False
            self.track_thresh = self.prev_track_thresh
            self.first_track_thresh = self.prev_first_track_thresh
            self.match_thresh = self.prev_match_thresh
            self.match_thresh_2 = self.prev_match_thresh_2
            self.match_thresh_3 = self.prev_match_thresh_3

    def conflict_solver_switch(self, use_reid_conflict_solver):
        if use_reid_conflict_solver:
            self.use_reid_conflict_solver = True
        else:
            self.use_reid_conflict_solver = False

    def verify_conflict_ids_consistency(self):
        possible_classes = set(self.conflict_classes)
        assert set(k for k in list(self.reid_models.keys()) if isinstance(k, int)) == possible_classes
        assert set(self.reid_memory_banks_args.keys()) == possible_classes
        assert set(self.conflict_iou_thresh_dict.keys()) == possible_classes
        assert set(self.max_ass_cost_thresh_dict.keys()) == possible_classes
        assert set(self.reid_args.keys()) == possible_classes

    def handle_none_reassignments(
        self, id_reassignments, id_assignments_including_self, ids_out_of_conflict, id_new_features
    ):
        """
        This function handles cases where reassignments result in `None` due to insufficient similarity for a confident match.
        Instead of reassigning, the function creates a new track ID and marks the original track as lost after reverting the Kalman filter.
        """
        none_reassignments = [k for k, v in id_reassignments.items() if v is None]
        for id_src in none_reassignments:
            for trk in self.tracked_stracks + self.lost_stracks + self.removed_stracks:
                if trk.track_id == id_src:
                    if trk.score > self.first_track_thresh and not self._is_tracker_too_small(trk):
                        # Create a new track if it meets the size and score criteria
                        new_trk = ReidSTrack(trk.tlwh, trk.score, trk.cls, trk.subclass, trk.det_id, trk.context)
                        new_trk.reid_vec = id_new_features[id_src].copy()
                        new_trk.activate(self.kalman_filter, self.frame_id)
                        new_trk.mean = trk.mean.copy()
                        new_trk.covariance = trk.covariance.copy()
                        new_trk.tracklet_len = self.n_init
                        new_trk.is_activated = True
                        self.tracked_stracks.append(new_trk)

                    # Remove the original track (sim too low) from the tracked list
                    self.tracked_stracks.remove(trk)
                    del id_reassignments[id_src]
                    if id_assignments_including_self is not None:
                        del id_assignments_including_self[id_src]

                    # Revert Kalman filter updates and mark the track as lost
                    trk.unupdate()
                    trk.mark_lost()
                    trk.frame_id = (
                        self.frame_id - self.max_time_lost + 3
                    )  # lost time "seen" to make it dissapear if not found in 3 frames

                    # Clean up references to the original track ID
                    self.lost_stracks.append(trk)
                    if id_src in ids_out_of_conflict:
                        del ids_out_of_conflict[id_src]
                    if id_src in id_new_features:
                        del id_new_features[id_src]
                    break

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

        img_h, img_w = img_info[0], img_info[1]
        scale = [img_size[1] / float(img_w), img_size[0] / float(img_h)]
        bboxes_scaled = bboxes * [*scale, *scale]

        self.img_h, self.img_w = img_h, img_w

        remain_inds = scores >= self.track_thresh
        inds_low = scores >= 0.0  #   the actual minimum is filtered in the yolo model
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        objects_keep = objects[remain_inds]
        objects_second = objects[remain_inds]
        classes_keep = subclasses[remain_inds]
        classes_second = subclasses[inds_second]

        det_ids_keep = det_ids[remain_inds]
        det_ids_second = det_ids[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                ReidSTrack(STrack.tlbr_to_tlwh(tlbr), s, c, sc, d, self)
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
        dists = iou_distance(strack_pool, detections, use_ciou=self.use_ciou)

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
                ReidSTrack(STrack.tlbr_to_tlwh(tlbr), s, c, sc, d, self)
                for (tlbr, s, c, sc, d) in zip(
                    dets_second, scores_second, objects_second, classes_second, det_ids_second
                )
            ]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = iou_distance(r_tracked_stracks, detections_second, use_ciou=self.use_ciou)

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
        dists = iou_distance(unconfirmed, detections, use_ciou=self.use_ciou)
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
            if track.score < self.first_track_thresh:
                continue

            if self._is_tracker_too_small(track):
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state """
        for track in self.lost_stracks:
            #   implement a function that takes the track.tlbr and check kf any of the coordinates is out of the image by max_kalman_border, you use img_h, img_w
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Step 6: Update class lists """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        """ Step 7: Update state if out of bounds """
        for track in self.lost_stracks:
            #   implement a function that takes the track.tlbr and check kf any of the coordinates is out of the image by max_kalman_border, you use img_h, img_w
            if is_track_outside_image_bounds(track.tlbr, img_h, img_w, tolerance_ratio=0):
                track.mark_removed()
                self.removed_stracks.append(track)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        """ Step 8: Handle conflicts """
        ids_conflicts = None
        if self.use_reid_conflict_solver:
            prev_id_conflicts = find_prev_id_conflicts(self.tracked_stracks + self.lost_stracks + self.removed_stracks)
            ids_conflicts, id_to_idet, id_to_cls = find_detection_conflicts(
                bboxes,
                objects,
                self.tracked_stracks,
                self.lost_stracks,
                self.removed_stracks,
                self.conflict_iou_thresh_dict,
                prev_id_conflicts,
                self.filter_objects,
            )
            ids_conflicts = filter_conflicts(
                ids_conflicts, self.tracked_stracks + self.lost_stracks + self.removed_stracks, self.ReIDMemmoryBank
            )
            ids_out_of_conflict = update_id_conflicts(self.tracked_stracks, ids_conflicts)
            ids_to_sample = handle_crop_sampling(
                self.tracked_stracks,
                self.lost_stracks,
                ids_conflicts,
                self.reid_freq,
                ids_out_of_conflict,
                self.ReIDMemmoryBank,
            )
            handle_reid_memory(
                ids_to_sample,
                self.ReIDMemmoryBank,
                frame_img,
                bboxes_scaled,
                id_to_cls,
                id_to_idet,
                self.tracked_stracks,
                self.reid_freq,
            )
            id_new_features, cur_features_update = run_reid(
                ids_out_of_conflict,
                self.ReIDMemmoryBank,
                id_to_idet,
                id_to_cls,
                bboxes_scaled,
                frame_img,
                self.reid_models,
            )
            cur_features = build_cur_features_dict(
                self.tracked_stracks + self.lost_stracks + self.removed_stracks,
                ids_out_of_conflict,
                id_new_features,
                cur_features_update,
            )
            id_reassignments, id_assignments_including_self = conflict_solver(
                ids_out_of_conflict,
                id_new_features,
                cur_features,
                self.maintain_id_cost_disscount,
                self.max_reid_cost,
                id_to_cls,
                self.max_ass_cost_thresh_dict,
                self.prepare_visualization,
            )
            self.handle_none_reassignments(
                id_reassignments, id_assignments_including_self, ids_out_of_conflict, id_new_features
            )
            update_trks(
                self.tracked_stracks + self.lost_stracks + self.removed_stracks, id_reassignments, ids_out_of_conflict
            )
            update_reid_vectors(
                self.tracked_stracks + self.lost_stracks + self.removed_stracks, id_new_features, id_reassignments
            )

            # for trk in self.tracked_stracks:
            #     self.all_activated_ids.add(trk.track_id)

            ids_conflicts_updated = {
                id_reassignments.get(key, key): {"ids": [id_reassignments.get(id, id) for id in value["ids"]]}
                for key, value in ids_conflicts.items()
            }

            if self.prepare_visualization:
                ids_out_of_conflict_updated = {
                    id_assignments_including_self.get(key, key): [
                        id_assignments_including_self.get(id, id) for id in value
                    ]
                    for key, value in ids_out_of_conflict.items()
                }
                self.visualiziation_ids_conflicts = ids_conflicts_updated
                self.visualiziation_ids_out_of_conflict = ids_out_of_conflict_updated
                self.visualiziation_ids_run_reid = ids_to_sample

        """ Step 9: Outputs """
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
            in_conflict = int(ids_conflicts_updated is not None and t.track_id in ids_conflicts_updated)
            found = np.where(det_ids == det_id)
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
            strack for strack in self.removed_stracks if (self.frame_id > strack.end_frame + self.max_time_removed)
        ]
        available_ids = set(
            [track.track_id for track in self.tracked_stracks + self.lost_stracks + self.removed_stracks]
        )
        for trk in self.tracked_stracks + self.lost_stracks + self.removed_stracks:
            trk.conflicts = [conflict for conflict in trk.conflicts if conflict in available_ids]
            if (len(trk.conflicts) == 1) and (trk.conflicts[0] == trk.track_id):
                trk.conflicts.clear()

    def _is_tracker_too_small(self, track: ReidSTrack):
        track_w, track_h = track.tlwh[2:]
        track_h, track_w = track_h / self.img_h, track_w / self.img_w
        sub_cls = track.subclass
        if sub_cls in self.max_filtered_subclasses and np.any([track_h, track_w] < self.max_filter_start[sub_cls]):
            return True
        return False
