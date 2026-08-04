from typing import Dict, Set

import cv2
from cython_bbox import bbox_overlaps as bbox_ious
import numpy as np
from scipy.optimize import linear_sum_assignment


def find_conflicts(iou_matrix, areas, threshold_dict, row_objs, coloumn_objs, row_prev_conflicts_cols, filter_objects):
    """
    Identifies conflicts among detections based on Intersection over Union (IoU) thresholds.

    Parameters:
        iou_matrix (np.ndarray): A 2D matrix where each element [i][j] represents the IoU between the i-th and j-th detections.
        areas (np.ndarray): A 1D array where each element represents the area of the corresponding detection.
    threshold (float): The IoU threshold above which detections are considered to be in conflict.

    Returns:
    dict: A dictionary where keys are detection indices and values are lists of indices of conflicting detections.
    """
    n1 = iou_matrix.shape[0]
    n2 = iou_matrix.shape[1]

    # Apply union-find for indices with IoU at or above the threshold
    conflicts_dict = {}
    for i in range(n1):
        for j in range(n2):
            iou = iou_matrix[i][j]
            if i == j:
                continue
            # Calculate each intersection area compared to box size
            iou_overlap_i = (iou / (iou + 1)) * ((areas[i] + areas[j]) / areas[i])
            iou_overlap_j = (iou / (iou + 1)) * ((areas[i] + areas[j]) / areas[j])
            maximal_overlap = max(iou_overlap_i, iou_overlap_j)
            # maximal_overlap = iou
            # Union the sets if the maximal overlap is above the threshold
            if row_objs[i] == coloumn_objs[j]:
                if row_objs[i] in threshold_dict:
                    if row_objs[i] not in filter_objects:
                        was_in_conflict = (i in row_prev_conflicts_cols) and (j in row_prev_conflicts_cols[i])

                        cur_thresh = (
                            threshold_dict[row_objs[i]]["start"]
                            if not was_in_conflict
                            else threshold_dict[row_objs[i]]["continue"]
                        )  #   conflict hysteresis
                        if maximal_overlap >= cur_thresh:
                            if i not in conflicts_dict:
                                conflicts_dict[i] = [i]
                            conflicts_dict[i].append(j)
    return conflicts_dict


def find_detection_conflicts(
    bboxes, objects, tracked_trks, lost_trks, removed_trks, conflict_iou_thresh_dict, prev_id_conflicts, filter_objects
):
    """
    Determines detection conflicts based on IoU and updates the detection objects with conflict information.

    Parameters:
    raw_dets (np.ndarray): A 2D array where each row represents a detection bounding box in the format [x1, y1, x2, y2].
    detections (list): A list of detection objects that will be updated with conflict information.
    conflict_iou_thresh (float): The IoU threshold above which detections are considered to be in conflict.

    Returns:
    list: The updated list of detection objects with conflict information.
    """

    if len(tracked_trks) < 1:
        return {}, {}, {}

    id_to_cls = get_id_to_cls(tracked_trks, lost_trks, removed_trks)
    idet_to_id, id_to_idet = translate_idet_to_id(tracked_trks)

    tracked_bboxes = [bboxes[trk.det_id] for trk in tracked_trks]
    tracked_objects = [objects[trk.det_id] for trk in tracked_trks]
    tracked_ids = [trk.track_id for trk in tracked_trks]

    trcked_dets = np.stack(tracked_bboxes)
    trcked_objs = np.stack(tracked_objects)
    trcked_dets = np.ascontiguousarray(trcked_dets, dtype=float)  #   (n, 4)

    if len(lost_trks):
        lost_dets = np.ascontiguousarray(np.stack([trk.tlbr for trk in lost_trks]))
        lost_objs = np.ascontiguousarray(np.stack([trk.cls for trk in lost_trks]))
        lost_ids = [trk.track_id for trk in lost_trks]  #   (m)
        compare_dets = np.concatenate([trcked_dets, lost_dets])  #   (n+m, 4)
        compare_objs = np.concatenate([trcked_objs, lost_objs])
    else:
        lost_ids = []
        compare_dets = trcked_dets
        compare_objs = trcked_objs

    if len(compare_dets) < 2:
        return {}, id_to_idet, id_to_cls

    row_ind_to_id_dict, coloumn_ind_to_id_dict = build_ind_to_id_dict(
        tracked_ids, lost_ids, len(trcked_dets), len(compare_dets)
    )

    row_prev_conflicts_cols = {}
    for row in row_ind_to_id_dict:
        row_id = row_ind_to_id_dict[row]["id"]
        if row_id in prev_id_conflicts:
            row_conflicts_in_ids = prev_id_conflicts[row_id]
            row_prev_conflicts_cols[row] = []
            for col in coloumn_ind_to_id_dict:
                if coloumn_ind_to_id_dict[col]["id"] in row_conflicts_in_ids:
                    row_prev_conflicts_cols[row].append(col)

    iou_distances = bbox_ious(trcked_dets, compare_dets)  # Reimplement to avoid double calculation
    areas = (compare_dets[:, 2] - compare_dets[:, 0]) * (compare_dets[:, 3] - compare_dets[:, 1])
    conflicts_dict = find_conflicts(
        iou_distances,
        areas,
        conflict_iou_thresh_dict,
        trcked_objs,
        compare_objs,
        row_prev_conflicts_cols,
        filter_objects,
    )  #   detection to detection
    ids_conflicts = translate_conflicts(
        conflicts_dict, row_ind_to_id_dict, coloumn_ind_to_id_dict
    )  #   transform to ids

    return ids_conflicts, id_to_idet, id_to_cls


def build_ind_to_id_dict(tracked_ids, lost_ids, row_count, coloumn_count):
    """
    This function builds two dictionaries: one mapping row indices to track IDs (row_ind_to_id_dict)
    and another mapping column indices to track IDs (coloumn_ind_to_id_dict). The row_ind_to_id_dict
    contains mappings for live tracks, while the coloumn_ind_to_id_dict includes both live and lost tracks.
    """
    row_ind_to_id_dict = {}
    coloumn_ind_to_id_dict = {}
    for i in range(coloumn_count):
        if i < row_count:  #   live tracks
            row_ind_to_id_dict[i] = {"ind": i, "id": tracked_ids[i]}
            coloumn_ind_to_id_dict[i] = row_ind_to_id_dict[i]
        else:  #   lost tracks
            lost_ind = i - row_count
            coloumn_ind_to_id_dict[i] = {"ind": None, "id": lost_ids[lost_ind]}
    return row_ind_to_id_dict, coloumn_ind_to_id_dict


def translate_idet_to_id(tracks):
    idet_to_id = {}
    id_to_idet = {}
    for trk in tracks:
        idet_to_id[trk.det_id] = trk.track_id

    for v, k in idet_to_id.items():
        id_to_idet[k] = v
    return idet_to_id, id_to_idet


def get_id_to_cls(*track_lists):
    """
    This function takes one or more lists of track objects and builds a dictionary (id_to_cls)
    that maps each track's ID (track_id) to its associated class (cls). It processes each list
    of tracks provided as an argument.
    """
    id_to_cls = {}
    for tracks in track_lists:
        for trk in tracks:
            id_to_cls[trk.track_id] = trk.cls
    return id_to_cls


def translate_conflicts(detection_conflicts, source_dict, target_dict):
    """
    This function translates detection conflicts from index-based references to ID-based references.
    It takes a dictionary of detection conflicts, where each key is a source index, and maps these
    to the corresponding IDs using source_dict and target_dict. If a source index is not found in
    source_dict, it is skipped. The resulting id_conflicts dictionary maps source IDs to their
    corresponding conflicting IDs and indices from the target dictionary.
    """
    id_conflicts = {}
    for ind_source, idets_conflicts in detection_conflicts.items():

        #   The detection was not tracked
        if ind_source not in source_dict:
            continue

        id_source = source_dict[ind_source]["id"]
        id_conflicts[id_source] = {
            "ids": [target_dict[i]["id"] for i in idets_conflicts],
            "idets": [target_dict[i]["ind"] for i in idets_conflicts],
        }

    return id_conflicts


def build_ids_state(trks):
    ids_state = {}
    for trk in trks:
        ids_state[trk.track_id] = trk.state
    return ids_state


def update_id_conflicts(tracked_tracks, ids_conflicts):
    """
    This function updates the conflict information for each tracked track. It maintains a dictionary
    (out_of_conflict_ids) for tracks that are no longer in conflict but had conflicts before. The function
    processes each track and updates its conflict list by adding IDs from ids_conflicts if they are still
    alive or lost but not yet included in the current conflicts. It also resets conflicts to an empty list
    if the track is not in ids_conflicts. Finally, it removes entries from out_of_conflict_ids if their
    corresponding list of conflicts is empty.
    """
    out_of_conflict_ids = {}

    for trk in tracked_tracks:
        if len(trk.conflicts) > 1:
            if trk.track_id not in ids_conflicts:
                out_of_conflict_ids[trk.track_id] = [id for id in trk.conflicts]
                out_of_conflict_ids[trk.track_id].sort()

        if trk.track_id in ids_conflicts:
            trk.conflicts = trk.conflicts + [
                id for id in ids_conflicts[trk.track_id]["ids"] if (id >= 0) and (id not in trk.conflicts)
            ]  #   append to current conflicts if they are alive or lost
        else:
            trk.conflicts = []

    return out_of_conflict_ids


def is_track_outside_image_bounds(track_tlbr, img_h, img_w, tolerance_ratio=0.05):
    """
    This function checks if a track is outside the image bounds based on top-left-bottom-right (tlbr) coordinates
    and if any single one of them is out of bounds.
    This situation is possible due to the kalman filter, as detections cannot be outside the bounds.
    Atolerance (new_parameter) to account for slight  overflows outside the image bounds.
    It checks if the track's top-left-bottom-right (tlbr) coordinates are
    outside the image dimensions (height and width) with an additional buffer zone.
    """
    # extrach h and w of tlbr (xyxy)
    h_box = track_tlbr[3] - track_tlbr[1]
    w_box = track_tlbr[2] - track_tlbr[0]

    # assert h_box > 0 and w_box > 0

    left, top, right, bottom = track_tlbr
    delta_w = w_box * tolerance_ratio
    delta_h = h_box * tolerance_ratio

    return right > img_w + delta_w or left < -delta_w or top > img_h + delta_h or bottom < -delta_h


def filter_out_of_conflict(ids_out_of_conflict, ids_list, img_h, img_w):
    """
    This function filters out tracks that are considered out of conflict based on whether they are outside the image bounds.
    It first identifies the tracks corresponding to the IDs in ids_out_of_conflict and checks if they are outside the image
    bounds and they are are added to a filter list and removed from the ids_out_of_conflict dictionary.
    """

    filter_list = []
    for id in ids_out_of_conflict:
        cur_trk = None
        for trk in ids_list:
            if trk.track_id == id:
                cur_trk = trk
                break

        tlbr = cur_trk.tlbr
        outside_bounds = is_track_outside_image_bounds(tlbr, img_h, img_w)
        if outside_bounds:
            filter_list.append(id)

    # perform filtering
    ids_out_of_conflict = {k: v for k, v in ids_out_of_conflict.items() if k not in filter_list}

    return ids_out_of_conflict


def find_prev_id_conflicts(tracks):
    """
    This function identifies and returns previous ID conflicts for tracks that use re-identification (reid).
    The resulting dictionary maps track IDs to their corresponding conflicts.
    """
    prev_id_conflicts = {}
    for trk in tracks:
        if trk.use_reid:
            prev_id_conflicts[trk.track_id] = trk.conflicts
    return prev_id_conflicts


def handle_crop_sampling(tracked_stracks, lost_stracks, ids_conflicts, reid_freq, ids_out_of_conflict, memory_bank):
    """
    This function manages the sampling count for both tracked and lost tracks, determining which tracks
    need to be pushed to memory for reid, which will run if needed. A reid count of zero indicates that it's
    time for sampling; this happens if the ID is not in conflict or if the ID is in conflict but lacks a reid vector
    prior to the current frame. The function also updates the reid count for lost IDs, but these counts remain at zero
    until the IDs are no longer lost and can be sampled. Additionally, the function ensures that tracks coming out of conflict
    have their reid count reset appropriately since they are going to run reid anyway.
    """
    reids_to_run = []

    for trk in tracked_stracks:
        if trk.reid_count == 0 and trk.use_reid:  #   need to run reid
            in_memory = memory_bank.is_id_in_memory(trk.track_id)  #   check if id in memory
            if (trk.track_id not in ids_conflicts) or (trk.reid_vec is None and not (in_memory)):
                if trk.is_activated:
                    reids_to_run.append(trk.track_id)

    #   Update count
    for trk in tracked_stracks:
        if (trk.track_id not in ids_conflicts) or (trk.reid_count != 0) or (trk.track_id in reids_to_run):
            if trk.use_reid:
                if trk.is_activated:
                    trk.reid_count = (trk.reid_count + 1) % reid_freq

    #  lost tracks
    for trk in lost_stracks:
        if trk.reid_count != 0 and trk.use_reid:
            trk.reid_count = (trk.reid_count + 1) % reid_freq

    #   remove ids from out_of_conflict_ids, set their count to 1, because  they are checked anyway..
    reids_to_run = [
        id for id in reids_to_run if id not in ids_out_of_conflict
    ]  #   dont run again for ids that are checked because out of conflicts
    for trk in tracked_stracks:
        if trk.track_id in ids_out_of_conflict:
            trk.reid_count = 1  #   the reid is running anyway

    return reids_to_run


def handle_reid_memory(ids_to_sample, reid_memory_banks, frame_img, dets, id_to_cls, id_to_idet, tracks, reid_freq):
    """
    This function processes tracks that might need re-identification (reid) by extracting image crops and storing them in memory.
    For each track ID in ids_to_sample, it retrieves the corresponding detection bounding box (crop_xyxy) from dets,
    extracts the relevant portion of the frame image, and then pushes this crop into the reid memory banks with the associated
    class label.
    """
    id_to_trkind = {trk.track_id: i for i, trk in enumerate(tracks)}
    for id in ids_to_sample:
        crop_xyxy = dets[id_to_idet[id]]
        crop_xyxy = frame_img[int(crop_xyxy[1]) : int(crop_xyxy[3]), int(crop_xyxy[0]) : int(crop_xyxy[2])]
        success = reid_memory_banks.push_new_crop(id, crop_xyxy, id_to_cls[id])
        if not (success):  #   memory full
            try:
                tracks[id_to_trkind[id]].reid_count = (tracks[id_to_trkind[id]].reid_count - 1) % reid_freq
            except:
                raise ValueError


def run_reid(ids_out_of_conflict, reid_memory_banks, id_to_idet, id_to_cls, dets, frame_img, reid_models):
    """
    This function runs the re-identification (reid) process for tracks that are out of conflict. It extracts new image crops
    for each track and retrieves any memory crops stored in the reid memory banks. For efficiency, crops for each class are
    concatenated into a list. The function then computes features for these crops using the appropriate reid models,
    taking into account the maximum batch size. The resulting features are returned for both the out-of-conflict tracks
    and any necessary updates to features extracted from memory crops.
    """

    all_out_of_conflict_crops = {}
    all_memory_crops = {}
    cls_list = []

    for id in ids_out_of_conflict:
        id_cls = id_to_cls[id]
        cls_list.append(id_cls)
        crop_xyxy = dets[id_to_idet[id]]
        all_out_of_conflict_crops[id] = frame_img[
            int(crop_xyxy[1]) : int(crop_xyxy[3]), int(crop_xyxy[0]) : int(crop_xyxy[2])
        ]
        if reid_memory_banks.is_id_in_memory(id):
            all_memory_crops[id] = reid_memory_banks.get_crops_release_id(id)
            cls_list += [id_cls * len(all_memory_crops[id])]
        for conflicted_id in ids_out_of_conflict[id]:
            if (conflicted_id not in all_memory_crops) and (conflicted_id not in ids_out_of_conflict):
                if reid_memory_banks.is_id_in_memory(conflicted_id):
                    all_memory_crops[conflicted_id] = reid_memory_banks.get_crops_release_id(conflicted_id)
                    cls_list += [id_cls * len(all_memory_crops[conflicted_id])]

    if len(all_out_of_conflict_crops) == 0 and len(all_memory_crops) == 0:
        return {}, {}

    #   find features for each crop
    out_of_conflict_features = {}
    memory_features_update = {}

    if len(all_out_of_conflict_crops) > 0:
        # get unique cls list
        unique_cls_list = list(set(cls_list))
        for cls in unique_cls_list:

            out_of_conflict_crops = {
                id: all_out_of_conflict_crops[id] for id in all_out_of_conflict_crops if id_to_cls[id] == cls
            }
            memory_crops = {id: all_memory_crops[id] for id in all_memory_crops if id_to_cls[id] == cls}

            if len(out_of_conflict_crops) == 0:
                continue

            #   convert a dict of lists to a big list
            memory_crops_list = (
                [item for sublist in memory_crops.values() for item in sublist] if len(memory_crops) > 0 else []
            )

            # Create a dict of keys with their corresponding indices in the big list
            memory_crops_indices = {}
            current_index = 0
            for key, sublist in memory_crops.items():
                memory_crops_indices[key] = list(range(current_index, current_index + len(sublist)))
                current_index += len(sublist)

            #   run reid
            crop_list = list(out_of_conflict_crops.values()) + list(memory_crops_list)
            batch = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crop_list]
            batch_features = reid_models[cls].forward_on_crop_list(batch)
            features = [feat["descriptor"] if isinstance(feat, dict) else feat for feat in batch_features]

            # reid_models['reid_used_count'] += len(features)
            # reid_models['batches_runs_count'] += b+1

            for id, feat in zip(out_of_conflict_crops.keys(), features[: len(out_of_conflict_crops)]):
                out_of_conflict_features[id] = feat
            for id, indexes in memory_crops_indices.items():
                features_memory = [features[len(out_of_conflict_crops) + ind] for ind in indexes]
                if len(features_memory) > 0:
                    # prev
                    vec = np.mean(features_memory, axis=0)
                    vec = vec / (np.linalg.norm(vec))

                    memory_features_update[id] = {"amount": len(features_memory), "vec": vec}
                else:
                    memory_features_update[id] = {"amount": 0, "vec": None}

    return out_of_conflict_features, memory_features_update


def conflict_solver(
    ids_out_of_conflict,
    new_features,
    old_features,
    maintain_id_cost_disscount,
    max_cost,
    id_to_cls=None,
    max_ass_cost_thresh_dict=None,
    build_include_self=False,
):
    """
    This function resolves conflicts between track IDs that have been identified as out of conflict. It uses a cost matrix to
    determine the optimal assignment of IDs based on feature similarity, applying a discount to maintain the same ID as the original tracker predicted.
    It also takes a maximal cost for assignment and, if the cost is not within the acceptable threshold, assigns a None value,
    which later translates to the creation of a new ID.
    """
    # Create a list to store all unique ids
    all_ids = set()
    for id, id_c_list in ids_out_of_conflict.items():
        all_ids.add(id)
        all_ids.update(id_c_list)

    # all_ids = sorted(all_ids)
    id_count = len(all_ids)

    # Initialize the cost matrix with max_cost
    cost_matrix = np.ones((id_count, id_count)) * max_cost
    used_reid_matrix = np.zeros((id_count, id_count), dtype=bool)

    # Create a mapping from ids to their row/column index in the cost matrix
    id_to_rowcol = {id: index for index, id in enumerate(all_ids)}
    rowcol_to_id = {index: id for index, id in enumerate(all_ids)}

    # Fill the cost matrix with the actual costs
    for id, idc_list in ids_out_of_conflict.items():
        old_features_matrix = np.array([old_features[id] for id in idc_list])
        sim = new_features[id] @ old_features_matrix.T
        cost = 1 - sim
        row = id_to_rowcol[id]
        for idc, cur_cost, cur_sim in zip(idc_list, cost, sim):
            col = id_to_rowcol[idc]
            if (id == idc) and (cur_sim == 0):  #   no reid vec available in the past, assign minimal cost
                cost_matrix[row, col] = 0.0
            else:
                cost_matrix[row, col] = cur_cost
            used_reid_matrix[row, col] = True

    #   give "disscount" to maintain same id
    for id in all_ids:
        row_ind = id_to_rowcol[id]
        col_ind = id_to_rowcol[id]
        if cost_matrix[row_ind, col_ind] == 1:
            cost_matrix[row_ind, col_ind] = cost_matrix[row_ind, col_ind] - 0.001
        else:
            cost_matrix[row_ind, col_ind] = cost_matrix[row_ind, col_ind] - maintain_id_cost_disscount

    #   use maintain_id_cost_disscount
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Update the existing dictionary with optimal assignments if ids are not the same
    if build_include_self:
        id_assignments_including_self = {}
    else:
        id_assignments_including_self = None

    id_assignments = {}
    for row, col in zip(row_ind, col_ind):
        row_id = rowcol_to_id[row]
        col_id = rowcol_to_id[col]
        cost = cost_matrix[row, col]

        max_ass_cost = max_ass_cost_thresh_dict[id_to_cls[row_id]]

        if (max_ass_cost_thresh_dict is not None) and (cost > max_ass_cost) and (used_reid_matrix[row, col]):
            new_id = None
        else:
            new_id = col_id

        if not (
            (row_id not in ids_out_of_conflict) and (new_id is None)
        ):  #   do not assign new track to one you did not intended to solve
            if build_include_self:
                id_assignments_including_self[row_id] = new_id
            if row_id != new_id:
                id_assignments[row_id] = new_id

    return id_assignments, id_assignments_including_self


def update_reid_vectors(trk_list, id_new_features, id_reassignments):
    """
    This is used after all the conflict solver is done.
    This function updates the re-identification (reid) vectors for a list of tracks using the current new reid vectors.
    It first reverses the ID assignments  to find which old ID corresponds to a new ID after reassignment.
    For each track, it checks if the track's ID should
    be reassigned, then updates the reid vector using the new features. If the track already has a reid vector, the new vector
    is averaged with the existing one, normalized by the number of reid runs.
    """

    reverse_id_reassignments = {value: key for key, value in id_reassignments.items()}

    for trk in trk_list:

        if trk.track_id in reverse_id_reassignments:
            id_to_use = reverse_id_reassignments[trk.track_id]
        else:
            id_to_use = trk.track_id

        if id_to_use in id_new_features:
            trk.reid_run_count += 1
            if trk.reid_vec is None:
                trk.reid_vec = id_new_features[id_to_use]
            else:
                trk.reid_vec = (
                    (trk.reid_vec * (trk.reid_run_count - 1)) + id_new_features[id_to_use]
                ) / trk.reid_run_count
                trk.reid_vec = trk.reid_vec / np.linalg.norm(trk.reid_vec)


def update_trks(trk_list, id_reassignments, ids_out_of_conflict):
    """
    This is used after all the conflict solver is done, to update the track ids and conflicts.
    This function updates the track IDs and conflicts for each track in the given list. It handles the reassignment of IDs
    based on a provided mapping (id_reassignments) and adjusts the conflict lists accordingly.
    """
    for trk in trk_list:
        if trk.track_id in id_reassignments:
            trk.track_id = id_reassignments[trk.track_id]

        #   if any id in conflict list is out for conflict, remove it from list
        trk.conflicts = [c for c in trk.conflicts if (c not in ids_out_of_conflict) or (c == trk.track_id)]
        if len(trk.conflicts) == 1 and (trk.conflicts[0] == trk.track_id):
            trk.conflicts = []

        #   handle conflicts too
        trk.conflicts = [c if (c not in id_reassignments) else id_reassignments[c] for c in trk.conflicts]


def build_cur_features_dict(trk_list, ids_out_of_conflict, id_new_features, cur_features_update):
    """
    This function builds a dictionary of current features (reid vectors before current frame) for a list of tracks. It updates the features based on
    re-identification (reid) runs from memory and handles any necessary adjustments for tracks that have been out of conflict.
    """

    cur_features = {trk.track_id: trk.reid_vec for trk in trk_list if trk.reid_vec is not None}

    #   handle dead ids
    for trk in trk_list:
        if trk.track_id in cur_features_update:
            if trk.track_id not in cur_features:
                cur_features[trk.track_id] = cur_features_update[trk.track_id]["vec"]
                trk.reid_run_count = cur_features_update[trk.track_id]["amount"]
            else:
                new_feature_sum = (
                    cur_features[trk.track_id] * trk.reid_run_count
                    + cur_features_update[trk.track_id]["vec"] * cur_features_update[trk.track_id]["amount"]
                )
                trk.reid_run_count += cur_features_update[trk.track_id]["amount"]
                cur_features[trk.track_id] = new_feature_sum / trk.reid_run_count

            cur_features[trk.track_id] = cur_features[trk.track_id] / np.linalg.norm(cur_features[trk.track_id])
            trk.reid_vec = cur_features[trk.track_id]

    #   handle dead ids
    for new_id, prev_conflicts in ids_out_of_conflict.items():
        for p in prev_conflicts:
            if p not in cur_features:
                cur_features[p] = 0 * id_new_features[new_id].copy()

    return cur_features


def filter_conflicts(conflicts, tracks, memory_bank):
    """
    This function filters out conflicts based on certain criteria related to track IDs. Specifically, it removes conflicts
    associated with track IDs that are not stored in memory, do not have a reid vector, or belong to classes that do not use reid.
    Previously, ids that were part of other conflicts were also removed, but this functionality has been commented out.
    """

    avoid_ids = []
    for trk in tracks:
        in_memory = memory_bank.is_id_in_memory(trk.track_id)
        reid_exists = trk.reid_vec is not None
        if not (in_memory or reid_exists) or not (trk.use_reid):
            avoid_ids.append(trk.track_id)

    for id in avoid_ids:
        if id in conflicts:
            del conflicts[id]

    # delete_keys = []
    # for id in conflicts:
    #     mask = [i for i, c in enumerate(conflicts[id]['ids']) if c not in avoid_ids]
    #     conflicts[id]['idets'] = [conflicts[id]['idets'][i] for i in mask]
    #     conflicts[id]['ids'] = [conflicts[id]['ids'][i] for i in mask]
    #     if len(conflicts[id]['ids']) <= 1:
    #         delete_keys.append(id)

    # for key in delete_keys:
    #     del conflicts[key]

    return conflicts


def update_size_filter(new_filter: Dict, factor: float) -> (Dict, Set):
    class_max_filter = {}
    filtered_subclasses = {int(c) for c in new_filter.keys()}
    for c in filtered_subclasses:
        class_max_filter[c] = np.array([new_filter[c]["h"], new_filter[c]["w"]]) * factor
    return class_max_filter, filtered_subclasses
