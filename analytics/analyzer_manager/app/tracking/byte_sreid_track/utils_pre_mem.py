from cython_bbox import bbox_overlaps as bbox_ious
import numpy as np
from scipy.optimize import linear_sum_assignment
from .basetrack import TrackState


def find_conflicts(iou_matrix, areas, threshold_dict, row_objs, coloumn_objs, row_prev_conflicts_cols, col_prev_conflicts_rows):
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
                    was_in_conflict = ((i in row_prev_conflicts_cols) and (j in row_prev_conflicts_cols[i])) #or ((j in col_prev_conflicts_rows) and (i in col_prev_conflicts_rows[j]))

                    
                    cur_thresh = threshold_dict[row_objs[i]]['start'] if not was_in_conflict else threshold_dict[row_objs[i]]['continue']   #   conflict hysteresis
                    if maximal_overlap >= cur_thresh:
                        if i not in conflicts_dict:
                            conflicts_dict[i] = [i]
                        conflicts_dict[i].append(j)
    return conflicts_dict


def find_detection_conflicts(bboxes, objects, tracked_trks, lost_trks, removed_trks, conflict_iou_thresh_dict, prev_id_conflicts):
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
    
    tracked_bboxes = []
    tracked_objects = []
    tracked_ids = []
    for trk in tracked_trks:
        tracked_bboxes.append(bboxes[trk.det_id])
        tracked_objects.append(objects[trk.det_id])
        # tracked_bboxes.append(trk.tlbr)
        tracked_ids.append(trk.track_id)

    trcked_dets = np.stack(tracked_bboxes)
    trcked_objs = np.stack(tracked_objects)
    trcked_dets = np.ascontiguousarray(trcked_dets, dtype=float)  #   (n, 4)
    idet_to_id, id_to_idet = translate_idet_to_id(tracked_trks)
    id_to_cls = get_id_to_cls(tracked_trks, lost_trks, removed_trks)


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
    
    row_ind_to_id_dict, coloumn_ind_to_id_dict = build_ind_to_id_dict(tracked_ids, lost_ids, len(trcked_dets), len(compare_dets))

    #     
    row_prev_conflicts_cols = {}
    for row in row_ind_to_id_dict:
        row_id = row_ind_to_id_dict[row]['id']
        if row_id in prev_id_conflicts:
            row_conflicts_in_ids = prev_id_conflicts[row_id]
            row_prev_conflicts_cols[row] = []
            for col in coloumn_ind_to_id_dict:
                if coloumn_ind_to_id_dict[col]['id'] in row_conflicts_in_ids:
                    row_prev_conflicts_cols[row].append(col)

    # now for col to rows
    col_prev_conflicts_rows = {}
    for col in coloumn_ind_to_id_dict:
        col_id = coloumn_ind_to_id_dict[col]['id']
        if col_id in prev_id_conflicts:
            col_conflicts_in_ids = prev_id_conflicts[col_id]
            col_prev_conflicts_rows[col] = []
            for row in row_ind_to_id_dict:
                if row_ind_to_id_dict[row]['id'] in col_conflicts_in_ids:
                    col_prev_conflicts_rows[col].append(row)


    iou_distances = bbox_ious(trcked_dets, compare_dets)  # Reimplement to avoid double calculation
    areas = (compare_dets[:, 2] - compare_dets[:, 0]) * (compare_dets[:, 3] - compare_dets[:, 1])
    conflicts_dict = find_conflicts(iou_distances, areas, conflict_iou_thresh_dict, trcked_objs, compare_objs, row_prev_conflicts_cols, col_prev_conflicts_rows)    #   detection to detection

    ids_conflicts = translate_conflicts(conflicts_dict, row_ind_to_id_dict, coloumn_ind_to_id_dict)  #   transform to ids

    return ids_conflicts, id_to_idet, id_to_cls



def build_ind_to_id_dict(tracked_ids, lost_ids, row_count, coloumn_count):
    row_ind_to_id_dict = {}
    coloumn_ind_to_id_dict = {}
    for i in range(coloumn_count):
        if i < row_count: #   live tracks
            row_ind_to_id_dict[i] = {
                'ind' : i,
                'id' : tracked_ids[i]}
            coloumn_ind_to_id_dict[i] = row_ind_to_id_dict[i]
        else:   #   lost tracks
            lost_ind = i - row_count  
            coloumn_ind_to_id_dict[i] = {
                'ind' : None,
                'id' : lost_ids[lost_ind]}
    return row_ind_to_id_dict, coloumn_ind_to_id_dict

# def split_conflicts_dict(conflicts_dict, max_dict_num):
#     dict_1 = {}
#     dict_2 = {}
#     for det, conflict_dets in conflicts_dict.items():
#         dict_1[det] = [c_id for c_id in conflict_dets if c_id <  max_dict_num ]
#         dict_2[det] = [c_id-max_dict_num for c_id in conflict_dets if c_id >= max_dict_num ]
#     return dict_1, dict_2


# def add_lost_ids(ids_conflicts, detection_conflicts_dict_lost, idet_to_id, lost_ids):
#     for idet, irow_lost_conflicts in detection_conflicts_dict_lost.items():
#         if (idet in idet_to_id) and len(irow_lost_conflicts):
#             id = idet_to_id[idet]
#             new_id_conflicts = [lost_ids[i] for i in irow_lost_conflicts]
#             if id not in ids_conflicts:
#                 ids_conflicts[id] = {
#                 'ids' : [id],
#                 'idets' : [idet],
#                 }

#             ids_conflicts[id]['ids'].extend(new_id_conflicts)
#             ids_conflicts[id]['idets'].extend([None] * len(new_id_conflicts))   #   because not detected on this frame
#     return ids_conflicts


def translate_idet_to_id(tracks):
    idet_to_id = {}
    id_to_idet = {}
    for trk in tracks:
        idet_to_id[trk.det_id] = trk.track_id

    for v, k in idet_to_id.items():
        id_to_idet[k] = v
    return idet_to_id, id_to_idet

def get_id_to_cls(*track_lists):
    id_to_cls = {}
    for tracks in track_lists:
        for trk in tracks:
            id_to_cls[trk.track_id] = trk.cls
    return id_to_cls

def translate_conflicts(detection_conflicts, source_dict, target_dict):

    id_conflicts = {}
    for ind_source, idets_conflicts in detection_conflicts.items():
        
        #   The detection was not tracked
        if ind_source not in source_dict:
            continue
        
        id_source = source_dict[ind_source]['id']
        id_conflicts[id_source] = {
                'ids' : [target_dict[i]['id'] for i in idets_conflicts],
                'idets' : [target_dict[i]['ind'] for i in idets_conflicts],
                }


        
        # for idet_conflict in idets_conflicts:
        #     if idet_conflict in idet_to_id:
        #         if idet_conflict in idet_to_id:
        #             id_conflicts[idet_to_id[idet_source]]['ids'].append(idet_to_id[idet_conflict])
        #         else:
        #             id_conflicts[idet_to_id[idet_source]]['ids'].append(-1)

    return id_conflicts
        
def build_ids_state(trks):
    ids_state = {}
    for trk in trks:
        ids_state[trk.track_id] = trk.state
    return ids_state

def update_id_conflicts(tracked_tracks, ids_conflicts, lost_tracks, removed_tracks):
    out_of_conflict_ids = {}

    ids_state = build_ids_state(tracked_tracks + lost_tracks + removed_tracks)

    for trk in tracked_tracks:
        if len(trk.conflicts) == 1 and (trk.conflicts[0] != trk.track_id):
            raise ValueError("how come 1?")
        
        if len(trk.conflicts) > 1:
            if trk.track_id not in ids_conflicts:
                # out_of_conflict_ids[trk.track_id] = trk.conflicts
                if not(trk.reid_vec is None):
                    out_of_conflict_ids[trk.track_id] = [id for id in trk.conflicts if id in ids_state] #and ids_state[id] != TrackState.Removed]
                    out_of_conflict_ids[trk.track_id].sort()

        if trk.track_id in ids_conflicts:
            trk.conflicts = trk.conflicts + [id for id in ids_conflicts[trk.track_id]['ids'] if (id >=0) and (id not in trk.conflicts)] #   append to current conflicts if they are alive or lost
            # trk.conflicts = [id for id in ids_conflicts[trk.track_id]['ids'] if id >=0 ]
        else:
            trk.conflicts = []
        
        if len(trk.conflicts) == 1 and (trk.conflicts[0] != trk.track_id):
            raise ValueError("how come 1?")
    
    #   remove ids from out_of_conflict_ids if their values list length is lower than 1
    out_of_conflict_ids = {k: v for k, v in out_of_conflict_ids.items() if len(v) > 0}

    return out_of_conflict_ids


def is_track_outside_image_bounds(track_tlbr, img_h, img_w):
    # Check if all coordinates are completely outside the image bounds (without margins)
    left, top, right, bottom = track_tlbr
    return right > img_w or left < 0 or top > img_h or bottom < 0
    

def is_track_outside_image_bounds_2(track_tlbr, img_h, img_w):
    # Check if all coordinates are completely outside the image bounds (without margins)
    new_parameter = 0.05

    # extrach h and w of tlbr (xyxy)
    h_box = track_tlbr[3] - track_tlbr[1]
    w_box = track_tlbr[2] - track_tlbr[0]

    assert h_box > 0 and w_box > 0

    left, top, right, bottom = track_tlbr
    delta_w = w_box * new_parameter
    delta_h = h_box * new_parameter

    return right > img_w + delta_w or left < -delta_w or top > img_h + delta_h or bottom < -delta_h

# def is_track_too_smaller(track, min_area_smaller_ratio=0.05):
#     # Check if all coordinates are completely outside the image bounds (without margins)
#     left, top, right, bottom = track.tlbr
#     cur_area = (right - left) * (bottom - top)
#     if cur_area / track.max_area < min_area_smaller_ratio:
#         return True
#     return False


def filter_out_of_conflict(ids_out_of_conflict, ids_list, img_h, img_w):

    filter_list = []
    for id in ids_out_of_conflict:
        cur_trk = None
        for trk in ids_list:
            if trk.track_id == id:
                cur_trk = trk
                break

        tlbr = cur_trk.tlbr
        outside_bounds = is_track_outside_image_bounds_2(tlbr, img_h, img_w)
        if outside_bounds:
            filter_list.append(id)
    
    # perform filtering
    ids_out_of_conflict = {k: v for k, v in ids_out_of_conflict.items() if k not in filter_list}
    
    return ids_out_of_conflict

    

def update_reid_count(tracked_stracks, lost_stracks, ids_conflicts, reid_freq, ids_out_of_conflict):
    reids_to_run = []

    for trk in tracked_stracks:            
        if trk.reid_count == 0:
            if (trk.track_id not in ids_conflicts) or (trk.reid_vec is None):
                if trk.is_activated:
                    reids_to_run.append(trk.track_id)

    for trk in tracked_stracks:
        if (trk.track_id not in ids_conflicts) or (trk.reid_count != 0):
            if trk.is_activated:
                trk.reid_count = (trk.reid_count + 1) % reid_freq

    for trk in lost_stracks:
        if trk.reid_count != 0:
            trk.reid_count = (trk.reid_count + 1) % reid_freq

    reids_to_run = [id for id in reids_to_run if id not in ids_out_of_conflict] #   dont run again for ids that are checked because out of conflicts
    for trk in tracked_stracks:            
        if trk.track_id in ids_out_of_conflict:
            trk.reid_count = 1  #   the reid is running anyway

    return reids_to_run


def run_reid(ids_out_of_conflict, ids_run_reid, id_to_idet, id_to_cls, dets, frame_img, reid_models):
    image_crops_people = {}
    image_crops_vehicles = {}

    for id in ids_out_of_conflict:
        if id_to_cls[id] == 0:    #   person
            crop_xyxy = dets[id_to_idet[id]]
            image_crops_people[id] = frame_img[int(crop_xyxy[1]):int(crop_xyxy[3]), int(crop_xyxy[0]):int(crop_xyxy[2])]
        elif id_to_cls[id] == 1:    #   vehicle
            crop_xyxy = dets[id_to_idet[id]]
            image_crops_vehicles[id] = frame_img[int(crop_xyxy[1]):int(crop_xyxy[3]), int(crop_xyxy[0]):int(crop_xyxy[2])]


    for id in ids_run_reid:
        crop_xyxy = dets[id_to_idet[id]]

        if id in ids_out_of_conflict:
            raise ValueError

        if id_to_cls[id] == 0:    #   person
            image_crops_people[id] = frame_img[int(crop_xyxy[1]):int(crop_xyxy[3]), int(crop_xyxy[0]):int(crop_xyxy[2])]
        elif id_to_cls[id] == 1:    #   vehicle
            image_crops_vehicles[id] = frame_img[int(crop_xyxy[1]):int(crop_xyxy[3]), int(crop_xyxy[0]):int(crop_xyxy[2])]
                                                 
    if len(image_crops_people) == 0 and len(image_crops_vehicles) == 0:
        return {}, {}


    #   find features for each crop
    feature_dict = {}

    if len(image_crops_people) > 0:
        features_1 = reid_models['person'].forward_on_crop_list(list(image_crops_people.values()))  # run reid
        reid_models['reid_used_count'] += features_1.shape[0]
        for id, feat in zip(image_crops_people.keys(), features_1):
            feature_dict[id] = feat['descriptor'] if isinstance(feat, dict) else feat

    if len(image_crops_vehicles) > 0:
        features_2 = reid_models['vehicle'].forward_on_crop_list(list(image_crops_vehicles.values()))  # run reid
        reid_models['reid_used_count'] += features_2.shape[0]
        for id, feat in zip(image_crops_vehicles.keys(), features_2):
            feature_dict[id] = feat['descriptor'] if isinstance(feat, dict) else feat
            
    return feature_dict




def conflict_solver(ids_out_of_conflict, new_features, old_features, maintain_id_cost_disscount, max_cost, id_to_cls = None, max_ass_cost_thresh_dict = None):
    # Create a list to store all unique ids
    all_ids = set()
    for id, id_c_list in ids_out_of_conflict.items():
        all_ids.add(id)
        all_ids.update(id_c_list)

    # all_ids = sorted(all_ids)
    id_count = len(all_ids)
    
    # Initialize the cost matrix with max_cost
    cost_matrix = np.ones((id_count, id_count)) * max_cost
    used_reid_matrix = np.zeros((id_count, id_count), dtype = np.bool)

    
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
                cost_matrix[row, col] = 0.
                if cur_sim == 0:
                    raise ValueError
            else:
                cost_matrix[row, col] = cur_cost
            used_reid_matrix[row, col] = True

    #   give "disscount" to maintain same id
    for id in all_ids:
        row_ind = id_to_rowcol[id]
        col_ind = id_to_rowcol[id]
        cost_matrix[row_ind, col_ind] = cost_matrix[row_ind, col_ind] - maintain_id_cost_disscount
    
    #   use maintain_id_cost_disscount
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Update the existing dictionary with optimal assignments if ids are not the same
    id_assignments_including_self = {}
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


        if not((row_id not in ids_out_of_conflict) and (new_id is None)):   #   do not assign new track to one you did not intended to solve
            id_assignments_including_self[row_id] = new_id
            if row_id != new_id:
                id_assignments[row_id] = new_id

    return id_assignments, id_assignments_including_self




def update_reid_vectors(trk_list, id_new_features, id_reassignments, reid_running_average_lambda):

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
                trk.reid_vec = ((trk.reid_vec * (trk.reid_run_count - 1)) + id_new_features[id_to_use]) / trk.reid_run_count
                
                

def update_trks(trk_list, id_reassignments, ids_out_of_conflict):
    for trk in trk_list:
        if trk.track_id in id_reassignments:
            trk.track_id = id_reassignments[trk.track_id]
            
        #   handle conflicts too
        trk.conflicts = [c if (c not in id_reassignments ) else id_reassignments[c] for c in trk.conflicts]

        #   if any id in conflict list is out for conflict, remove it from list
        trk.conflicts = [c for c in trk.conflicts if (c not in ids_out_of_conflict) or (c == trk.track_id)]
        if len(trk.conflicts) == 1 and (trk.conflicts[0] == trk.track_id):
            trk.conflicts = []



def build_old_features_dict(trk_list, ids_out_of_conflict, id_new_features):
    old_features = {trk.track_id: trk.reid_vec for trk in trk_list if trk.reid_vec is not None}

    #   handle dead ids
    for new_id, prev_conflicts in ids_out_of_conflict.items():
            for p in prev_conflicts:
                if p not in old_features:
                    try:
                        old_features[p] = 0 * id_new_features[new_id].copy()
                    except:
                        id_new_features[new_id] = 0

    return old_features
