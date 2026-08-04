import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious

from tracking.byte_track.kalman_filter import chi2inv95

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

# def second_opt(a_in, b_in, ious):
#     ax, ay = a_in.copy(), a_in.copy()
#     bx, by = b_in.copy(), b_in.copy()

#     for a1 in ax:
#         w = a1[2] - a1[0]
#         a1[3] = a1[1] + w
#     for b1 in bx:
#         w = b1[2] - b1[0]
#         b1[3] = b1[1] + w
#     for a2 in ay:
#         h = a2[2] - a2[0]
#         a2[3] = a2[1] + h
#     for b2 in by:
#         h = b2[2] - b2[0]
#         b2[3] = b2[1] + h

#     ious_x = bbox_ious(
#         np.ascontiguousarray(ax, dtype=np.float),
#         np.ascontiguousarray(bx, dtype=np.float))

#     ious_y = bbox_ious(
#         np.ascontiguousarray(ay, dtype=np.float),
#         np.ascontiguousarray(by, dtype=np.float))

#     ious_new = (ious_x + ious_y)/2

#     return ious_new


def ious(atlbrs, btlbrs, use_ciou):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)

    if not(use_ciou):
        iou_func = bbox_ious
        max_iou = 1
    else:
        iou_func = ciou_efficient_bbox_ious_np
        max_iou = 2

    if ious.size == 0:
        return ious, max_iou

    ious = iou_func(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
        )
    
    return ious, max_iou


def iou_distance(atracks, btracks, use_ciou = False):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

        acls = [track.cls for track in atracks]
        bcls = [track.cls for track in btracks]

    
    _ious, max_cost = ious(atlbrs, btlbrs, use_ciou)
    cost_matrix = 1 - _ious
    cost_matrix = class_filter_cost_matrix(cost_matrix, acls, bcls, max_cost)

    return cost_matrix


def class_filter_cost_matrix(cost_matrix, acls, bcls, max_cost = 1.):
    '''
    maximal cost if classes are not shared
    '''
    if len(acls) == 0 or len(bcls) == 0:
        return cost_matrix
    
    acls_mat = np.expand_dims(acls, axis = 1).repeat(axis = 1, repeats = len(bcls))
    bcls_mat = np.expand_dims(bcls, axis = 0).repeat(axis = 0, repeats = len(acls))

    diff_cls_mask = (acls_mat != bcls_mat)
    cost_matrix[diff_cls_mask] = max_cost #   maximal cost for different class

    return cost_matrix
    
def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost



def ciou_efficient_bbox_ious_np(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = np.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = np.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    a = bboxes1.shape[0]
    b = bboxes2.shape[0]

    inter_max_xy = np.minimum(bboxes1[:, 2:].reshape(a, 1, 2).repeat(b, axis=1),
                              bboxes2[:, 2:].reshape(1, b, 2).repeat(a, axis=0))
    inter_min_xy = np.maximum(bboxes1[:, :2].reshape(a, 1, 2).repeat(b, axis=1),
                              bboxes2[:, :2].reshape(1, b, 2).repeat(a, axis=0))

    out_max_xy = np.maximum(bboxes1[:, 2:].reshape(a, 1, 2).repeat(b, axis=1),
                            bboxes2[:, 2:].reshape(1, b, 2).repeat(a, axis=0))
    out_min_xy = np.minimum(bboxes1[:, :2].reshape(a, 1, 2).repeat(b, axis=1),
                            bboxes2[:, :2].reshape(1, b, 2).repeat(a, axis=0))

    inter = np.clip((inter_max_xy - inter_min_xy), a_min=0, a_max=None)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_1 = np.stack([center_x1, center_y1], axis=1)
    center_2 = np.stack([center_x2, center_y2], axis=1)
    inter_diag = np.sqrt(np.sum((center_1[:, None, :] - center_2[None, :, :]) ** 2, axis=2))

    outer = np.clip(out_max_xy - out_min_xy, a_min=0, a_max=None)
    outer_diag = (outer[:,:, 0] * 2) + (outer[:,:, 1] * 2)


    union = area1[:, None] + area2[None, :] - inter_area
    u = inter_diag / outer_diag
    iou = inter_area / union


    v = (4 / (np.pi ** 2)) * np.power((np.arctan(w2 / h2)[None, :] - np.arctan(w1 / h1)[:, None]), 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        S = 1 - iou  # Ensure this matches with your PyTorch version
        alpha = v / (S + v)
        cious = iou - (u + alpha * v)  # Ensure this matches with your PyTorch version

    cious = np.clip(cious, a_min=-1.0, a_max=1.0)  # Clamping range corrected to match PyTorch version

    cious[iou == 1] =  1
    cious[iou == 0] = -1

    if np.isnan(cious).any():
        masker = np.isnan(cious)
        raise ValueError(cious[masker][0], iou[masker][0], S[masker][0], u[masker][0], v[masker][0], alpha[masker][0])
    # cious = np.clip(cious, a_5min=-100.0, a_max=1.0)  # Clamping range corrected to match PyTorch version

    if exchange:
        cious = cious.T
        iou = iou.T

    return cious