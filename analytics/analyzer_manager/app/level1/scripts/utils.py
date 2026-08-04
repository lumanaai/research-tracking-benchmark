import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxA[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def filter_pose_with_yolo(yolo_results, pose_results):

    #   Filter pose
    iou_tresh = 0.07
    
    for b in range(len(yolo_results)):  # each image in batch
        
        # leave only people
        yolo_indx = np.arange(len(yolo_results[b]))
        current_yolo_people = yolo_results[b][yolo_results[b][:, 5] == 0, :4].clone()   #   xyxy
        yolo_indx = yolo_indx[(yolo_results[b][:, 5].cpu() == 0).numpy()]   #   original yolo indxes

        pose_yolo_iou = np.zeros((len(pose_results[b]),len(current_yolo_people)))   #   pose-yolo iou matrix
        current_yolo_people[:, 2:4] -= current_yolo_people[:, 0:2]    #   xyxy to xywh
        valid_list = []
        for i, pose in enumerate(pose_results[b]):
            max_pose_iou = 0
            for j, current_yolo in enumerate(current_yolo_people):
                pose_bb = pose.bbox
                iou = bb_intersection_over_union(current_yolo, pose_bb)    # compare bounding boxes (xywh)
                if iou > iou_tresh:
                    max_pose_iou = max(max_pose_iou, iou)
                    pose_yolo_iou[i, j] = iou
            
            if max_pose_iou > iou_tresh:
                valid_list.append(pose_results[b][i])
        
        pose_results[b] = valid_list

        #   Assign pose to yolo
        pose_yolo_iou = pose_yolo_iou[pose_yolo_iou.sum(axis=1) > 0, :] #   leave only valid
        while pose_yolo_iou.sum() != 0:
            i, j = np.unravel_index(np.argmax(pose_yolo_iou, axis=None), pose_yolo_iou.shape)
            pose_yolo_iou[i, :] = 0
            pose_yolo_iou[:, j] = 0
            pose_results[b][i].yolo_index = yolo_indx[j]

    return pose_results
