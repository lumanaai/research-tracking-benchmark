import csv
import glob
import os
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from math import floor, ceil
from shutil import copy2
from typing import Set, List, Dict

import cv2
import numpy as np
import singlestoredb as s2
from pathlib import PurePosixPath, Path
import concurrent.futures

from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from general.core import ClassHandler
from general.img_utils import motion_score, is_image_monochrome
from level1.face.face_analyzer import FaceAnalyzer, conf_order_map, gen_face_mask
from level1.face.face_utils import FaceConfidence
from level1.face.matchings import FaceRecord, conf2cosine_th, GroupRecord, conf2factor
from utils.infra.s3_io import S3Handler

from itertools import groupby
from operator import itemgetter

base_path = "/mnt/c/Temp/faces"
base_db_path = "/mnt/c/Temp/face_db"
index_file_name = "index.csv"


def download_faces_groups():
    conn = s2.connect(
        host="svc-a519c79a-48fd-4531-9e71-5220fc82e61c-dml.gcp-iowa-1.svc.singlestore.com:3306",
        port=3360,
        user="grafana-readonly",
        password="DAI4VhetILRgc6CFXY5Dix9O7VbWw9SN",
        database="lumix_production",
        results_type="dicts",
    )

    query_g = 'SELECT edgeId, cameraId, zoomImage, groupId from lumix_production.trackers WHERE groupId <> ""  AND zoomImage IS NOT NULL AND orgIdHash = 1322041909 AND timestamp > 1704018227 ORDER BY groupId'
    query_low = 'SELECT edgeId, cameraId, zoomImage, groupId from lumix_production.trackers WHERE groupId IS NULL AND zoomImage IS NOT NULL AND orgIdHash = 1322041909 AND timestamp > 1704018227  AND faceConfidence = "low" '
    s3 = S3Handler()
    with conn.cursor() as cur:
        cur.execute(query_g)
        groups = cur.fetchall()

        cur.execute(query_low)
        lows = cur.fetchall()

    print("downloading groups")
    for tracker in tqdm(groups):
        try:
            s3_path = PurePosixPath("crop").joinpath(tracker["edgeId"], tracker["cameraId"], tracker["zoomImage"])
            local_dir = os.path.join(base_path, str(tracker["groupId"]))
            os.makedirs(local_dir, exist_ok=True)
            s3.download_single_file_s3(str(s3_path), os.path.join(local_dir, tracker["zoomImage"]))
        except Exception as e:
            print(f"could not download {tracker['zoomImage']}: {e}")

    local_low_dir = os.path.join(base_path, "low_conf")
    os.makedirs(local_low_dir, exist_ok=True)
    print("downloading low confidence")
    for tracker in tqdm(lows):
        try:
            s3_path = PurePosixPath("crop").joinpath(tracker["edgeId"], tracker["cameraId"], tracker["zoomImage"])
            s3.download_single_file_s3(str(s3_path), os.path.join(local_low_dir, tracker["zoomImage"]))
        except Exception as e:
            print(f"could not download {tracker['zoomImage']}: {e}")


def copy_to_single_folder():
    target_folder = os.path.join(base_path, "all")
    os.makedirs(target_folder, exist_ok=True)
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if dir != "all":
                files = glob.glob(os.path.join(root, dir, "*.*"))
                for file in files:
                    copy2(file, os.path.join(target_folder, f"{dir}_{Path(file).name}"))


def gather_face_stats():
    import matplotlib.pyplot as plt

    detector = FaceDetectorFactory.create("RetinaFace", {})
    all_landmarks = []
    all_bbox = []
    out_folder = os.path.join(base_path, "stats")
    os.makedirs(out_folder, exist_ok=True)
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            dir_images = []
            if dir != "all":
                files = glob.glob(os.path.join(root, dir, "*.*"))
                for file in files:
                    image = cv2.imread(file)
                    dir_images.append(image)
                bbox, landmarks, confidence = detector.forward_on_crop_list(dir_images)
                for idx, landmk in enumerate(landmarks):
                    if landmk is not None:
                        eyes_dist = int(round((landmk[1, 0] - landmk[0, 0]) / 5) * 5)
                        copy2(files[idx], os.path.join(out_folder, f"{eyes_dist}_{Path(files[idx]).name}"))
                all_bbox += bbox
                all_landmarks += landmarks
    arr = np.array([x[1, 0] - x[0, 0] for x in all_landmarks if x is not None])

    min_val = np.min(arr)
    max_val = np.max(arr)
    average_val = np.mean(arr)

    print(f"Minimum: {min_val}, Maximum: {max_val}, Average: {average_val}")

    # Create a histogram
    plt.hist(arr, bins=30, edgecolor="black")
    plt.title("Histogram of Array Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Save the histogram
    plt.savefig(os.path.join(base_path, "stat.png"))

    plt.show()


def download_all_faces():
    conn = s2.connect(
        host="svc-a519c79a-48fd-4531-9e71-5220fc82e61c-dml.gcp-iowa-1.svc.singlestore.com:3306",
        port=3360,
        user="grafana-readonly",
        password="DAI4VhetILRgc6CFXY5Dix9O7VbWw9SN",
        database="lumix_production",
        results_type="dicts",
    )

    query = """
    SELECT edgeId, cameraId, zoomImage, orgId, groupId, faceIdBin, faceConfidence from trackers
    WHERE zoomImage IS NOT NULL AND timestamp/1000 > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 90 DAY) AND trackerTypeId = 1
    ORDER BY orgId ASC, timestamp ASC
    """

    with conn.cursor() as cur:
        cur.execute(query)
        row_list = cur.fetchall()

    grouped = groupby(row_list, key=itemgetter("orgId"))

    s3 = S3Handler()

    for orgId, faces in tqdm(grouped, desc="Downloading Org"):
        local_dir = os.path.join(base_db_path, orgId)
        os.makedirs(local_dir, exist_ok=True)
        face_list = list(faces)

        with open(os.path.join(local_dir, "index.csv"), mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=face_list[0].keys())
            writer.writeheader()
            for row in face_list:
                writer.writerow(row)

        def get_file_label_by_index(index):
            try:
                tracker = face_list[index]
                s3_path = PurePosixPath("crop").joinpath(tracker["edgeId"], tracker["cameraId"], tracker["zoomImage"])
                target_path = os.path.join(local_dir, tracker["zoomImage"])
                if not os.path.exists(target_path):
                    s3.download_single_file_s3(str(s3_path), target_path)

                return 1
            except Exception as e:
                print(f"An error occur trying to download {face_list[index]['zoomImage']}: {e}")
                return 0

        file_iter = range(len(face_list))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(get_file_label_by_index, file_iter),
                    total=len(file_iter),
                    desc="Downloading faces",
                    leave=False,
                )
            )


def classify_faces():
    analyzer = FaceAnalyzer(
        {"detector": "retinaface", "recognizer": "sface"}, context=Namespace(class_handler=ClassHandler())
    )
    detector = analyzer.detector

    out_folder = os.path.join(base_path, "classification", "v2")
    os.makedirs(out_folder, exist_ok=True)
    for conf in FaceConfidence:
        os.makedirs(os.path.join(out_folder, str(conf)), exist_ok=True)

    for root, dirs, files in os.walk(base_db_path):
        for org in dirs:
            files = glob.glob(os.path.join(root, org, "*.jpg"))
            for file in files:
                try:
                    image = cv2.imread(file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bbox, landmarks, confidence = detector.forward_on_crop_list([image])
                    if landmarks[0] is None:
                        continue
                    face_class = analyzer.classify_face(image, landmarks[0], bbox[0], confidence[0])
                    ann = draw_detection_on_image(image, bbox[0], landmarks[0])
                    cv2.imwrite(
                        os.path.join(out_folder, f"{face_class.value}_{Path(file).name}"),
                        # os.path.join(out_folder, str(face_class), Path(file).name),
                        cv2.cvtColor(ann, cv2.COLOR_RGB2BGR),
                    )

                    # copy2(file, os.path.join(out_folder, str(face_class), Path(file).name))
                except Exception:
                    pass


def laplacian_score(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def laplacian_score_masked(image, mask):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    return np.nanvar(lap[mask == 1])



def face_quality_estimator(image, mask):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    return np.nanvar(lap[mask == 1]), np.std(image[mask == 1])


def motion_score_masked(image_crop, mask, low_pctl=10, low_th=30, min_grad_len=250) -> float:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)

    # blur speckle noise
    image = cv2.medianBlur(gray, 5)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grads = np.abs(sobelx) * mask
    min_grad_len = min(np.sum(mask) / 10, min_grad_len)
    grads_arr = grads[grads > low_th]
    if len(grads_arr) > min_grad_len:
        low_grad = np.percentile(grads_arr, low_pctl)
        score = np.count_nonzero(grads_arr > low_grad * 2) / len(grads_arr)
    else:
        score = 0
    return score


def bluriness_test():
    analyzer = FaceAnalyzer(
        {"detector": "retinaface", "recognizer": "sface"}, context=Namespace(class_handler=ClassHandler())
    )
    detector = analyzer.detector
    recognizer = analyzer.recognizer
    # lcan_out_folder = os.path.join(base_path, "laplace_canonized")
    # lcrop_out_folder = os.path.join(base_path, "laplace_crop")
    # lm_out_folder = os.path.join(base_path, "laplace_mask")
    # b_out_folder = os.path.join(base_path, "blur_full")
    # bc_out_folder = os.path.join(base_path, "blur_crop")
    # bm_out_folder = os.path.join(base_path, "blur_mask")
    out_folder = os.path.join(base_path, "contrast")
    # os.makedirs(lcan_out_folder, exist_ok=True)
    # os.makedirs(lcrop_out_folder, exist_ok=True)
    # os.makedirs(b_out_folder, exist_ok=True)
    # os.makedirs(bc_out_folder, exist_ok=True)
    # os.makedirs(lm_out_folder, exist_ok=True)
    # os.makedirs(bm_out_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    for root, dirs, files in os.walk(base_db_path):
        for org in dirs:
            files = glob.glob(os.path.join(root, org, "*.jpg"))
            for file in files:
                try:
                    image = cv2.imread(file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bbox, landmarks, confidence = detector.forward_on_crop_list([image])
                    if landmarks[0] is None:
                        continue
                    rec_crop = recognizer.canonize_face(image, landmarks[0])
                    face_crop = image[floor(bbox[0][0]) : ceil(bbox[0][2]), floor(bbox[0][1]) : ceil(bbox[0][3])]
                    mask = gen_face_mask(image, bbox[0], landmarks[0])

                    # blur = motion_score(image)
                    # blur_cr = motion_score(face_crop, min_grad_len = 200, crop_sz=1)
                    # lap_cr = laplacian_score(image[floor(bbox[0][0]):ceil(bbox[0][2]), floor(bbox[0][1]): ceil(bbox[0][3])])
                    # lap_can = laplacian_score(rec_crop)

                    # copy2(file, os.path.join(b_out_folder, f"{blur:.2f}_{Path(file).name}"))
                    # copy2(file, os.path.join(bc_out_folder, f"{blur_cr:.2f}_{Path(file).name}"))
                    # copy2(file, os.path.join(lcrop_out_folder, f"{lap_cr:.2f}_{Path(file).name}"))
                    # copy2(file, os.path.join(lcan_out_folder, f"{lap_can:.2f}_{Path(file).name}"))

                    _, contrast = face_quality_estimator(image, mask)
                    # lap_m = laplacian_score_masked(image, mask)

                    # copy2(file, os.path.join(bm_out_folder, f"{blur_m:.2f}_{Path(file).name}"))
                    # copy2(file, os.path.join(lm_out_folder, f"{lap_m:.2f}_{Path(file).name}"))

                    copy2(file, os.path.join(out_folder, f"{contrast:.2f}_{Path(file).name}"))

                except Exception as e:
                    print(e)


def draw_detection_on_image(image, bbox, landmarks):
    if bbox is None:
        return image
    pos = np.asarray(bbox).astype(int)
    image = cv2.rectangle(image, tuple(pos[0:2]), tuple(pos[2:4]), (255, 0, 0), 2)
    for lm in landmarks:
        image = cv2.circle(image, np.asarray(lm).astype(int), 2, (0, 255, 0), -1)
    return image


def check_detections():
    analyzer = FaceAnalyzer(
        {"detector": "retinaface", "recognizer": "sface"}, context=Namespace(class_handler=ClassHandler())
    )
    detector = analyzer.detector
    out_folder = os.path.join(base_path, "detection")
    os.makedirs(out_folder, exist_ok=True)
    for root, dirs, files in os.walk(base_db_path):
        for org in dirs:
            files = glob.glob(os.path.join(root, org, "*.jpg"))
            for file in files:
                try:
                    image = cv2.imread(file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bbox, landmarks, confidence = detector.forward_on_crop_list([image])
                    if landmarks[0] is None:
                        continue
                    mask = gen_face_mask(image, bbox[0], landmarks[0])
                    lap = laplacian_score_masked(image, mask)
                    ann = draw_detection_on_image(image, bbox[0], landmarks[0])
                    cv2.imwrite(os.path.join(out_folder, Path(file).name), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

                except Exception as e:
                    print(e)


def read_index_file(org_dir):
    index_path = os.path.join(base_db_path, org_dir, index_file_name)
    list_of_dicts = []

    with open(index_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            list_of_dicts.append(row)
    return list_of_dicts


def classify_face_new(self, image, landmarks, bbox, det_conf):
    yaw_levels: Dict = {"high": 15, "medium": 20, "low": 30}
    pitch_levels: Dict = {"high": 7.5, "medium": 15, "low": 25}
    roll_levels: Dict = {"high": 20, "medium": 30, "low": 40}

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[0:5]
    eye_dist = np.linalg.norm(np.array(left_eye - right_eye), 2)
    mask = gen_face_mask(image, landmarks, bbox, self.args.dilation_factor)
    sharpness, contrast = face_quality_estimator(image, mask)
    is_ir = is_image_monochrome(image)
    criteria = [0] * 8
    yaw, pitch, roll = calc_ang_projection_rot(landmarks)

    def get_score_from_range_ge(value, range_dict):
        if value >= range_dict["high"]:
            return 3
        elif value >= range_dict["medium"]:
            return 2
        elif value >= range_dict["low"]:
            return 1
        return 0

    def get_score_from_range_ang(value, range_dict):
        if value <= range_dict["high"]:
            return 3
        elif value <= range_dict["medium"]:
            return 2
        elif value <= range_dict["low"]:
            return 1
        return 0

    criteria[0] = get_score_from_range_ge(eye_dist, self.args.eye_dist_levels)
    criteria[1] = get_score_from_range_ge(det_conf, self.args.conf_threshold)
    criteria[2] = get_score_from_range_ge(contrast, self.args.contrast_levels)
    criteria[3] = get_score_from_range_ge(sharpness, self.args.sharpness_levels)
    criteria[4] = get_score_from_range_ang(abs(yaw), self.args.yaw_levels)
    criteria[5] = get_score_from_range_ang(pitch, self.args.pitch_levels)
    criteria[6] = get_score_from_range_ang(abs(roll), self.args.roll_levels)
    criteria[7] = 1 if is_ir else 3

    conf = np.min(np.array(criteria))
    values = [FaceConfidence.NONE, FaceConfidence.LOW, FaceConfidence.MEDIUM, FaceConfidence.HIGH]
    return values[conf]


def calc_ang_projection(landmarks):
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[0:5]
    eyes_axis = right_eye - left_eye
    mount_axis = right_mouth - left_mouth

    def calc_proj_ratio(A, B, C):
        AB = B - A
        AC = C - A
        dot_product = np.dot(AB, AC)
        length_square = np.dot(AB, AB)

        # Calculate the projection scalar
        t = dot_product / length_square
        return t

    eyes_t = calc_proj_ratio(left_eye, right_eye, nose)
    mouth_t = calc_proj_ratio(left_mouth, right_mouth, nose)
    yaw = ((eyes_t + mouth_t) / 2 - 0.5) * 45

    left_t = calc_proj_ratio(left_eye, left_mouth, nose)
    right_t = calc_proj_ratio(right_eye, right_mouth, nose)
    pitch = ((left_t + right_t) / 2 - 0.5) * 45
    return yaw, pitch


def calc_ang_projection_rot(landmarks):
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[0:5]

    # Calculate the angle theta
    roll = np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])

    # Define the rotation matrix for -theta
    R = np.array([[np.cos(-roll), -np.sin(-roll)], [np.sin(-roll), np.cos(-roll)]])

    # Function to rotate a point
    def translate_point(point):
        translated_point = point - right_eye
        rotated_point = R.dot(translated_point)
        return rotated_point

    left_eye_r = translate_point(left_eye)
    right_mouth_r = translate_point(right_mouth)
    left_mouth_r = translate_point(left_mouth)
    nose_r = translate_point(nose)

    eyes_t = nose_r[0] / left_eye_r[0]
    mouth_t = (right_mouth_r[0] - nose_r[0]) / (right_mouth_r[0] - left_mouth_r[0])
    yaw = ((eyes_t + mouth_t) / 2 - 0.5) * 45

    left_t = nose_r[1] / left_mouth_r[1]
    right_t = nose_r[1] / right_mouth_r[1]
    pitch_d = 0.6 - (left_t + right_t) / 2
    # pitch = np.arcsin(pitch_d) * 180 / np.pi
    # pitch = np.sign(pitch_d) * (pitch_d / 0.1) ** 2 / 10 * 45
    pitch = pitch_d * 45
    roll_deg = np.degrees(np.pi - roll)
    return yaw, pitch, roll_deg if roll_deg < 180 else roll_deg - 360


def replay_pipe():
    similarity_threshold = 0.45

    analyzer = FaceAnalyzer(
        {"detector": "retinaface", "recognizer": "sface"}, context=Namespace(class_handler=ClassHandler())
    )
    detector = analyzer.detector
    recognizer = analyzer.recognizer
    out_folder = os.path.join(base_path, "replay", "v2")
    for root, dirs, files in os.walk(base_db_path):
        for org in dirs:
            records = read_index_file(org)

            org_dir = os.path.join(out_folder, org)
            os.makedirs(org_dir, exist_ok=True)
            org_groups = {}
            org_db = {}
            group_idx = 0
            org_mat = np.zeros((len(records), 128), dtype=float)
            for idx, record in enumerate(records):
                file = os.path.join(root, org, record["zoomImage"])
                if os.path.exists(file):
                    image = cv2.imread(file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bbox, landmarks, confidence = detector.forward_on_crop_list([image.copy()])
                    if landmarks[0] is None:
                        # copy2(file, os.path.join(org_dir, "undetected_" + record["zoomImage"]))
                        org_db[idx] = FaceRecord(idx, -1, FaceConfidence.NONE, is_detected=False)
                        continue
                    face_class = analyzer.classify_face(image, landmarks[0], bbox[0], confidence[0])
                    rec_crop = recognizer.canonize_face(image, landmarks[0])
                    features = np.array(recognizer.forward_on_crop_list([rec_crop])[0])
                    org_mat[idx, :] = features
                    group_num = -1
                    if idx > 0:
                        similarity_score = np.matmul(org_mat[:idx], features.T)
                        similarity_argmax = np.argmax(similarity_score)
                        similarity_max_score = similarity_score[similarity_argmax]
                        if similarity_max_score >= conf2cosine_th[face_class]:
                            group_num = org_db[similarity_argmax].group_id
                    if group_num < 0:
                        if face_class in [FaceConfidence.NONE, FaceConfidence.LOW]:
                            # copy2(file, os.path.join(org_dir, "low_conf_unmatched_" + record["zoomImage"]))
                            org_db[idx] = FaceRecord(idx, -1, FaceConfidence.NONE)
                        else:
                            group_num = group_idx
                            org_groups[group_num] = GroupRecord(group_num, [idx])
                            org_groups[group_num].members.add(idx)
                            group_idx += 1

                    else:
                        org_groups[group_num].members.add(idx)
                        # here should be code to rearrange the representatives
                    org_db[idx] = FaceRecord(idx, group_num, face_class)

            # for gid, group in org_groups.items():
            #    for face_ind in group.members:
            #        source_name = records[face_ind]["zoomImage"]
            #        conf_str = str(org_db[face_ind].confidence.value)
            #        copy2(os.path.join(root, org, source_name), os.path.join(org_dir, f"{gid}_{conf_str}_{source_name}"))
            all_scores = np.matmul(org_mat, org_mat.T)
            np.fill_diagonal(all_scores, 0)
            conf_factors = np.array(
                [
                    conf2factor[org_db.get(i, FaceRecord(i, -1, FaceConfidence.NONE, False)).confidence]
                    for i in range(len(records))
                ]
            )
            for face_ind, face in org_db.items():
                if face.group_id < 0:
                    similarity_score = all_scores[face_ind]
                    similarity_argmax = np.argmax(
                        similarity_score * np.minimum(conf_factors, conf2factor[face.confidence])
                    )
                    similarity_max_score = similarity_score[similarity_argmax]
                    if similarity_max_score >= conf2cosine_th[face.confidence]:
                        if org_db[similarity_argmax].group_id < 0:
                            org_db[similarity_argmax].group_id = group_idx
                            group_idx += 1
                        face.group_id = org_db[similarity_argmax].group_id

            for face_ind, face in org_db.items():
                source_name = records[face_ind]["zoomImage"]
                conf_str = str(face.confidence.value)
                file = os.path.join(root, org, source_name)
                if face.group_id >= 0:
                    copy2(
                        os.path.join(root, org, source_name),
                        os.path.join(org_dir, f"{face.group_id}_{conf_str}_{source_name}"),
                    )
                elif face.is_detected:
                    copy2(file, os.path.join(org_dir, f"unmatched_{conf_str}_{source_name}"))
                else:
                    if os.path.exists(file):
                        copy2(file, os.path.join(org_dir, f"undetected_{conf_str}_{source_name}"))


def replay_new_pipe():
    analyzer = FaceAnalyzer(
        {"detector": "retinaface", "recognizer": "sface"}, context=Namespace(class_handler=ClassHandler())
    )
    sim_groups = False
    out_folder = os.path.join(base_path, "replay", "v3")
    for root, dirs, files in os.walk(base_db_path):
        for org in dirs:
            org_dir = os.path.join(out_folder, org)
            os.makedirs(org_dir, exist_ok=True)
            org_db, org_mat, records = get_org_data(org, analyzer)
            records = read_index_file(org)
            org_groups = {}
            gid = 1000

            # simulate addition of person from recommendation and add person
            if sim_groups:
                n_ids = len(records) // 2
                buckets = person_bucketing(org_db, org_mat[:n_ids], org_groups)
                sorted_buckets = sorted(buckets.values(), key=lambda item: len(item), reverse=True)
                if len(sorted_buckets[0]) > 3:
                    add_person_from_bucket(sorted_buckets[0], gid, org_groups, org_db, org_mat)
                    gid += 1
                if len(sorted_buckets) > 1:
                    found = False
                    for bucket in sorted_buckets[1:]:
                        for p in bucket:
                            if org_db.get(p, FaceRecord()).confidence in [FaceConfidence.MEDIUM, FaceConfidence.HIGH]:
                                add_person(p, gid, org_groups, org_db, org_mat[:n_ids])
                                gid += 1
                                found = True
                                break
                        if found:
                            break
            buckets = person_bucketing(org_db, org_mat, org_groups, FaceConfidence.NONE)
            for gid, bucket in sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True):
                for face_ind in bucket:
                    face = org_db.get(face_ind, None)
                    if face:
                        source_name = records[face_ind]["zoomImage"]
                        conf_str = str(face.confidence.value)
                        file = os.path.join(root, org, source_name)
                        if not face.is_detected:
                            if os.path.exists(file):
                                copy2(file, os.path.join(org_dir, f"undetected_{conf_str}_{source_name}"))
                        elif len(bucket) > 1:
                            copy2(
                                os.path.join(root, org, source_name),
                                os.path.join(org_dir, f"{gid}_{conf_str}_{source_name}"),
                            )
                        else:
                            copy2(file, os.path.join(org_dir, f"unmatched_{conf_str}_{source_name}"))
        break


def get_org_data(org: str, analyzer: FaceAnalyzer):
    detector = analyzer.detector
    recognizer = analyzer.recognizer
    records = read_index_file(org)
    org_db = {}
    org_mat = np.zeros((len(records), 128), dtype=float)
    for idx, record in enumerate(records):
        file = os.path.join(base_db_path, org, record["zoomImage"])
        if os.path.exists(file):
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox, landmarks, confidence = detector.forward_on_crop_list([image.copy()])
            if landmarks[0] is None:
                org_db[idx] = FaceRecord(idx, -1, FaceConfidence.NONE, is_detected=False)
                continue
            face_class = analyzer.classify_face(image, landmarks[0], bbox[0], confidence[0])
            rec_crop = recognizer.canonize_face(image, landmarks[0])
            features = np.array(recognizer.forward_on_crop_list([rec_crop])[0])
            org_mat[idx, :] = features
            org_db[idx] = FaceRecord(idx, -1, face_class)
    return org_db, org_mat, records


def get_closest_match(similarity_score: np.array, conf_factors: np.array, face_conf):
    similarity_argmax = np.argmax(similarity_score * np.minimum(conf_factors, conf2factor[face_conf]))
    similarity_max_score = similarity_score[similarity_argmax]
    if similarity_max_score >= conf2cosine_th[face_conf]:
        return similarity_argmax
    return None


def find_representatives(group: GroupRecord, org_db: Dict[int, FaceRecord], org_mat: np.array):
    group_list = list(group.members)
    feature_matrix = org_mat[np.array(group_list).astype(int)]
    if feature_matrix.shape[0] < 10:
        k_meds = KMedoids(n_clusters=1, random_state=0).fit(feature_matrix)
    else:
        k_meds = KMedoids(n_clusters=3, random_state=0).fit(feature_matrix)
    group.reps = [group_list[i] for i in k_meds.medoid_indices_]


def find_all_matches(similarity_score: np.array, conf_ths: np.array, face_conf):
    similarities = np.argwhere(similarity_score > np.maximum(conf_ths, conf2cosine_th[face_conf]))
    return similarities.flatten().tolist()


def add_person_from_bucket(bucket, gid, groups, org_db, org_mat):
    groups[gid] = GroupRecord(group_id=gid, reps=[], members=set(bucket))
    find_representatives(groups[gid], org_db, org_mat)


def add_person(person, gid, groups, org_db, org_mat):
    all_scores = np.matmul(org_mat, org_mat[person].T)
    default_rec = FaceRecord(-1, -1, FaceConfidence.NONE, False)
    conf_ths = np.array([conf2cosine_th[org_db.get(i, default_rec).confidence] for i in range(len(org_mat))])
    matches = find_all_matches(all_scores, conf_ths, org_db[person].confidence)
    groups[gid] = GroupRecord(group_id=gid, reps=[], members=set(matches))
    find_representatives(groups[gid], org_db, org_mat)


def person_bucketing(
    org_db: Dict[int, FaceRecord],
    org_mat: np.array,
    groups: Dict[int, GroupRecord],
    min_conf: FaceConfidence = FaceConfidence.LOW,
):
    n_persons = len(org_mat)
    all_scores = np.matmul(org_mat, org_mat.T)
    np.fill_diagonal(all_scores, 0)

    default_rec = FaceRecord(-1, -1, FaceConfidence.NONE, False)
    conf_factors = np.array([conf2factor[org_db.get(i, default_rec).confidence] for i in range(n_persons)])
    bucket_idx = 1
    bucket_mapping = {i: org_db.get(i, default_rec).group_id for i in range(n_persons)}

    conf_ths = np.array([conf2cosine_th[org_db.get(i, default_rec).confidence] for i in range(n_persons)])
    below_conf_th = np.array(
        [conf_order_map[org_db.get(i, default_rec).confidence] < conf_order_map[min_conf] for i in range(n_persons)]
    )
    all_scores[below_conf_th] *= 0
    all_scores[:, below_conf_th] *= 0

    in_groups = set()

    for gid, group in groups.items():
        in_groups.update(group.members)
        for rep in group.reps:
            matches = find_all_matches(all_scores[rep], conf_ths, org_db[rep].confidence)
            in_groups.update(matches)
            group.members.update(matches)
            for m in matches:
                org_db[m].group_id = gid
                bucket_mapping[m] = gid
        find_representatives(group, org_db, org_mat)
    if in_groups:
        np_groups = np.array(list(in_groups))
        all_scores[np_groups] *= 0
        all_scores[:, np_groups] *= 0

    for idx in range(n_persons):
        face = org_db.get(idx, default_rec)
        if bucket_mapping[idx] >= 0:
            continue
        matches = find_all_matches(all_scores[idx], conf_ths, face.confidence)
        matches_buckets = set([bucket_mapping[m] for m in matches]) - {-1}
        if len(matches_buckets) == 1:
            b_idx = matches_buckets.pop()
        elif len(matches_buckets) > 1:  # handle conflict
            grouped = np.array([m for m in matches if bucket_mapping[m] >= 0])
            max_bucket_scores = np.argmax(all_scores[idx][grouped] * conf_factors[grouped])
            b_idx = bucket_mapping[grouped[max_bucket_scores]]
        else:  # new group
            b_idx = bucket_idx
            bucket_idx += 1
        bucket_mapping[idx] = b_idx
        for m in matches:
            bucket_mapping[m] = b_idx

    buckets = defaultdict(set)
    for key, value in bucket_mapping.items():
        buckets[value].add(key)

    return buckets


if __name__ == "__main__":
    # download_faces_groups()
    # copy_to_single_folder()
    # gather_face_stats()
    # download_all_faces()
    # classify_faces()
    # bluriness_test()
    # check_detections()
    # replay_pipe()
    replay_new_pipe()
