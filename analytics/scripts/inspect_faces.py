import glob
import json
import shutil
from typing import List

import cv2
import numpy as np

from level1.face.face_detection import FaceDetectorFactory
from level1.face.face_recognition import FaceRecognizerFactory
from utils.infra.GcpHandler import GcpStorageHandler
from utils.infra.s3_io import S3Handler
import os
from tqdm import tqdm
import concurrent.futures
from pathlib import Path, PurePosixPath
from utils.infra.credentials import credentials_file_to_env

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"))
from utils.infra.singlestore_io import SingleStoreDb

db = SingleStoreDb()

conf_2_int = {"high": 7, "medium": 5, "low": 3, "none": 1}
recognizer_threshold = {"sface": 0.5, "arcface": 0.45, "webface":0.4}
def download_single_file(s3_handler, gcs_handler, s3_path, local_path):
    try:
        s3_handler.download_single_file_s3(str(s3_path), local_path)
        print(f"Downloaded {s3_path} from S3")
    except Exception as e:
        try:
            gcs_handler.download_single_file_s3(str(s3_path), local_path)
            print(f"Downloaded {s3_path} from GCS")
        except Exception as e:
            print(f"An error occur trying to download {s3_path}: {e}")
            return False
    return True


def download_person_faces(org_hash: int = None, use_personId: bool = True):
    skip_representive = False
    skip_trackers = True
    s3 = S3Handler("lumixai-training-thumbnails")
    gcs = GcpStorageHandler("lumixai-training-thumbnails")
    s3_assets = S3Handler(bucket_name="lumixai-face-assets")
    gcs_assets = GcpStorageHandler(bucket_name="lumixai-face-assets")
    s3_groups = S3Handler(bucket_name="lumixai-groups-images")
    gcs_groups = GcpStorageHandler(bucket_name="lumixai-groups-images")

    query = "SELECT personId, orgId, orgIdHash, bestImage from persons"
    if org_hash is not None:
        query += f" WHERE orgIdHash = {org_hash}"

    base_dir = "/mnt/d/ALIA_faces"
    os.makedirs(base_dir, exist_ok=True)

    all_persons_results = db.run_query(query)
    for res in all_persons_results:
        person_id = res["personId"]
        org_id_hash = res["orgIdHash"]
        org_id = res["orgId"]
        best_image = res["bestImage"]
        if use_personId:
            person_query = f"""
             SELECT edgeId, cameraId, zoomImage, id as tracker_id
             from trackers2
             WHERE orgIdHash = {org_id_hash} and personId = {person_id} and timestamp > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 5 DAY) * 1000
             """
        else:

            person_query = f"""
            WITH a AS (
            SELECT faceIdBin AS faceIdBinA, id as rep_id
            FROM person_representatives
            WHERE orgIdHash = {org_id_hash} AND personId in ({person_id})
            )
            SELECT t.edgeId, t.cameraId, t.zoomImage, t.id as tracker_id,
            DOT_PRODUCT(faceIdBinA, t.faceIdBin) AS score
            from trackers2 t, a
            WHERE t.orgIdHash = {org_id_hash} AND score > 0.46 AND t.timestamp > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 30 DAY) * 1000
            group by tracker_id limit 1000;
            """
        results = db.run_query(person_query)

        local_dir = os.path.join(base_dir, str(res["personId"]))
        os.makedirs(local_dir, exist_ok=True)
        if not skip_representive:
            # local_path = os.path.join(local_dir, best_image)
            local_path = os.path.join(local_dir, f"0_representative_{person_id}.jpg")
            s3_path = PurePosixPath("").joinpath(org_id, best_image)
            download_single_file(s3_assets, gcs_assets, s3_path, local_path)
            s3_path = PurePosixPath("crop").joinpath(org_id, best_image)
            download_single_file(s3_groups, gcs_groups, s3_path, local_path)

        def get_file_label_by_index(index):
            tracker = results[index]

            id_ = tracker["tracker_id"]
            img = tracker["zoomImage"]
            target_path = os.path.join(local_dir, f"{person_id}_{id_}.jpg")

            if img is None:
                return 0
            elif os.path.exists(target_path):
                return 1

            s3_path = PurePosixPath("crop").joinpath(tracker["edgeId"], tracker["cameraId"], img)
            return download_single_file(s3, gcs, s3_path, target_path)

        file_iter = range(len(results))

        # for i in tqdm(file_iter, desc="Downloading images", leave=False):
        # get_file_label_by_index(i)
        if not skip_trackers:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                summary = list(
                    tqdm(
                        executor.map(get_file_label_by_index, file_iter),
                        total=len(file_iter),
                        desc="Downloading images",
                        leave=False,
                    )
                )
            print(f"Downloaded {sum(summary)} images for person {person_id} from org {org_id_hash}")


def download_person_representing_images():
    s3 = S3Handler("lumixai-groups-images")
    gcs = GcpStorageHandler("lumixai-groups-images")
    query = f"SELECT personId, orgId, bestImage from persons"
    base_dir = "/mnt/d/person_faces/representatives"
    os.makedirs(base_dir, exist_ok=True)
    results = db.run_query(query)

    def get_file_label_by_index(index):
        tracker = results[index]

        person_id = tracker["personId"]
        img = tracker["bestImage"]
        target_path = os.path.join(base_dir, f"{person_id}.jpg")
        if img is None:
            return 0
        elif os.path.exists(target_path):
            return 1

        s3_path = PurePosixPath("crop").joinpath(tracker["orgId"], img)
        return download_single_file(s3, gcs, s3_path, target_path)

    file_iter = range(len(results))

    # for i in tqdm(file_iter, desc="Downloading images", leave=False):
    # get_file_label_by_index(i)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summary = list(
            tqdm(
                executor.map(get_file_label_by_index, file_iter),
                total=len(file_iter),
                desc="Downloading images",
                leave=False,
            )
        )


def run_face_rec_on_metadata(metadata: dict, recognizer: str, face_dir: str):
    is_update = False
    face_pool = set(glob.glob(os.path.join(face_dir, "*.jpg")))
    rec_data = metadata.get(recognizer, {})
    work_batch = list(face_pool - set(rec_data.keys()))
    if work_batch:
        detector = FaceDetectorFactory.create("retinaface", {})
        recognizer_engine = FaceRecognizerFactory.create(recognizer, {})
        classifier = FaceRecognizerFactory.create("magface", {})
        unrec = []
        for i in tqdm(range(0, len(work_batch), 8), desc="Processing batches"):
            batch_files = work_batch[i : i + 8]
            images = [cv2.cvtColor(cv2.imread(os.path.join(face_dir, file)), cv2.COLOR_BGR2RGB) for file in batch_files]

            bboxes, landmarks_list, confidences = detector.forward_on_crop_list(images)
            rec_crops = []
            rec_jobs = []
            for j in range(len(batch_files)):
                if bboxes[j] is None:
                    unrec.append(batch_files[j])
                    continue
                rec_crops.append(recognizer_engine.canonize_face(images[j], landmarks_list[j]))
                rec_jobs.append(batch_files[j])
            if rec_crops:
                desc = recognizer_engine.forward_on_crop_list(rec_crops)
                classifications, class_scores = classifier.forward_on_crop_list(rec_crops)
                for j in range(len(rec_jobs)):
                    # save face descriptor as hex
                    rec_data[rec_jobs[j]] = {
                        "descriptor": desc[j].tobytes().hex(),
                        "confidence": classifications[j],
                        "score": class_scores[j],
                    }
        for urec in unrec:
            rec_data[urec] = {
                "descriptor": np.zeros(recognizer_engine.descriptor_length, dtype=np.float32).tobytes().hex(),
                "confidence": "none",
                "score": 0,
            }
        if recognizer not in metadata:
            metadata[recognizer] = {}
        metadata[recognizer].update(rec_data)
        is_update = True
    return is_update


def investigate_person_faces(org_dir, faces: List[int] = None, org_hash: int = 0, recognizer:str = "sface", reset:bool=False):
    visualize_images = True
    visualize_plots = False
    filter_faces = True
    fix_reps = True

    if faces is None:
        # get all subfolders
        faces = [int(f) for f in os.listdir(org_dir) if os.path.isdir(os.path.join(org_dir, f))]
    for face in faces:
        face_dir = str(os.path.join(org_dir, str(face)))
        output_dir = os.path.join(face_dir, "output")
        metadata_file = os.path.join(face_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {recognizer: {}}

        face_rep = f"0_representative_{face}.jpg"
        if recognizer not in metadata or reset:
            metadata[recognizer] = {}
        is_update = run_face_rec_on_metadata(metadata, recognizer, face_dir)
        if is_update:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
        rec_data = metadata[recognizer]
        # un hex descriptors
        rec_desc = {k: np.frombuffer(bytes.fromhex(v["descriptor"]), dtype=np.float32) for k, v in rec_data.items()}
        rep_desc = rec_desc.pop(os.path.join(face_dir, face_rep))
        keys = list(rec_desc.keys())
        desc_mat = np.stack([rec_desc[k] for k in keys])
        matches = rep_desc @ desc_mat.T
        orig_matches = matches.copy()
        confidences = np.array([conf_2_int[rec_data[k]["confidence"]] for k in keys])
        cls_scores = np.array([rec_data[k]["score"] for k in keys])
        match_th = recognizer_threshold[recognizer]

        if visualize_plots:
            # plot matches
            from matplotlib import pyplot as plt

            # plot histogram with bins of 0.1
            plt.hist(matches, bins=20)
            plt.xlabel("Match score")
            plt.ylabel("Frequency")
            plt.title("Histogram of face matches for original representative using " + recognizer)
            plt.grid(True)
            plt.show()

            if "sface" in metadata:
                if recognizer != "sface":
                    sface_data = metadata["sface"]
                    desc_mat = np.stack([np.frombuffer(bytes.fromhex(sface_data[k]["descriptor"]), dtype=np.float32) for k in keys])

                from scipy.stats import pearsonr

                query = f"SELECT * from person_representatives where personId = {face}"
                results = db.run_query(query)
                for res in results:
                    rep_id = res["id"]
                    rep_vec = np.frombuffer(res["faceIdBin"], dtype=np.float32)
                    matches = rep_vec @ desc_mat.T
                    corr, _ = pearsonr(orig_matches, matches)
                    print(f"Pearson correlation for rep {rep_id}: {corr}")
                    # plot histogram with bins of 0.1
                    plt.hist(np.abs(orig_matches - matches), bins=20)
                    plt.xlabel("abs of difference from original rep")
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram of face matches difference for rep {rep_id} using sface")
                    plt.grid(True)
                    plt.show()

        if filter_faces:
            best_rep = np.argmax(np.sum(desc_mat @ desc_mat.T > match_th, axis=0))
            orig_matches = desc_mat[best_rep] @ desc_mat.T
            if visualize_images:
                high_conf_idx = np.argsort(orig_matches).flatten()
                output_rec_dir = os.path.join(output_dir, recognizer)

                if output_rec_dir is not None:
                    if os.path.exists(output_rec_dir):
                        shutil.rmtree(output_rec_dir)
                    os.makedirs(output_rec_dir, exist_ok=False)

                    # copy to output folder
                    for idx in high_conf_idx:
                        if orig_matches[idx] > -1:
                            src = Path(keys[idx])
                            dst = os.path.join(output_rec_dir, f"{orig_matches[idx]:.2f}_{src.name}")
                            shutil.copy2(src, dst)
            low_conf_idx = np.argwhere(orig_matches < match_th).flatten()
            trackers_to_remove = []
            for idx in low_conf_idx:
                src = Path(keys[idx])
                parts = src.name.split("_")
                if len(parts) == 2:
                    trackers_to_remove.append(parts[1].split(".")[0])
            print(f"found {len(trackers_to_remove)} low confidence trackers to remove for person {face}")
            query = f"UPDATE trackers2 set personId=0 where orgIdHash={org_hash} and personId={face} and id in ({','.join(trackers_to_remove)})"
            # save as text file
            with open(os.path.join(face_dir, "low_conf_trackers.txt"), "w") as f:
                f.write(query)

        if fix_reps:
            reps_to_remove = []
            reps_to_add = []
            sim_mat = desc_mat @ desc_mat.T
            sm = np.sum(sim_mat > match_th, axis=0)
            mx = np.max(sm)
            best_reps = np.argwhere(sm >= mx ).flatten()
            best_rep = np.argwhere(sm == mx).flatten()[0]
            new_matches = sim_mat[best_rep]

            # copy new best rep to make sure its indeed the correct person
            cum_rep = np.argsort(confidences + cls_scores/100 + np.mean(np.clip(sim_mat, 0, 1), axis=0))
            reps_dir = os.path.join(face_dir, "representatives")

            selectd = best_reps.tolist() + cum_rep[-5:].tolist()
            for br in selectd:
                os.makedirs(reps_dir, exist_ok=True)
                shutil.copy2(keys[br], os.path.join(reps_dir, Path(keys[br]).name))
                reps_to_add.append(keys[br])

            if "sface" not in metadata:
                # run sface on all images and save to metadata
                run_face_rec_on_metadata(metadata, "sface", face_dir)
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=4)

            sface_data = metadata["sface"]
            sface_desc_mat = np.stack([np.frombuffer(bytes.fromhex(sface_data[k]["descriptor"]), dtype=np.float32) for k in keys])
            sface_math_th = recognizer_threshold["sface"]

            query = f"SELECT * from person_representatives where personId = {face}"
            results = db.run_query(query)
            org_id = None
            for res in results:
                rep_id = res["id"]
                org_id = res["orgId"]
                rep_vec = np.frombuffer(res["faceIdBin"], dtype=np.float32)
                matches = rep_vec @ sface_desc_mat.T

                # measure precison, recall and f1 score based on "new_matches"
                tp = np.sum((matches > sface_math_th) & (new_matches > match_th))
                fp = np.sum((matches < sface_math_th) & (new_matches > match_th))
                fn = np.sum((matches > sface_math_th) & (new_matches < match_th))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                print(f"Rep {rep_id} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

                if not res["manualUpload"] and f1 < 0.9:
                    reps_to_remove.append(res)

            # generate update query
            with open(os.path.join(face_dir, "reps_fix.txt"), "w") as f:
                if reps_to_remove:
                    rem_query = f"DELETE FROM person_representatives where personId = {face} and id in ({','.join([str(r['id']) for r in reps_to_remove])});\n"
                    print(f"found {len(reps_to_remove)} representatives to remove for person {face}")
                    f.write(rem_query)
                for r in reps_to_add:
                    conf = conf_2_int[sface_data[r]["confidence"]]
                    face_hex = sface_data[r]["descriptor"]
                    add_query = f'INSERT INTO person_representatives (personId, orgId, orgIdHash, faceIdBin, faceConfidence, manualUpload) VALUES ({face}, "{org_id}", {org_hash}, UNHEX("{face_hex}"), {conf}, 1);\n'
                    # save as text file
                    f.write(add_query)


if __name__ == "__main__":
    print("started")
    org_hash_id = 2585734239832998
    # download_person_faces()
    # download_person_faces(org_hash_id)
    investigate_person_faces("/mnt/d/ALIA_faces", None, org_hash_id, recognizer="webface", reset=False)
