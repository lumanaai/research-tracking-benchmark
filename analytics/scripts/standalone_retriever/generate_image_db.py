import concurrent
import heapq
import json
import os
from pathlib import PurePosixPath, Path

from tqdm import tqdm

from infra.GcpHandler import GcpStorageHandler
from infra.credentials import credentials_file_to_env
from infra.singlestore_io import SingleStoreDb

credentials_file_to_env(Path(__file__).parent.joinpath("env_cred.json"), env="prod")
db = SingleStoreDb()
root_dir = "/mnt/d/alerts_20251123/"
playbook_root = "/home/aviadz/sources/playbook"
gcs = GcpStorageHandler("lumixai-alert-thumbnails")
crop_gcs = GcpStorageHandler()

alert_types = {3: "weapon", 9: "fire", 10: "fall", 11: "violence", 13: "brandishing", 4000000: "protectivegear"}

bad_org_hashes = [
    4498244278661834,
    2580440212172753,
    4498244278661834,
    3628711157450028,
    634709024819640,
    2771300933355345,
    4094724414476479,
    1386261417423039,
    4272828014638899,
    643919798520473,
    68946614113802,
    2407643435878654,
    801678643081693,
    1625503694628242,
    3057294456990585,
    4168210973426054,
    448448726528987,
    2887519206752484,
    3618215448434039,
]
org_hash_to_avoid = [
    3629336386847755,  #  "MAY INSTITUTE"
    4216993766007150,  #  "WELLS FARGO DEMO"
    926389631442275,  #  "HANSON"
    3175881254331297,  #  ,"NYU"
    104419465223451,  #  "NEBIUS"
    760350377506117,  #  "NEBIUS RESIDENCE"
    117992671875358,  #  "WELLS FARGO"
    4047729532298004,  #  "Bank HaPoalim"
]

def generate_camera_dataset(n_top_per_camera: int = 140, max_camera_workers: int = 8, max_download_workers: int = 16):
    local_dir = "/mnt/d/images_db"
    os.makedirs(local_dir, exist_ok=True)

    orgs_to_avoid = set(bad_org_hashes + org_hash_to_avoid)
    orgs_str = ",".join(str(o) for o in orgs_to_avoid)
    gcs_training = GcpStorageHandler("lumixai-training-thumbnails")

    # 1. Find all relevant camera hashes
    hash_query = f"""
    SELECT DISTINCT cameraIdHash FROM training_thumbnails
    WHERE timestamp > UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL 30 DAY)) * 1000
      AND orgIdHash NOT IN ({orgs_str})
      AND edgeMetadata IS NOT NULL AND edgeMetadata::$detections IS NOT NULL
    ORDER BY edgeId ASC
    """
    camera_hashes = [row["cameraIdHash"] for row in db.run_query(hash_query)]
    print(f"Found {len(camera_hashes)} cameras to process")

    thumb_query = """
    SELECT cameraId, edgeId, thumbnail, edgeMetadata::$detections AS detections
    FROM training_thumbnails
    WHERE cameraIdHash = %s
      AND edgeMetadata IS NOT NULL AND edgeMetadata::$detections IS NOT NULL
      AND timestamp > UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL 30 DAY)) * 1000
    """

    # Shared download pool for all cameras
    download_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_download_workers)

    import threading
    _thread_local = threading.local()

    def _get_db():
        """Get a thread-local DB connection to avoid packet sequence errors."""
        if not hasattr(_thread_local, "db"):
            _thread_local.db = SingleStoreDb()
        return _thread_local.db

    # 2. Process each camera in parallel
    def process_camera(cam_hash):
        try:
            # a. Grab all thumbnails with detections
            local_db = _get_db()
            cam_results = local_db.run_query_with_params(thumb_query, (cam_hash,))
            if not cam_results:
                return 0
            os.makedirs(os.path.join(local_dir, cam_results[0]["edgeId"]), exist_ok=True)

            # b. Score each thumbnail by count of rows where first element is 0
            def _score(row):
                dets = row["detections"]
                if not dets:
                    return 0
                if isinstance(dets, str):
                    dets = json.loads(dets)
                return sum(1 for d in dets if d and d[0] == 0)

            scores = list(map(_score, cam_results))

            # c. Download top N (highest scores) - partial sort with heapq
            top_indices = heapq.nlargest(n_top_per_camera, range(len(scores)), key=scores.__getitem__)
            top = [(cam_results[i], scores[i]) for i in top_indices if scores[i] > 0]

            # Submit downloads to shared pool
            def _download(row_score):
                row, score = row_score
                eid = row["edgeId"]
                cid = row["cameraId"]
                thumb = row["thumbnail"]
                full_path = os.path.join(local_dir, eid, f"{eid}_{cid}_{thumb}")
                try:
                    if not os.path.exists(full_path):
                        crop_gcs.download_single_file_s3(
                            str(PurePosixPath("training").joinpath(eid, cid, thumb)), full_path
                        )
                except Exception as e:
                    print(f"Failed to download {full_path}: {e}")
                return 0

            futures = [download_pool.submit(_download, item) for item in top]
            downloaded = sum(f.result() for f in concurrent.futures.as_completed(futures))
            return downloaded
        except Exception as e:
            print(f"Error processing camera {cam_hash}: {e}")
            return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_camera_workers) as camera_pool:
        results = list(
            tqdm(
                camera_pool.map(process_camera, camera_hashes),
                total=len(camera_hashes),
                desc="Processing cameras",
            )
        )
    download_pool.shutdown(wait=True)
    print(f"Done. Downloaded {sum(results)} thumbnails from {len(camera_hashes)} cameras")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download top training thumbnails per camera")
    parser.add_argument("--n_top", type=int, default=140, help="Top N thumbnails per camera to download")
    parser.add_argument("--camera_workers", type=int, default=8, help="Max parallel camera processing threads")
    parser.add_argument("--download_workers", type=int, default=16, help="Max parallel download threads")
    args = parser.parse_args()
    generate_camera_dataset(
        n_top_per_camera=args.n_top,
        max_camera_workers=args.camera_workers,
        max_download_workers=args.download_workers,
    )
