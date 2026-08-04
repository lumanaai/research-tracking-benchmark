import concurrent
from pathlib import Path, PosixPath

import numpy as np
from tqdm import tqdm

from infra.GcpHandler import GcpStorageHandler
from utils.infra.credentials import credentials_file_to_env
from utils.infra.singlestore_io import SingleStoreDb

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"))
db = SingleStoreDb()
handler = GcpStorageHandler(bucket_name="lumixai-training-thumbnails")


def download_annotated_images(camera_id, days_back=30):
    get_hashes_query = "select orgIdHash, edgeIdHash, cameraIdHash from orgdevices where cameraId=%s limit 1"
    hashes = db.run_query_with_params(get_hashes_query, (camera_id,))[0]
    query = """SELECT *
               FROM general_thumbnails
               WHERE orgIdHash = %s
                 AND edgeIdHash = %s
                 AND cameraIdHash = %s
                 AND thumType = 2
                 AND timestamp > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL %s DAY) * 1000 
                 AND annotationData IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT 5000
            """
    results = db.run_query_with_params(
        query, (hashes["orgIdHash"], hashes["edgeIdHash"], hashes["cameraIdHash"], days_back)
    )
    output_dir = Path("/mnt/d/training_images").joinpath(camera_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    def download_file_by_idx(idx):
        result = results[idx]
        image_data = result["annotationData"].get("detections", [])
        image_name = result["thumbnail"]
        bucket_path = PosixPath("training").joinpath(result["edgeId"], result["cameraId"], image_name)
        local_path = output_dir.joinpath(image_name)
        local_label_path = local_path.with_suffix(".txt")
        try:
            if not local_path.exists():
                handler.download_single_file_s3(str(bucket_path), str(local_path))
            if not local_label_path.exists():
                det_array = np.atleast_2d(image_data)
                np.savetxt(local_label_path, det_array)
            return 1
        except Exception as e:
            print(f"Failed to download {bucket_path} to {local_path} due to {e}")
            return 0

    file_iter = range(len(results))
    if False:
        download_file_by_idx(1)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(download_file_by_idx, file_iter),
                    total=len(file_iter),
                    desc="Downloading images",
                    leave=False,
                )
            )
    print(f"Downloaded {len(results)} images to {output_dir}")


if __name__ == "__main__":
    camera_id = "694d0f890626b1a66ebb0478"
    download_annotated_images(camera_id, days_back=14)
