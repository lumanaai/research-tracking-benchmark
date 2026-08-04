import concurrent
import json
import os
import subprocess

from tqdm import tqdm

from analyzer_manager.cuda_library.build_cuda import print_subprocess_results
from utils.infra.singlestore_io import SingleStoreDb
from utils.infra.credentials import credentials_file_to_env
from utils.infra.s3_io import S3Handler
from pathlib import PurePosixPath, Path

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"))
db = SingleStoreDb()

def download_ppe_crops():
    local_dir = "/mnt/c/Temp/ppe/jul24"
    s3 = S3Handler()

    query = """select edgeId, cameraId, bestImage, pt.table_col AS pt_type, pt_type::%value AS value from trackers2 
    JOIN table(json_to_array(attributes::protectiveGearType)) pt
    where cameraIdHash in (2431625118737604, 3710315589539385, 505691887282468) and trackerTypeId=1 and bestImage is not NULL and value={value}
    and timestamp < 1721278573681
    order by  timestamp desc limit 1000"""

    params = {"no_hard_hat": 2} #{"hard_hat": 1, "no_hard_hat": 2}
    for key,val in params.items():
        os.makedirs(os.path.join(local_dir, key), exist_ok=True)
        results = db.run_query(query.format(value=val))
        def get_file_label_by_index(index):
            try:
                tracker = results[index]
                s3_path = PurePosixPath("crop").joinpath(tracker["edgeId"], tracker["cameraId"], tracker["bestImage"])
                target_path = os.path.join(local_dir, key,  tracker["bestImage"])
                if not os.path.exists(target_path):
                    s3.download_single_file_s3(str(s3_path), target_path)

                return 1
            except Exception as e:
                print(f"An error occur trying to download {results[index]['zoomImage']}: {e}")
                return 0

        file_iter = range(len(results))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(get_file_label_by_index, file_iter),
                    total=len(file_iter),
                    desc="Downloading images",
                    leave=False,
                )
            )


def download_wc_data():
    local_dir = "/mnt/c/Temp/weapons/sep24"
    s3 = S3Handler(bucket_name="lumixai-alert-thumbnails")
    s3_thumb = S3Handler()
    query = """select eventId, edgeId, cameraId, thumbnails, timestamp, vcc.verified from alerts_instances
    LEFT JOIN(SELECT alertInstanceId, verified  from vcc_log  where type = 3 and success=1) as vcc
    ON alerts_instances.alertInstanceId = vcc.alertInstanceId
    where flowType = 3 and category = 0 AND timestamp > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 30 DAY) * 1000  
    order by timestamp desc"""

    verified_dir = os.path.join(local_dir, "vcc_true")
    unverified_dir = os.path.join(local_dir, "vcc_false")
    unknown = os.path.join(local_dir, "vcc_unknown")
    path_selector = {1: verified_dir, 0: unverified_dir, None: unknown}

    os.makedirs(verified_dir, exist_ok=True)
    os.makedirs(unverified_dir, exist_ok=True)
    os.makedirs(unknown, exist_ok=True)

    results = db.run_query(query)
    def get_file_label_by_index(index):
        try:
            res = results[index]
            im_name = res["thumbnails"][0]
            s3_path = PurePosixPath("alerts").joinpath(res["edgeId"], res["cameraId"], im_name)
            target_path = os.path.join(path_selector[res["verified"]], im_name)

            ts = res["timestamp"]
            event_id = res["eventId"]
            snap_name = f"snapshot-{ts}-ooc-{event_id}.jpg"
            if not os.path.exists(target_path):
                s3.download_single_file_s3(str(s3_path), target_path)

            snapshot_s3_path = PurePosixPath("snapshot").joinpath(res["edgeId"], res["cameraId"], snap_name)
            target_path = os.path.join(path_selector[res["verified"]], snap_name)
            if not os.path.exists(target_path):
                s3_thumb.download_single_file_s3(str(snapshot_s3_path), target_path)
            return 1
        except Exception as e:
            print(f"An error occur trying to download {event_id}: {e}")
            return 0

    file_iter = range(len(results))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(get_file_label_by_index, file_iter),
                total=len(file_iter),
                desc="Downloading images",
                leave=False,
            )
        )
    download_vcc_images("weapons", path = os.path.join(local_dir, "vcc"))

def download_fall_vids():
    local_dir = "/mnt/c/Temp/fall"
    query = "select * from alerts_instances where flowType = 10 and category = 0  and timestamp > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 30 DAY) * 1000 order by timestamp desc"
    os.makedirs(local_dir, exist_ok=True)
    results = db.run_query(query)
    playbook = "playbooks/download_archive.yml"
    def get_file_label_by_index(index):
        try:
            res = results[index]
            eid = res["edgeId"]
            cid = res["cameraId"]
            ts = str(res["timestamp"])
            full_path = os.path.join(local_dir, f"{eid}_{cid}_{ts}.mp4")
            if not os.path.exists(full_path):
                p = subprocess.run(
                    args=["sh","playbook.sh",playbook, "-p", "-e",eid,"--cameraId", cid, "--timestamp", ts, "--dir_path", full_path],
                    cwd="/home/aviadz/sources/playbook",
                )
                print_subprocess_results(p)
                if os.path.exists(full_path):
                    with open(Path(full_path).with_suffix(".json"), "wt") as f:
                        json.dump(res, f)
        except Exception as e:
            print(f"An error occur trying to download {full_path}: {e}")

    file_iter = range(len(results))
    for i in tqdm(file_iter):
        get_file_label_by_index(i)
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #    list(
    #        tqdm(
    #            executor.map(get_file_label_by_index, file_iter),
    #            total=len(file_iter),
    #            desc="Downloading images",
    #            leave=False,
    #        )
    #    )


def download_vcc_images(alert_type: str = "weapons", path: str = None):

    if alert_type == "fire":
        _type = 9
    elif alert_type == "weapons":
        _type = 3
    elif alert_type == "fall":
        _type = 10
    else:
        raise ValueError(f"Invalid alert type: {alert_type}")
    if path is None:
        local_dir = os.path.join("/mnt/c/Temp/vcc", alert_type)
    else:
        local_dir = path
    query = f"select * from vcc_log where type = {_type} and timestamp > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 30 DAY) * 1000 order by timestamp desc"
    os.makedirs(local_dir, exist_ok=True)
    results = db.run_query(query)
    s3 = S3Handler(bucket_name="lumixai-alert-thumbnails")

    def get_file_label_by_index(index):
        try:
            res = results[index]
            eid = res["edgeId"]
            cid = res["cameraId"]
            img = str(res["imagePath"])
            full_path = os.path.join(local_dir,img)
            if not os.path.exists(full_path):
                s3.download_single_file_s3(str(PurePosixPath("alerts").joinpath(eid, cid, img)), full_path)
                with open(Path(full_path).with_suffix(".json"), "wt") as f:
                    res["timestamp"] = str(res["timestamp"])
                    json.dump(res, f, indent=4)
        except Exception as e:
            print(f"An error occur trying to download {full_path}: {e}")

    file_iter = range(len(results))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(get_file_label_by_index, file_iter),
                total=len(file_iter),
                desc="Downloading images",
                leave=False,
            )
        )

if __name__ == "__main__":
    download_wc_data()