import concurrent
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import PurePosixPath, Path
from typing import List, Dict, Union

from tqdm import tqdm

from analyzer_manager.cuda_library.build_cuda import print_subprocess_results
from infra.GcpHandler import GcpStorageHandler
from utils.infra.credentials import credentials_file_to_env
from utils.infra.singlestore_io import SingleStoreDb

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"), env="prod")
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
avoided_orgs = list(set(bad_org_hashes + org_hash_to_avoid))


def download_video_image_from_tracker(tracker_data: Dict, storage_dir, ts_offset=3, no_vid: bool = False) -> str:
    eid = tracker_data["edgeId"]
    cid = tracker_data["cameraId"]
    ts = str(tracker_data["timestamp"] // 1000 - ts_offset)
    duration = round(tracker_data.get("dwell") or 5 + 0.5) + ts_offset
    full_path = os.path.join(storage_dir, f"{cid}_{ts}.mp4")
    playbook = "playbooks/download_archive_gcp.yml"
    try:
        if not os.path.exists(full_path):

            cmd = [
                "sh",
                "playbook.sh",
                playbook,
                "-p",
                "-e",
                eid,
                "--cameraId",
                cid,
                "--timestamp",
                ts,
                "--dir_path",
                storage_dir,
                "--duration",
                str(duration),
            ]
            if not no_vid:
                print(f"Running command: {' '.join(cmd)}")
                p = subprocess.run(args=cmd, cwd=playbook_root)
                print_subprocess_results(p)
            if os.path.exists(full_path) or no_vid:
                try:
                    tracker_data.pop("timestampPartition", None)
                    with open(Path(full_path).with_suffix(".json"), "wt") as f:
                        json.dump(tracker_data, f, indent=4)
                    img_target_path = str(Path(full_path).with_suffix(".jpg"))
                    thumb = tracker_data["bestImage"]
                    if thumb:
                        crop_gcs.download_single_file_s3(
                            str(PurePosixPath("crop").joinpath(eid, cid, thumb)), img_target_path
                        )
                except Exception as e:
                    print(f"An error occur trying to download image for {full_path}: {e}")
    except Exception as e:
        print(f"An error occur trying to download {full_path}: {e}")
    return full_path


def download_vcc_vids(
    alert_query: Union[int, Dict] = 10, local_dir=None, days_before=14, ts_offset=0, duration=40, limit=1000
):
    if isinstance(alert_query, int):
        alert_str = alert_types[alert_query]
        if local_dir is None:
            local_dir = os.path.join(root_dir, alert_str, "")
        last_ts = 0
        if os.path.exists(local_dir):
            last_ts = max(
                [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(local_dir) if f.endswith(".mp4")],
                default=0,
            )
        else:
            os.makedirs(local_dir, exist_ok=True)

        max_period = datetime.now() - timedelta(days=days_before)
        max_period = int(max_period.timestamp())
        last_ts = max(last_ts, max_period) * 1000

        query = """
                select UNIX_TIMESTAMP(timestamp) AS ts, *
                from vcc_log
                where type = %s
                  and alertInstanceId > 0
                  and timestamp
                    > %s
                  and verified = 1
                order by timestamp desc limit %s; \
                """
        results = db.run_query_with_params(query, (alert_query, last_ts, limit))

    else:
        key = list(alert_query.keys())[0]
        if local_dir is None:
            local_dir = os.path.join(root_dir, key)
        filters = " AND ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in alert_query.items()])
        query = f"""SELECT UNIX_TIMESTAMP(timestamp) AS ts, * FROM vcc_log 
                    WHERE timestamp > CURRENT_DATE - INTERVAL %s DAY AND {filters} ORDER BY timestamp DESC LIMIT {limit}
                """
        results = db.run_query_with_params(query, (days_before))
    playbook = "playbooks/download_archive_gcp.yml"
    items_sorted = sorted(results, key=lambda x: x["ts"])
    duration = str(duration)
    os.makedirs(local_dir, exist_ok=True)

    # items_sorted.reverse()
    def get_file_label_by_index(index):

        res = items_sorted[index]
        eid = res["edgeId"]
        cid = res["cameraId"]
        img_path = res["imagePath"]
        ts = str(int(img_path.split("-")[2]) // 1000 - ts_offset)
        res["timestamp"] = ts
        full_path = os.path.join(local_dir, f"{cid}_{ts}.mp4")
        try:
            if not os.path.exists(full_path):
                cmd = [
                    "sh",
                    "playbook.sh",
                    playbook,
                    "-p",
                    "-e",
                    eid,
                    "--cameraId",
                    cid,
                    "--timestamp",
                    ts,
                    "--dir_path",
                    local_dir,
                    "--duration",
                    duration,
                ]
                print(f"Running command: {' '.join(cmd)}")
                p = subprocess.run(args=cmd, cwd=playbook_root)
                print_subprocess_results(p)
                if os.path.exists(full_path):
                    with open(Path(full_path).with_suffix(".json"), "wt") as f:
                        json.dump(res, f, indent=4)
                    try:
                        img_target_path = str(Path(full_path).with_suffix(".jpg"))
                        gcs.download_single_file_s3(
                            str(PurePosixPath("alerts").joinpath(eid, cid, res["imagePath"])), img_target_path
                        )
                    except Exception as e:
                        print(f"An error occur trying to download image for {full_path}: {e}")
        except Exception as e:
            print(f"An error occur trying to download {full_path}: {e}")

    file_iter = range(len(items_sorted))

    # for i in tqdm(file_iter):
    #    get_file_label_by_index(i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(
            tqdm(
                executor.map(get_file_label_by_index, file_iter),
                total=len(file_iter),
                desc="Downloading images",
                leave=False,
            )
        )


def download_vcc_images(alert_type: int = 3, path: str = None, filter_cores: List = None):

    if path is None:
        local_dir = os.path.join(root_dir, alert_types[alert_type])
    else:
        local_dir = path
    if filter_cores is not None:
        filter_str = " AND edgeId IN ({}) ".format(",".join([f"'{core}'" for core in filter_cores]))
    else:
        filter_str = ""
    query = f"""
        select * from vcc_log where type = {alert_type} and timestamp > CURRENT_DATE - INTERVAL 30 DAY 
        AND verified=1 {filter_str}
        order by timestamp desc limit 10000
    """
    os.makedirs(local_dir, exist_ok=True)
    results = db.run_query(query)

    def get_file_label_by_index(index):
        res = results[index]
        eid = res["edgeId"]
        cid = res["cameraId"]
        img = str(res["imagePath"])
        full_path = os.path.join(local_dir, img)
        try:
            if not os.path.exists(full_path):
                gcs.download_single_file_s3(str(PurePosixPath("alerts").joinpath(eid, cid, img)), full_path)
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


def redownload_vids():
    # query = """
    # select ai.timestamp as ts, v.* from vcc_log v
    # JOIN alerts_instances ai ON v.alertInstanceId = ai.alertInstanceId
    # where v.type=%s and v.alertInstanceId > 0 and ai.timestamp > %s and v.verified = 1 order by ai.timestamp desc limit 10;
    # """
    # query = "select UNIX_TIMESTAMP(timestamp) as ts, * from vcc_log where type=%s and alertInstanceId > 0 and ts > %s and verified = 1 order by timestamp desc limit 5000"

    import re

    with open("/exports/falling_filtered_list.txt", "r") as f:
        files = [line.strip() for line in f if line.strip()]

    pairs = []
    for f in files:
        m = re.match(r"^([a-f0-9]+)_(\d+)\.jpg$", f)
        if m:
            camera_id, timestamp = m.group(1), m.group(2)
            pairs.append((camera_id, int(timestamp)))

    camera_is = list(set([p[0] for p in pairs]))
    tss = [p[1] for p in pairs]
    query = f"""
    SELECT ai.timestamp as ts, v.* , UNIX_TIMESTAMP(v.timestamp) as vts
    FROM vcc_log v JOIN alerts_instances ai ON v.alertInstanceId = ai.alertInstanceId
    WHERE vts in ({",".join([str(ts) for ts in tss])})
    """
    #         #",".join([f"'{cid}'" for cid in camera_is]), and v.cameraId in (%s)
    results = db.run_query(query)
    print(results)


def download_faces(mx_n_to_download: int = 10000, is_excluded=False, by_org=True):
    local_dir = "/mnt/d/newFaceDB"
    days_before = 5
    if is_excluded:

        query = f"""SELECT edgeId, cameraId, zoomImage, timestamp FROM trackers2 
                    where faceConfidence > 0 and zoomImage IS NOT NULL and timestamp > %s  and orgIdHash NOT IN ({','.join([str(oh) for oh in avoided_orgs])}) 
                    order by timestamp ASC limit 10000"""
    else:
        # org_hashes = [1900761844117543, 1211398195034909, 1921860897710873]
        query = f"SELECT DISTINCT orgIdHash FROM orgdevices WHERE orgIdHash NOT IN ({','.join([str(oh) for oh in avoided_orgs])})"
        org_hashes = db.run_query(query)
        org_hashes = [row["orgIdHash"] for row in org_hashes]
        query = f"""SELECT orgIdHash, edgeId, cameraId, zoomImage, timestamp FROM trackers2 
                            where faceConfidence > 0 and zoomImage IS NOT NULL and timestamp > %s  and orgIdHash = %s
                            order by timestamp ASC limit 2000"""
    if by_org:
        org_data = db.run_query("SELECT DISTINCT orgIdHash, orgName FROM orgdevices")
        org_dict = {row["orgIdHash"]: row["orgName"] for row in org_data}
    else:
        org_dict = {}

    max_period = datetime.now() - timedelta(days=days_before)
    os.makedirs(local_dir, exist_ok=True)

    for org_hash in org_hashes:
        last_ts = int(max_period.timestamp()) * 1000
        num_downloaded = 0
        if by_org:
            org_name = org_dict.get(org_hash, str(org_hash))
            org_path = os.path.join(local_dir, org_name.replace(" ", "_"))
        else:
            org_path = local_dir
        os.makedirs(org_path, exist_ok=True)
        while num_downloaded < mx_n_to_download:
            results = db.run_query_with_params(query, (last_ts, org_hash))
            n_res = len(results)
            if n_res == 0:
                break

            def get_file_label_by_index(index):
                res = results[index]
                eid = res["edgeId"]
                cid = res["cameraId"]
                img = str(res["zoomImage"])
                full_path = os.path.join(org_path, img)
                try:
                    if not os.path.exists(full_path):
                        crop_gcs.download_single_file_s3(str(PurePosixPath("crop").joinpath(eid, cid, img)), full_path)
                except Exception as e:
                    print(f"An error occur trying to download {full_path}: {e}")

            num_downloaded += n_res
            last_ts = int(results[-1]["timestamp"])
            file_iter = range(n_res)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(
                    tqdm(
                        executor.map(get_file_label_by_index, file_iter),
                        total=len(file_iter),
                        desc="Downloading images",
                        leave=False,
                    )
                )


def download_alerts_videos_by_event_id(event_id: str, local_dir: str = "/mnt/d/event_clips", ts_offset=5, duration=10):
    storage_dir = os.path.join(local_dir, event_id)
    ts = 0
    os.makedirs(storage_dir, exist_ok=True)
    query = "select * from alerts_instances where eventId=%s and timestamp  > %s order by timestamp asc LIMIT 300"
    results = db.run_query_with_params(query, (event_id, ts))

    playbook = "playbooks/download_archive_gcp.yml"
    items_sorted = sorted(results, key=lambda x: x["timestamp"])

    def get_file_label_by_index(index):

        res = items_sorted[index]
        eid = res["edgeId"]
        cid = res["cameraId"]
        ts = str(res["timestamp"] // 1000 - ts_offset)
        full_path = os.path.join(storage_dir, f"{cid}_{ts}.mp4")
        try:
            if not os.path.exists(full_path):
                cmd = [
                    "sh",
                    "playbook.sh",
                    playbook,
                    "-p",
                    "-e",
                    eid,
                    "--cameraId",
                    cid,
                    "--timestamp",
                    ts,
                    "--dir_path",
                    storage_dir,
                    "--duration",
                    str(duration),
                ]
                print(f"Running command: {' '.join(cmd)}")
                p = subprocess.run(args=cmd, cwd=playbook_root)
                print_subprocess_results(p)
                if os.path.exists(full_path):
                    with open(Path(full_path).with_suffix(".json"), "wt") as f:
                        json.dump(res, f, indent=4)
                    try:
                        img_target_path = str(Path(full_path).with_suffix(".jpg"))
                        thumb = res["thumbnails"]
                        if thumb:
                            gcs.download_single_file_s3(
                                str(PurePosixPath("alerts").joinpath(eid, cid, thumb[0])), img_target_path
                            )
                    except Exception as e:
                        print(f"An error occur trying to download image for {full_path}: {e}")
        except Exception as e:
            print(f"An error occur trying to download {full_path}: {e}")

    file_iter = range(len(items_sorted))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(
            tqdm(
                executor.map(get_file_label_by_index, file_iter),
                total=len(file_iter),
                desc="Downloading images",
                leave=False,
            )
        )


def download_alerts_thumbnails(local_dir: str, query_parameters: Dict, days_before=7, with_vcc=True, max_res=1000):
    os.makedirs(local_dir, exist_ok=True)
    if with_vcc:
        os.makedirs(os.path.join(local_dir, "vcc_unknown"), exist_ok=True)
        os.makedirs(os.path.join(local_dir, "vcc_true"), exist_ok=True)
        os.makedirs(os.path.join(local_dir, "vcc_false"), exist_ok=True)

    max_period = datetime.now() - timedelta(days=days_before)
    max_period = int(max_period.timestamp())
    query_vars = [f"{k}='{v}'" for k, v in query_parameters.items() if not isinstance(v, list)]
    query_list = [
        f"{k} IN ({','.join(repr(item) for item in v)})" for k, v in query_parameters.items() if isinstance(v, list)
    ]
    additional_query = " AND ".join(query_list + query_vars)
    limit = 10000
    query = f"""
    SELECT * from alerts_instances_for_debug where timestamp > %s AND {additional_query} 
    AND thumbnails IS NOT NULL AND orgIdHash NOT IN ({','.join([str(oh) for oh in avoided_orgs])}) 
    order by timestamp asc limit %s;
    """

    results = db.run_query_with_params(query, (max_period * 1000, min(limit, max_res)))
    num_res = max_res - limit
    while results:
        vcc_results_dict = {}
        if with_vcc:
            vcc_parms = [res["alertInstanceId"] for res in results]
            placeholders = ",".join(["%s"] * len(vcc_parms))
            vcc_query = f"SELECT alertInstanceId, success, verified, responseMessage from vcc_verification_log  where alertInstanceId in ({placeholders})"
            vcc_results = db.run_query_with_params(vcc_query, tuple(vcc_parms))
            vcc_results_dict = {res["alertInstanceId"]: res for res in vcc_results}

        def get_file_label_by_index(index):
            res = results[index]
            res.update(vcc_results_dict.get(str(res["alertInstanceId"]), {}))
            eid = res["edgeId"]
            cid = res["cameraId"]
            thumbs = res["thumbnails"] + res["validationThumbnails"]
            if len(thumbs) == 0:
                print(f"No thumbnails for {eid}/{cid} - instance {res['alertInstanceId']}")
                return None
            store_dir = local_dir
            if with_vcc:
                vcc_dir = "vcc_unknown"
                if res.get("success", None):
                    vcc_dir = "vcc_true" if res.get("verified") > 0 else "vcc_false"
                store_dir = os.path.join(local_dir, vcc_dir)
            for thumb in thumbs:
                full_path = os.path.join(store_dir, thumb)
                try:
                    if not os.path.exists(full_path):
                        gcs.download_single_file_s3(str(PurePosixPath("alerts").joinpath(eid, cid, thumb)), full_path)
                except Exception as e:
                    print(f"An error occur trying to download {full_path}: {e}")

            debug_data = res.pop("debugData", None)
            first_thumb = os.path.join(store_dir, thumbs[0])
            if os.path.exists(first_thumb):
                json_path = Path(first_thumb).with_suffix(".json")
                with open(json_path, "wt") as f:
                    json.dump(res, f, indent=4)

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
        if num_res >= 0:
            results = db.run_query_with_params(query, (results[-1]["timestamp"], min(limit, num_res)))
            num_res -= limit
        else:
            break


def download_custom_obj_representatives(custom_object_id: str, out_dir: str):
    """
    select * from lumana_prod.orgdevices where cameraId='692f54915c8e963b37f7859f'
    select * from lumana_prod.custom_objects where orgId='6776cfe0a4903acd9b11f6dd'
    select * from lumana_prod.custom_objects_representative where customObjectId='4503599633000006'
    """
    from infra.GcpHandler import GcpStorageHandler

    gcp = GcpStorageHandler(bucket_name="lumanaai-custom-obj-representatives")
    out_dir = os.path.join(out_dir, custom_object_id)
    shutil.rmtree(out_dir, ignore_errors=True)
    neg_dir = os.path.join(out_dir, "neg")
    pos_dir = os.path.join(out_dir, "pos")
    os.makedirs(neg_dir, exist_ok=True)
    os.makedirs(pos_dir, exist_ok=True)

    query = f"select * from lumana_prod.custom_objects_representative where customObjectId=%s"
    results = db.run_query_with_params(query, (custom_object_id,))
    print(f"Downloading {len(results)} representatives to {out_dir}")
    for row in results:
        gcp_path = f"crop/{row['orgId']}/{row['bestImage']}"
        subdir = "pos" if row["isPositive"] == 1 else "neg"
        target_path = os.path.join(out_dir, subdir, row["bestImage"])
        try:
            gcp.download_single_file_s3(gcp_path, target_path)
        except Exception as e:
            # print(f"Failed to download {gcp_path}: {e}")
            print(target_path)


def download_free_text_vids(target_dir, prompt: str, limit=1000, image_only: bool = False, filters=None):
    from general.clip_encoder import ClipVisionEncoder
    import numpy as np
    from time import time

    is_serial = False
    os.makedirs(target_dir, exist_ok=True)

    cache_file = prompt.replace(" ", "_") + "_query_results.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            results1 = json.load(f)
    else:
        weight_path = Path(os.getcwd()).parent.joinpath("weights/all/clip/clip_ViT-B-16-SigLIP2-wise.pt")
        clip = ClipVisionEncoder({"force_full": True, "weights": str(weight_path)}, is_local=True)

        vectors = clip.encode_text([prompt])
        prompt_hex = vectors[0].astype(np.float32).tobytes().hex()
        filters_str = ""
        if filters is not None:
            if isinstance(filters, str):
                filters_str = f"AND {filters} "

        query1 = f"""
            SELECT
                ct.trackerId, ct.cameraId, ct.edgeId,
                ct.vecClip <*> UNHEX('{prompt_hex}') AS score
            FROM clips_trackers ct
            WHERE ct.vecClip IS NOT NULL
            AND score > 0.1 {filters_str}
            order by score desc limit {limit};
        """

        s = time()
        results1 = db.run_query(query1)
        print(f"Found {len(results1)} clips in {time() - s:.2f} seconds")

    results = []
    batch_size = 100
    for i in range(0, len(results1), batch_size):
        batch = results1[i : i + batch_size]
        tracker_ids = tuple(row["trackerId"] for row in batch)
        if len(tracker_ids) == 1:
            tracker_ids = f"('{tracker_ids[0]}')"
        query2 = f"SELECT * FROM trackers2 WHERE id IN {tracker_ids}"
        s = time()
        results2 = db.run_query(query2)
        print(f"Batch {i // batch_size + 1}: Got {len(results2)} bestImages in {time() - s:.2f} seconds")

        if is_serial:
            for res in results2:
                download_video_image_from_tracker(res, target_dir, ts_offset=5, no_vid=image_only)
        else:

            # Download concurrently
            def download_by_index(index):
                download_video_image_from_tracker(results2[index], target_dir, ts_offset=5, no_vid=image_only)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                list(
                    tqdm(
                        executor.map(download_by_index, range(len(results2))),
                        total=len(results2),
                        desc="Downloading clips",
                        leave=False,
                    )
                )


def download_lpr():
    local_dir = "/mnt/d/uk_lpr"
    last_ts = 1778072646528  # from previous collection
    query = """select edgeId, cameraId, zoomImage, timestamp, attributes from trackers2 
               where orgIdHash = 4218568786730365 and  licensePlate is not NULL 
                 and zoomImage is not null and timestamp > %s order by timestamp asc limit 50000"""
    os.makedirs(local_dir, exist_ok=True)

    results = db.run_query_with_params(query, (last_ts,))
    while results:

        def get_file_label_by_index(index):
            res = results[index]
            eid = res["edgeId"]
            cid = res["cameraId"]
            img = str(res["zoomImage"])
            full_path = os.path.join(local_dir, img)
            try:
                if not os.path.exists(full_path):
                    crop_gcs.download_single_file_s3(str(PurePosixPath("crop").joinpath(eid, cid, img)), full_path)
                if os.path.exists(full_path):
                    json_path = Path(full_path).with_suffix(".json")
                    with open(json_path, "wt") as f:
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

        last_ts = int(results[-1]["timestamp"])
        results = db.run_query_with_params(query, (last_ts,))


if __name__ == "__main__":
    # download_faces()
    # download_alerts_videos_by_event_id("69108ef64e79db003fe95124")  # washing hands
    # download_alerts_videos_by_event_id("692ffef08fe9cb0f6cfd6b2a", ts_offset=5, duration=10)  # shoplifting

    # download_vcc_vids(11, ts_offset=5, duration=13)  # violence
    # download_vcc_vids(10, ts_offset=5, duration=13)  # violence
    # cam_id = "6847159dc890df1f03965f58"
    # local_dir = os.path.join("/mnt/d/event_clips", cam_id)
    # download_vcc_vids(
    #     {"cameraId": cam_id, "type": 7, "verified": 1}, local_dir=local_dir, ts_offset=5, duration=13, limit=100
    # )  # shoplifting
    # download_vcc_vids(
    #     {"cameraId": cam_id, "type": 7, "verified": 0}, local_dir=local_dir, ts_offset=5, duration=13, limit=100
    # )  # shoplifting

    # download_vcc_images(9)
    # download_vcc_images(3)
    # download_vcc_images(4000000, filter_cores=["681e4c710ca268f5431ee376"])
    # redownload_vids()
    # download_alerts_thumbnails("/mnt/d/alerts_thumbnails/lpc/", {"eventId": "6925a51e126ac5a50a5f2f54"}, days_before=10)
    # download_custom_obj_representatives("3377699728000002", "/mnt/d/custom_objects/")
    # download_free_text_vids(
    #     "/mnt/d/horse",
    #     "horse",
    #     limit=500,
    #     image_only=True,
    #     filters="orgIdHash = 3108480583228308 AND idBase > 1773327600000 AND idBase < 1773813600000",
    # )
    # download_lpr()
    download_alerts_thumbnails("/mnt/d/weapons_2026_20_05/", {"eventTypeId": [3, 13]}, days_before=30, max_res=50000)
