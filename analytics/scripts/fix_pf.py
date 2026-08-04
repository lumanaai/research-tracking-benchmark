import csv
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PosixPath

from tqdm import tqdm

from analyzer_manager.cuda_library.build_cuda import print_subprocess_results
from infra.s3_io import S3Handler
from utils.infra.credentials import credentials_file_to_env
from utils.infra.singlestore_io import SingleStoreDb

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"), env="prod")
db = SingleStoreDb()
root_dir = Path("/mnt/d/pf_fix_dir")
s3_pf = S3Handler(bucket_name="lumixai-properfitting-weights")
s3_assets = S3Handler(bucket_name="lumixai-camera-assets")
playbook_root = "/home/aviadz/sources/playbook"
nas_base = Path("/mnt/y/proper_fitting/prod")


def fix_arch_issue():
    os.makedirs(str(root_dir), exist_ok=True)
    bad_pf_query = """
        SELECT edgeId, cameraId, lastPfFile, edgeVersion from camera_training_metadata where
        lastPfFile is not NULL  and 
        ((edgeVersion in ("L4", "6.1.0") and edgeArchitecture= "orin-jp5") or edgeId="698999abbebd30693c8deba6")
        """
    result_table = []
    results = db.run_query(bad_pf_query)
    for result in tqdm(results):
        edge_id = result["edgeId"]
        camera_id = result["cameraId"]
        pf_file = result["lastPfFile"]
        true_hw = "l4" if result["edgeVersion"] == "L4" else "orin-jp6"
        target_pf = root_dir.joinpath(pf_file)
        target_json = target_pf.with_suffix(".json")

        file_ok = target_pf.is_file() and target_json.is_file()
        if not file_ok:
            try:
                # attempt to download from NAS:
                nas_file = nas_base.joinpath(edge_id, camera_id, pf_file)
                if nas_file.is_file():
                    shutil.copy2(nas_file, target_pf)
                    shutil.copy2(nas_file.with_suffix(".json"), target_json)
                    file_ok = True
            except Exception as e:
                print(f"NAS: failed to copy {nas_file} with error {e}")

        if not file_ok:
            # download file from s3 assets
            asset_path = PosixPath(edge_id, camera_id, pf_file)
            try:
                s3_assets.download_single_file_s3(str(asset_path), str(target_pf))
                s3_assets.download_single_file_s3(str(asset_path.with_suffix(".json")), str(target_json))
                file_ok = True
            except Exception as e:
                print(f"S3: failed to download {asset_path} with error {e}")
        pf_uploaded = False
        if file_ok:
            try:
                # first fix json - read and replace the "edge_hw" field
                with open(target_json, "r") as f:
                    json_data = json.load(f)
                json_data["edge_hw"] = true_hw
                with open(target_json, "w") as f:
                    json.dump(json_data, f, indent=4)
                # then upload the pf file to the correct location in s3_pf
                target_s3 = PosixPath("awaiting_jobs", true_hw, pf_file)
                s3_pf.upload_file(str(target_pf), str(target_s3))
                s3_pf.upload_file(str(target_json), str(target_s3.with_suffix(".json")))
                pf_uploaded = True
            except Exception as e:
                print(f"S3: failed to upload {target_pf} with error {e}")
        # last part is to delete the file from edge
        is_deleted = False
        try:
            edge_path = Path("assets").joinpath(edge_id, camera_id, "*.engine")
            cmd = ["./xterm.sh", "-t", "10000", "-p", "-e", edge_id, "-c", f"rm {str(edge_path)}"]
            p = subprocess.run(args=cmd, cwd=playbook_root)
            print_subprocess_results(p)
            is_deleted = True
        except Exception as e:
            print(f"Xterm: failed to delete {edge_path} with error {e}")

        output_results = result.copy()
        output_results["true_hw"] = true_hw
        output_results["downloaded"] = file_ok
        output_results["uploaded"] = pf_uploaded
        output_results["deleted"] = is_deleted
        result_table.append(output_results)

    # print result table to csv
    if result_table:
        csv_path = root_dir / "fix_pf_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
            writer.writeheader()
            writer.writerows(result_table)
        print(f"Results saved to {csv_path}")


def retry_failed_downloads():
    """
    Read the results CSV from fix_arch_issue, iterate over entries where 'downloaded' is False,
    and attempt to recover the .pt + .json files via S3 (or tar.gz fallback).
    """
    csv_path = root_dir / "fix_pf_results.csv"
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}")
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    result_table = []
    for row in tqdm(rows):
        if row.get("downloaded", "").strip().lower() == "true":
            result_table.append(row)
            continue

        edge_id = row["edgeId"]
        camera_id = row["cameraId"]
        pf_file = row["lastPfFile"]
        true_hw = row["true_hw"]

        target_pf = root_dir / pf_file
        target_json = target_pf.with_suffix(".json")

        file_ok = target_pf.is_file()
        json_ok = target_json.is_file()

        # Step 1: try downloading the .pt from s3 assets (if not already local)
        if not file_ok:
            asset_path = str(PosixPath(edge_id, camera_id, pf_file))
            try:
                s3_assets.download_single_file_s3(asset_path, str(target_pf))
                file_ok = True
                print(f"  Downloaded .pt from S3: {asset_path}")
            except Exception as e:
                print(f"  S3 .pt download failed for {asset_path}: {e}")

        # Step 2: try downloading the tar.gz variant to extract the json
        if not json_ok:
            tar_name = Path(pf_file).stem + "_orin-jp5.tar.gz"
            tar_s3_path = str(PosixPath(edge_id, camera_id, tar_name))
            tar_local = root_dir / tar_name
            try:
                s3_assets.download_single_file_s3(tar_s3_path, str(tar_local))
                print(f"  Downloaded tar.gz from S3: {tar_s3_path}")
                # extract only the json from the tar
                json_filename = Path(pf_file).with_suffix(".json").name
                with tarfile.open(tar_local, "r:gz") as tar:
                    members = tar.getnames()
                    json_member = next((m for m in members if m.endswith(".json")), None)
                    if json_member:
                        extracted = tar.extractfile(json_member)
                        if extracted:
                            with open(target_json, "wb") as jf:
                                jf.write(extracted.read())
                            json_ok = True
                            print(f"  Extracted JSON: {json_member}")
                    else:
                        print(f"  No .json found inside {tar_name}. Members: {members}")
                # clean up tar
                tar_local.unlink(missing_ok=True)
            except Exception as e:
                print(f"  S3 tar.gz download failed for {tar_s3_path}: {e}")

        # Step 3: fix json and upload
        pf_uploaded = False
        if file_ok and json_ok:
            try:
                with open(target_json, "r") as f:
                    json_data = json.load(f)
                json_data["edge_hw"] = true_hw
                with open(target_json, "w") as f:
                    json.dump(json_data, f, indent=4)

                target_s3 = PosixPath("awaiting_jobs", true_hw, pf_file)
                s3_pf.upload_file(str(target_pf), str(target_s3))
                s3_pf.upload_file(str(target_json), str(target_s3.with_suffix(".json")))
                pf_uploaded = True
                print(f"  Uploaded to S3 pf: {target_s3}")
            except Exception as e:
                print(f"  Upload failed for {pf_file}: {e}")

        row["downloaded"] = file_ok and json_ok
        row["uploaded"] = pf_uploaded
        result_table.append(row)

    # overwrite csv with updated results
    if result_table:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
            writer.writeheader()
            writer.writerows(result_table)
        print(f"Updated results saved to {csv_path}")


def upload_flir_model_0():
    """
    For each camera from the query:
    1. Look for an existing .pt file in base_dir whose name contains the cameraId.
       If found, use it; otherwise copy the template .pt.
       Target name format: yolov8s-flir_14cls_{cameraId}_{date}.pt
    2. Create a .json sidecar (copy of template json) with edge_id, camera_id, and hw fields updated.
    3. Upload both .pt and .json to s3_pf under awaiting_jobs/<hw>/
    """
    from datetime import date

    base_dir = Path("/mnt/d/flir/pf")
    template_pt = base_dir / "yolo8s-flir-14cls_1_2.pt"
    template_json = base_dir / "yolo8s-flir-14cls_1_2.json"
    query = "select edgeId, cameraId, edgeVersion from camera_training_metadata where cameraMode = 1 and orgId = '676a5c6f57f077ed52f6a19a'"

    results = db.run_query(query)
    today_str = date.today().strftime("%Y_%m_%d")

    for result in tqdm(results):
        edge_id = result["edgeId"]
        camera_id = result["cameraId"]
        if result["edgeVersion"] == "L4":
            hw = "l4"
        elif result["edgeVersion"].startswith("6.1"):
            hw = "orin-jp6"
        elif result["edgeVersion"].startswith("5.1"):
            hw = "orin-jp5"

        # Target filename
        target_name = f"yolov8s-flir_14cls_{camera_id}_{today_str}.pt"
        target_pt = base_dir / target_name

        # Step 1: find existing .pt with cameraId in name, or copy template
        existing = [f for f in base_dir.glob("*.pt") if camera_id in f.name]
        if existing:
            # use the first match
            target_pt = Path(existing[0])
            print(f"  Using existing pt: {target_pt}")
        else:
            shutil.copy2(template_pt, target_pt)
            print(f"  Copied template pt -> {target_pt}")

        # Step 2: prepare json - copy template and update fields
        target_json = target_pt.with_suffix(".json")
        target_s3 = PosixPath("awaiting_jobs", hw, target_name)

        shutil.copy2(template_json, target_json)
        with open(target_json, "r") as f:
            json_data = json.load(f)
        json_data["edge_id"] = edge_id
        json_data["cam_id"] = camera_id
        json_data["edge_hw"] = hw
        with open(target_json, "w") as f:
            json.dump(json_data, f, indent=4)

        # Step 3: upload to s3 pf bucket
        try:
            s3_pf.upload_file(str(target_pt), str(target_s3))
            s3_pf.upload_file(str(target_json), str(target_s3.with_suffix(".json")))
            print(f"  Uploaded: {target_s3}")
        except Exception as e:
            print(f"  Upload failed for {target_name}: {e}")


if __name__ == "__main__":
    # fix_arch_issue()
    # retry_failed_downloads()
    upload_flir_model_0()
