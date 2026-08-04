import concurrent.futures
import glob
from pathlib import Path, PosixPath

from utils.infra.GcpHandler import GcpStorageHandler
from utils.infra.credentials import credentials_file_to_env

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"))
handler = GcpStorageHandler(bucket_name="lumix-analytics-images-api")


def upload_file(handler, local_path, object_name):
    try:
        handler.upload_file(local_path, object_name)
        print(f"Uploaded: {local_path} -> {object_name}")
    except Exception as e:
        print(f"Failed to upload {local_path}: {e}")


def upload_event_vids():
    local_path = "/mnt/d/event_clips/6913498d6de0fbe7870fb327"
    files = glob.glob(f"{local_path}/*.mp4")
    # files = files[:10]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for local_path in files:
            file_name = Path(local_path).name
            object_name = str(PosixPath("encord/washing_hands").joinpath(file_name))
            futures.append(executor.submit(upload_file, handler, local_path, object_name))
        # Optionally, wait for all uploads to finish and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            future.result()


def upload_db(local_path: str, gcp_path: str, ext: str = "jpg"):
    files = glob.glob(f"{local_path}/*.{ext}")
    # files = files[:10]
    if "encord" in gcp_path:
        gcp_posix = PosixPath(gcp_path)
    else:
        gcp_posix = PosixPath(f"encord/{gcp_path}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for local_path in files:
            file_name = Path(local_path).name
            object_name = str(gcp_posix.joinpath(file_name))
            futures.append(executor.submit(upload_file, handler, local_path, object_name))
        # Optionally, wait for all uploads to finish and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            future.result()


def upload_single_file_and_presign():

    local_path = "/mnt/d/sagi.png"
    file_name = Path(local_path).name
    object_name = str(PosixPath("test_gcs").joinpath(file_name))
    handler.upload_file(local_path, object_name)
    presigned = handler.generate_presigned_url(object_name)
    print(f"Presigned URL: {presigned}")
    x = handler.get_s3_object_to_memory(object_name)
    print(len(x))


if __name__ == "__main__":
    # upload_db("/mnt/d/alerts_20251123/violence", "violence", "mp4")
    upload_db("/mnt/d/synth_guns_3", "synthetic_guns", "png")
