import time
from pathlib import Path, PosixPath

from infra.confluent_kafka_io import KafkaHandler
from infra.credentials import credentials_file_to_env
from infra.s3_io import S3Handler
from infra.singlestore_io import SingleStoreDb

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"), env="prod")


def send_pf_kafka(kafka_handler, camera_res: dict, filename: str) -> bool:

    topic = "proper_fitting"

    current_timestamp = int(time.time() * 1000)  # utc timestamp
    message = {
        "orgId": camera_res["orgId"],
        "locationId": camera_res["locationId"],
        "edgeId": camera_res["edgeId"],
        "cameraId": camera_res["cameraId"],
        "type": 0,
        "filename": filename,
        "timestamp": current_timestamp,
    }

    key = f"{message['orgId']}-{message['edgeId']}-{message['cameraId']}-{current_timestamp}"
    try:
        kafka_handler.send(key, message, topic=topic)
        print("sent pf message to kafka for cameraId: " + camera_res["cameraId"])
        return True
    except Exception as e:
        print(f"error occurred when sending pf message to kafka: {str(e)}")
        return False


def resend_pf_files():
    db = SingleStoreDb()
    kafka_handler = KafkaHandler(mode="producer")

    query = """
            select * from camera_training_metadata where
                (cloudRecievedTs is NULL or edgeStatus = 0) and lastPfFile is not NULL and edgeVersion <> 'L4' and
                lastPfTime > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 45 DAY) * 1000 and
                lastPfTime < UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 3 HOUR ) * 1000
            """
    results = db.run_query(query)
    for camera_res in results:
        filename = Path(camera_res["lastPfFile"]).stem + f"_{camera_res['edgeArchitecture']}.tar.gz"
        send_pf_kafka(kafka_handler, camera_res, filename)


def resend_door_files():
    db = SingleStoreDb()
    kafka_handler = KafkaHandler(mode="producer")
    s3 = S3Handler(bucket_name="lumixai-camera-assets")
    query = "select * from doors where timestamp  < UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 1 HOUR)*1000 and timestamp  > UNIX_TIMESTAMP(CURRENT_DATE - INTERVAL 30 DAY)*1000 order by timestamp desc"
    results = db.run_query(query)
    for door_res in results:
        s3_path = str(PosixPath(door_res["edgeId"]).joinpath(door_res["cameraId"]))
        assets = s3.list_files_in_folder(s3_path, name_filter="door*")
        if assets:
            send_pf_kafka(kafka_handler, door_res, Path(assets[-1]).name)


if __name__ == "__main__":
    resend_door_files()
