# to be tested from inside the docker to make sure everything is working
import os
import time

import cv2
from pprint import pprint

from general.core import EndpointFactory
from general.proj import project_path, load_config
from general.offline_analytics import (
    encode_image_to_base64,
)
import sys
import multiprocessing as mp

from general.triton_utils import is_triton_live

sys.path.append(str(os.path.join(project_path(), "analyzer_manager", "offline", "app")))

from weapon_worker import weapon_task, WeaponPayload


def main():
    app_config = load_config()
    inference_server_settings = app_config.get("analytics", {}).get("inferenceServerUri", {})
    endpoint_factory = EndpointFactory(inference_server_settings)
    mock_endpoint = endpoint_factory.create("")
    if is_triton_live(mock_endpoint):
        print("Triton connection successful.")
    else:
        print("Triton connection failed. Exiting worker.")
        print("endpoint: ", mock_endpoint)
        raise Exception("Triton connection failed. Exiting worker.")


    out_queue = mp.Queue()
    payload = WeaponPayload(
        camera_id="test_camera",
        image=encode_image_to_base64(cv2.imread("snapshot_1.png")),
        batch_data=[[0.1,0.1,0.9,0.9,0.9,0]],
        timestamp=1200000000,
    )
    state_dict = {}
    weapon_task(payload, state_dict, out_queue)
    time.sleep(2)
    alert = out_queue.get_nowait()
    pprint(alert)


if __name__ == "__main__":
    main()
