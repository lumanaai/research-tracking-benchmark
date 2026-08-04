import glob
import json
import os

import requests

env_str = "" # "-dev"

url = f"https://api{env_str}.lumix.ai/doors"


template = {
    "edgeId": "",
    "timestamp": 0,
    "cameraId": "",
    "data": {"type": 0, "position": [], "openState": "", "closeState": ""},
}

if __name__ == "__main__":
    doors_dir = "/mnt/d/doors_info"
    with open(os.path.join(doors_dir, f"Authorization{env_str}.txt"), "rt") as f:
        auth = f.read()
    headers = {"Authorization": auth, "Content-Type": "application/json"}

    files = glob.glob(os.path.join(doors_dir, "*.json"))
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            # Process the data as needed
            # For example, you can send it to the API
            payload = template.copy()
            payload["edgeId"] = data["edge_id"]
            payload["timestamp"] = data["utc_time"] * 1000
            payload["cameraId"] = data["cam_id"]
            payload["data"]["type"] = data["type"]
            payload["data"]["position"] = data["position"]
            payload["data"]["openState"] = data["openState"]
            payload["data"]["closeState"] = data["closeState"]
            print(f"uploading door to camera {data['cam_id']}")

            response = requests.post(url, headers=headers, json=payload)

            if response.status_code < 300:
                print("Data uploaded successfully.")
                os.rename(file, file + ".uploaded")
            else:
                print("Status Code:", response.status_code)
                print("Response Body:", response.text)
