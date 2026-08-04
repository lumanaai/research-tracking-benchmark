import os
from typing import List, Dict

import requests
import json
from pathlib import PurePosixPath

from scripts.helpers import datetime_to_timestamp, training_bucket
from utils.infra.s3_io import S3Handler


def sql_request(from_time, to_time, req, gear):
    url = f"https://api.lumix.ai/search/analytics?page=0&size=10000&sort=-1&start={from_time}&end={to_time}"
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InczcGdZVFUxZlNaYXJ3c0FIeUM3RiJ9.eyJodHRwczovL2FwaS5sdW1peC5haS9vcmdhbml6YXRpb24vcm9sZXMiOlsib3duZXIiXSwiaHR0cHM6Ly9hcGkubHVtaXguYWkvb3JnYW5pemF0aW9uL29yZ0lkIjoiNjRhMzBlZDBkMGYyMmUyOWQ0MDJkZTkyIiwiaHR0cHM6Ly9hcGkubHVtaXguYWkvb3JnYW5pemF0aW9uL3N1cGVyIjpbInVzZXIiLCJhZG1pbiIsImRldmVsb3BlciJdLCJnaXZlbl9uYW1lIjoiTHVtaXgiLCJmYW1pbHlfbmFtZSI6Ikx1bWl4Iiwibmlja25hbWUiOiJzdXBwb3J0IiwibmFtZSI6InN1cHBvcnRAbHVtaXguYWkiLCJwaWN0dXJlIjoiaHR0cHM6Ly9zLmdyYXZhdGFyLmNvbS9hdmF0YXIvYWJlYjU0YWE5MmNjMmU4NDUwZGNkMzEyYTQ0OTExMWM_cz00ODAmcj1wZyZkPWh0dHBzJTNBJTJGJTJGY2RuLmF1dGgwLmNvbSUyRmF2YXRhcnMlMkZzdS5wbmciLCJ1cGRhdGVkX2F0IjoiMjAyMy0xMC0yM1QxNDo1MzoxOC4zODhaIiwiaXNzIjoiaHR0cHM6Ly9sdW1peGFpcHJvZC51cy5hdXRoMC5jb20vIiwiYXVkIjoiYWVUU3F0R05tcG0yVkQ0OW95MGV4OVVEdlgxY2tadHoiLCJpYXQiOjE2OTgwNzI4MjMsImV4cCI6MTcwMDY2NDgyMywic3ViIjoiYXV0aDB8NjNmNjY3MWY1MjE5OWIxODg2ZmIwYWMxIiwiYXRfaGFzaCI6ImVRT2szSnpndExNY1FRaDZHU3Y2amciLCJzaWQiOiJ4LXNiUEFnRXpoQzNUUWhUUVR2eFBzWXB4M2VCLTNsTCIsIm5vbmNlIjoiMWl0NzA3Z29Za0Jpb1k3QnV2bzFFd0xrTUdRbzJTNjAifQ.b1Rb2xjsUH5glesyyaNOBO4_E8X2-iuj5Ra9LNwjLpJQQo1POAPsW-kzEfWeISgno84awhD-O8aUp8-vqM6RhBjjksHwOmyIk4-pDnoHWjMP639jKz4wVS4JgWWnvUSI3o_HRvCUHfF4EnxBwW6l5tt58MF11G6o4bVZbF6V61-qIGj13DAsP1L66OEY6uHKlIGGmc9WL1W3xgKTTzaa6pToRMXVywNJFStXFMmiv_0CmzwTcP8nM8ha1pHd_iiAjmNqYTAMCyK9mDCUHtiJstvbOnUNd9VKUk4UaIfQM8U3_VIy4dTGO8W8ozx3o5lwPyuh2nkh4c7TfzxG4mK3FA",
        "Content-Type": "application/json",
        "Referer": "https://app.lumix.ai/",
        "Sec-Ch-Ua": '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    }

    data = {
        "searchSelections": [
            {
                "type": "person",
                "operator": "AND",
                "properties": [
                    {"name": "genderType", "enabled": False, "value": [], "colors": []},
                    {"name": "ageType", "enabled": False, "value": [], "colors": []},
                    {"name": "footwearType", "enabled": False, "value": [], "colors": []},
                    {"name": "lowerbodyType", "enabled": False, "value": [], "colors": []},
                    {"name": "upperbodyType", "enabled": False, "value": [], "colors": []},
                    {"name": "hairType", "enabled": False, "value": [], "colors": []},
                    {"name": "accessoryType", "enabled": False, "value": [], "colors": []},
                    {"name": "carryingType", "enabled": False, "value": [], "colors": []},
                    {"name": "protectivegearType", "enabled": gear, "value": [req], "colors": []},
                ],
                "groupId": None,
                "groups": None,
                "groupIdCollapsed": False,
                "enabled": True,
                "collapsed": True,
            }
        ],
        "useSingleStore": False,
        "operator": "AND",
        "cameras": [{"edgeId": "651f1051960d3e6758ccc30d", "cameraId": "6529711a39096692dfe8158c"}],
        "maxStart": 1698008400000,
        "maxEnd": 1698044400000,
        "highConfidence": True,
        "dwellRange": {"period": {"unit": "minutes", "start": 0, "end": 61}, "start": 0, "end": None},
    }

    response = requests.post(url, headers=headers, json=data)

    # Handle the response as needed:
    response_dict = json.loads(response.text)
    results = []
    objects = response_dict.get("searchObjects", [])
    for idx, obj_list in enumerate(objects):
        for obj in obj_list:
            results.append(
                {
                    "idBase": obj["idBase"],
                    "idIndex": obj["idIndex"],
                    "image": obj["url"],
                    "cam_id": response_dict["cameraIds"][idx],
                    "edge_id": response_dict["edgeIds"][idx],
                }
            )
    return results


def main():
    handler = S3Handler(bucket_name=training_bucket)

    from_time = datetime_to_timestamp(2023, 10, 20, 0, 0)
    to_time = datetime_to_timestamp(2023, 10, 26, 0, 0)
    no_hat_res = sql_request(from_time, to_time, "no_hard_hat", True)
    hat_res = sql_request(from_time, to_time, "hard_hat", True)
    all_res = sql_request(from_time, to_time, "hard_hat", False)
    undef = [r for r in all_res if r not in hat_res and r not in no_hat_res]
    print(f"number of no hats: {len(no_hat_res)}")
    print(f"number of hats: {len(hat_res)}")
    print(f"undefined: {len(undef)}")

    base_dir = "/mnt/c/Temp/ppe"

    def download_samples(results: List[Dict], local_path: str):
        os.makedirs(local_path, exist_ok=True)
        for res in results:
            image = res["image"]
            try:
                if image is not None:
                    local_file_path = os.path.join(local_path, image)
                    if not os.path.exists(local_file_path):
                        s3_path = PurePosixPath("crop").joinpath(res["edge_id"], res["cam_id"], image)
                        handler.download_single_file_s3(str(s3_path), local_file_path)
            except Exception as e:
                print(e)
        pass

    download_samples(no_hat_res, os.path.join(base_dir, "no_hard_hat"))
    download_samples(hat_res, os.path.join(base_dir, "hard_hat"))
    download_samples(undef, os.path.join(base_dir, "undefined"))


if __name__ == "__main__":
    main()
