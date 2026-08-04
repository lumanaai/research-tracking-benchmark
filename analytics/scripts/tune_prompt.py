import base64
import concurrent
import io
import json
import os
import shutil
import time
from pathlib import PurePosixPath, Path

import cv2
from PIL import Image
from openai import OpenAI
from pydantic import ConfigDict, BaseModel
from tqdm import tqdm

from infra.GcpHandler import GcpStorageHandler
from utils.infra.credentials import credentials_file_to_env
from utils.infra.singlestore_io import SingleStoreDb

credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"), env="prod")
root_dir = "/mnt/d/carfour/"
gcs = GcpStorageHandler("lumixai-alert-thumbnails")
client = OpenAI(base_url="http://192.168.101.27:8000/v1", api_key="none", timeout=20.0)
# client = OpenAI()
model = "gpt-5.1"


def download_vcc_images(query_params: dict, path: str = None, days_back: int = 10, limit=1000):
    db = SingleStoreDb()

    if path is None:
        local_dir = os.path.join(root_dir)
    else:
        local_dir = path
    if query_params:
        filters = " AND ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in query_params.items()])
        filters = " AND " + filters
    else:

        filters = ""

    query = f"""
        select * from vcc_log where timestamp > CURRENT_DATE - INTERVAL {days_back} DAY
        {filters} 
        order by timestamp desc limit {limit}
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


def encode_image2(image_path: str):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    with io.BytesIO() as bio:
        img = Image.fromarray(img)

        try:
            img.save(bio, format="JPEG")
        finally:
            img.close()
        return base64.b64encode(bio.getvalue()).decode("utf-8")


class CoTStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    final_answer: bool


def send_to_verifier(image_path: str, prompt: str, details="low", use_cot=False):
    url = f"data:image/jpeg;base64,{encode_image2(image_path)}"
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": url, "detail": details},
            },
        ],
    }
    if use_cot:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[message],
            response_format=CoTStructuredOutput,
            max_completion_tokens=100,
        )
        parsed = completion.choices[0].message.parsed
        return parsed.final_answer
    response = client.chat.completions.create(
        model=model,
        messages=[message],
        max_completion_tokens=50,
    )
    resp = response.choices[0].message.content.lower()
    return "yes" in resp[:5], resp


if __name__ == "__main__":
    cam_id = "6847159dc890df1f03965f58"
    db_path = os.path.join(root_dir, cam_id)
    search = """person with products, shopping bags, shopping cart with items or anything that should be declared in a supermarket register and that doesnt seems like he is going to declare them? 
    answer for the person in the middle of the frame only. 
    if the person seems to stay at the register, or carry empty bags or personal stuff only answer no. also answer no if it seems they are interacting with the registry or cashier.
    is there such a person"""
    prompt = (
        f"does this image contains {search}? Answer yes or no only. then follow up with an explanation why is it so"
    )
    do_neg = True
    need_download = False
    request_delay = 3
    if need_download:

        download_vcc_images(
            {"cameraId": cam_id, "type": 7, "verified": 1},
            path=os.path.join(root_dir, cam_id, "verified"),
            days_back=30,
        )
        download_vcc_images(
            {"cameraId": cam_id, "type": 7, "verified": 0},
            path=os.path.join(root_dir, cam_id, "non_verified"),
            days_back=30,
            limit=100,
        )
    pos_files = list(Path(os.path.join(db_path, "verified")).rglob("*.jpg"))
    pos_res = {True: [], False: []}
    print("Starting positive files verification...")
    for file in pos_files:
        answer, explanation = send_to_verifier(str(file), prompt, details="low")
        if not answer:
            print(f"File: {file}: {explanation}")

        pos_res[answer].append(file)
        time.sleep(request_delay)  # to avoid rate limit

    neg_files = list(Path(os.path.join(db_path, "non_verified")).rglob("*.jpg"))
    neg_res = {True: [], False: []}
    if do_neg:
        print("Starting negative files verification...")
        for file in neg_files:
            answer, explanation = send_to_verifier(str(file), prompt, details="low")
            if answer:
                print(f"File: {file}: {explanation}")
            neg_res[answer].append(file)
            time.sleep(request_delay)  # to avoid rate limit

    print("Summary - confusion matrix:")
    print(f"True Positives: {len(pos_res[True])}")
    print(f"False Negatives: {len(pos_res[False])}")
    print(f"False Positives: {len(neg_res[True])}")
    print(f"True Negatives: {len(neg_res[False])}")

    prompt_str = search.replace(" ", "_").replace("?", "")[:50]

    res_dir_fp = os.path.join(db_path, "results", prompt_str, "false_positives")
    res_dir_fn = os.path.join(db_path, "results", prompt_str, "false_negatives")
    os.makedirs(res_dir_fp, exist_ok=True)
    os.makedirs(res_dir_fn, exist_ok=True)
    for file in pos_res[False]:
        shutil.copyfile(file, os.path.join(res_dir_fn, os.path.basename(file)))
    for file in neg_res[True]:
        shutil.copyfile(file, os.path.join(res_dir_fp, os.path.basename(file)))
