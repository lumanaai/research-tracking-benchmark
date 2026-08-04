import argparse
import json
import os
import pathlib
import posixpath
import sys
from copy import deepcopy

import trt_exporter
from export_common import in_docker, export_dir, determine_host
from general.analyzer_general import InferenceType

if not in_docker:
    this_path = pathlib.Path(__file__)
    sys.path.append(os.path.join(this_path.parent.parent.parent.resolve(), "automation"))
    from utils.infra.logger import get_logger
    export_dir = os.path.join(pathlib.Path.home(), "exports")
else:
    from utils.infra.logger import get_logger

from utils.infra.s3_io import S3Handler


def get_command_line_args():
    parser = argparse.ArgumentParser(
        prog="ProperFittingTrainer", description="Searches s3 for devices and performs proper fitting"
    )
    parser.add_argument("-b", "--batch_sz", type=int, default=None, help="batch size to be used")
    parser.add_argument("-v", "--verbose", type=bool, default=None, help="verbosity of export procedure")
    parser.add_argument("-ha", "--half", type=bool, default=None, help="use fp16 instead of fp32")
    parser.add_argument("-j", "--jobs_number", type=int, default=-1, help="number of jobs to process")

    return parser.parse_args()

log = None
if __name__ == "__main__":
    if not in_docker:
        log = get_logger(report_dir=export_dir, log_name="pf_exporter_s3", continue_log_file=True)
        log.info("running locally")
    else:
        log = get_logger(report_dir="/logs", log_name="pf_exporter_s3", continue_log_file=True)
        log.info("running in docker")
    pf_bucket = "lumixai-properfitting-weights"
    host = str(determine_host().value)
    s3_prefix = os.path.join("awaiting_jobs", host)
    yolo_params_name = "yolov5_export_params.json"
    opts = get_command_line_args()
    yolo_export_params_path = os.path.join(pathlib.Path(__file__).parent, yolo_params_name)
    with open(yolo_export_params_path, "rt") as f:
        yolo_export_params = json.load(f)

    opts_dict = vars(opts)

    for opt in opts_dict:
        if opts_dict[opt] is not None:
            yolo_export_params[opt] = opts_dict[opt]

    os.makedirs(export_dir, exist_ok=True)
    s3_handler = S3Handler(bucket_name=pf_bucket)
    awaiting_jobs = s3_handler.list_files_in_folder(s3_prefix, name_filter="*.pt")
    if len(awaiting_jobs) == 0 :
        log.info("no jobs - exiting")
        exit(0)

    if opts.jobs_number > 0:
        awaiting_jobs = awaiting_jobs[:opts.jobs_number]
    for job in awaiting_jobs:
        job_path = pathlib.Path(job)
        job_local_path = os.path.join(export_dir, job_path.name)
        s3_handler.download_single_file_s3(job, job_local_path)
        json_path = job_path.with_suffix(".json")
        json_local_path = os.path.join(export_dir, json_path.name)
        s3_handler.download_single_file_s3(str(json_path), json_local_path)
        with open(json_local_path, "rt") as f:
            job_dict = json.load(f)
        current_params = deepcopy(yolo_export_params)
        detector_params = job_dict.get("detector_config", {}).get("l0_detect", {})
        if "imgsz" in detector_params:
            current_params["image_sz"] = detector_params["imgsz"]

        params = argparse.Namespace(**current_params)
        params.weights = job_local_path

        # run the conversion
        if job_path.name.startswith("yolov8"):
            from export_yolov8 import export_model
            result_file = export_model(params)
        elif job_path.name.startswith("yolov5"):
            from detection.yolov5 import export_tensorrt
            result_file = export_tensorrt.run(params)
            pass
        else:
            pref = job_path.name.split("_")[0]
            if pref in trt_exporter.TRT_SUPPORTED_ARCH:
                result_file, _ = trt_exporter.main(["-w", job_local_path, "-v", str(opts.verbose)])
            elif pref.startswith(InferenceType.STATE_OBJECT.value):
                model_info = job_dict.get("model_params", {})
                result_file, _ = trt_exporter.main(["-w", job_local_path, "-v", str(opts.verbose)], model_info)
            else:
                log.warning(f"unsupported weights file {job_path.name}")
                continue

        # if result valid, upload to assets
        if result_file is None:
            log.error(f"failed to export {job_path.name}")
        else:
            log.info(f"successfully export {job_path.name}")
            # upload assets
            result_name = pathlib.Path(result_file).stem + f"_{host}.engine"
            result_json = pathlib.Path(result_name).with_suffix(".json").name

            s3_handler.upload_file(result_file, posixpath.join("completed_jobs", result_name))
            s3_handler.upload_file(json_local_path, posixpath.join("completed_jobs", result_json))

            # remove job from awaiting jobs
            s3_handler.remove_objects([job, str(json_path)])

            log.info("job removed from awaiting jobs")
