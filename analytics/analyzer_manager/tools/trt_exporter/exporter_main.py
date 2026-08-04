import json
import os
import pathlib
import sys
import glob
from argparse import Namespace

in_docker = True if os.getenv("IN_DOCKER") and os.getenv("IN_DOCKER") == "True" else False
export_dir = "/exports"

if not in_docker:
    this_path = pathlib.Path(__file__)
    sys.path.append(os.path.join(this_path.parent.parent.parent.resolve(), "app"))
    sys.path.append(os.path.join(this_path.parent.parent.parent.resolve(), "app", "level1"))
    export_dir = os.path.join(pathlib.Path.home(), "exports")

yolo_default_params = {"image_sz": [384, 640], "batch_sz": 8, "half": True, "workspace": 2, "verbose": False,
                       "weights": ""}


def load_params(param_file: str = None, default_params: dict = {}) -> Namespace:
    params = default_params
    if param_file and os.path.exists(param_file):
        with open(param_file, "rt") as f:
            param_data = json.load(f)
        params.update(param_data)
    return Namespace(**params)


def main(export_path: str = export_dir):
    files = glob.glob(os.path.join(export_dir, "*.pt*"))
    for file in files:
        file_name = os.path.basename(file)
        if file_name.startswith("yolov5"):
            from detection.yolov5 import export_tensorrt
            params = load_params(os.path.join(export_path, "yolov5_export_params.json"), yolo_default_params)
            params.weights = file
            export_tensorrt.run(params)
        elif file_name.startswith("pa_net"):
            import trt_exporter as exp
            params = load_params(os.path.join(export_path, "pa_net_export_params.json"), exp.pa_net_default_params)
            params.weights = file
            exp.main(vars(params))
        elif file_name.startswith("reid"):
            import trt_exporter as exp
            params = load_params(os.path.join(export_path, "reid_export_params.json"), exp.default_reid_params)
            params.weights = file
            exp.main(vars(params))


if __name__ == "__main__":
    main()
