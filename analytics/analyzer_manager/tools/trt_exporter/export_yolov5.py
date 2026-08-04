import argparse
import json
import os
import pathlib
import shutil
from argparse import Namespace
from detection.yolov5 import export_tensorrt

yolo_default_params = {"image_sz": [384, 640], "batch_sz": 4, "half": True, "workspace": 2, "verbose": False,
                       "weights": ""}


def load_params(param_file: str = None, default_params: dict = {}) -> Namespace:
    params = default_params
    if param_file and os.path.exists(param_file):
        with open(param_file, "rt") as f:
            param_data = json.load(f)
        params.update(param_data)
    return Namespace(**params)


def main(model_path, export_path: str = None):
        params = Namespace(**yolo_default_params)
        params.weights = model_path
        out_file = export_tensorrt.run(params)
        if out_file is not None and export_path is not None:
            outfile_name = pathlib.Path(out_file).name
            shutil.copy2(out_file, os.path.join(export_path, outfile_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str)
    parser.add_argument("-o", "--output", type=str, default=None, help="output path")
    opt = parser.parse_args()
    main(opt.weights, opt.output)

