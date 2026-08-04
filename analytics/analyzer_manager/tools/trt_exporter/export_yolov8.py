import argparse
import os
from typing import Optional
from pathlib import Path
from ultralytics import YOLO

from general import proj


def trim_v8(weights, file_out):
    with open(weights, "rb") as f, open(file_out, "wb") as t:
        meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
        f.read(meta_len)  # read metadata
        model = f.read()  # read engine
        t.write(model)


def export_model(opts) -> Optional[str]:
    weights = None
    # Load a model
    model = YOLO(str(opts.weights))
    is_half = not opts.float if "float" in opts else opts.half
    output_path = None if "output" not in opts else opts.output
    try:
        # Export the model
        weights = model.export(
            format="engine",
            half=is_half,
            batch=abs(opts.batch_sz),
            imgsz=opts.image_sz,
            workspace=opts.workspace,
            device="0",
            dynamic=opts.batch_sz < 0
        )
    except Exception as e:
        print(e)
    out_file = None
    if weights is not None:
        if output_path is None:
            out_file = weights
            weights = out_file + ".orig"
            os.rename(out_file, weights)
        else:
            out_file = os.path.join(opts.output, Path(weights).name)

        if os.path.isfile(out_file):
            os.remove(out_file)
        trim_v8(weights, out_file)
        os.remove(weights)
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, default="yolov8s.pt", help="model.pt path")
    parser.add_argument("-sz", "--image-sz", nargs="+", type=int, default=[384, 640], help="image (h, w)")
    parser.add_argument("-b", "--batch-sz", type=int, default=4, help="batch size")
    parser.add_argument("-fp32", "--float", action="store_true", help="fp32 engine instead of fp16")
    parser.add_argument("-ws", "--workspace", type=int, default=2, help="TensorRT workspace size (GB)")
    parser.add_argument("-o", "--output", type=str, default=None, help="output folder path")
    parser.add_argument("-t", "--test", action="store_true", help="run test with default parameters")
    opt = parser.parse_args()
    if opt.test:
        opt.weights = os.path.join(proj.project_path(), "weights", "all", "yolov8", "yolov8m-expert-1_2.pt")
        # opt.weights = os.path.join(proj.project_path(), "weights", "all", "yolov8", "yolov8s-31cls_1_1.pt")
        # opt.weights = os.path.join(proj.project_path(), "weights", "all", "expert-detect", "yolov8m-guns-1280_1_1.pt")
        opt.output = None
        opt.float = False
        opt.batch_sz = 4
        opt.image_sz = [384, 640]
    export_model(opt)
