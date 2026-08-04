import json
from typing import Dict

import numpy as np
import tensorrt as trt

from general.analyzer_general import logger as analyzer_logger
from general.trt_utils import TrtInfer


class DetectorTrt(TrtInfer):

    input_name = "images"
    output_name = "output"

    def __init__(self, weights, input_name: str = None, max_size_allowed: Dict = None):
        super().__init__(weights, input_name, max_size_allowed)
        if self.is_dynamic:
            self.context.set_binding_shape(0, self.bindings[self.input_name].data.shape)

    def infer_directly(self):
        self.context.execute_v2(list(self.binding_addrs.values()))
        return self.bindings[self.output_name].data.get()

    @property
    def output_buffer(self):
        return self.bindings[self.output_name].data

    def infer(self, images):
        if self.is_dynamic:
            predictions = super().infer(images)
        else:
            predictions = self.infer_directly()
        import torch

        return torch.from_numpy(np.asarray(predictions))

    def __post_init__(self):
        self.concat_func = lambda x: np.concatenate(x, axis=0)


class DetectorTrtV8(DetectorTrt):
    output_name = "output0"

    def _read_model_file(self, weights):
        from general.trt_utils import logger as trt_logger

        # yolo v8 regular export has metadat on it, ours doesn't - support both options
        try:
            analyzer_logger.info("loading yolov8 engine file. this might fail if the export wasn't done lately")
            model = super()._read_model_file(weights)
        except Exception:
            analyzer_logger.warning("loading yolov8 engine file failed. trying to read the file in deprecated form")
            model = None
        if model is None:
            with open(weights, "rb") as f, trt.Runtime(trt_logger) as runtime:
                meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                model = runtime.deserialize_cuda_engine(f.read())  # read engine
                print(metadata)
            if model is not None:
                analyzer_logger.info("loading yolov8 engine file in deprecated form succeeded")
            else:
                analyzer_logger.error("loading yolov8 engine file in deprecated form failed as well and cannot be used")
        return model
