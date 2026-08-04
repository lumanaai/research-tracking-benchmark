from collections import OrderedDict, namedtuple
from typing import Dict

import cupy as cp
import numpy as np
import tensorrt as trt

from general.analyzer_general import DEFAULT_TRT_MAX_BATCH_ALLOWED
from general.core import BaseInfer

Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

logger = trt.Logger(trt.Logger.INFO)

trt.init_libnvinfer_plugins(None, "")


class TrtInfer(BaseInfer):
    input_name = "input"

    def __init__(self, weights, input_name: str = None, max_size_allowed: Dict = None):
        if input_name is not None:
            self.input_name = input_name
        if max_size_allowed is None:
            self.max_size_allowed = {self.input_name: [DEFAULT_TRT_MAX_BATCH_ALLOWED]}
        else:
            self.max_size_allowed = max_size_allowed
        model = self._read_model_file(weights)
        bindings = OrderedDict()
        self.output_shapes = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = model.get_binding_shape(index)
            if shape[0] < 0:
                shape[0] = self.max_size_allowed[self.input_name][0]
                self.is_dynamic = True
            if len(self.max_size_allowed.get(name, [])) > 1:
                shape = self.max_size_allowed[name]
            data = cp.zeros(shape, dtype=np.dtype(dtype))
            if name != self.input_name:
                self.output_shapes[name] = list(shape)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data))
        self.batch_size = bindings[self.input_name].shape[0]
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        self.context = model.create_execution_context()
        self.bindings = bindings
        self.data_type = bindings[self.input_name].dtype
        self.im_size = bindings[self.input_name].shape[-2:]

    def _read_model_file(self, weights):
        with open(weights, "rb") as weight_file, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(weight_file.read())
        return model

    def run_engine(self, crop_stack: np.ndarray):
        if self.is_dynamic:
            partial_array = self.bindings[self.input_name].data[: len(crop_stack)]
            cp.copyto(partial_array, cp.asarray(crop_stack))
            self.context.set_binding_shape(0, crop_stack.shape)
        else:
            self.bindings[self.input_name].data[:] = cp.asarray(crop_stack)
        self.context.execute_v2(list(self.binding_addrs.values()))
        return len(crop_stack)

    def get_results(self, results, output_name: str) -> np.array:
        if self.is_dynamic:
            return self.bindings[output_name].data.get()[:results]
        else:
            return self.bindings[output_name].data.get()

    # assume data is already in the buffer memory
    def infer_directly(self):
        self.context.execute_v2(list(self.binding_addrs.values()))
        results = []
        for output in self.output_shapes:
            results.append(self.bindings[output].data.get())
        return tuple(results)

    @property
    def output_buffer(self):
        output_name = list(self.output_shapes.keys())[-1]
        return self.bindings[output_name].data

    @property
    def input_buffer(self):
        return self.bindings[self.input_name].data
