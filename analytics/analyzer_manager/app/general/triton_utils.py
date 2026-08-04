import time
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .analyzer_general import logger, DEFAULT_TRT_MAX_BATCH_ALLOWED
from .core import BaseInfer, EndpointInfo

triton_datatype_to_np = {
    "FP16": np.float16,
    "FP32": np.float32,
    "INT32": np.int32,
    "UINT8": np.uint8,
}


class TritonInfer(BaseInfer):
    retries = 3

    def __init__(self, endpoint: EndpointInfo, max_size_allowed: Dict = None):
        self.endpoint_info = endpoint
        self.inputs = []
        self.output_shapes = OrderedDict()
        self.im_size = []
        if max_size_allowed is None:
            max_size_allowed = {}

        metadata = None
        for _ in range(self.retries):
            try:
                self.triton_client = grpcclient.InferenceServerClient(
                    url=endpoint.url,
                    verbose=endpoint.verbose,
                )
                metadata = self.triton_client.get_model_metadata(
                    model_name=self.endpoint_info.model_name, model_version=str(self.endpoint_info.model_version)
                )
                config = self.triton_client.get_model_config(
                    model_name=self.endpoint_info.model_name,
                    model_version=str(self.endpoint_info.model_version),
                )
                break
            except InferenceServerException as e:
                logger.warning(
                    f"exception caught when trying to load model{endpoint.model_name} in url {endpoint.url}: {e.message()}"
                )
                load_triton_model(self.endpoint_info)

        if metadata is None:
            msg = f"could not initialize inference model {endpoint.model_name} using triton in url {endpoint.url}"
            raise RuntimeError(msg)

        for input_ in metadata.inputs:
            self.inputs.append(grpcclient.InferInput(input_.name, input_.shape, input_.datatype))
        self.is_dynamic = metadata.inputs[0].shape[0] < 0
        input_shape = metadata.inputs[0].shape
        self.im_size = input_shape[-2:] if len(input_shape) == 4 else input_shape
        self.data_type = triton_datatype_to_np[metadata.inputs[0].datatype]
        for output in metadata.outputs:
            self.output_shapes[output.name] = max_size_allowed.get(output.name, output.shape)
        if self.is_dynamic:
            self.batch_size = config.config.max_batch_size or DEFAULT_TRT_MAX_BATCH_ALLOWED
        else:
            self.batch_size = metadata.inputs[0].shape[0]

    def run_engine(self, crop_stack: np.ndarray):
        for _ in range(self.retries):
            try:
                if self.is_dynamic:
                    self.inputs[0].set_shape(crop_stack.shape)
                self.inputs[0].set_data_from_numpy(crop_stack)
                return self.triton_client.infer(
                    model_name=self.endpoint_info.model_name,
                    inputs=self.inputs,
                    client_timeout=self.endpoint_info.timeout,
                    model_version=str(self.endpoint_info.model_version),
                )
            except InferenceServerException as e:
                logger.warning(f"exception caught when trying to infer {self.endpoint_info.model_name}: {e.message()}")
                load_triton_model(self.endpoint_info)

    def get_results(self, results, output_name: str) -> np.array:
        return results.as_numpy(output_name)


def is_triton_available(endpoint_info: EndpointInfo):
    t0 = time.time()
    name = endpoint_info.model_name
    try:
        logger.info(f"[triton-probe] {name}: creating grpc client for {endpoint_info.url}")
        triton_client = grpcclient.InferenceServerClient(
            url=endpoint_info.url,
            verbose=endpoint_info.verbose,
        )
        logger.info(f"[triton-probe] {name}: calling load_model (+{time.time() - t0:.2f}s)")
        triton_client.load_model(model_name=name)
        logger.info(f"[triton-probe] {name}: load_model returned (+{time.time() - t0:.2f}s)")
        triton_client.get_model_metadata(
            model_name=name, model_version=str(endpoint_info.model_version)
        )
        logger.info(f"[triton-probe] {name}: get_model_metadata returned (+{time.time() - t0:.2f}s), available")
        return True
    except Exception as e:
        logger.warning(f"[triton-probe] {name}: failed after {time.time() - t0:.2f}s: {type(e).__name__}: {e}")
        return False


def is_triton_live(endpoint_info: EndpointInfo):
    try:
        triton_client = grpcclient.InferenceServerClient(url=endpoint_info.url)
        return triton_client.is_server_live()
    except Exception:
        return False


def load_triton_model(endpoint_info: EndpointInfo):
    try:
        triton_client = grpcclient.InferenceServerClient(url=endpoint_info.url)
        triton_client.load_model(model_name=endpoint_info.model_name)
        return True
    except Exception:
        return False


def unload_triton_model(endpoint_info: EndpointInfo):
    try:
        triton_client = grpcclient.InferenceServerClient(url=endpoint_info.url)
        triton_client.unload_model(model_name=endpoint_info.model_name)
        return True
    except Exception:
        return False


class DetectorTriton(TritonInfer):
    output_name = "output"

    def infer_directly(self, data: np.ndarray):
        results = self.run_engine(data)
        return torch.from_numpy(results.as_numpy(self.output_name))

    def infer(self, data: np.ndarray):
        return torch.from_numpy(super().infer(data))

    def __post_init__(self):
        self.concat_func = lambda x: np.concatenate(x, axis=0)
