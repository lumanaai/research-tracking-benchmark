import os
import shutil
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import onnx
import tensorrt as trt
import torch
from torch._C._onnx import OperatorExportTypes

from level1.text.text_recognizer import CrnnTextRecognition

workspace_sz = 2  # GB


def export_crnn_to_trt():
    try:
        logger_trt = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(None, "")
        crnn = CrnnTextRecognition({"enable": True, "force_full": True, "unique_weights": None}, is_local=True)
        model = crnn.model
        model.eval()

        model_path = Path(crnn.args.weights)
        onnx_path = model_path.with_suffix(".onnx")

        print("initializing model")
        init_params = {
            "unique_weights": str(model_path.absolute()),
            "dynamic_size": [-1, 1, 64, -1],
            "min_size": [1, 1, 64, 32],
            "opt_size": [4, 1, 64, 192],
            "max_size": [16, 1, 64, 640],
            "half": crnn.args.half,
            "enable": True,
            "net_type": "crnn",
        }
        params = Namespace(**init_params)
        pprint(f"parameters: {init_params}")

        output_names = params.output_names if "output_names" in params else ["output"]
        naming = ["batch", "channel", "height", "width"]

        dynamic_axes = {"input": {}}
        for i, named in enumerate(naming):
            if params.dynamic_size[i] < 0:
                dynamic_axes["input"][i] = named

        for name in output_names:
            dynamic_axes[name] = deepcopy(dynamic_axes["input"])

        im_input = torch.rand(*params.opt_size).to("cuda")
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # | 1 << int(trt.BuilderFlag.DEBUG)

        # Update model
        if params.half:
            im_input, model = im_input.half(), model.half()  # to FP16

        print(f"converting to onnx....")
        torch.onnx.export(
            model,
            im_input,
            onnx_path,
            input_names=["input"],
            output_names=output_names,
            export_params=True,
            operator_export_type=OperatorExportTypes.ONNX,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=False,
            dynamic_axes=dynamic_axes,
        )

        print(f"onnx conversion done, checking result model")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        print(f"onnx conversion completed successfully")

        builder = trt.Builder(logger_trt)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger_trt)

        print("Beginning ONNX file parsing")
        with open(onnx_path, "rb") as model:
            parser.parse(model.read())
        print("Completed parsing of ONNX file")

        print("Building an engine...")
        file_out = os.path.join(
            Path(model_path).parent.resolve(),
            f"{model_path.stem}_fp{16 if params.half else 32}.engine",
        )  # TensorRT engine filename

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_sz * (1 << 30))
        if params.half:
            config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape("input", params.min_size, params.opt_size, params.max_size)
        config.add_optimization_profile(profile)
        serialized_engine = builder.build_serialized_network(network, config)
        with open(file_out, "wb") as t:
            t.write(serialized_engine)
        print(f"Completed creating Engine file: {file_out}")

        # checking folder exists and creating if not
        os.makedirs(os.path.join("/exports", params.net_type), exist_ok=True)

        export_file_path = os.path.join("/exports", params.net_type, Path(file_out).name)
        if os.path.exists(export_file_path):
            print("file already exists there in target location, removing")
            os.remove(export_file_path)
        shutil.copy2(file_out, export_file_path)
        print(f"weights file was copied to exports: {export_file_path}")

        return file_out, export_file_path
    except Exception as e:
        print(f"export failure: {e}")
        return None


if __name__ == "__main__":
    export_crnn_to_trt()
