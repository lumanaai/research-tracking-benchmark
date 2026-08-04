import glob
import os
import shutil
import subprocess
import sys
from argparse import Namespace, ArgumentParser, ArgumentTypeError
from pathlib import Path, PosixPath
from pprint import pprint
from typing import List, Union, Dict, Type

import onnx
import tensorrt as trt
import torch
from botocore.exceptions import ClientError

from detection.yolov5.yolov8_detector import (
    YoloV8ExpertDetector,
    YoloV8Detector,
    YoloV8ContainerOcdDetector,
    YoloV8WeaponExpertDetector,
)
from export_common import determine_host
from general import image_encoder
from general.analyzer_general import DEFAULT_TRT_MAX_BATCH_ALLOWED, InferenceType
from general.clip_encoder import ClipVisionEncoder
from general.common_models import ClassificationWrapper
from general.image_quality_monitor.iq_monitor import ImageQualityNetMetrics
from general.inference import InferenceWrapper
from level1.doors.doors_classifier import DoorsClassifier
from level1.face.models import MagFace, WebFace, EDifFIQA
from level1.face.models.retinaface import RetinaFace
from level1.face.models.sface import SFace
from level1.face.models.yunet import YuNet
from level1.license_plate.lp_attributes import LPAttributesModel
from level1.license_plate.lp_recognition import LicensePlateRecognition
from level1.pa_recognition.hands import HandsModel
from level1.pa_recognition.person_attributes import AttributesModel, PersonReid
from level1.pa_recognition.phone_detector import PhoneModel
from level1.pa_recognition.ppe import PpeModel
from level1.skeleton.rtmpose import RtmPose
from level1.skeleton.stgcn import StgcnActionRecognition
from level1.text.text_detector import CraftTextDetector
from level1.text.text_recognizer import CrnnTextRecognition
from level1.vehicle.vehicle_parsing import BoTWithClassifier
from level1.violence.violence_models import ViolenceDetector, ViolenceClassifier
from level1.weapons_classifier.weapon_analyzer import WCResnet18Classifier
from level1.shipping_container.ocr_models import ContainerOcr
from tracking.byte_sreid_track.reid_models import TrackerPersonReid, TrackerVehicleReid
from utils.infra.s3_io import S3Handler

WORKSPACE = 4

# Models that default to mixed precision export (FP16 globally, sensitive layers in FP32).
# These models get near-FP32 accuracy with near-FP16 size/speed.
MIXED_PRECISION_MODELS = {
    InferenceType.PPE_ATTR.value,
}

# Layer name patterns forced to FP32 in mixed precision mode.
# Matched case-insensitively against TensorRT layer names from the ONNX graph.
MIXED_PRECISION_FP32_PATTERNS = [
    "fc", "classifier", "head", "linear",  # classification layers
    "bn", "batch_norm", "norm",             # normalization layers
    "mean", "reduce",                        # reduction operations
    "softmax", "sigmoid",                    # activation layers
]


def _should_force_fp32(layer_name: str) -> bool:
    """Check if a TRT layer should be forced to FP32 based on name patterns."""
    if not layer_name:
        return False
    layer_lower = layer_name.lower()
    return any(pattern in layer_lower for pattern in MIXED_PRECISION_FP32_PATTERNS)


def run_command(command: str):
    try:
        env = os.environ.copy()
        env["PATH"] = f"{Path(sys.executable).parent}:{env['PATH']}"
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, env=env)
        print(f"Output:\n{result.stdout}")

        if result.stderr:
            print(f"Error occur:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Exception raised! executing command: {command}\n{e.stderr}")


def get_resource_root():
    in_docker = True if os.getenv("IN_DOCKER") and os.getenv("IN_DOCKER") == "True" else False
    if in_docker:
        return "/usr/src/"
    else:
        return Path(__file__).parent.parent.parent.resolve()


def make_batch_dynamic(onnx_path, output_path):
    model = onnx.load(onnx_path)
    input_tensor = model.graph.input[0]

    # Modify batch dim from fixed value to dynamic (dim_param)
    input_tensor.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    input_tensor.type.tensor_type.shape.dim[0].ClearField("dim_value")

    # Optional: do the same for output if needed
    output_tensor = model.graph.output[0]
    output_tensor.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    output_tensor.type.tensor_type.shape.dim[0].ClearField("dim_value")

    onnx.save(model, output_path)


def check_make_dynamic_onnx(onnx_path: str) -> str:
    model = onnx.load(onnx_path)
    input_tensor = model.graph.input[0]
    shape = input_tensor.type.tensor_type.shape.dim
    if isinstance(shape[0].dim_value, int) and shape[0].dim_value > 0:
        # If the batch dimension is fixed, make it dynamic
        tmp_onnx_file = Path(onnx_path).with_suffix(".dynamic.onnx")
        print(f"Converting {onnx_path} to dynamic batch size.")
        make_batch_dynamic(onnx_path, tmp_onnx_file)
        return tmp_onnx_file
    return onnx_path


TRT_SUPPORTED_ARCH: Dict[str, Type[InferenceWrapper]] = {
    InferenceType.PERSON_ATTR.value: AttributesModel,
    InferenceType.PERSON_REID.value: PersonReid,
    InferenceType.WEAPONS_CLASSIFICATION.value: WCResnet18Classifier,
    InferenceType.IMAGE_ENCODER.value: image_encoder.ImageEncoder,
    InferenceType.DOORS_CLASSIFICATION.value: DoorsClassifier,
    InferenceType.PPE_ATTR.value: PpeModel,
    InferenceType.RETINA_FACE.value: RetinaFace,
    InferenceType.CRAFT.value: CraftTextDetector,
    InferenceType.VEHICLE.value: BoTWithClassifier,
    InferenceType.RTMPOSE.value: RtmPose,
    InferenceType.SFACE.value: SFace,
    InferenceType.YUNET.value: YuNet,
    InferenceType.CLIP_ENCODER.value: ClipVisionEncoder,
    InferenceType.TRACKER_PERSON_REID.value: TrackerPersonReid,
    InferenceType.TRACKER_VEHICLE_REID.value: TrackerVehicleReid,
    InferenceType.HANDS.value: HandsModel,
    InferenceType.PHONE.value: PhoneModel,
    InferenceType.LICENSE_PLATE.value: LicensePlateRecognition,
    InferenceType.CRNN.value: CrnnTextRecognition,
#    InferenceType.YOLOV5.value: YoloV5Detector,
    InferenceType.YOLOV8.value: YoloV8Detector,
    InferenceType.STGCN.value: StgcnActionRecognition,
    InferenceType.MAGFACE.value: MagFace,
    InferenceType.EDIFFIQA.value: EDifFIQA,
    InferenceType.WEBFACE.value: WebFace,
    InferenceType.EXPERT_DETECT.value: YoloV8ExpertDetector,
    InferenceType.WEAPON_EXPERT.value: YoloV8WeaponExpertDetector,
    InferenceType.VIOLENCE_DETECT.value: ViolenceDetector,
    InferenceType.VIOLENCE.value: ViolenceClassifier,
    InferenceType.STATE_OBJECT.value: ClassificationWrapper,
    InferenceType.IMAGEQUALITY.value: ImageQualityNetMetrics,
    InferenceType.LPC_ATTR.value: LPAttributesModel,
    InferenceType.CONTAINER_OCD.value: YoloV8ContainerOcdDetector,
    InferenceType.CONTAINER_OCR.value: ContainerOcr,
}

TRT_SPECIAL_ARCH = [
    InferenceType.CRNN.value,
    InferenceType.YOLOV5.value,
    InferenceType.YOLOV8.value,
    InferenceType.STGCN.value,
    InferenceType.MAGFACE.value,
    InferenceType.EDIFFIQA.value,
    InferenceType.EXPERT_DETECT.value,
    InferenceType.WEAPON_EXPERT.value,
    InferenceType.VIOLENCE.value,
    InferenceType.CONTAINER_OCD.value,
    InferenceType.CONTAINER_OCR.value,
    InferenceType.VIOLENCE_DETECT.value,
]


def execute_special_arch_tasks(inference_type: str, model_path: PosixPath, init_params: dict, batch_sz):
    out_folder = Path("/exports").joinpath(inference_type)
    export_file_path = None
    run_command(f"mkdir -p {out_folder}")

    if inference_type == InferenceType.CRNN.value:
        from dtrt_exporter import export_crnn_to_trt

        file_out, export_file_path = export_crnn_to_trt()

    elif inference_type == InferenceType.YOLOV5.value:
        run_command(f"python3 -m export_yolov5 -w {model_path} -o {out_folder}")
        out_files = list(out_folder.glob(model_path.stem + "*.engine"))
        if len(out_files) == 0:
            raise Exception("no engine file found after export")
        export_file_path = str(out_files[0])

    elif inference_type == InferenceType.YOLOV8.value:
        run_command(f"python3 -m export_yolov8 -w {model_path} -o {out_folder}")
        out_files = list(out_folder.glob(model_path.stem + "*.engine"))
        if len(out_files) == 0:
            raise Exception("no engine file found after export")
        export_file_path = str(out_files[0])

    elif inference_type == InferenceType.STGCN.value:
        fp_type = 16 if init_params["half"] else 32
        fp_flag = "--fp16" if init_params["half"] else ""
        export_file_path = out_folder.joinpath(f"{model_path.stem}_fp{fp_type}_b{batch_sz}.engine")
        cmd_suffix = (
            " --trt-min-shapes input:[1,1,1,25,15,3]"
            " --trt-opt-shapes input:[1,1,1,25,15,3]"
            " --trt-max-shapes input:[8,1,1,25,15,3]"
        )

        run_command(f"polygraphy convert {model_path} -o {export_file_path} {fp_flag}{cmd_suffix}")

    elif inference_type == InferenceType.MAGFACE.value:
        fp_type = 16 if init_params["half"] else 32
        fp_flag = "--fp16" if init_params["half"] else ""
        export_file_path = out_folder.joinpath(f"{model_path.stem}_b{batch_sz}_fp{fp_type}.engine")
        cmd_suffix = (
            " --trt-min-shapes input:[1,3,112,112]"
            " --trt-opt-shapes input:[2,3,112,112]"
            " --trt-max-shapes input:[8,3,112,112]"
        )

        if init_params["half"]:
            cmd_suffix += " --tensor-datatypes input:float16 output:float16" " --precision-constraints obey"

        run_command(f"polygraphy convert {model_path} -o {export_file_path} {fp_flag}{cmd_suffix}")

    elif inference_type == InferenceType.EDIFFIQA.value:
        fp_type = 16 if init_params["half"] else 32
        fp_flag = "--fp16" if init_params["half"] else ""
        export_file_path = out_folder.joinpath(f"{model_path.stem}_b{batch_sz}_fp{fp_type}.engine")
        cmd_suffix = (
            " --trt-min-shapes input:[1,3,112,112]"
            " --trt-opt-shapes input:[2,3,112,112]"
            " --trt-max-shapes input:[8,3,112,112]"
        )

        if init_params["half"]:
            cmd_suffix += " --tensor-datatypes input:float16 output:float16" " --precision-constraints obey"

        run_command(f"polygraphy convert {model_path} -o {export_file_path} {fp_flag}{cmd_suffix}")

    elif inference_type in [
        InferenceType.EXPERT_DETECT.value,
        InferenceType.CONTAINER_OCD.value,
        InferenceType.WEAPON_EXPERT.value,
    ]:
        arch = TRT_SUPPORTED_ARCH[inference_type](init_params, is_local=True)
        img_sz = f"{arch.args.im_size[0]},{arch.args.im_size[1]}"
        # Step 1: Export YOLO model to ONNX
        onnx_model_path = model_path.with_suffix(".onnx")  # Convert .pt path to .onnx
        export_command = f"yolo export model={model_path} format=onnx device=cpu dynamic=True imgsz=[{img_sz}]"
        run_command(export_command)

        # Step 2: Convert ONNX model to TensorRT engine
        fp_type = 16 if init_params["half"] else 32
        fp_flag = "--fp16" if init_params["half"] else ""
        export_file_path = out_folder.joinpath(f"{onnx_model_path.stem}_fp{fp_type}_b{batch_sz}.engine")
        cmd_suffix = (
            f" \\\n"
            f"--trt-min-shapes images:[1,3,{img_sz}]"
            f" --trt-opt-shapes images:[2,3,{img_sz}]"
            f" --trt-max-shapes images:[{arch.args.max_dynamic_batch},3,{img_sz}]"
        )

        if init_params["half"]:
            cmd_suffix += " --tensor-datatypes images:float16 output0:float16" " --precision-constraints obey"

        run_command(f"polygraphy convert {onnx_model_path} -o {export_file_path} {fp_flag}{cmd_suffix}")

    elif inference_type == InferenceType.VIOLENCE.value:
        # Step 1: Build model in fp32 for ONNX export (TRT handles fp16 later)
        init_params["enable"] = False
        arch = ViolenceClassifier(init_params, is_local=True)
        im_h, im_w = init_params["im_size"]
        seq_len = arch.args.T

        model = arch._build_model()
        state_dict = torch.load(str(model_path), map_location=arch.args.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        onnx_model_path = out_folder / f"{model_path.stem}.onnx"
        dummy_input = torch.randn(1, seq_len, 3, im_h, im_w, device=arch.args.device)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_model_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

        # Step 2: Convert ONNX to TensorRT engine
        fp_type = 16 if init_params["half"] else 32
        fp_flag = "--fp16" if init_params["half"] else ""
        export_file_path = out_folder.joinpath(f"{model_path.stem}_b{batch_sz}_fp{fp_type}.engine")
        cmd_suffix = (
            f" --trt-min-shapes input:[1,{seq_len},3,{im_h},{im_w}]"
            f" --trt-opt-shapes input:[1,{seq_len},3,{im_h},{im_w}]"
            f" --trt-max-shapes input:[{arch.args.max_dynamic_batch},{seq_len},3,{im_h},{im_w}]"
        )

        if init_params["half"]:
            cmd_suffix += " --tensor-datatypes input:float16 output:float16" " --precision-constraints obey"

        run_command(f"polygraphy convert {onnx_model_path} -o {export_file_path} {fp_flag}{cmd_suffix}")

    elif inference_type == InferenceType.VIOLENCE_DETECT.value:
        im_h, im_w = init_params["im_size"]
        onnx_model_path = model_path.with_suffix(".onnx")

        # Step 1: export to ONNX only if weights are not already .onnx
        if model_path.suffix == ".onnx" and model_path.exists():
            onnx_model_path = check_make_dynamic_onnx(str(model_path))
            onnx_model_path = Path(onnx_model_path)
        else:
            # Resolve .pt source: use model_path directly if it's .pt,
            # otherwise try swapping the extension
            pt_path = model_path if model_path.suffix == ".pt" else model_path.with_suffix(".pt")
            if not pt_path.exists():
                raise FileNotFoundError(
                    f"Cannot find weights for {inference_type}: tried '{model_path}' and '{pt_path}'"
                )
            export_command = f"yolo export model={pt_path} format=onnx device=cpu dynamic=True imgsz=[{im_h},{im_w}]"
            run_command(export_command)
            onnx_model_path = pt_path.with_suffix(".onnx")

        # Step 2: Convert ONNX to TensorRT engine
        fp_type = 16 if init_params["half"] else 32
        fp_flag = "--fp16" if init_params["half"] else ""
        export_file_path = out_folder.joinpath(f"{onnx_model_path.stem}_fp{fp_type}_b{batch_sz}.engine")
        arch = ViolenceDetector(init_params, is_local=True)
        cmd_suffix = (
            f" --trt-min-shapes images:[1,3,{im_h},{im_w}]"
            f" --trt-opt-shapes images:[2,3,{im_h},{im_w}]"
            f" --trt-max-shapes images:[{arch.args.max_dynamic_batch},3,{im_h},{im_w}]"
        )

        if init_params["half"]:
            cmd_suffix += " --tensor-datatypes images:float16 output0:float16" " --precision-constraints obey"

        run_command(f"polygraphy convert {onnx_model_path} -o {export_file_path} {fp_flag}{cmd_suffix}")

    elif inference_type == InferenceType.CONTAINER_OCR.value:
        init_params["enable"] = False
        arch = TRT_SUPPORTED_ARCH[inference_type](init_params, is_local=True)
        imgH, imgW = init_params["im_size"]
        onnx_model_path = model_path if model_path.suffix == ".onnx" else model_path.with_suffix(".onnx")
        if not onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        onnx_model_path = Path(check_make_dynamic_onnx(str(onnx_model_path)))

        # Read input/output tensor names from ONNX
        onnx_model = onnx.load(str(onnx_model_path))
        input_name = onnx_model.graph.input[0].name
        output_name = onnx_model.graph.output[0].name

        fp_type = 16 if init_params["half"] else 32
        fp_flag = "--fp16" if init_params["half"] else ""
        export_file_path = out_folder.joinpath(f"{model_path.stem}_fp{fp_type}_b{batch_sz}.engine")
        max_batch = arch.args.max_dynamic_batch
        cmd_suffix = (
            f" --trt-min-shapes {input_name}:[1,3,{imgH},{imgW}]"
            f" --trt-opt-shapes {input_name}:[2,3,{imgH},{imgW}]"
            f" --trt-max-shapes {input_name}:[{max_batch},3,{imgH},{imgW}]"
        )
        if init_params["half"]:
            cmd_suffix += f" --tensor-datatypes {input_name}:float16 {output_name}:float16 --precision-constraints obey"

        run_command(f"polygraphy convert {onnx_model_path} -o {export_file_path} {fp_flag}{cmd_suffix}")

    elif inference_type == InferenceType.CLIP_ENCODER.value:
        # special case where we first export to onnx in the usual flow and then convert to trt using polygraphy
        if init_params["half"] == False:
            print("WARNING: CLIP encoder is not supported in FP32 due to performance issue, using FP16")
        export_file_path = out_folder.joinpath(f"{model_path.stem}_b{batch_sz}_fp16.engine")
        cmd_suffix = (
            " --trt-min-shapes input:[1,3,224,224]"
            " --trt-opt-shapes input:[2,3,224,224]"
            " --trt-max-shapes input:[8,3,224,224]"
        )
        cmd_suffix += " --fp16 --precision-constraints prefer"
        lp_suffix = ""
        model = onnx.load(str(model_path))
        for i, node in enumerate(model.graph.node):
            if node.op_type in ["ReduceMean", "Pow"]:
                lp_suffix += f" {node.name}:float32"
        if lp_suffix != "":
            cmd_suffix += f" --layer-precisions lp_suffix"
        run_command(f"polygraphy convert {model_path} -o {export_file_path} {cmd_suffix}")
    else:
        raise Exception(f"Unknown inference type: {inference_type}")

    if not export_file_path:
        raise Exception("No file_out issue!")

    return str(model_path), str(export_file_path)


def export_to_trt(
    model_name: str,
    weight: str = None,
    half: bool = None,
    batch_sz: int = -1,
    verbose: bool = False,
    model_params: dict = None,
    mixed_precision: bool = None,
):
    try:
        # Determine if mixed precision should be used:
        # None = auto (use if model is in MIXED_PRECISION_MODELS), True = force, False = disable
        if mixed_precision is None:
            mixed_precision = model_name in MIXED_PRECISION_MODELS
        if mixed_precision:
            print(f"[Mixed Precision] Enabled for {model_name}")

        logger_trt = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(None, "")
        init_dict = {"unique_weights": weight, "force_full": True}
        if model_params is not None:
            init_dict.update(model_params)
        if half is not None:
            init_dict["half"] = half

        # For special arch models, disable model loading — we only need the config
        if model_name in TRT_SPECIAL_ARCH:
            init_dict["enable"] = False
        arch = TRT_SUPPORTED_ARCH[model_name](init_dict, is_local=True)
        model_path = Path(arch.args.weights)
        onnx_path = model_path.with_suffix(".onnx")
        if half is None:
            half = arch.export_is_half
        image_sz = arch.args.im_size

        print("initializing model")
        init_params = {
            "unique_weights": str(model_path.absolute()),
            "im_size": image_sz,
            "half": half,
            "enable": True,
        }
        pprint(f"parameters: {init_params}")

        if model_name in TRT_SPECIAL_ARCH:
            return execute_special_arch_tasks(model_name, model_path, init_params, batch_sz)

        output_names = arch.output_names if arch.output_names else ["output"]
        input_name = arch.input_name if arch.input_name else "input"
        if batch_sz < 0:  # currently not working
            im_input = torch.rand(1, 3, *image_sz).to("cuda")
            dynamic_axes = {
                input_name: {0: "batch_size"},  # Set the first dimension (batch size) as dynamic
            }
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size"}
            max_batch = (
                DEFAULT_TRT_MAX_BATCH_ALLOWED if arch.args.max_dynamic_batch is None else arch.args.max_dynamic_batch
            )
        else:
            max_batch = batch_sz
            im_input = torch.rand(batch_sz, 3, *image_sz).to("cuda")
            dynamic_axes = None
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # | 1 << int(trt.BuilderFlag.DEBUG)

        if arch.args.weights.endswith(".onnx"):

            onnx_path = check_make_dynamic_onnx(arch.args.weights)
        else:
            model = arch.model
            model.eval()
            # For mixed precision: export in FP32 (TRT handles FP16 internally)
            if mixed_precision:
                model.float()
                im_input = im_input.float()
            elif half:
                model = model.half()
                im_input = im_input.half()

            print(f"converting to onnx....")
            torch.onnx.export(
                model,
                im_input,
                onnx_path,
                input_names=[input_name],
                output_names=output_names,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )

            print(f"onnx conversion done, checking result model")
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            print(f"onnx conversion completed successfully")
        if model_name == InferenceType.CLIP_ENCODER.value:
            return execute_special_arch_tasks(model_name, onnx_path, init_params, batch_sz)

        builder = trt.Builder(logger_trt)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger_trt)

        print("Beginning ONNX file parsing")
        with open(onnx_path, "rb") as model:
            parser.parse(model.read())
        print("Completed parsing of ONNX file")

        # For mixed precision: force FP32 I/O so input/output are not quantized
        if mixed_precision:
            for i in range(network.num_inputs):
                network.get_input(i).dtype = trt.float32
            for i in range(network.num_outputs):
                network.get_output(i).dtype = trt.float32

        print("Building an engine...")
        if mixed_precision:
            file_out = os.path.join(
                Path(model_path).parent.resolve(),
                f"{model_path.stem}_mixed_b{batch_sz}.engine",
            )
        else:
            file_out = os.path.join(
                Path(model_path).parent.resolve(),
                f"{model_path.stem}_fp{16 if half else 32}_b{batch_sz}.engine",
            )  # TensorRT engine filename

        config = builder.create_builder_config()
        # allow TensorRT to use up to 1GB of GPU memory for tactic selection
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE * 1 << 30)
        if mixed_precision:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            # Force sensitive layers to FP32
            num_fp32 = 0
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if _should_force_fp32(layer.name):
                    layer.precision = trt.float32
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.float32)
                    num_fp32 += 1
            print(f"[Mixed Precision] {num_fp32} layers forced FP32, {network.num_layers - num_fp32} layers FP16")
        elif half:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if batch_sz < 0:
            opt_batch = (max_batch + 1) // 2
            profile = builder.create_optimization_profile()
            profile.set_shape(input_name, [1, 3, *image_sz], [opt_batch, 3, *image_sz], [max_batch, 3, *image_sz])
            config.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, config)
        with open(file_out, "wb") as t:
            t.write(plan)
        print(f"Completed creating Engine file: {file_out}")

        try:
            if not verbose or not arch.args.weights.endswith(".onnx"):
                os.remove(str(onnx_path))
        except Exception as e:
            print(f"failed to remove onnx file: {e}")

        # checking folder exists and creating if not
        os.makedirs(os.path.join("/exports", model_name), exist_ok=True)

        export_file_path = os.path.join("/exports", model_name, Path(file_out).name)
        if os.path.exists(export_file_path):
            print("file already exists there in target location, removing")
            os.remove(export_file_path)
        shutil.copy2(file_out, export_file_path)
        print(f"weights file was copied to exports: {export_file_path}")

        return file_out, export_file_path
    except Exception as e:
        print(f"export failure: {e}")
        return None, None


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    elif v.lower() in ("none", ""):
        return None
    else:
        raise ArgumentTypeError("Boolean or None value expected.")


def get_command_line_args(arg_list: List[str] = None):
    parser = ArgumentParser(
        prog="General TRT exporter", description="can convert to trt pa net, reid from defaults or specific file"
    )
    parser.add_argument(
        "-t",
        "--type",
        default=None,
        nargs="+",  # Allows multiple values as a list
        type=str,
        choices=list(TRT_SUPPORTED_ARCH.keys()),
        help="net to compile to TensorRT (single value as string, multiple as list)",
    )

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",  # Sets to True if -u is provided, False otherwise
        help="Upload to S3/GCP if specified",
    )

    parser.add_argument("-w", "--weights", type=str, default=None, help="path to weights file")
    parser.add_argument("-b", "--batch_sz", type=int, default=-1, help="batch size to be used")
    parser.add_argument("-v", "--verbose", type=str2bool, default=None, help="verbosity of export procedure")
    parser.add_argument("-ha", "--half", type=str2bool, default=None, help="use fp16 instead of fp32")
    parser.add_argument("-a", "--all", action="store_true", help="export all supported nets: pa, reid, wc, etc.")
    parser.add_argument("-i", "--input-dir", type=str, default=None, help="path to base dir file")
    parser.add_argument("--aws-key", type=str, default=None, help="path to base dir file")
    parser.add_argument("--aws-access-key", type=str, default=None, help="path to base dir file")
    parser.add_argument("--debug", action="store_true", help="Debug mode - default on")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite if file already exists in S3")
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision even for models that default to it (e.g. PPE)",
    )

    if arg_list is not None:
        args, additional_args = parser.parse_known_args(arg_list)
    else:
        args = parser.parse_args()

    if args.aws_key is None:
        args.aws_key = os.environ.get("AWS_SECRET_KEY", None)
    if args.aws_access_key is None:
        args.aws_access_key = os.environ.get("AWS_ACCESS_KEY", None)

    return args


def main(opts: Union[Dict, List[str]] = None, model_params: dict = None):

    if opts is None or isinstance(opts, list):
        opts_dict = vars(get_command_line_args(opts))
    else:
        opts_dict = opts
    weights = None
    if opts_dict.get("weights", None) is not None:
        weights = opts_dict["weights"]
    if opts_dict.get("type") is not None:
        net_type = opts_dict["type"]
    elif weights is not None:
        net_type = os.path.basename(weights).split("_")[0]
        if net_type.startswith(InferenceType.STATE_OBJECT.value):
            net_type = InferenceType.STATE_OBJECT.value
        elif net_type not in TRT_SUPPORTED_ARCH:
            raise RuntimeError(f"unexpected model type {net_type}")
    else:
        if opts_dict.get("all", False):
            raise RuntimeError("all flag is only supported when running from __main__")
        raise RuntimeError("no network was given")

    return export_to_trt(
        net_type,
        weight=weights,
        batch_sz=opts_dict["batch_sz"],
        half=opts_dict["half"],
        verbose=opts_dict["verbose"],
        model_params=model_params,
        mixed_precision=False if opts_dict.get("no_mixed_precision") else None,
    )


def check_remote_file_exists(cloud_handler: str, upload_path: str, **kwargs) -> bool:
    """Check if a file already exists at the remote upload destination.
    Supports 's3' (and can be extended for 'gcp' etc.).
    """
    if cloud_handler == "s3":
        s3: S3Handler = kwargs["s3"]
        try:
            s3._s3_client.head_object(Bucket=s3._s3_bucket_name, Key=upload_path)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
    elif cloud_handler == "gcp":
        raise NotImplementedError("GCP remote file existence check is not yet implemented")
    else:
        raise ValueError(f"Unsupported cloud handler: {cloud_handler}")


def rename_with_host(file_out: str, host: str):
    if file_out is None:
        print("no file to rename - skipping")
    else:
        new_name = file_out.replace(".engine", f"_{host}.engine")
        os.rename(file_out, new_name)
        print(f"file renamed to {new_name}")
        return new_name


if __name__ == "__main__":
    cloud_handler = "s3"
    opts_in: Namespace = get_command_line_args()
    host = str(determine_host().value)
    if opts_in.debug:
        host += "-debug"

    if opts_in.all:
        nets = list(TRT_SUPPORTED_ARCH.keys())
        nets.remove(InferenceType.STATE_OBJECT.value)
    else:
        nets = opts_in.type
    for i, net in enumerate(nets):
        if opts_in.weights is not None:
            weights = opts_in.weights
        elif opts_in.input_dir is not None:
            weights = glob.glob(os.path.join(opts_in.input_dir, net, ".pt*"))
            if len(weights) == 0:
                print(f"no weights found for {net} in {opts_in.input_dir}")
                continue
            # sort by date
            weights = sorted(weights, key=os.path.getmtime)[-1]
        else:
            weights = None
        print("*" * 80)
        print(f"exporting {net} with weights: {weights}")
        file_out, export_file_path = main(
            {
                "type": net,
                "weights": weights,
                "batch_sz": opts_in.batch_sz,
                "half": opts_in.half,
                "verbose": opts_in.verbose,
                "no_mixed_precision": opts_in.no_mixed_precision,
            }
        )
        rename_with_host(file_out, host)
        local_file_name = rename_with_host(export_file_path, host)

        if local_file_name is None:
            raise RuntimeError(f"Export failed for {net} — no output file produced")

        if opts_in.upload:
            upload_path = os.path.join(host, local_file_name.replace("/exports/", "", 1).lstrip("/"))
            if cloud_handler == "s3":
                s3 = S3Handler(
                    bucket_name="lumix-analytics-default-weights",
                    key=opts_in.aws_key,
                    access_key=opts_in.aws_access_key,
                )

                if check_remote_file_exists(cloud_handler, upload_path, s3=s3):
                    if opts_in.force:
                        print(f"WARNING: Overwriting existing file at 'lumix-analytics-default-weights/{upload_path}'")
                    else:
                        raise RuntimeError(
                            f"Upload aborted: a file already exists at "
                            f"'lumix-analytics-default-weights/{upload_path}'. "
                            f"Overwriting is not allowed to prevent accidental loss of shared weights. "
                            f"Use --force to overwrite."
                        )

                upload_success = s3.upload_file(local_file_name, upload_path)
                print("*" * 80)
                if upload_success:
                    print(f"Upload successful! AWS path: lumix-analytics-default-weights/{str(upload_path)}")
                else:
                    raise Exception("Upload failed.")
            elif cloud_handler == "gcp":
                print("1")
