import argparse
import os

from .export import export_onnx
from models.experimental import attempt_load
from models.common import Conv
from models.yolo import Detect
import torch
import torch.nn as nn
from .utils.torch_utils import select_device
from .utils.activations import SiLU
from .utils.general import LOGGER, check_img_size, check_requirements, check_version, colorstr, file_size
from pathlib import Path


def export_engine(
    model, im, batch_size, file, train, half, simplify, workspace=2, verbose=False, prefix=colorstr("TensorRT:")
):
    try:
        check_requirements(("tensorrt",))
        import tensorrt as trt

        check_version(trt.__version__, "8.0.0", hard=True)
        export_onnx(model, im, file, 13, train, False, simplify)
        onnx = file.with_suffix(".onnx")

        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        assert im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
        assert onnx.exists(), f"failed to export ONNX file: {onnx}"
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)

        export_fp16 = builder.platform_has_fast_fp16 and half

        f = os.path.join(
            Path(file).parent.resolve(), f"{file.stem}_fp{16 if export_fp16 else 32}_b{batch_size}.engine"
        )  # TensorRT engine filename

        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f"failed to load ONNX file: {onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        LOGGER.info(f"{prefix} Network Description:")
        for inp in inputs:
            LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        LOGGER.info(f"{prefix} building FP{16 if export_fp16 else 32} engine in {f}")
        if export_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, "wb") as t:
            t.write(engine.serialize())
        LOGGER.info(f"{prefix} export success, saved as {f} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        LOGGER.info(f"\n{prefix} export failure: {e}")
        return None


def run(opt):
    device = select_device("0")
    model = attempt_load(opt.weights, map_location=device, inplace=True, fuse=True)  # load PyTorch model
    nc, names = model.nc, model.names  # number of classes, class names
    assert nc == len(names), f"Model class count {nc} != len(names) {len(names)}"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in opt.image_sz]  # verify img_size are gs-multiples
    im = torch.zeros(opt.batch_sz, 3, *imgsz).to(device)

    # Update model
    if opt.half:
        im, model = im.half(), model.half()  # to FP16
    model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False
            if hasattr(m, "forward_export"):
                m.forward = m.forward_export

    for _ in range(2):
        y = model(im)  # dry runs
    shape = tuple(y[0].shape)
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {opt.weights} with output shape {shape} ({file_size(opt.weights):.1f} MB)"
    )
    file = Path(opt.weights)
    return export_engine(
        model=model,
        im=im,
        batch_size=opt.batch_sz,
        file=file,
        train=False,
        half=opt.half,
        simplify=False,
        workspace=opt.workspace,
        verbose=opt.verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, default="yolov5m.pt", help="model.pt path")
    parser.add_argument("-sz", "--image-sz", nargs="+", type=int, default=[384, 640], help="image (h, w)")
    parser.add_argument("-b", "--batch-sz", type=int, default=1, help="batch size")
    parser.add_argument("-v", "--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("-fp16", "--half", action="store_true", help="fp16 engine")
    parser.add_argument("-ws", "--workspace", type=int, default=2, help="TensorRT workspace size (GB)")
    opt = parser.parse_args()
    run(opt)
