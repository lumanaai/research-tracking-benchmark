#!/usr/bin/env python3
"""
Convert a PaddlePaddle inference model to ONNX format.

Standalone script — intended to be dropped into any repo that needs to convert
Paddle inference models (*.json / *.pdmodel + *.pdiparams) to ONNX.

Dependencies (install into a fresh venv / conda env)::

    pip install paddle2onnx paddlepaddle-gpu onnxsim  # or paddlepaddle (CPU)

Usage::

    # Minimal — converts the model in the given directory
    python convert_paddle_to_onnx.py \
        --model_dir weights/OCR/inference \
        --output    weights/OCR/onnx/inference.onnx

    # With explicit filenames (when the directory has non-standard names)
    python convert_paddle_to_onnx.py \
        --model_dir    weights/OCR/inference \
        --model_file   inference.json \
        --params_file  inference.pdiparams \
        --output       weights/OCR/onnx/inference.onnx \
        --opset 14

    # Validate the exported model (requires onnx + onnxruntime)
    python convert_paddle_to_onnx.py \
        --model_dir weights/OCR/inference \
        --output    weights/OCR/onnx/inference.onnx \
        --validate

Notes on PIR format models (as of paddle2onnx 2.1.0 / PaddlePaddle 3.3.1)::

    Models saved in PaddlePaddle's new PIR format (.json) may contain ops that
    paddle2onnx cannot directly convert (e.g. ``pd_op.linear_v2``,
    ``pd_op.shape64``).  This script handles them automatically by:

    1. Loading the PIR program via ``paddle.static.load_inference_model``
    2. Decomposing unsupported ops into equivalent basic ops
       (linear_v2 → matmul + add, shape64 → shape)
    3. Saving the decomposed program to a temp directory
    4. Running paddle2onnx on the decomposed model
    5. Optimising with onnxsim (constant folding, node reduction)

    The resulting ONNX model produces identical outputs to the original Paddle
    model (validated on PP-OCRv5_server_rec with 100% exact match).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import shutil


# ────────────────────────── Auto-detect model file ───────────────────────────

_MODEL_EXTENSIONS = (".json", ".pdmodel")
_PARAMS_EXTENSIONS = (".pdiparams",)


def _find_file(directory: Path, extensions: tuple[str, ...]) -> str | None:
    """Return the first file matching any of *extensions* in *directory*."""
    for f in sorted(directory.iterdir()):
        if f.suffix in extensions and f.is_file():
            return f.name
    return None


# ───────────────────── PIR program decomposition ─────────────────────────────

def _is_pir_model(model_file: str) -> bool:
    """Check if the model uses PIR format (.json extension)."""
    return model_file.endswith(".json")


def _decompose_pir_program(model_dir: Path, model_file: str, params_file: str) -> Path:
    """Load a PIR model, decompose unsupported ops, save to a temp directory.

    Handles: pd_op.linear_v2 → matmul + add, pd_op.shape64 → shape.
    Returns the path to the temp directory containing the decomposed model.
    """
    import os
    os.environ.setdefault("FLAGS_enable_pir_api", "1")

    import paddle
    import paddle.pir  # noqa: F401

    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    model_prefix = str(model_dir / model_file.rsplit(".", 1)[0])
    prog, feed_names, fetch_names = paddle.static.load_inference_model(
        model_prefix, exe
    )

    block = prog.global_block()

    # Collect ops to replace
    linear_ops = [op for op in block.ops if op.name() == "pd_op.linear_v2"]
    shape64_ops = [op for op in block.ops if op.name() == "pd_op.shape64"]

    if linear_ops or shape64_ops:
        print(f"  Decomposing: {len(linear_ops)} linear_v2, {len(shape64_ops)} shape64")

        # Replace linear_v2(x, weight, bias) → matmul(x, weight) + add(bias)
        for op in linear_ops:
            x = op.operand(0).source()
            weight = op.operand(1).source()
            bias = op.operand(2).source()
            transpose_w = op.attrs().get("transpose_weight", False)
            output = op.result(0)

            with paddle.pir.core.program_guard(prog):
                paddle.pir.set_insertion_point(op)
                matmul_out = paddle._C_ops.matmul(x, weight, False, transpose_w)
                add_out = paddle._C_ops.add(matmul_out, bias)
                output.replace_all_uses_with(add_out)
            op.erase()

        # Replace shape64 → shape
        shape64_ops = [op for op in block.ops if op.name() == "pd_op.shape64"]
        for op in shape64_ops:
            x = op.operand(0).source()
            output = op.result(0)
            with paddle.pir.core.program_guard(prog):
                paddle.pir.set_insertion_point(op)
                shape_out = paddle._C_ops.shape(x)
                output.replace_all_uses_with(shape_out)
            op.erase()

    # Find feed/fetch Values for saving
    feed_vars = [op.result(0) for op in block.ops if op.name() == "pd_op.data"]
    fetch_vars = [op.operand(0).source() for op in block.ops if op.name() == "pd_op.fetch"]
    if not fetch_vars:
        fetch_vars = fetch_names if not isinstance(fetch_names[0], str) else []

    # Save decomposed model to temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="paddle_decomposed_"))
    save_prefix = str(tmp_dir / "inference")
    paddle.static.save_inference_model(save_prefix, feed_vars, fetch_vars, exe, program=prog)
    print(f"  Saved decomposed model to: {tmp_dir}")
    return tmp_dir


# ──────────────────────────── Core conversion ────────────────────────────────

def convert(
    model_dir: str,
    model_file: str | None = None,
    params_file: str | None = None,
    output: str = "model.onnx",
    opset: int = 14,
    simplify: bool = True,
) -> str:
    """Convert a Paddle inference model to ONNX and return the output path.

    Parameters
    ----------
    model_dir : str
        Directory containing the Paddle inference model files.
    model_file : str, optional
        Model definition filename (e.g. ``inference.json`` or
        ``inference.pdmodel``).  Auto-detected if omitted.
    params_file : str, optional
        Parameters filename (e.g. ``inference.pdiparams``).
        Auto-detected if omitted.
    output : str
        Destination path for the ONNX file.
    opset : int
        ONNX opset version (default 14).
    simplify : bool
        Run onnxsim after conversion (default True).

    Returns
    -------
    str
        Absolute path of the saved ONNX file.
    """
    if shutil.which("paddle2onnx") is None:
        sys.exit(
            "ERROR: paddle2onnx CLI not found.\n"
            "  pip install paddle2onnx paddlepaddle-gpu   # (or paddlepaddle for CPU)"
        )

    model_dir_path = Path(model_dir).resolve()
    if not model_dir_path.is_dir():
        sys.exit(f"ERROR: model_dir does not exist: {model_dir}")

    # Auto-detect filenames when not specified
    if model_file is None:
        model_file = _find_file(model_dir_path, _MODEL_EXTENSIONS)
        if model_file is None:
            sys.exit(
                f"ERROR: could not find a model file ({_MODEL_EXTENSIONS}) "
                f"in {model_dir}"
            )
        print(f"[auto-detect] model file  → {model_file}")

    if params_file is None:
        params_file = _find_file(model_dir_path, _PARAMS_EXTENSIONS)
        if params_file is None:
            sys.exit(
                f"ERROR: could not find a params file ({_PARAMS_EXTENSIONS}) "
                f"in {model_dir}"
            )
        print(f"[auto-detect] params file → {params_file}")

    model_path = model_dir_path / model_file
    params_path = model_dir_path / params_file

    if not model_path.is_file():
        sys.exit(f"ERROR: model file not found: {model_path}")
    if not params_path.is_file():
        sys.exit(f"ERROR: params file not found: {params_path}")

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"Model dir  : {model_dir_path}")
    print(f"Model file : {model_file}")
    print(f"Params file: {params_file}")
    print(f"Output     : {output_path}")
    print(f"Opset      : {opset}")
    print(f"{'─' * 60}\n")

    # For PIR models (.json), decompose unsupported ops first
    conversion_dir = model_dir_path
    tmp_dir = None
    if _is_pir_model(model_file):
        print("[PIR format detected] Decomposing unsupported ops...")
        tmp_dir = _decompose_pir_program(model_dir_path, model_file, params_file)
        conversion_dir = tmp_dir
        model_file = "inference.json"
        params_file = "inference.pdiparams"

    # Run paddle2onnx CLI
    print("[paddle2onnx] Converting to ONNX...")
    cmd = [
        "paddle2onnx",
        "--model_dir", str(conversion_dir),
        "--model_filename", model_file,
        "--params_filename", params_file,
        "--save_file", str(output_path),
        "--opset_version", str(opset),
        "--enable_auto_update_opset", "True",
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(f"ERROR: paddle2onnx failed with exit code {e.returncode}")

    # Clean up temp dir
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Simplify with onnxsim (constant folding, remove redundant nodes)
    if simplify:
        try:
            import onnx
            import onnxsim

            print("[onnxsim] Simplifying model...")
            model = onnx.load(str(output_path))
            model_opt, check = onnxsim.simplify(model)
            if check:
                onnx.save(model_opt, str(output_path))
                print(f"  Reduced: {len(model.graph.node)} → {len(model_opt.graph.node)} nodes")
            else:
                print("  WARNING: onnxsim validation failed, keeping original")
        except ImportError:
            print("  (onnxsim not installed, skipping simplification)")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ ONNX model saved: {output_path}  ({size_mb:.1f} MB)")
    return str(output_path)


# ──────────────────────────── Validation ─────────────────────────────────────

def validate(onnx_path: str) -> None:
    """Quick sanity checks on the exported ONNX model."""
    try:
        import onnx
    except ImportError:
        print("WARNING: 'onnx' package not installed — skipping graph check")
        onnx = None

    try:
        import onnxruntime as ort
    except ImportError:
        print("WARNING: 'onnxruntime' not installed — skipping runtime check")
        ort = None

    if onnx is not None:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model, full_check=True)
        print(f"✓ onnx.checker passed")

        for inp in model.graph.input:
            shape = [
                d.dim_param if d.dim_param else d.dim_value
                for d in inp.type.tensor_type.shape.dim
            ]
            print(f"  input : {inp.name}  shape={shape}")
        for out in model.graph.output:
            shape = [
                d.dim_param if d.dim_param else d.dim_value
                for d in out.type.tensor_type.shape.dim
            ]
            print(f"  output: {out.name}  shape={shape}")

    if ort is not None:
        import numpy as np

        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        sess = ort.InferenceSession(onnx_path, providers=providers)
        print(f"  ort providers: {sess.get_providers()}")

        meta = sess.get_inputs()[0]
        print(f"  ort input: name={meta.name}  shape={meta.shape}  "
              f"dtype={meta.type}")

        # Map ONNX tensor type strings to numpy dtypes
        _ONNX_TO_NP = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
            "tensor(bool)": bool,
        }
        dtype = _ONNX_TO_NP.get(meta.type, np.float32)

        # Build a dummy tensor matching the expected shape and dtype
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
        dummy = np.random.randn(*shape).astype(dtype)
        outputs = sess.run(None, {meta.name: dummy})
        print(f"  ort output shapes: {[o.shape for o in outputs]}")
        print(f"✓ onnxruntime inference passed (dummy input)")


# ─────────────────────────────── CLI ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PaddlePaddle inference model to ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_dir", default="container/weights/OCR/inference",
        help="Directory containing inference model files.",
    )
    parser.add_argument(
        "--model_file", default=None,
        help="Model definition filename inside model_dir (auto-detected).",
    )
    parser.add_argument(
        "--params_file", default=None,
        help="Params filename inside model_dir (auto-detected).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output ONNX file path (default: <model_dir>/model.onnx).",
    )
    parser.add_argument(
        "--opset", type=int, default=14,
        help="ONNX opset version (default: 14).",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation checks after conversion.",
    )
    parser.add_argument(
        "--no-simplify", action="store_true",
        help="Skip onnxsim simplification step.",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.model_dir, "model.onnx")

    onnx_path = convert(
        model_dir=args.model_dir,
        model_file=args.model_file,
        params_file=args.params_file,
        output=args.output,
        opset=args.opset,
        simplify=not args.no_simplify,
    )

    if args.validate:
        print()
        validate(onnx_path)


if __name__ == "__main__":
    main()
