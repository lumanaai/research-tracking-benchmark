"""
Export MobilenetLSTM violence classifier from PyTorch (.pt) to ONNX (.onnx).

Usage:
    python -m export_violence_classifier -w <weights.pt> -o <output_dir> [--im-size 384 384] [--seq-len 5]

The model expects input shape [batch, T, 3, H, W] where T=seq_len.
"""

import argparse
import os
from pathlib import Path

import torch

from level1.violence.violence_models import MobilenetLSTM


def export_to_onnx(weights_path: str, output_dir: str, im_size=(384, 384), seq_len=5):
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model and load weights
    model = MobilenetLSTM(num_classes=1, seq_length=seq_len).to(device)
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input: [batch=1, T=seq_len, 3, H, W]
    H, W = im_size
    dummy_input = torch.randn(1, seq_len, 3, H, W, device=device)

    onnx_path = output_dir / f"{weights_path.stem}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"Exported ONNX model to: {onnx_path}")
    print(f"  Input shape:  [batch, {seq_len}, 3, {H}, {W}]")
    print(f"  Output shape: [batch, 1]")
    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MobilenetLSTM violence classifier to ONNX")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to .pt weights file")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory for .onnx file")
    parser.add_argument("--im-size", type=int, nargs=2, default=[384, 384], help="Input image size (H W)")
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length T (default: 5)")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output_dir, tuple(args.im_size), args.seq_len)
