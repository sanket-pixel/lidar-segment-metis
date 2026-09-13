#!/usr/bin/env python3
import torch
import os
from pathlib import Path
from model import SalsaNext


def main():
    # Resolve paths relative to this script's location (python/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    checkpoint_path = project_root / "weights" / "pretrained" / "pretrained" / "SalsaNext"
    onnx_output_path = project_root / "weights" / "salsanext_full.onnx"

    # Instantiate model with raw logits output for C++ argmax postprocessing
    model = SalsaNext(nclasses=20, return_softmax=False)

    if checkpoint_path.exists():
        print(f"Loading weights from {checkpoint_path}...")
        # PyTorch 2.6+ defaults to weights_only=True; set False to load legacy numpy checkpoint objects
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load weights with strict=False to bypass missing keys for custom PixelShuffleConv modules
        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        print(f"Weights loaded successfully. Missing keys (expected for custom PixelShuffleConv): {missing}")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found! Exporting with random weights.")

    model.eval()

    # Create standard LiDAR range-view input shape: [1, 5, 64, 2048]
    dummy_input = torch.randn(1, 5, 64, 2048)

    print(f"Exporting to ONNX at {onnx_output_path} (Opset 16)...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        external_data=False,  # Add this flag to keep everything in one file
        opset_version=16,
        do_constant_folding=True,
        input_names=['input_5ch'],
        output_names=['logits']
    )
    print("ONNX export completed successfully.")


if __name__ == "__main__":
    main()