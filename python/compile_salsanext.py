#!/usr/bin/env python3
"""
Compiles SalsaNext full graph for Axelera Metis AIPU.
Input:  input_5ch (1, 5, 64, 2048)
Output: logits (1, 20, 64, 2048)

Usage:
    cd python && python compile_salsanext.py
"""

import numpy as np
from pathlib import Path
from axelera import compiler
from axelera.compiler import CompilerConfig

# Resolve relative paths from script location
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ONNX_PATH    = PROJECT_ROOT / "weights" / "salsanext_full.onnx"
COMPILED_DIR = PROJECT_ROOT / "weights" / "compiled_salsanext"
NUM_CALIB    = 10

# SalsaNext Input Dimensions: [Batch, Channels (Range, X, Y, Z, Remission), Height, Width]
INPUT_C = 5
INPUT_H = 64
INPUT_W = 2048


def calibration_generator():
    """
    Yields synthetic random 5-channel range-view tensors for quantization calibration.
    Replace this with actual range-view numpy loader when dataset is ready.
    """
    print(f"Generating {NUM_CALIB} synthetic calibration samples [1, {INPUT_C}, {INPUT_H}, {INPUT_W}]...")
    for _ in range(NUM_CALIB):
        # Generate random float32 tensor matching LiDAR range-view channel distribution
        sample = np.random.randn(1, INPUT_C, INPUT_H, INPUT_W).astype(np.float32)
        yield sample


def main():
    compiled_out  = COMPILED_DIR
    quantized_out = compiled_out / "quantized"

    compiled_out.mkdir(parents=True, exist_ok=True)
    quantized_out.mkdir(parents=True, exist_ok=True)

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at '{ONNX_PATH}'. Run export_onnx.py first!")

    # Configure Axelera compiler for full monolithic AIPU graph
    config = CompilerConfig(
        remove_output_dir=True,
        save_error_artifact=True,
        model_name="salsanext_full",
        output_dir=str(compiled_out),
        aipu_cores_used=1,              # Utilize all 4 Metis AIPU cores
        resources_used=0.25,             # Allocate 100% SRAM resources
        pipeline_spatial_tiles=True,    # Spatial tiling for large 2048 width
        pipeline_channel_tiles=True,   # Channel tiling for 32->256 feature maps
        inter_operator_async=True,
        use_hw_tokens=True,
        double_buffer=False,            # Disabled for lower memory pressure on large feature maps
        enable_buffer_promotion=True,
        enable_icr=True,
        enable_swicr=True,
        dma_dual_channel=True,
        run_graph_cleaner=False,
        quantization_scheme="per_tensor_histogram",
        dpu_allocation_algorithm="try_all",
        tiling_depth=1,
    )

    print(f"--- Quantizing SalsaNext ---")
    quantized = compiler.quantize(
        model=str(ONNX_PATH),
        calibration_dataset=calibration_generator(),
        config=config,
    )
    quantized.export(str(quantized_out))

    print(f"\n--- Compiling SalsaNext to Metis Hardware Artifacts ---")
    compiler.compile(
        model=quantized,
        config=config,
        output_dir=compiled_out,
    )
    print(f"\nSuccessfully compiled SalsaNext! Output location: '{compiled_out}'")


if __name__ == "__main__":
    main()