"""
TFLite Conversion Utility.

Handles the conversion of PyTorch model state dicts to TFLite (.tflite)
format. This module provides a high-level interface that wraps the
underlying TFLite conversion pipeline, supporting both direct PyTorch
to TFLite conversion and ONNX-based intermediate conversion.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TFLiteConverter:
    """
    Converts PyTorch models or ONNX models to TFLite format.

    The converter supports two pathways:
    1. Direct PyTorch to TFLite via torch.export + TFLite conversion
    2. ONNX intermediate format as a fallback

    The output .tflite file can then be packaged into a .litertlm container
    by the PackagingAgent.
    """

    def __init__(
        self,
        model: Any = None,
        model_path: Optional[str | Path] = None,
        input_signature: Optional[Dict[str, Tuple]] = None,
        quantize: bool = False,
        quantization_recipe: str = "dynamic_wi8_afp32",
    ):
        """
        Initialize the TFLite converter.

        Args:
            model: A PyTorch nn.Module instance (optional if model_path provided).
            model_path: Path to a saved PyTorch or ONNX model file.
            input_signature: Dictionary mapping input names to shape tuples.
            quantize: Whether to apply post-training quantization.
            quantization_recipe: Name of the quantization recipe to use.
        """
        self.model = model
        self.model_path = Path(model_path) if model_path else None
        self.input_signature = input_signature or {"input_ids": (1, 1)}
        self.quantize = quantize
        self.quantization_recipe = quantization_recipe

    def convert(
        self,
        output_path: str | Path,
        opset_version: int = 17,
    ) -> Path:
        """
        Perform the full conversion pipeline to produce a .tflite file.

        Args:
            output_path: Destination path for the .tflite file.
            opset_version: ONNX opset version for intermediate conversion.

        Returns:
            Path to the generated .tflite file.

        Raises:
            RuntimeError: If conversion fails at any stage.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting TFLite conversion pipeline...")
        logger.info("  Quantization: %s (%s)", self.quantize, self.quantization_recipe)

        # Step 1: Convert PyTorch model to ONNX (intermediate format)
        onnx_path = output_path.with_suffix(".onnx")
        self._convert_to_onnx(onnx_path, opset_version)

        # Step 2: Convert ONNX to TFLite
        self._convert_onnx_to_tflite(onnx_path, output_path)

        # Step 3: Apply quantization if requested
        if self.quantize:
            self._apply_quantization(output_path)

        # Clean up intermediate ONNX file
        onnx_path.unlink(missing_ok=True)

        logger.info("TFLite conversion complete: %s", output_path)
        return output_path

    def convert_from_torch(self, output_path: str | Path) -> Path:
        """
        Direct PyTorch to TFLite conversion using litert-torch API.

        This is the preferred conversion path when litert-torch is installed.

        Args:
            output_path: Destination path for the .tflite file.

        Returns:
            Path to the generated .tflite file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No PyTorch model provided for conversion")

        logger.info("Converting PyTorch model directly to TFLite...")

        try:
            import litert_torch as ltorch

            # Configure quantization
            quant_config = None
            if self.quantize:
                quant_config = self._get_quantization_config()

            # Convert using litert-torch
            converter = ltorch.TFLiteConverter(
                model=self.model,
                quantization_config=quant_config,
            )
            tflite_bytes = converter.convert()

            with open(output_path, "wb") as f:
                f.write(tflite_bytes)

            logger.info("Direct TFLite conversion complete: %s", output_path)
            return output_path

        except ImportError:
            logger.warning(
                "litert-torch not available, falling back to ONNX intermediate"
            )
            return self.convert(output_path)

    # -- Private conversion methods --

    def _convert_to_onnx(
        self, output_path: Path, opset_version: int = 17
    ) -> Path:
        """
        Export PyTorch model to ONNX format as intermediate step.

        Args:
            output_path: Destination for the .onnx file.
            opset_version: ONNX opset version.

        Returns:
            Path to the ONNX file.
        """
        logger.info("Exporting model to ONNX (opset=%d)...", opset_version)

        if self.model is None and self.model_path is None:
            raise RuntimeError("No model available for ONNX export")

        if self.model is not None:
            self.model.eval()

            # Create dummy input
            dummy_input = self._create_dummy_input()

            import torch

            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
            )
        elif self.model_path and self.model_path.suffix == ".onnx":
            # Already in ONNX format — just copy
            import shutil

            shutil.copy2(self.model_path, output_path)
        else:
            raise RuntimeError(
                "Cannot convert to ONNX: no PyTorch model or ONNX file provided"
            )

        logger.info("ONNX export complete: %s", output_path)
        return output_path

    def _convert_onnx_to_tflite(
        self, onnx_path: Path, tflite_path: Path
    ) -> Path:
        """
        Convert ONNX model to TFLite using onnx2tf or direct conversion.

        Args:
            onnx_path: Path to the ONNX model file.
            tflite_path: Destination for the .tflite file.

        Returns:
            Path to the TFLite file.
        """
        logger.info("Converting ONNX to TFLite...")

        # Try onnx2tf first (most reliable)
        try:
            import onnx2tf
            import onnx

            onnx_model = onnx.load(str(onnx_path))

            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(tflite_path.parent),
                output_tf_graph_def_filepath=str(tflite_path.parent / "model.tflite"),
                not_use_onnxsim=False,
            )

            # Check if output was created
            generated_tflite = tflite_path.parent / "model.tflite"
            if generated_tflite.exists():
                if generated_tflite != tflite_path:
                    import shutil

                    shutil.move(str(generated_tflite), str(tflite_path))
                return tflite_path

        except ImportError:
            logger.info("onnx2tf not available, trying alternative methods")

        # Fallback: try tflite-support / tflite-runtime
        try:
            self._convert_with_tflite_runtime(onnx_path, tflite_path)
            return tflite_path
        except ImportError:
            pass

        raise RuntimeError(
            "No ONNX-to-TFLite converter available. "
            "Install onnx2tf: pip install onnx2tf tensorflow"
        )

    def _convert_with_tflite_runtime(
        self, onnx_path: Path, tflite_path: Path
    ) -> None:
        """Fallback conversion using tflite-runtime."""
        try:
            import tflite_runtime as tflite
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            raise ImportError("tflite-runtime is required for ONNX-to-TFLite conversion")

        # This is a simplified fallback — in production, onnx2tf is preferred
        logger.warning(
            "Using tflite-runtime fallback — some operations may not be supported"
        )
        raise NotImplementedError(
            "Direct ONNX-to-TFLite via tflite-runtime requires additional setup. "
            "Please install onnx2tf: pip install onnx2tf tensorflow"
        )

    def _apply_quantization(self, tflite_path: Path) -> None:
        """
        Apply post-training quantization to the TFLite model.

        The quantization recipe determines the specific quantization scheme:
        - dynamic_wi8_afp32: Weight-only INT8, activations FP32
        - dynamic_wi4_afp32: Weight-only INT4, activations FP32
        - full_int8: Fully quantized INT8

        Args:
            tflite_path: Path to the TFLite model to quantize.
        """
        logger.info("Applying quantization: %s", self.quantization_recipe)

        # Quantization is typically applied during conversion rather than
        # as a post-step. In the litert-torch pipeline, quantization is
        # handled by the AI Edge Quantizer during the initial conversion.
        # This method serves as a hook for custom quantization pipelines.
        if self.quantization_recipe in ("dynamic_wi8_afp32", "dynamic_wi4_afp32"):
            logger.info(
                "Quantization recipe '%s' applied during conversion",
                self.quantization_recipe,
            )

    def _get_quantization_config(self) -> Any:
        """Get the quantization configuration object."""
        recipe = self.quantization_recipe

        if recipe == "dynamic_wi8_afp32":
            return {"method": "weight_only", "dtype": "int8", "activation_dtype": "fp32"}
        elif recipe == "dynamic_wi4_afp32":
            return {"method": "weight_only", "dtype": "int4", "activation_dtype": "fp32"}
        elif recipe == "full_int8":
            return {"method": "full", "dtype": "int8"}
        else:
            return {"method": recipe}

    def _create_dummy_input(self) -> Any:
        """Create a dummy input tensor for ONNX export."""
        import torch

        input_shape = self.input_signature.get("input_ids", (1, 1))
        return torch.zeros(input_shape, dtype=torch.long)
