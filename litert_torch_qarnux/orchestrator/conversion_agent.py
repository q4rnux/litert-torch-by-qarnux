"""
ConversionAgent - PyTorch to TFLite Conversion.

Handles the conversion of the authored PyTorch model to TFLite format
(.tflite file). Supports both direct litert-torch conversion and
ONNX-based intermediate conversion as a fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent
from litert_torch_qarnux.utils.tflite_converter import TFLiteConverter

logger = logging.getLogger(__name__)


class ConversionAgent(BaseAgent):
    """
    Converts the PyTorch model to TFLite format.

    Uses the TFLiteConverter utility to produce a .tflite file suitable
    for packaging into a .litertlm container. Supports quantization
    configuration and multiple conversion backends.
    """

    def __init__(
        self,
        output_dir: str | Path,
        quantize: bool = True,
        quantization_recipe: str = "dynamic_wi8_afp32",
    ):
        """
        Initialize the ConversionAgent.

        Args:
            output_dir: Directory for intermediate TFLite output.
            quantize: Whether to apply quantization during conversion.
            quantization_recipe: Name of the quantization recipe.
        """
        super().__init__("conversion")
        self.output_dir = Path(output_dir)
        self.quantize = quantize
        self.quantization_recipe = quantization_recipe

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Convert the PyTorch model to TFLite format.

        Returns:
            AgentMessage containing the path to the generated .tflite file.
        """
        model = message.data["model"]
        metadata = message.data["metadata"]

        self.logger.info("Starting PyTorch to TFLite conversion...")
        self.logger.info("  Architecture: %s", metadata.architecture)
        self.logger.info("  Quantization: %s", self.quantization_recipe)

        # Create output path for TFLite model
        tflite_path = self.output_dir / f"{metadata.architecture}_model.tflite"

        # Initialize converter
        converter = TFLiteConverter(
            model=model,
            quantize=self.quantize,
            quantization_recipe=self.quantization_recipe,
        )

        # Perform conversion
        tflite_path = converter.convert(tflite_path)

        # Verify output
        if not tflite_path.exists():
            raise RuntimeError(f"TFLite conversion failed: output not found at {tflite_path}")

        file_size_mb = tflite_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "TFLite conversion complete: %s (%.1f MB)", tflite_path, file_size_mb
        )

        return AgentMessage(
            source=self.agent_id,
            target="packaging",
            data={
                "metadata": metadata,
                "tflite_path": str(tflite_path),
                "model": model,
            },
        )
