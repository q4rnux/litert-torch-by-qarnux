"""
Model Quantizer.

High-level orchestrator that coordinates the quantization pipeline:
detecting format, quantizing, embedding templates, and writing behavior
metadata.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from litert_torch_qarnux.model_quantization.config import QuantizationProfile
from litert_torch_qarnux.model_quantization.formatters.registry import FormatRegistry

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Main entry point for model quantization.

    Usage:
        profile = QuantizationProfile.from_file("config.yaml")
        quantizer = ModelQuantizer(profile)
        result = quantizer.quantize()
    """

    def __init__(self, profile: QuantizationProfile):
        self.profile = profile
        self.handler = None

    def quantize(self) -> Dict[str, Any]:
        """
        Execute the full quantization pipeline.

        Returns:
            Dictionary with keys:
                - output_path: Path to the quantized model
                - format: Detected source format
                - metadata: Extracted model metadata
                - elapsed_seconds: Pipeline duration
        """
        start_time = time.time()
        input_path = Path(self.profile.input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input model not found: {input_path}")

        # Validate output path
        if self.profile.output_path:
            output_dir = Path(self.profile.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Model Quantization Pipeline")
        logger.info("=" * 60)
        logger.info("Input:      %s", input_path)
        logger.info("Method:     %s", self.profile.method.value)
        logger.info("Output dtype: %s", self.profile.output_dtype.value)
        logger.info("Output:     %s", self._get_output_path_str())
        logger.info("=" * 60)

        result: Dict[str, Any] = {}

        try:
            # Step 1: Detect format and get handler
            handler = FormatRegistry.detect_and_get_handler(input_path, self.profile)
            self.handler = handler
            detected_format = handler.detect_format(input_path)
            logger.info("Detected format: %s", detected_format)

            # Step 2: Load metadata
            metadata = handler.load_metadata(input_path)
            logger.info("Metadata loaded: %d fields", len(metadata))

            # Step 3: Quantize
            logger.info("Quantizing...")
            quantized_path = handler.quantize(input_path)
            logger.info("Quantization complete: %s", quantized_path)

            # Step 4: Embed templates
            if self._has_template_data():
                logger.info("Embedding templates...")
                handler.embed_template(quantized_path)

            # Step 5: Write behavior metadata
            if self._has_behavior_data():
                logger.info("Writing behavior metadata...")
                handler.write_behavior_metadata(quantized_path)

            # Step 6: Compile results
            elapsed = time.time() - start_time
            file_size = quantized_path.stat().st_size if quantized_path.exists() else 0

            result = {
                "success": True,
                "output_path": str(quantized_path),
                "format": detected_format,
                "metadata": metadata,
                "file_size_bytes": file_size,
                "elapsed_seconds": elapsed,
            }

            logger.info("=" * 60)
            logger.info("Quantization completed successfully!")
            logger.info("  Output: %s", quantized_path)
            logger.info("  Size: %.2f MB", file_size / (1024 * 1024))
            logger.info("  Time: %.1f seconds", elapsed)
            logger.info("=" * 60)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Quantization failed after %.1fs: %s", elapsed, e)
            result = {
                "success": False,
                "error": str(e),
                "elapsed_seconds": elapsed,
            }
            raise

        return result

    def _has_template_data(self) -> bool:
        """Check if any template data is configured."""
        t = self.profile.template
        return any([
            t.chat_template,
            t.system_prompt,
            t.personality,
            t.role,
            t.skill_md_content,
            t.skill_md_path,
        ])

    def _has_behavior_data(self) -> bool:
        """Check if any non-default behavior data is configured."""
        return any(
            abs(v.emphasis) > 0.01
            for v in self.profile.behavior.categories.values()
        )

    def _get_output_path_str(self) -> str:
        if self.profile.output_path:
            return self.profile.output_path
        return f"{self.profile.input_path}.quantized.{self.profile.output_dtype.value}"
