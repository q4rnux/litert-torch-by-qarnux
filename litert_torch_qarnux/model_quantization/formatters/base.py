"""
Base Format Handler.

Abstract base class for all model format handlers. Each handler
implements reading, quantizing, and writing for a specific model format.
"""
from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from litert_torch_qarnux.model_quantization.config import (
    BehaviorProfile,
    QuantizationProfile,
    TemplateConfig,
)

logger = logging.getLogger(__name__)


class BaseFormatHandler(abc.ABC):
    """Abstract base for format-specific quantization handlers."""

    FORMAT_NAME: str = "unknown"
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = ()

    def __init__(self, profile: QuantizationProfile):
        self.profile = profile

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Return True if this handler can process the given file."""
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def detect_format(self, path: Path) -> str:
        """
        Detect the exact format variant (e.g., GGUF Q4_K_M, ONNX opset 15).
        Returns a human-readable format string.
        """
        ...

    @abc.abstractmethod
    def load_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Extract metadata from the model file (architecture, dims, etc.).
        Returns a dict of metadata key-value pairs.
        """
        ...

    @abc.abstractmethod
    def quantize(self, path: Path) -> Path:
        """
        Perform quantization on the model file according to the profile.
        Returns the path to the quantized output file.
        """
        ...

    @abc.abstractmethod
    def embed_template(self, quantized_path: Path) -> None:
        """
        Embed chat template, skill.md, or system prompt into the
        quantized model file.
        """
        ...

    @abc.abstractmethod
    def write_behavior_metadata(self, quantized_path: Path) -> None:
        """
        Write behavior category emphasis fields into the quantized
        model's metadata section.
        """
        ...

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _get_output_path(self, input_path: Path) -> Path:
        """Determine the output file path."""
        if self.profile.output_path:
            return Path(self.profile.output_path)
        suffix = {
            "gguf": ".gguf",
            "onnx": ".onnx",
            "pytorch": ".pt",
            "safetensors": ".safetensors",
        }.get(self.profile.output_format.value, ".gguf")
        return input_path.parent / f"{input_path.stem}_quantized_{self.profile.output_dtype.value}{suffix}"

    def _log_info(self, msg: str) -> None:
        logger.info("[%s] %s", self.FORMAT_NAME, msg)
