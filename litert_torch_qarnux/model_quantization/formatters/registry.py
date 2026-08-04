"""
Format Registry.

Auto-detects model format from file extension or magic bytes and
returns the appropriate handler.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from litert_torch_qarnux.model_quantization.config import QuantizationProfile, SourceFormat
from litert_torch_qarnux.model_quantization.formatters.base import BaseFormatHandler

logger = logging.getLogger(__name__)

# Extension -> Format mapping
EXT_TO_FORMAT: Dict[str, SourceFormat] = {
    ".gguf": SourceFormat.GGUF,
    ".onnx": SourceFormat.ONNX,
    ".pt": SourceFormat.PYTORCH,
    ".pth": SourceFormat.PYTORCH,
    ".ckpt": SourceFormat.PYTORCH,
    ".safetensors": SourceFormat.SAFETENSORS,
    ".bin": SourceFormat.BINARY,
}


class FormatRegistry:
    """Registry of format handlers with auto-detection capabilities."""

    _handlers: Dict[SourceFormat, Type[BaseFormatHandler]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_registered(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        # Import handlers locally to avoid circular imports
        from litert_torch_qarnux.model_quantization.formatters.gguf_handler import GGUFFormatHandler
        from litert_torch_qarnux.model_quantization.formatters.onnx_handler import ONNXFormatHandler
        from litert_torch_qarnux.model_quantization.formatters.pytorch_handler import PyTorchFormatHandler
        from litert_torch_qarnux.model_quantization.formatters.safetensors_handler import SafeTensorsFormatHandler

        cls._handlers[SourceFormat.GGUF] = GGUFFormatHandler
        cls._handlers[SourceFormat.ONNX] = ONNXFormatHandler
        cls._handlers[SourceFormat.PYTORCH] = PyTorchFormatHandler
        cls._handlers[SourceFormat.SAFETENSORS] = SafeTensorsFormatHandler
        logger.debug("FormatRegistry initialized with %d handlers", len(cls._handlers))

    @classmethod
    def register(cls, fmt: SourceFormat, handler_class: Type[BaseFormatHandler]) -> None:
        cls._ensure_registered()
        cls._handlers[fmt] = handler_class
        logger.debug("Registered handler %s for format %s", handler_class.__name__, fmt.value)

    @classmethod
    def get_handler(cls, fmt: SourceFormat, profile: QuantizationProfile) -> BaseFormatHandler:
        cls._ensure_registered()
        handler_class = cls._handlers.get(fmt)
        if handler_class is None:
            raise ValueError(f"No handler registered for format: {fmt.value}")
        return handler_class(profile)

    @classmethod
    def detect_format(cls, path: Path) -> SourceFormat:
        """Auto-detect model format from file extension."""
        cls._ensure_registered()
        ext = path.suffix.lower()
        fmt = EXT_TO_FORMAT.get(ext)
        if fmt is not None:
            return fmt
        # Check if it's a directory (HuggingFace transformers format)
        if path.is_dir():
            if (path / "config.json").exists():
                return SourceFormat.HUGGINGFACE
        logger.warning("Could not auto-detect format for %s", path)
        return SourceFormat.AUTO

    @classmethod
    def detect_and_get_handler(cls, path: Path, profile: QuantizationProfile) -> BaseFormatHandler:
        """Auto-detect format and return the appropriate handler."""
        fmt = cls.detect_format(path)
        if fmt == SourceFormat.AUTO:
            raise ValueError(
                f"Could not auto-detect format for {path}. "
                f"Please specify source_format explicitly."
            )
        # Use the profile's source_format if it's not auto
        if profile.source_format != SourceFormat.AUTO:
            fmt = profile.source_format
        return cls.get_handler(fmt, profile)

    @classmethod
    def supported_formats(cls) -> List[str]:
        """Return list of supported format names."""
        cls._ensure_registered()
        return [fmt.value for fmt in cls._handlers.keys()]

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return list of all supported file extensions."""
        cls._ensure_registered()
        exts = set()
        for handler_class in cls._handlers.values():
            exts.update(handler_class.SUPPORTED_EXTENSIONS)
        return sorted(exts)
