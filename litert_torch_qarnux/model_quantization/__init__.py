"""
Model Quantization Module.

A production-quality module for quantizing models from various formats
(GGUF, ONNX, PyTorch .pt/.pth, SafeTensors, etc.) into optimized
quantized versions (INT4, INT8, FP16, etc.). Supports embedding skill.md
and chat templates, and categorized behavior emphasis fields.
"""
from litert_torch_qarnux.model_quantization.config import (
    QuantizationProfile,
    BehaviorCategory,
    BehaviorProfile,
    TemplateConfig,
)
from litert_torch_qarnux.model_quantization.quantizer import ModelQuantizer
from litert_torch_qarnux.model_quantization.agent import QuantizationAgent

__all__ = [
    "QuantizationProfile",
    "BehaviorCategory",
    "BehaviorProfile",
    "TemplateConfig",
    "ModelQuantizer",
    "QuantizationAgent",
]
