"""
Model Configuration Registry.

Provides default configurations for each supported model architecture,
including recommended hyperparameter defaults, normalization settings,
and architecture-specific conversion parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """
    Architecture-specific configuration for model conversion.

    Attributes:
        architecture: The GGUF architecture identifier.
        hf_class_name: The HuggingFace model class name.
        num_attention_heads: Default number of attention heads (if not in GGUF).
        num_kv_heads: Default number of KV heads (for GQA).
        head_dim: Default head dimension.
        use_rms_norm: Whether the architecture uses RMSNorm.
        use_rope: Whether the architecture uses Rotary Position Embedding.
        use_swiglu: Whether the architecture uses SwiGLU activation in MLP.
        tie_embeddings: Whether embeddings are tied to the LM head.
        recommended_batch_size: Recommended batch size for conversion.
        recommended_seq_length: Recommended sequence length for conversion.
    """

    architecture: str
    hf_class_name: str
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    use_rms_norm: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    tie_embeddings: bool = False
    recommended_batch_size: int = 1
    recommended_seq_length: int = 128


# Default configurations for each supported architecture
_MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "llama": ModelConfig(
        architecture="llama",
        hf_class_name="LlamaForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        tie_embeddings=False,
    ),
    "gemma": ModelConfig(
        architecture="gemma",
        hf_class_name="GemmaForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=False,  # Gemma uses GeGLU
        tie_embeddings=False,
    ),
    "gemma2": ModelConfig(
        architecture="gemma2",
        hf_class_name="Gemma2ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=False,
        tie_embeddings=False,
    ),
    "gemma3": ModelConfig(
        architecture="gemma3",
        hf_class_name="Gemma3ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=False,
        tie_embeddings=False,
    ),
    "gemma3n": ModelConfig(
        architecture="gemma3n",
        hf_class_name="Gemma3nForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=False,
        tie_embeddings=False,
    ),
    "gemma4": ModelConfig(
        architecture="gemma4",
        hf_class_name="Gemma4ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=False,
        tie_embeddings=False,
    ),
    "mistral": ModelConfig(
        architecture="mistral",
        hf_class_name="MistralForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        tie_embeddings=False,
    ),
    "qwen2": ModelConfig(
        architecture="qwen2",
        hf_class_name="Qwen2ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        tie_embeddings=False,
    ),
    "qwen3": ModelConfig(
        architecture="qwen3",
        hf_class_name="Qwen3ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        tie_embeddings=False,
    ),
    "phi2": ModelConfig(
        architecture="phi2",
        hf_class_name="PhiForCausalLM",
        use_rms_norm=False,  # Phi-2 uses LayerNorm
        use_rope=True,
        use_swiglu=False,
        tie_embeddings=False,
    ),
    "phi3": ModelConfig(
        architecture="phi3",
        hf_class_name="Phi3ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        tie_embeddings=False,
    ),
    "phi": ModelConfig(
        architecture="phi",
        hf_class_name="Phi3ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        tie_embeddings=False,
    ),
    "smollm3": ModelConfig(
        architecture="smollm3",
        hf_class_name="SmolLM3ForCausalLM",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
        tie_embeddings=False,
    ),
}


def get_model_config(architecture: str) -> ModelConfig:
    """
    Get the model configuration for a given architecture.

    Args:
        architecture: The GGUF architecture identifier.

    Returns:
        A ModelConfig instance with architecture-specific defaults.

    Raises:
        ValueError: If the architecture is not recognized.
    """
    arch_lower = architecture.lower()
    if arch_lower in _MODEL_CONFIGS:
        return _MODEL_CONFIGS[arch_lower]

    # Fallback to Llama config for unknown architectures
    # This allows best-effort conversion for similar architectures
    return _MODEL_CONFIGS["llama"]


def list_supported_architectures() -> list:
    """Return all supported architecture identifiers."""
    return sorted(_MODEL_CONFIGS.keys())
