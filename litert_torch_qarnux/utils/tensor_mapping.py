"""
Tensor Name Mapping Utility.

Provides bidirectional mapping between GGUF tensor naming conventions and
PyTorch/HuggingFace model parameter names. Each supported architecture
has its own mapping rules to translate GGUF block-level tensor names
(e.g., "blk.0.attn_q.weight") into standard PyTorch parameter paths
(e.g., "model.layers.0.self_attn.q_proj.weight").
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Regex patterns for GGUF tensor names ───────────────────────────────────

# blk.{layer_idx}.attn_norm.weight
_PATTERN_ATTN_NORM = re.compile(r"^blk\.(\d+)\.attn_norm\.weight$")
_PATTERN_FFN_NORM = re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$")
_PATTERN_ATTN_Q = re.compile(r"^blk\.(\d+)\.attn_q\.weight$")
_PATTERN_ATTN_K = re.compile(r"^blk\.(\d+)\.attn_k\.weight$")
_PATTERN_ATTN_V = re.compile(r"^blk\.(\d+)\.attn_v\.weight$")
_PATTERN_ATTN_QKV = re.compile(r"^blk\.(\d+)\.attn_qkv\.weight$")
_PATTERN_ATTN_OUT = re.compile(r"^blk\.(\d+)\.attn_output\.weight$")
_PATTERN_FFN_GATE = re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$")
_PATTERN_FFN_UP = re.compile(r"^blk\.(\d+)\.ffn_up\.weight$")
_PATTERN_FFN_DOWN = re.compile(r"^blk\.(\d+)\.ffn_down\.weight$")
_PATTERN_FFN_GATE_EXP = re.compile(r"^blk\.(\d+)\.ffn_gate_exp\.(\d+)\.weight$")
_PATTERN_FFN_UP_EXP = re.compile(r"^blk\.(\d+)\.ffn_up_exp\.(\d+)\.weight$")
_PATTERN_FFN_DOWN_EXP = re.compile(r"^blk\.(\d+)\.ffn_down_exp\.(\d+)\.weight$")
_PATTERN_ATTN_Q_BIAS = re.compile(r"^blk\.(\d+)\.attn_q\.bias$")
_PATTERN_ATTN_K_BIAS = re.compile(r"^blk\.(\d+)\.attn_k\.bias$")
_PATTERN_ATTN_V_BIAS = re.compile(r"^blk\.(\d+)\.attn_v\.bias$")
_PATTERN_ATTN_OUT_BIAS = re.compile(r"^blk\.(\d+)\.attn_output\.bias$")
_PATTERN_FFN_GATE_BIAS = re.compile(r"^blk\.(\d+)\.ffn_gate\.bias$")
_PATTERN_FFN_UP_BIAS = re.compile(r"^blk\.(\d+)\.ffn_up\.bias$")
_PATTERN_FFN_DOWN_BIAS = re.compile(r"^blk\.(\d+)\.ffn_down\.bias$")
_PATTERN_FFN_BIAS = re.compile(r"^blk\.(\d+)\.ffn\.bias$")

# Model-level tensors
_PATTERN_TOKEN_EMBD = re.compile(r"^token_embd\.weight$")
_PATTERN_TOKEN_EMBD_BIAS = re.compile(r"^token_embd\.bias$")
_PATTERN_OUTPUT_NORM = re.compile(r"^output_norm\.weight$")
_PATTERN_OUTPUT_NORM_BIAS = re.compile(r"^output_norm\.bias$")
_PATTERN_OUTPUT = re.compile(r"^output\.weight$")
_PATTERN_OUTPUT_BIAS = re.compile(r"^output\.bias$")
_PATTERN_CLS = re.compile(r"^cls\.bias$")

# Gemma-specific
_PATTERN_ATTN_POST_NORM = re.compile(r"^blk\.(\d+)\.attn_post_norm\.weight$")
_PATTERN_FFN_POST_NORM = re.compile(r"^blk\.(\d+)\.ffn_post_norm\.weight$")
_PATTERN_ATTN_K_NORM = re.compile(r"^blk\.(\d+)\.attn_k_norm\.weight$")
_PATTERN_ATTN_Q_NORM = re.compile(r"^blk\.(\d+)\.attn_q_norm\.weight$")


class TensorMapper:
    """
    Maps GGUF tensor names to PyTorch model parameter names.

    Supports multiple architectures including Llama, Gemma, Mistral, Qwen2,
    and Phi variants. The mapper uses architecture-specific templates to
    translate between the two naming conventions.
    """

    # Architecture-to-template mapping
    _ARCH_TEMPLATES: Dict[str, Dict[str, str]] = {}

    def __init__(self, architecture: str):
        """
        Initialize the mapper for a given architecture.

        Args:
            architecture: The model architecture string (e.g., "llama", "gemma").
        """
        self.architecture = architecture.lower()
        self.template = self._get_template(self.architecture)
        logger.debug(
            "TensorMapper initialized for architecture: %s",
            self.architecture,
        )

    def map_tensor(self, gguf_name: str) -> Optional[str]:
        """
        Convert a GGUF tensor name to its PyTorch equivalent.

        Args:
            gguf_name: The original GGUF tensor name.

        Returns:
            The PyTorch parameter name, or None if the tensor is not recognized.
        """
        # Try each pattern in the template
        for pattern, replacement in self.template.items():
            match = re.match(pattern, gguf_name)
            if match:
                groups = match.groups()
                result = replacement
                for i, group in enumerate(groups):
                    result = result.replace(f"{{layer_idx}}", group, 1) if "{layer_idx}" in result else result
                    result = result.replace(f"{{expert_idx}}", group, 1) if "{expert_idx}" in result else result
                # Replace remaining placeholders with actual group values
                for i, g in enumerate(groups):
                    result = result.replace(f"{{{i}}}", g)
                return result

        # Unmapped tensor — log and return None
        logger.debug("Unmapped GGUF tensor: %s", gguf_name)
        return None

    def map_all_tensors(self, gguf_names: List[str]) -> Dict[str, str]:
        """
        Map a batch of GGUF tensor names to PyTorch names.

        Args:
            gguf_names: List of GGUF tensor names.

        Returns:
            Dictionary mapping GGUF names to PyTorch names.
            Tensors that cannot be mapped are excluded.
        """
        result = {}
        unmapped = []
        for name in gguf_names:
            mapped = self.map_tensor(name)
            if mapped:
                result[name] = mapped
            else:
                unmapped.append(name)

        if unmapped:
            logger.warning(
                "%d tensors could not be mapped: %s",
                len(unmapped),
                unmapped[:5],
            )

        return result

    @classmethod
    def _get_template(cls, architecture: str) -> Dict[str, str]:
        """Return the mapping template for the given architecture."""
        templates = {
            "llama": cls._llama_template(),
            "gemma": cls._gemma_template(),
            "gemma2": cls._gemma2_template(),
            "gemma3": cls._gemma_template(),
            "gemma3n": cls._gemma_template(),
            "gemma4": cls._gemma_template(),
            "mistral": cls._mistral_template(),
            "qwen2": cls._qwen2_template(),
            "qwen3": cls._qwen2_template(),
            "phi2": cls._phi2_template(),
            "phi3": cls._phi3_template(),
            "phi": cls._phi3_template(),
            "smollm3": cls._llama_template(),
        }
        return templates.get(architecture, cls._llama_template())

    @staticmethod
    def _llama_template() -> Dict[str, str]:
        """Mapping template for Llama-family models."""
        return {
            r"^token_embd\.weight$": "model.embed_tokens.weight",
            r"^token_embd\.bias$": "model.embed_tokens.bias",
            r"^blk\.(\d+)\.attn_norm\.weight$": "model.layers.{0}.input_layernorm.weight",
            r"^blk\.(\d+)\.attn_norm\.bias$": "model.layers.{0}.input_layernorm.bias",
            r"^blk\.(\d+)\.ffn_norm\.weight$": "model.layers.{0}.post_attention_layernorm.weight",
            r"^blk\.(\d+)\.ffn_norm\.bias$": "model.layers.{0}.post_attention_layernorm.bias",
            r"^blk\.(\d+)\.attn_q\.weight$": "model.layers.{0}.self_attn.q_proj.weight",
            r"^blk\.(\d+)\.attn_k\.weight$": "model.layers.{0}.self_attn.k_proj.weight",
            r"^blk\.(\d+)\.attn_v\.weight$": "model.layers.{0}.self_attn.v_proj.weight",
            r"^blk\.(\d+)\.attn_output\.weight$": "model.layers.{0}.self_attn.o_proj.weight",
            r"^blk\.(\d+)\.attn_q\.bias$": "model.layers.{0}.self_attn.q_proj.bias",
            r"^blk\.(\d+)\.attn_k\.bias$": "model.layers.{0}.self_attn.k_proj.bias",
            r"^blk\.(\d+)\.attn_v\.bias$": "model.layers.{0}.self_attn.v_proj.bias",
            r"^blk\.(\d+)\.attn_output\.bias$": "model.layers.{0}.self_attn.o_proj.bias",
            r"^blk\.(\d+)\.ffn_gate\.weight$": "model.layers.{0}.mlp.gate_proj.weight",
            r"^blk\.(\d+)\.ffn_up\.weight$": "model.layers.{0}.mlp.up_proj.weight",
            r"^blk\.(\d+)\.ffn_down\.weight$": "model.layers.{0}.mlp.down_proj.weight",
            r"^blk\.(\d+)\.ffn_gate\.bias$": "model.layers.{0}.mlp.gate_proj.bias",
            r"^blk\.(\d+)\.ffn_up\.bias$": "model.layers.{0}.mlp.up_proj.bias",
            r"^blk\.(\d+)\.ffn_down\.bias$": "model.layers.{0}.mlp.down_proj.bias",
            r"^blk\.(\d+)\.ffn\.bias$": "model.layers.{0}.mlp.down_proj.bias",
            r"^blk\.(\d+)\.ffn_gate_exp\.(\d+)\.weight$": "model.layers.{0}.mlp.gate_proj.experts.{1}.weight",
            r"^blk\.(\d+)\.ffn_up_exp\.(\d+)\.weight$": "model.layers.{0}.mlp.up_proj.experts.{1}.weight",
            r"^blk\.(\d+)\.ffn_down_exp\.(\d+)\.weight$": "model.layers.{0}.mlp.down_proj.experts.{1}.weight",
            r"^output_norm\.weight$": "model.norm.weight",
            r"^output_norm\.bias$": "model.norm.bias",
            r"^output\.weight$": "lm_head.weight",
            r"^output\.bias$": "lm_head.bias",
        }

    @staticmethod
    def _gemma_template() -> Dict[str, str]:
        """Mapping template for Gemma-family models."""
        return {
            r"^token_embd\.weight$": "model.embed_tokens.weight",
            r"^token_embd\.bias$": "model.embed_tokens.bias",
            r"^blk\.(\d+)\.attn_norm\.weight$": "model.layers.{0}.input_layernorm.weight",
            r"^blk\.(\d+)\.attn_post_norm\.weight$": "model.layers.{0}.post_attention_layernorm.weight",
            r"^blk\.(\d+)\.ffn_norm\.weight$": "model.layers.{0}.pre_feedforward_layernorm.weight",
            r"^blk\.(\d+)\.ffn_post_norm\.weight$": "model.layers.{0}.post_feedforward_layernorm.weight",
            r"^blk\.(\d+)\.attn_q\.weight$": "model.layers.{0}.self_attn.q_proj.weight",
            r"^blk\.(\d+)\.attn_k\.weight$": "model.layers.{0}.self_attn.k_proj.weight",
            r"^blk\.(\d+)\.attn_v\.weight$": "model.layers.{0}.self_attn.v_proj.weight",
            r"^blk\.(\d+)\.attn_output\.weight$": "model.layers.{0}.self_attn.o_proj.weight",
            r"^blk\.(\d+)\.attn_q_norm\.weight$": "model.layers.{0}.self_attn.q_norm.weight",
            r"^blk\.(\d+)\.attn_k_norm\.weight$": "model.layers.{0}.self_attn.k_norm.weight",
            r"^blk\.(\d+)\.ffn_gate\.weight$": "model.layers.{0}.mlp.gate_proj.weight",
            r"^blk\.(\d+)\.ffn_up\.weight$": "model.layers.{0}.mlp.up_proj.weight",
            r"^blk\.(\d+)\.ffn_down\.weight$": "model.layers.{0}.mlp.down_proj.weight",
            r"^output_norm\.weight$": "model.norm.weight",
            r"^output_norm\.bias$": "model.norm.bias",
            r"^output\.weight$": "lm_head.weight",
            r"^output\.bias$": "lm_head.bias",
        }

    @staticmethod
    def _gemma2_template() -> Dict[str, str]:
        """Mapping template for Gemma2-family models (same as Gemma with extras)."""
        base = TensorMapper._gemma_template()
        return base

    @staticmethod
    def _mistral_template() -> Dict[str, str]:
        """Mapping template for Mistral-family models."""
        return {
            r"^token_embd\.weight$": "model.embed_tokens.weight",
            r"^blk\.(\d+)\.attn_norm\.weight$": "model.layers.{0}.input_layernorm.weight",
            r"^blk\.(\d+)\.ffn_norm\.weight$": "model.layers.{0}.post_attention_layernorm.weight",
            r"^blk\.(\d+)\.attn_q\.weight$": "model.layers.{0}.self_attn.q_proj.weight",
            r"^blk\.(\d+)\.attn_k\.weight$": "model.layers.{0}.self_attn.k_proj.weight",
            r"^blk\.(\d+)\.attn_v\.weight$": "model.layers.{0}.self_attn.v_proj.weight",
            r"^blk\.(\d+)\.attn_output\.weight$": "model.layers.{0}.self_attn.o_proj.weight",
            r"^blk\.(\d+)\.ffn_gate\.weight$": "model.layers.{0}.mlp.gate_proj.weight",
            r"^blk\.(\d+)\.ffn_up\.weight$": "model.layers.{0}.mlp.up_proj.weight",
            r"^blk\.(\d+)\.ffn_down\.weight$": "model.layers.{0}.mlp.down_proj.weight",
            r"^output_norm\.weight$": "model.norm.weight",
            r"^output\.weight$": "lm_head.weight",
        }

    @staticmethod
    def _qwen2_template() -> Dict[str, str]:
        """Mapping template for Qwen2/Qwen3-family models."""
        return {
            r"^token_embd\.weight$": "model.embed_tokens.weight",
            r"^blk\.(\d+)\.attn_norm\.weight$": "model.layers.{0}.input_layernorm.weight",
            r"^blk\.(\d+)\.ffn_norm\.weight$": "model.layers.{0}.post_attention_layernorm.weight",
            r"^blk\.(\d+)\.attn_q\.weight$": "model.layers.{0}.self_attn.q_proj.weight",
            r"^blk\.(\d+)\.attn_k\.weight$": "model.layers.{0}.self_attn.k_proj.weight",
            r"^blk\.(\d+)\.attn_v\.weight$": "model.layers.{0}.self_attn.v_proj.weight",
            r"^blk\.(\d+)\.attn_output\.weight$": "model.layers.{0}.self_attn.o_proj.weight",
            r"^blk\.(\d+)\.attn_q\.bias$": "model.layers.{0}.self_attn.q_proj.bias",
            r"^blk\.(\d+)\.attn_k\.bias$": "model.layers.{0}.self_attn.k_proj.bias",
            r"^blk\.(\d+)\.attn_v\.bias$": "model.layers.{0}.self_attn.v_proj.bias",
            r"^blk\.(\d+)\.attn_output\.bias$": "model.layers.{0}.self_attn.o_proj.bias",
            r"^blk\.(\d+)\.ffn_gate\.weight$": "model.layers.{0}.mlp.gate_proj.weight",
            r"^blk\.(\d+)\.ffn_up\.weight$": "model.layers.{0}.mlp.up_proj.weight",
            r"^blk\.(\d+)\.ffn_down\.weight$": "model.layers.{0}.mlp.down_proj.weight",
            r"^output_norm\.weight$": "model.norm.weight",
            r"^output\.weight$": "lm_head.weight",
        }

    @staticmethod
    def _phi2_template() -> Dict[str, str]:
        """Mapping template for Phi-2 models."""
        return {
            r"^token_embd\.weight$": "model.embed_tokens.weight",
            r"^token_embd\.bias$": "model.embed_tokens.bias",
            r"^blk\.(\d+)\.attn_norm\.weight$": "model.layers.{0}.input_layernorm.weight",
            r"^blk\.(\d+)\.attn_norm\.bias$": "model.layers.{0}.input_layernorm.bias",
            r"^blk\.(\d+)\.ffn_norm\.weight$": "model.layers.{0}.post_attention_layernorm.weight",
            r"^blk\.(\d+)\.ffn_norm\.bias$": "model.layers.{0}.post_attention_layernorm.bias",
            r"^blk\.(\d+)\.attn_qkv\.weight$": "model.layers.{0}.self_attn.qkv_proj.weight",
            r"^blk\.(\d+)\.attn_qkv\.bias$": "model.layers.{0}.self_attn.qkv_proj.bias",
            r"^blk\.(\d+)\.attn_output\.weight$": "model.layers.{0}.self_attn.dense.weight",
            r"^blk\.(\d+)\.attn_output\.bias$": "model.layers.{0}.self_attn.dense.bias",
            r"^blk\.(\d+)\.ffn_up\.weight$": "model.layers.{0}.mlp.fc1.weight",
            r"^blk\.(\d+)\.ffn_up\.bias$": "model.layers.{0}.mlp.fc1.bias",
            r"^blk\.(\d+)\.ffn_down\.weight$": "model.layers.{0}.mlp.fc2.weight",
            r"^blk\.(\d+)\.ffn_down\.bias$": "model.layers.{0}.mlp.fc2.bias",
            r"^output_norm\.weight$": "model.final_layernorm.weight",
            r"^output_norm\.bias$": "model.final_layernorm.bias",
            r"^lm_head\.weight$": "lm_head.weight",
            r"^lm_head\.bias$": "lm_head.bias",
        }

    @staticmethod
    def _phi3_template() -> Dict[str, str]:
        """Mapping template for Phi-3 models."""
        return {
            r"^token_embd\.weight$": "model.embed_tokens.weight",
            r"^blk\.(\d+)\.attn_norm\.weight$": "model.layers.{0}.input_layernorm.weight",
            r"^blk\.(\d+)\.ffn_norm\.weight$": "model.layers.{0}.post_attention_layernorm.weight",
            r"^blk\.(\d+)\.attn_q\.weight$": "model.layers.{0}.self_attn.q_proj.weight",
            r"^blk\.(\d+)\.attn_k\.weight$": "model.layers.{0}.self_attn.k_proj.weight",
            r"^blk\.(\d+)\.attn_v\.weight$": "model.layers.{0}.self_attn.v_proj.weight",
            r"^blk\.(\d+)\.attn_output\.weight$": "model.layers.{0}.self_attn.o_proj.weight",
            r"^blk\.(\d+)\.ffn_gate\.weight$": "model.layers.{0}.mlp.gate_up_proj.weight",
            r"^blk\.(\d+)\.ffn_down\.weight$": "model.layers.{0}.mlp.down_proj.weight",
            r"^output_norm\.weight$": "model.norm.weight",
            r"^output\.weight$": "lm_head.weight",
        }
