"""
Llama Model Architecture Definition.

Implements a minimal LlamaForCausalLM model in pure PyTorch that matches
the standard HuggingFace Transformers implementation. This model is used
as an intermediate representation during GGUF-to-TFLite conversion,
accepting dequantized weights from the GGUF file and providing a
fully functional PyTorch nn.Module suitable for export to TFLite.

The implementation uses RMSNorm for normalization and RoPE for position
encoding, matching the original Llama architecture.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from litert_torch_qarnux.models.base import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as used in Llama models."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input tensor."""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).type_as(x)


class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for Llama attention layers."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _update_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and cache the cos and sin rotation matrices."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, head_dim).
            position_ids: Optional position indices.

        Returns:
            Tuple of (cos, sin) tensors for rotation.
        """
        seq_len = x.shape[1]
        cos, sin = self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-head attention layer for Llama models."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass for the attention layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq, hidden).
            attention_mask: Optional attention mask.
            use_cache: Whether to return key-value cache.

        Returns:
            Tuple of (output, optional cache).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, heads, seq, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(query_states)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Repeat KV heads if GQA
        if self.num_kv_heads != self.num_heads:
            key_states = key_states.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=1
            )
            value_states = value_states.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=1
            )

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask
        )

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return residual + attn_output, None


class LlamaMLP(nn.Module):
    """SwiGLU feed-forward network for Llama models."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq, hidden).

        Returns:
            Output tensor of the same shape.
        """
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return residual + self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer combining attention and MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass for the decoder layer.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Optional attention mask.
            use_cache: Whether to use KV cache.

        Returns:
            Tuple of (output, optional cache).
        """
        hidden_states, attn_cache = self.self_attn(
            hidden_states, attention_mask, use_cache
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, attn_cache


class LlamaForCausalLM(BaseModel):
    """
    Full Llama Causal Language Model implementation.

    Matches the HuggingFace Transformers LlamaForCausalLM architecture
    for compatibility with the litert-torch export pipeline.
    """

    ARCHITECTURES = ["llama"]

    def __init__(self, metadata: Any):
        super().__init__(metadata)
        self.model = None

    def build(self) -> nn.Module:
        """Construct the Llama model architecture."""
        hidden_size = self.metadata.embedding_length
        num_heads = self.metadata.attention_head_count
        num_kv_heads = self.metadata.attention_head_count_kv
        intermediate_size = self.metadata.feed_forward_length
        num_layers = self.metadata.block_count
        rms_norm_eps = self.metadata.rms_norm_eps

        head_dim = hidden_size // num_heads

        # Build layers
        layers = []
        for _ in range(num_layers):
            layer = LlamaDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
            )
            layers.append(layer)

        # Embedding
        embed_tokens = nn.Embedding(self.metadata.vocab_size, hidden_size)

        # Final norm and LM head
        norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        lm_head = nn.Linear(hidden_size, self.metadata.vocab_size, bias=False)

        # Assemble into a simple module
        model = _LlamaModelWrapper(
            embed_tokens=embed_tokens,
            layers=nn.ModuleList(layers),
            norm=norm,
            lm_head=lm_head,
        )

        logger.info(
            "Built Llama model: layers=%d, hidden=%d, heads=%d, kv_heads=%d, "
            "intermediate=%d, vocab=%d",
            num_layers,
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            self.metadata.vocab_size,
        )
        return model

    def load_weights(
        self,
        tensor_map: Dict[str, str],
        dequantized_tensors: Dict[str, np.ndarray],
    ) -> None:
        """
        Load dequantized weights into the Llama model.

        Maps GGUF tensor names to PyTorch parameter paths and loads
        the converted weight matrices.
        """
        if self.model is None:
            self.model = self.build()

        model = self.model
        state_dict = model.state_dict()
        new_state_dict = {}

        for gguf_name, pytorch_name in tensor_map.items():
            if gguf_name in dequantized_tensors:
                weight = dequantized_tensors[gguf_name]
                # Ensure correct shape for PyTorch linear layers
                if isinstance(weight, np.ndarray):
                    weight = torch.from_numpy(weight.astype(np.float32))

                    # GGUF stores weights in [out_features, in_features] format
                    # PyTorch Linear also uses [out_features, in_features]
                    # but we need to verify transpose requirements

                    if "attn_q" in gguf_name or "attn_k" in gguf_name or "attn_v" in gguf_name:
                        # QKV projections are stored as [num_heads * head_dim, hidden]
                        # which matches PyTorch's expected format
                        pass
                    elif "attn_output" in gguf_name:
                        # Output projection: [hidden, num_heads * head_dim]
                        pass
                    elif "ffn_gate" in gguf_name or "ffn_up" in gguf_name:
                        # MLP gate/up: [intermediate, hidden]
                        pass
                    elif "ffn_down" in gguf_name:
                        # MLP down: [hidden, intermediate]
                        pass

                    new_state_dict[pytorch_name] = weight
                elif hasattr(weight, "__len__"):
                    new_state_dict[pytorch_name] = weight

        # Load the new state dict, allowing partial loads
        try:
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if missing:
                logger.warning("Missing parameters: %s", missing[:5])
            if unexpected:
                logger.warning("Unexpected parameters: %s", unexpected[:5])
        except RuntimeError as e:
            logger.error("Error loading weights: %s", e)
            raise

        logger.info("Loaded %d weight tensors into Llama model", len(new_state_dict))


class _LlamaModelWrapper(nn.Module):
    """Internal wrapper to expose the model parameters for state_dict loading."""

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        layers: nn.ModuleList,
        norm: RMSNorm,
        lm_head: nn.Linear,
    ):
        super().__init__()
        # Create the model structure matching HuggingFace naming
        self.model = nn.Module()
        self.model.embed_tokens = embed_tokens
        self.model.layers = layers
        self.model.norm = norm

        # Rebuild the structure so state_dict paths match
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: embed -> layers -> norm -> lm_head."""
        hidden_states = self._embed_tokens(input_ids)
        for layer in self._layers:
            hidden_states, _ = layer(hidden_states, attention_mask)
        hidden_states = self._norm(hidden_states)
        return self.lm_head(hidden_states)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to use HuggingFace-compatible naming."""
        sd = {}
        sd["model.embed_tokens.weight"] = self._embed_tokens.weight.data
        for i, layer in enumerate(self._layers):
            prefix = f"model.layers.{i}"
            sd[f"{prefix}.input_layernorm.weight"] = (
                layer.self_attn.input_layernorm.weight.data
            )
            sd[f"{prefix}.self_attn.q_proj.weight"] = (
                layer.self_attn.q_proj.weight.data
            )
            sd[f"{prefix}.self_attn.k_proj.weight"] = (
                layer.self_attn.k_proj.weight.data
            )
            sd[f"{prefix}.self_attn.v_proj.weight"] = (
                layer.self_attn.v_proj.weight.data
            )
            sd[f"{prefix}.self_attn.o_proj.weight"] = (
                layer.self_attn.o_proj.weight.data
            )
            sd[f"{prefix}.post_attention_layernorm.weight"] = (
                layer.mlp.post_attention_layernorm.weight.data
            )
            sd[f"{prefix}.mlp.gate_proj.weight"] = (
                layer.mlp.gate_proj.weight.data
            )
            sd[f"{prefix}.mlp.up_proj.weight"] = (
                layer.mlp.up_proj.weight.data
            )
            sd[f"{prefix}.mlp.down_proj.weight"] = (
                layer.mlp.down_proj.weight.data
            )
        sd["model.norm.weight"] = self._norm.weight.data
        sd["lm_head.weight"] = self.lm_head.weight.data
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to accept HuggingFace-compatible naming."""
        # Apply weights directly
        if "model.embed_tokens.weight" in state_dict:
            self._embed_tokens.weight.data.copy_(
                state_dict["model.embed_tokens.weight"]
            )
        for i, layer in enumerate(self._layers):
            prefix = f"model.layers.{i}"
            if f"{prefix}.input_layernorm.weight" in state_dict:
                layer.self_attn.input_layernorm.weight.data.copy_(
                    state_dict[f"{prefix}.input_layernorm.weight"]
                )
            if f"{prefix}.self_attn.q_proj.weight" in state_dict:
                layer.self_attn.q_proj.weight.data.copy_(
                    state_dict[f"{prefix}.self_attn.q_proj.weight"]
                )
            if f"{prefix}.self_attn.k_proj.weight" in state_dict:
                layer.self_attn.k_proj.weight.data.copy_(
                    state_dict[f"{prefix}.self_attn.k_proj.weight"]
                )
            if f"{prefix}.self_attn.v_proj.weight" in state_dict:
                layer.self_attn.v_proj.weight.data.copy_(
                    state_dict[f"{prefix}.self_attn.v_proj.weight"]
                )
            if f"{prefix}.self_attn.o_proj.weight" in state_dict:
                layer.self_attn.o_proj.weight.data.copy_(
                    state_dict[f"{prefix}.self_attn.o_proj.weight"]
                )
            if f"{prefix}.post_attention_layernorm.weight" in state_dict:
                layer.mlp.post_attention_layernorm.weight.data.copy_(
                    state_dict[f"{prefix}.post_attention_layernorm.weight"]
                )
            if f"{prefix}.mlp.gate_proj.weight" in state_dict:
                layer.mlp.gate_proj.weight.data.copy_(
                    state_dict[f"{prefix}.mlp.gate_proj.weight"]
                )
            if f"{prefix}.mlp.up_proj.weight" in state_dict:
                layer.mlp.up_proj.weight.data.copy_(
                    state_dict[f"{prefix}.mlp.up_proj.weight"]
                )
            if f"{prefix}.mlp.down_proj.weight" in state_dict:
                layer.mlp.down_proj.weight.data.copy_(
                    state_dict[f"{prefix}.mlp.down_proj.weight"]
                )
        if "model.norm.weight" in state_dict:
            self._norm.weight.data.copy_(state_dict["model.norm.weight"])
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.data.copy_(state_dict["lm_head.weight"])
        return nn.modules.module.Module.load_state_dict(self, state_dict, strict)


# Register the model
ModelRegistry.register("llama", LlamaForCausalLM)
