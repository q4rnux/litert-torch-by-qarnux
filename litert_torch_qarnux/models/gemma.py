"""
Gemma Model Architecture Definition.

Implements a minimal GemmaForCausalLM model in pure PyTorch that matches
the standard HuggingFace Transformers implementation. Gemma models use
RMSNorm with a +1 offset in the normalization weights (Gemma-specific),
post-attention layer norms (pre-FFN and post-FFN), and RMSNorm for
both Q and K projections (query-key normalization).
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


class GemmaRMSNorm(nn.Module):
    """RMSNorm with the Gemma-specific +1 offset on the weight."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))  # initialized to 0, becomes +1
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gemma-style RMSNorm."""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight + 1.0) * x


class GemmaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Gemma models."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
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

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for rotary embeddings."""
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class GemmaAttention(nn.Module):
    """Multi-head attention with Gemma-specific Q/K normalization."""

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

        self.q_norm = nn.Parameter(torch.zeros(num_heads * head_dim))  # +1 offset
        self.k_norm = nn.Parameter(torch.zeros(num_kv_heads * head_dim))

        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = GemmaRotaryEmbedding(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Gemma attention."""
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Apply Q/K RMSNorm
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + 1e-6) * (self.q_norm + 1.0)
        k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + 1e-6) * (self.k_norm + 1.0)

        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q)
        # Apply RoPE
        q1, q2 = q[..., : self.head_dim // 2], q[..., self.head_dim // 2 :]
        k1, k2 = k[..., : self.head_dim // 2], k[..., self.head_dim // 2 :]
        q = torch.cat((-q2, q1), dim=-1) * cos[..., : self.head_dim] + q * cos[..., : self.head_dim]
        k = torch.cat((-k2, k1), dim=-1) * cos[..., : self.head_dim] + k * cos[..., : self.head_dim]

        # GQA repeat
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return residual + self.post_attention_layernorm(attn_output)


class GemmaMLP(nn.Module):
    """Gelu-based MLP for Gemma models."""

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
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(intermediate_size, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for Gemma MLP."""
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        gate = F.gelu(self.gate_proj(hidden_states), approximate="tanh")
        up = self.up_proj(hidden_states)
        return residual + self.down_proj(self.post_feedforward_layernorm(gate * up))


class GemmaForCausalLM(BaseModel):
    """Full Gemma Causal Language Model implementation."""

    ARCHITECTURES = ["gemma", "gemma2", "gemma3", "gemma3n", "gemma4"]

    def __init__(self, metadata: Any):
        super().__init__(metadata)
        self.model = None

    def build(self) -> nn.Module:
        """Construct the Gemma model architecture."""
        hidden_size = self.metadata.embedding_length
        num_heads = self.metadata.attention_head_count
        num_kv_heads = self.metadata.attention_head_count_kv
        intermediate_size = self.metadata.feed_forward_length
        num_layers = self.metadata.block_count
        rms_norm_eps = self.metadata.rms_norm_eps
        head_dim = hidden_size // num_heads

        layers = []
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "self_attn": GemmaAttention(
                    hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps
                ),
                "mlp": GemmaMLP(hidden_size, intermediate_size, rms_norm_eps),
            })
            layers.append(layer)

        model = nn.ModuleDict({
            "embed_tokens": nn.Embedding(self.metadata.vocab_size, hidden_size),
            "layers": nn.ModuleList(layers),
            "norm": GemmaRMSNorm(hidden_size, eps=rms_norm_eps),
            "lm_head": nn.Linear(hidden_size, self.metadata.vocab_size, bias=False),
        })

        logger.info(
            "Built Gemma model: layers=%d, hidden=%d, heads=%d, vocab=%d",
            num_layers, hidden_size, num_heads, self.metadata.vocab_size,
        )
        return model

    def load_weights(
        self,
        tensor_map: Dict[str, str],
        dequantized_tensors: Dict[str, np.ndarray],
    ) -> None:
        """Load dequantized weights into the Gemma model."""
        if self.model is None:
            self.model = self.build()

        model = self.model
        for gguf_name, pytorch_name in tensor_map.items():
            if gguf_name in dequantized_tensors:
                weight = dequantized_tensors[gguf_name]
                if isinstance(weight, np.ndarray):
                    weight = torch.from_numpy(weight.astype(np.float32))
                    self._set_param(model, pytorch_name, weight)

        logger.info("Loaded %d weight tensors into Gemma model", len(tensor_map))

    def _set_param(self, model: nn.Module, param_path: str, value: torch.Tensor) -> None:
        """Set a parameter by dotted path."""
        parts = param_path.split(".")
        obj = model
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        if parts[-1] == "weight":
            obj.weight.data.copy_(value)
        elif parts[-1] == "bias":
            obj.bias.data.copy_(value)
        else:
            # Could be a parameter like q_norm
            setattr(obj, parts[-1], nn.Parameter(value))


# Register all Gemma variants
for arch in ["gemma", "gemma2", "gemma3", "gemma3n", "gemma4"]:
    ModelRegistry.register(arch, GemmaForCausalLM)
