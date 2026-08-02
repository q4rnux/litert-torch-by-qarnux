"""
Qwen2/Qwen3 Model Architecture Definition.

Implements a minimal Qwen2ForCausalLM model in pure PyTorch.
Qwen2 models are architecturally similar to Llama with RMSNorm, RoPE,
and SwiGLU MLP, but include optional attention and MLP biases.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from litert_torch_qarnux.models.base import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)


class Qwen2RMSNorm(nn.Module):
    """RMSNorm for Qwen2 models."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).type_as(x)


class Qwen2RotaryEmbedding(nn.Module):
    """RoPE for Qwen2 models."""

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Qwen2Attention(nn.Module):
    """Attention layer for Qwen2 with optional biases."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        use_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=use_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=use_bias)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(head_dim, max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q)
        q1, q2 = q[..., : self.head_dim // 2], q[..., self.head_dim // 2 :]
        k1, k2 = k[..., : self.head_dim // 2], k[..., self.head_dim // 2 :]
        q = torch.cat((-q2, q1), dim=-1) * cos + q * cos
        k = torch.cat((-k2, k1), dim=-1) * cos + k * cos

        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        return residual + self.o_proj(attn_output)


class Qwen2MLP(nn.Module):
    """SwiGLU MLP for Qwen2 with optional biases."""

    def __init__(self, hidden_size: int, intermediate_size: int, rms_norm_eps: float = 1e-6, use_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.post_attention_layernorm = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen2ForCausalLM(BaseModel):
    """Full Qwen2/Qwen3 Causal Language Model."""

    ARCHITECTURES = ["qwen2", "qwen3"]

    def __init__(self, metadata: Any):
        super().__init__(metadata)
        self.model = None

    def build(self) -> nn.Module:
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
                "self_attn": Qwen2Attention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps),
                "mlp": Qwen2MLP(hidden_size, intermediate_size, rms_norm_eps),
            })
            layers.append(layer)

        model = nn.ModuleDict({
            "embed_tokens": nn.Embedding(self.metadata.vocab_size, hidden_size),
            "layers": nn.ModuleList(layers),
            "norm": Qwen2RMSNorm(hidden_size, eps=rms_norm_eps),
            "lm_head": nn.Linear(hidden_size, self.metadata.vocab_size, bias=False),
        })

        logger.info(
            "Built Qwen2 model: layers=%d, hidden=%d, heads=%d, kv_heads=%d, vocab=%d",
            num_layers, hidden_size, num_heads, num_kv_heads, self.metadata.vocab_size,
        )
        return model

    def load_weights(
        self,
        tensor_map: Dict[str, str],
        dequantized_tensors: Dict[str, np.ndarray],
    ) -> None:
        if self.model is None:
            self.model = self.build()

        model = self.model
        for gguf_name, pytorch_name in tensor_map.items():
            if gguf_name in dequantized_tensors:
                weight = dequantized_tensors[gguf_name]
                if isinstance(weight, np.ndarray):
                    weight = torch.from_numpy(weight.astype(np.float32))
                    self._set_param(model, pytorch_name, weight)

        logger.info("Loaded %d weight tensors into Qwen2 model", len(tensor_map))

    def _set_param(self, model: nn.ModuleDict, param_path: str, value: torch.Tensor) -> None:
        parts = param_path.split(".")
        obj = model
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = obj[part] if isinstance(obj, nn.ModuleDict) else getattr(obj, part)
        if parts[-1] == "weight":
            obj.weight.data.copy_(value)
        elif parts[-1] == "bias":
            obj.bias.data.copy_(value)


ModelRegistry.register("qwen2", Qwen2ForCausalLM)
ModelRegistry.register("qwen3", Qwen2ForCausalLM)
