"""PyTorch model definitions for supported GGUF architectures."""

from litert_torch_qarnux.models.base import BaseModel, ModelRegistry
from litert_torch_qarnux.models.llama import LlamaForCausalLM
from litert_torch_qarnux.models.gemma import GemmaForCausalLM
from litert_torch_qarnux.models.mistral import MistralForCausalLM
from litert_torch_qarnux.models.qwen2 import Qwen2ForCausalLM

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "LlamaForCausalLM",
    "GemmaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
]
