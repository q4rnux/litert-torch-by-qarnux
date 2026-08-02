"""
GGUF File Parser Agent Utility.

Parses GGUF binary files to extract model metadata, architecture information,
hyperparameters, and tensor descriptors. This module serves as the data
extraction layer for the ParserAgent in the multi-agent orchestration pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gguf
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GGUFMetadata:
    """Container for GGUF model metadata extracted from the file header."""

    architecture: str = ""
    name: str = ""
    file_type: Optional[int] = None
    quantization_version: Optional[int] = None
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    attention_head_count: int = 0
    attention_head_count_kv: int = 0
    feed_forward_length: int = 0
    expert_count: int = 0
    expert_used_count: int = 0
    rope_dimension_count: int = 0
    rope_freq_base: float = 0.0
    vocab_size: int = 0
    layer_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    raw_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorInfo:
    """Describes a single tensor within the GGUF file."""

    name: str
    shape: List[int]
    tensor_type: int
    data: np.ndarray
    data_type: str  # human-readable quantization type

    @property
    def num_elements(self) -> int:
        """Total number of elements in the tensor."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def total_size_bytes(self) -> int:
        """Approximate memory footprint in bytes."""
        return self.data.nbytes if hasattr(self.data, "nbytes") else 0


class GGUFParser:
    """
    Parses GGUF binary files and extracts model metadata and tensor information.

    This class wraps the official `gguf.GGUFReader` and provides a higher-level
    API for accessing architecture-specific hyperparameters needed by downstream
    agents in the conversion pipeline.
    """

    # Architecture-specific metadata key prefixes
    _ARCH_PREFIXES = {
        "llama": "llama",
        "gemma": "gemma",
        "gemma2": "gemma",
        "gemma3": "gemma",
        "gemma3n": "gemma",
        "gemma4": "gemma",
        "mistral": "mistral",
        "qwen2": "qwen2",
        "qwen3": "qwen",
        "phi2": "phi",
        "phi3": "phi",
        "phi": "phi",
        "smollm3": "smollm",
    }

    def __init__(self, model_path: str | Path):
        """
        Initialize the GGUF parser.

        Args:
            model_path: Path to the .gguf model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the file is not a valid GGUF file.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model file not found: {self.model_path}")

        logger.info("Opening GGUF file: %s", self.model_path)
        self.reader = gguf.GGUFReader(str(self.model_path))

    def get_field(self, key: str) -> Any:
        """
        Retrieve a metadata field value by key.

        Args:
            key: The metadata key string.

        Returns:
            The field value, or None if not found.
        """
        if key in self.reader.fields:
            field = self.reader.fields[key]
            return field.parts[field.data][: field.data_len]
        return None

    def extract_metadata(self) -> GGUFMetadata:
        """
        Extract all relevant metadata from the GGUF file.

        Returns:
            A GGUFMetadata instance populated with architecture and
            hyperparameter information.
        """
        metadata = GGUFMetadata()

        # Core identification fields
        arch = self._get_str_field("general.architecture")
        metadata.architecture = arch if arch else ""

        metadata.name = self._get_str_field("general.name") or ""
        metadata.file_type = self._get_int_field("general.file_type")
        metadata.quantization_version = self._get_int_field("general.quantization_version")

        # Build the architecture-specific prefix
        prefix = self._get_arch_prefix(metadata.architecture)

        # Embedding dimension
        metadata.embedding_length = (
            self._get_int_field(f"{prefix}.embedding_length") or 0
        )

        # Block / layer count
        metadata.block_count = (
            self._get_int_field(f"{prefix}.block_count") or 0
        )

        # Attention heads
        metadata.attention_head_count = (
            self._get_int_field(f"{prefix}.attention.head_count") or 0
        )
        metadata.attention_head_count_kv = (
            self._get_int_field(f"{prefix}.attention.head_count_kv")
            or metadata.attention_head_count
        )

        # Feed-forward dimension
        metadata.feed_forward_length = (
            self._get_int_field(f"{prefix}.feed_forward_length") or 0
        )

        # Context window
        metadata.context_length = (
            self._get_int_field(f"{prefix}.context_length")
            or self._get_int_field("llama.context_length")
            or 0
        )

        # MoE fields
        metadata.expert_count = self._get_int_field(f"{prefix}.expert_count") or 0
        metadata.expert_used_count = (
            self._get_int_field(f"{prefix}.expert_used_count") or 0
        )

        # RoPE
        metadata.rope_dimension_count = (
            self._get_int_field(f"{prefix}.rope.dimension_count") or 0
        )
        metadata.rope_freq_base = (
            self._get_float_field(f"{prefix}.rope.freq_base") or 10000.0
        )

        # Normalization
        metadata.rms_norm_eps = (
            self._get_float_field(f"{prefix}.attention.layer_norm_rms_epsilon")
            or self._get_float_field(f"{prefix}.attention.layer_norm_epsilon")
            or 1e-6
        )
        metadata.layer_norm_eps = metadata.rms_norm_eps

        # Vocabulary
        metadata.vocab_size = (
            self._get_int_field(f"{prefix}.vocab_size") or 0
        )
        if metadata.vocab_size == 0:
            # Fallback: try tokenizer model length
            tok_vocab = self._get_int_field("tokenizer.ggml.tokens_count")
            if tok_vocab:
                metadata.vocab_size = tok_vocab

        # Store all raw fields for reference
        for key in self.reader.fields:
            try:
                val = self.get_field(key)
                if val is not None:
                    metadata.raw_fields[key] = val
            except Exception:
                metadata.raw_fields[key] = "<unreadable>"

        logger.info(
            "Extracted metadata: arch=%s, layers=%d, hidden=%d, heads=%d, "
            "kv_heads=%d, ffn=%d, ctx=%d",
            metadata.architecture,
            metadata.block_count,
            metadata.embedding_length,
            metadata.attention_head_count,
            metadata.attention_head_count_kv,
            metadata.feed_forward_length,
            metadata.context_length,
        )

        return metadata

    def list_tensors(self) -> List[TensorInfo]:
        """
        Enumerate all tensors in the GGUF file.

        Returns:
            A list of TensorInfo objects describing each tensor.
        """
        tensors = []
        for gguf_tensor in self.reader.tensors:
            tensor_type_str = self._tensor_type_name(gguf_tensor.tensor_type)
            info = TensorInfo(
                name=gguf_tensor.name,
                shape=list(gguf_tensor.shape),
                tensor_type=gguf_tensor.tensor_type,
                data=gguf_tensor.data,
                data_type=tensor_type_str,
            )
            tensors.append(info)

        logger.info("Found %d tensors in GGUF file", len(tensors))
        return tensors

    def get_tensor(self, name: str) -> Optional[TensorInfo]:
        """
        Retrieve a specific tensor by name.

        Args:
            name: The GGUF tensor name (e.g., "blk.0.attn_q.weight").

        Returns:
            A TensorInfo object, or None if not found.
        """
        for gguf_tensor in self.reader.tensors:
            if gguf_tensor.name == name:
                tensor_type_str = self._tensor_type_name(gguf_tensor.tensor_type)
                return TensorInfo(
                    name=gguf_tensor.name,
                    shape=list(gguf_tensor.shape),
                    tensor_type=gguf_tensor.tensor_type,
                    data=gguf_tensor.data,
                    data_type=tensor_type_str,
                )
        return None

    def get_tokenizer_vocab(self) -> Optional[np.ndarray]:
        """
        Extract the tokenizer vocabulary tokens from GGUF metadata.

        Returns:
            A numpy array of token strings, or None if not found.
        """
        vocab = self.get_field("tokenizer.ggml.tokens")
        if vocab is not None:
            return np.array(vocab, dtype=object)
        return None

    def get_tokenizer_scores(self) -> Optional[np.ndarray]:
        """Extract tokenizer token scores (merge weights)."""
        scores = self.get_field("tokenizer.ggml.token_type")
        if scores is not None:
            return np.array(scores)
        return None

    def get_tokenizer_merges(self) -> Optional[List[str]]:
        """Extract BPE merge rules if present."""
        merges = self.get_field("tokenizer.ggml.merges")
        if merges is not None:
            return [m.decode() if isinstance(m, bytes) else str(m) for m in merges]
        return None

    def get_tokenizer_model(self) -> str:
        """Return the tokenizer model type (e.g., 'llama', 'spm', 'bpe')."""
        model_type = self._get_str_field("tokenizer.ggml.model")
        return model_type if model_type else "unknown"

    def get_tensor_types(self) -> Dict[int, int]:
        """
        Count tensors by their quantization type.

        Returns:
            A dictionary mapping tensor_type int to count.
        """
        type_counts: Dict[int, int] = {}
        for t in self.reader.tensors:
            type_counts[t.tensor_type] = type_counts.get(t.tensor_type, 0) + 1
        return type_counts

    # -- Private helpers --

    def _get_str_field(self, key: str) -> Optional[str]:
        """Read a string-typed metadata field."""
        val = self.get_field(key)
        if val is not None:
            if isinstance(val, (bytes, bytearray)):
                return val.decode("utf-8", errors="replace")
            if isinstance(val, np.ndarray):
                for item in val.flat:
                    if isinstance(item, (bytes, bytearray)):
                        return item.decode("utf-8", errors="replace")
            return str(val)
        return None

    def _get_int_field(self, key: str) -> Optional[int]:
        """Read an integer-typed metadata field."""
        val = self.get_field(key)
        if val is not None:
            try:
                if isinstance(val, np.ndarray):
                    return int(val.item())
                return int(val)
            except (ValueError, TypeError):
                pass
        return None

    def _get_float_field(self, key: str) -> Optional[float]:
        """Read a float-typed metadata field."""
        val = self.get_field(key)
        if val is not None:
            try:
                if isinstance(val, np.ndarray):
                    return float(val.item())
                return float(val)
            except (ValueError, TypeError):
                pass
        return None

    def _get_arch_prefix(self, architecture: str) -> str:
        """Map architecture name to its GGUF metadata prefix."""
        arch_lower = architecture.lower()
        if arch_lower in self._ARCH_PREFIXES:
            return self._ARCH_PREFIXES[arch_lower]
        # Fallback: use the architecture name directly
        return arch_lower

    @staticmethod
    def _tensor_type_name(tensor_type: int) -> str:
        """Convert GGUF tensor type integer to human-readable string."""
        type_map = {
            0: "F32",
            1: "F16",
            2: "Q4_0",
            3: "Q4_1",
            6: "Q5_0",
            7: "Q5_1",
            8: "Q8_0",
            9: "Q8_1",
            10: "Q2_K",
            11: "Q3_K_S",
            12: "Q3_K_M",
            13: "Q3_K_L",
            14: "Q4_K_S",
            15: "Q4_K_M",
            16: "Q5_K_S",
            17: "Q5_K_M",
            18: "Q6_K",
            19: "Q8_K",
            20: "IQ2_XXS",
            21: "IQ2_XS",
            22: "IQ3_XXS",
            23: "IQ1_S",
            24: "IQ4_NL",
            25: "IQ3_S",
            26: "IQ2_S",
            27: "IQ4_XS",
            28: "I8",
            29: "I16",
            30: "I32",
            31: "I64",
            32: "F64",
            33: "IQ1_M",
            34: "BF16",
        }
        return type_map.get(tensor_type, f"UNKNOWN_{tensor_type}")
