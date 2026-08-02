"""
Tests for the GGUF parser utility module.

Validates metadata extraction, tensor enumeration, and architecture
detection from sample GGUF metadata structures.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from litert_torch_qarnux.utils.gguf_parser import GGUFParser, GGUFMetadata


def _make_field(value):
    """Helper to create a mock GGUFReader field.

    The GGUFParser.get_field method does:
        field.parts[field.data][:field.data_len]

    So parts[0] should contain the actual data array, data=0, data_len=len.
    """
    f = MagicMock()
    if isinstance(value, list):
        f.parts = [value]
        f.data = 0
        f.data_len = len(value)
    elif isinstance(value, np.ndarray):
        f.parts = [value]
        f.data = 0
        f.data_len = len(value)
    elif isinstance(value, (bytes, str)):
        f.parts = [value]
        f.data = 0
        f.data_len = len(value)
    else:
        f.parts = [value]
        f.data = 0
        f.data_len = 1
    return f


class TestGGUFMetadata:
    """Tests for the GGUFMetadata dataclass."""

    def test_default_values(self):
        meta = GGUFMetadata()
        assert meta.architecture == ""
        assert meta.name == ""
        assert meta.embedding_length == 0
        assert meta.block_count == 0
        assert meta.attention_head_count == 0
        assert meta.vocab_size == 0
        assert meta.rms_norm_eps == 1e-6
        assert meta.raw_fields == {}

    def test_custom_values(self):
        meta = GGUFMetadata(
            architecture="llama",
            name="test-model",
            embedding_length=4096,
            block_count=32,
            attention_head_count=32,
            vocab_size=32000,
        )
        assert meta.architecture == "llama"
        assert meta.embedding_length == 4096
        assert meta.block_count == 32
        assert meta.attention_head_count == 32
        assert meta.vocab_size == 32000


class TestGGUFParser:
    """Tests for the GGUFParser class."""

    def test_init_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.gguf"
        with pytest.raises(FileNotFoundError):
            GGUFParser(missing)

    @patch("litert_torch_qarnux.utils.gguf_parser.gguf.GGUFReader")
    def test_extract_metadata(self, mock_reader_cls, tmp_path):
        mock_path = tmp_path / "model.gguf"
        mock_path.touch()

        mock_reader = MagicMock()
        mock_reader_cls.return_value = mock_reader

        mock_reader.fields = {
            "general.architecture": _make_field(b"llama"),
            "general.name": _make_field("test"),
            "llama.embedding_length": _make_field(np.array([4096])),
            "llama.block_count": _make_field(np.array([32])),
            "llama.attention.head_count": _make_field(np.array([32])),
            "llama.attention.head_count_kv": _make_field(np.array([8])),
            "llama.feed_forward_length": _make_field(np.array([11008])),
            "llama.context_length": _make_field(np.array([4096])),
            "llama.rope.freq_base": _make_field(np.array([10000.0])),
            "llama.attention.layer_norm_rms_epsilon": _make_field(np.array([1e-5])),
            "tokenizer.ggml.tokens_count": _make_field(np.array([32000])),
        }

        parser = GGUFParser(mock_path)
        metadata = parser.extract_metadata()

        assert metadata.architecture == "llama"
        assert metadata.name == "test"
        assert metadata.embedding_length == 4096
        assert metadata.block_count == 32
        assert metadata.attention_head_count == 32
        assert metadata.attention_head_count_kv == 8
        assert metadata.feed_forward_length == 11008
        assert metadata.context_length == 4096
        assert metadata.vocab_size == 32000

    @patch("litert_torch_qarnux.utils.gguf_parser.gguf.GGUFReader")
    def test_list_tensors(self, mock_reader_cls, tmp_path):
        mock_path = tmp_path / "model.gguf"
        mock_path.touch()

        mock_reader = MagicMock()
        mock_reader_cls.return_value = mock_reader

        mock_tensor1 = MagicMock()
        mock_tensor1.name = "blk.0.attn_q.weight"
        mock_tensor1.shape = [4096, 4096]
        mock_tensor1.tensor_type = 2  # Q4_0
        mock_tensor1.data = np.zeros((1,))

        mock_tensor2 = MagicMock()
        mock_tensor2.name = "token_embd.weight"
        mock_tensor2.shape = [32000, 4096]
        mock_tensor2.tensor_type = 0  # F32
        mock_tensor2.data = np.zeros((1,))

        mock_reader.tensors = [mock_tensor1, mock_tensor2]

        parser = GGUFParser(mock_path)
        tensors = parser.list_tensors()

        assert len(tensors) == 2
        assert tensors[0].name == "blk.0.attn_q.weight"
        assert tensors[0].data_type == "Q4_0"
        assert tensors[1].name == "token_embd.weight"
        assert tensors[1].data_type == "F32"

    @patch("litert_torch_qarnux.utils.gguf_parser.gguf.GGUFReader")
    def test_get_tokenizer_vocab(self, mock_reader_cls, tmp_path):
        mock_path = tmp_path / "model.gguf"
        mock_path.touch()

        mock_reader = MagicMock()
        mock_reader_cls.return_value = mock_reader

        tokens = [b"<unk>", b"hello", b"world"]
        mock_reader.fields = {
            "tokenizer.ggml.tokens": _make_field(tokens),
        }

        parser = GGUFParser(mock_path)
        vocab = parser.get_tokenizer_vocab()

        assert vocab is not None
        assert len(vocab) == 3
