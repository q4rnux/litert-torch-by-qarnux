"""
Tokenizer Conversion Utility.

Extracts tokenizer data from GGUF metadata fields and reconstructs it into
SentencePiece-compatible format for packaging into the .litertlm container.
Supports BPE, SentencePiece (SPM), and WordPiece tokenizer types commonly
found in GGUF files.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TokenizerConverter:
    """
    Converts tokenizer data from GGUF format into SentencePiece model files.

    GGUF files store tokenizer information as raw arrays in the metadata section.
    This class extracts those arrays and reconstructs them into a format
    compatible with the SentencePiece library for use with LiteRT-LM.
    """

    # Token type constants matching GGUF spec
    TOKEN_TYPE_NORMAL = 1
    TOKEN_TYPE_UNKNOWN = 2
    TOKEN_TYPE_CONTROL = 3
    TOKEN_TYPE_USER_DEFINED = 4
    TOKEN_TYPE_UNUSED = 5
    TOKEN_TYPE_BYTE = 6

    # Special token type mapping (initialized in __init_subclass__)
    SPECIAL_TOKENS: dict

    def __init__(
        self,
        tokens: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        token_types: Optional[np.ndarray] = None,
        merges: Optional[List[str]] = None,
        model_type: str = "unknown",
    ):
        """
        Initialize the tokenizer converter.

        Args:
            tokens: Array of token string values from GGUF metadata.
            scores: Array of token scores (log-likelihoods or merge weights).
            token_types: Array of token type identifiers.
            merges: List of BPE merge rules (if applicable).
            model_type: The tokenizer model type string from GGUF.
        """
        self.tokens = tokens
        self.scores = scores
        self.token_types = token_types
        self.merges = merges
        self.model_type = model_type
        self.special_tokens = {
            "<unk>": TokenizerConverter.TOKEN_TYPE_UNKNOWN,
            "<s>": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "</s>": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "<bos>": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "<eos>": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "<pad>": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "[PAD]": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "[UNK]": TokenizerConverter.TOKEN_TYPE_UNKNOWN,
            "[CLS]": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "[SEP]": TokenizerConverter.TOKEN_TYPE_CONTROL,
            "[MASK]": TokenizerConverter.TOKEN_TYPE_CONTROL,
        }
        TokenizerConverter.SPECIAL_TOKENS = self.special_tokens

    @classmethod
    def from_gguf_reader(cls, reader) -> "TokenizerConverter":
        """
        Create a TokenizerConverter from a GGUFReader instance.

        Args:
            reader: A gguf.GGUFReader instance.

        Returns:
            A populated TokenizerConverter instance.
        """
        # Extract tokens
        tokens = None
        if "tokenizer.ggml.tokens" in reader.fields:
            field = reader.fields["tokenizer.ggml.tokens"]
            raw = field.parts[field.data][: field.data_len]
            tokens = np.array(
                [
                    t.decode("utf-8", errors="replace")
                    if isinstance(t, (bytes, bytearray))
                    else str(t)
                    for t in raw
                ],
                dtype=object,
            )

        # Extract scores
        scores = None
        if "tokenizer.ggml.scores" in reader.fields:
            field = reader.fields["tokenizer.ggml.scores"]
            raw = field.parts[field.data][: field.data_len]
            scores = np.array(raw, dtype=np.float32)

        # Extract token types
        token_types = None
        if "tokenizer.ggml.token_type" in reader.fields:
            field = reader.fields["tokenizer.ggml.token_type"]
            raw = field.parts[field.data][: field.data_len]
            token_types = np.array(raw, dtype=np.int32)

        # Extract merges
        merges = None
        if "tokenizer.ggml.merges" in reader.fields:
            field = reader.fields["tokenizer.ggml.merges"]
            raw = field.parts[field.data][: field.data_len]
            merges = []
            for m in raw:
                if isinstance(m, (bytes, bytearray)):
                    merges.append(m.decode("utf-8", errors="replace"))
                else:
                    merges.append(str(m))

        # Get model type
        model_type = "unknown"
        if "tokenizer.ggml.model" in reader.fields:
            field = reader.fields["tokenizer.ggml.model"]
            raw = field.parts[field.data][: field.data_len]
            for item in raw:
                if isinstance(item, (bytes, bytearray)):
                    model_type = item.decode("utf-8", errors="replace")
                    break

        return cls(
            tokens=tokens,
            scores=scores,
            token_types=token_types,
            merges=merges,
            model_type=model_type,
        )

    def convert_to_sentencepiece(
        self, output_path: str | Path, add_special_tokens: bool = True
    ) -> Path:
        """
        Convert GGUF tokenizer data to a SentencePiece model file.

        This method builds a minimal SentencePiece-compatible model using
        the protobuf protocol buffer format. For models that already have
        an SPM model, this serves as a reconstruction; for BPE models,
        it creates a compatible representation.

        Args:
            output_path: Destination path for the .model file.
            add_special_tokens: Whether to add <bos>, <eos>, <unk> tokens.

        Returns:
            Path to the generated SentencePiece model file.

        Raises:
            ValueError: If no token vocabulary is available.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.tokens is None or len(self.tokens) == 0:
            raise ValueError("No token vocabulary available for conversion")

        logger.info(
            "Converting tokenizer (%d tokens, type=%s) to %s",
            len(self.tokens),
            self.model_type,
            output_path,
        )

        try:
            self._build_spm_proto(output_path, add_special_tokens)
            logger.info("Successfully wrote SentencePiece model to %s", output_path)
        except ImportError:
            # Fallback: write a JSON-based tokenizer file
            logger.warning(
                "sentencepiece not available, writing JSON tokenizer fallback"
            )
            self._write_json_tokenizer(output_path)

        return output_path

    def get_vocab_dict(self) -> Dict[str, int]:
        """
        Build a vocabulary dictionary mapping tokens to their IDs.

        Returns:
            Dictionary with token strings as keys and integer IDs as values.
        """
        if self.tokens is None:
            return {}
        return {token: idx for idx, token in enumerate(self.tokens)}

    def get_special_token_ids(self) -> Dict[str, int]:
        """
        Identify special token IDs from the vocabulary.

        Returns:
            Dictionary mapping special token names to their vocabulary IDs.
        """
        special = {}
        vocab = self.get_vocab_dict()
        for name in ["<unk>", "<s>", "</s>", "<bos>", "<eos>", "<pad>"]:
            if name in vocab:
                special[name] = vocab[name]
        return special

    # -- Private methods --

    def _build_spm_proto(
        self, output_path: Path, add_special_tokens: bool = True
    ) -> None:
        """
        Build a SentencePiece model file using the protobuf format.

        This creates a minimal but valid SentencePiece model that can be
        loaded by the SentencePiece library. The model uses a simple
        unigram language model with all tokens from the vocabulary.
        """
        import sentencepiece as spm

        # Create a temporary training file with the vocabulary
        vocab_file = output_path.with_suffix(".vocab.tmp")
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in enumerate(self.tokens):
                score = 0.0
                if self.scores is not None and idx < len(self.scores):
                    score = float(self.scores[idx])
                # Escape tabs in token strings
                safe_token = token.replace("\t", "\\t").replace("\n", "\\n")
                f.write(f"{safe_token}\t{score}\n")

        # Use SentencePiece to train a minimal model
        vocab_size = len(self.tokens)
        spm.SentencePieceTrainer.train(
            input=str(vocab_file),
            model_prefix=str(output_path.with_suffix("")),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            pad_id=-1 if "<pad>" not in self.get_special_token_ids() else 0,
            unk_id=self.get_special_token_ids().get("<unk>", 0),
            bos_id=-1 if "<bos>" not in self.get_special_token_ids() else 1,
            eos_id=self.get_special_token_ids().get("<eos>", 2),
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            num_threads=1,
            shuffle_input_sentence=False,
            max_sentencepiece_length=0,
        )

        # Clean up temporary file
        vocab_file.unlink(missing_ok=True)

    def _write_json_tokenizer(self, output_path: Path) -> None:
        """
        Fallback: write a JSON tokenizer file when SentencePiece is unavailable.

        Args:
            output_path: Destination path for the JSON tokenizer file.
        """
        tokenizer_data = {
            "version": "1.0",
            "model_type": self.model_type,
            "vocab": self.get_vocab_dict(),
            "special_tokens": self.get_special_token_ids(),
        }
        json_path = output_path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
        logger.info("Wrote JSON tokenizer fallback to %s", json_path)
