"""
TokenizerAgent - Tokenizer Extraction and Conversion.

Extracts tokenizer data from the GGUF file's metadata fields and
converts it into the appropriate format for packaging into the
.litertlm container. Supports BOTH SentencePiece tokenizer (.model)
and HuggingFace tokenizer (tokenizer.json) paths.

Detection is based on the GGUF tokenizer model type:
- "llama", "spm", "bpe" with SentencePiece markers → SentencePiece
- "bpe", "wpm", "wordpiece", "unigram" → HuggingFace tokenizer.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent
from litert_torch_qarnux.utils.tokenizer_converter import TokenizerConverter

logger = logging.getLogger(__name__)

# Tokenizer model types that should use HuggingFace format
_HF_TOKENIZER_TYPES = {"bpe", "wpm", "wordpiece", "unigram"}

# Tokenizer model types that should use SentencePiece format
_SPM_TOKENIZER_TYPES = {"llama", "spm"}


class TokenizerAgent(BaseAgent):
    """
    Extracts and converts the tokenizer from GGUF metadata.

    Supports both SentencePiece (.model) and HuggingFace (tokenizer.json)
    output formats, choosing the appropriate path based on the GGUF
    tokenizer model type.
    """

    def __init__(self, output_dir: str | Path):
        """
        Initialize the TokenizerAgent.

        Args:
            output_dir: Directory for the output tokenizer file.
        """
        super().__init__("tokenizer")
        self.output_dir = Path(output_dir)

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Extract and convert the tokenizer.

        Returns:
            AgentMessage containing the path to the tokenizer file,
            tokenizer model type, vocab size, and special tokens.
        """
        vocab = message.data.get("vocab")
        scores = message.data.get("scores")
        merges = message.data.get("merges")
        tokenizer_model = message.data.get("tokenizer_model", "unknown")

        if vocab is None:
            self.logger.warning("No tokenizer vocabulary found in GGUF file")
            return AgentMessage(
                source=self.agent_id,
                target="packaging",
                data={
                    "tokenizer_path": None,
                    "tokenizer_model": tokenizer_model,
                    "tokenizer_format": "none",
                },
            )

        self.logger.info(
            "Converting tokenizer: %d tokens, model=%s",
            len(vocab),
            tokenizer_model,
        )

        # Create tokenizer converter
        converter = TokenizerConverter(
            tokens=vocab,
            scores=scores,
            merges=merges,
            model_type=tokenizer_model,
        )

        # Get special tokens
        special_tokens = converter.get_special_token_ids()
        self.logger.info("Special tokens: %s", special_tokens)

        # Determine tokenizer format
        tokenizer_format = self._detect_tokenizer_format(tokenizer_model)
        self.logger.info("Detected tokenizer format: %s", tokenizer_format)

        if tokenizer_format == "hf":
            # Generate HuggingFace tokenizer.json
            tokenizer_path = self._convert_to_hf_tokenizer(converter, special_tokens)
        else:
            # Generate SentencePiece model
            tokenizer_path = self.output_dir / "tokenizer.model"
            tokenizer_path = converter.convert_to_sentencepiece(tokenizer_path)

        return AgentMessage(
            source=self.agent_id,
            target="packaging",
            data={
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_model": tokenizer_model,
                "tokenizer_format": tokenizer_format,
                "vocab_size": len(vocab),
                "special_tokens": special_tokens,
            },
        )

    def _detect_tokenizer_format(self, tokenizer_model: str) -> str:
        """
        Detect whether to use SentencePiece or HuggingFace tokenizer format.

        Args:
            tokenizer_model: The tokenizer model type from GGUF metadata.

        Returns:
            "spm" for SentencePiece, "hf" for HuggingFace, or "spm" as default.
        """
        model_lower = tokenizer_model.lower()

        if model_lower in _HF_TOKENIZER_TYPES:
            return "hf"
        elif model_lower in _SPM_TOKENIZER_TYPES:
            return "spm"
        else:
            # Default to SentencePiece for unknown types
            self.logger.info(
                "Unknown tokenizer model type '%s', defaulting to SentencePiece",
                tokenizer_model,
            )
            return "spm"

    def _convert_to_hf_tokenizer(
        self, converter: TokenizerConverter, special_tokens: Dict[str, int]
    ) -> Path:
        """
        Convert GGUF tokenizer data to HuggingFace tokenizer.json format.

        Args:
            converter: The TokenizerConverter instance with tokenizer data.
            special_tokens: Dictionary of special token names to IDs.

        Returns:
            Path to the generated tokenizer.json file.
        """
        tokenizer_path = self.output_dir / "tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

        # Build HF tokenizer.json structure
        vocab = converter.get_vocab_dict()

        # Build HF tokenizer model configuration
        hf_tokenizer = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": None,
                "byte_fallback": False,
                "vocab": vocab,
                "merges": [],
            },
        }

        # Add merges if available
        if converter.merges:
            hf_tokenizer["model"]["merges"] = converter.merges

        # Add special tokens
        special_token_names = {"<unk>", "<s>", "</s>", "<bos>", "<eos>", "<pad>"}
        for name, token_id in special_tokens.items():
            if name in special_token_names:
                hf_tokenizer["added_tokens"].append({
                    "id": token_id,
                    "content": name,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                })

        # Set unknown token
        if "<unk>" in special_tokens:
            hf_tokenizer["model"]["unk_token"] = "<unk>"

        # Write tokenizer.json
        with open(tokenizer_path, "w", encoding="utf-8") as f:
            json.dump(hf_tokenizer, f, indent=2, ensure_ascii=False)

        self.logger.info("Successfully wrote HuggingFace tokenizer to %s", tokenizer_path)
        return tokenizer_path
