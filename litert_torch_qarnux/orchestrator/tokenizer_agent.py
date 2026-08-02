"""
TokenizerAgent - Tokenizer Extraction and Conversion.

Extracts tokenizer data from the GGUF file's metadata fields and
converts it into a SentencePiece model file suitable for packaging
into the .litertlm container.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent
from litert_torch_qarnux.utils.tokenizer_converter import TokenizerConverter

logger = logging.getLogger(__name__)


class TokenizerAgent(BaseAgent):
    """
    Extracts and converts the tokenizer from GGUF metadata.

    Reads the tokenizer vocabulary, scores, and merge rules from the
    GGUF file and reconstructs them into a SentencePiece-compatible
    model file.
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
            AgentMessage containing the path to the tokenizer file.
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

        # Convert to SentencePiece format
        tokenizer_path = self.output_dir / "tokenizer.model"
        tokenizer_path = converter.convert_to_sentencepiece(tokenizer_path)

        # Get special tokens
        special_tokens = converter.get_special_token_ids()
        self.logger.info("Special tokens: %s", special_tokens)

        return AgentMessage(
            source=self.agent_id,
            target="packaging",
            data={
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_model": tokenizer_model,
                "vocab_size": len(vocab),
                "special_tokens": special_tokens,
            },
        )
