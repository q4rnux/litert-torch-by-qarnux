"""
ParserAgent - GGUF File Parsing and Metadata Extraction.

Responsible for reading the GGUF binary file and extracting:
- Model architecture identifier
- Hyperparameters (hidden size, layer count, head count, etc.)
- Tokenizer vocabulary and merge rules
- Tensor descriptors (names, shapes, quantization types)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent
from litert_torch_qarnux.utils.gguf_parser import GGUFParser, GGUFMetadata, TensorInfo

logger = logging.getLogger(__name__)


class ParserAgent(BaseAgent):
    """
    Parses GGUF files and extracts all metadata needed by downstream agents.

    This agent performs the initial file I/O and produces structured
    metadata objects that are passed through the pipeline as AgentMessages.
    """

    def __init__(self, model_path: str | Path):
        """
        Initialize the ParserAgent.

        Args:
            model_path: Path to the GGUF model file to parse.
        """
        super().__init__("parser")
        self.model_path = Path(model_path)
        self.parser = GGUFParser(self.model_path)

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Parse the GGUF file and extract all metadata.

        Returns:
            AgentMessage containing GGUFMetadata, tensor info list,
            tokenizer data, and raw GGUFReader reference.
        """
        self.logger.info("Parsing GGUF file: %s", self.model_path)

        # Extract metadata
        metadata = self.parser.extract_metadata()

        # Enumerate tensors
        tensors = self.parser.list_tensors()

        # Extract tokenizer data
        vocab = self.parser.get_tokenizer_vocab()
        scores = self.parser.get_tokenizer_scores()
        merges = self.parser.get_tokenizer_merges()
        tok_model = self.parser.get_tokenizer_model()

        # Get quantization type distribution
        type_counts = self.parser.get_tensor_types()

        # Log summary
        self.logger.info(
            "Parsed: arch=%s, layers=%d, hidden=%d, vocab=%d, tensors=%d",
            metadata.architecture,
            metadata.block_count,
            metadata.embedding_length,
            metadata.vocab_size,
            len(tensors),
        )
        if type_counts:
            self.logger.info("Quantization types: %s", type_counts)

        return AgentMessage(
            source=self.agent_id,
            target="dequantization",
            data={
                "metadata": metadata,
                "tensors": tensors,
                "vocab": vocab,
                "scores": scores,
                "merges": merges,
                "tokenizer_model": tok_model,
                "reader": self.parser.reader,
                "file_path": str(self.model_path),
            },
        )
