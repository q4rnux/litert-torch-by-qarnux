"""
PackagingAgent - LiteRT-LM Container Building.

Assembles all conversion artifacts (TFLite model, tokenizer, metadata)
into a single .litertlm container file using the litert-lm-builder
Python package.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent

logger = logging.getLogger(__name__)


class PackagingAgent(BaseAgent):
    """
    Builds the final .litertlm container file.

    Uses the litert-lm-builder library to package the TFLite model,
    SentencePiece tokenizer, and system metadata into a unified
    container suitable for the LiteRT-LM runtime.
    """

    def __init__(self, output_path: str | Path):
        """
        Initialize the PackagingAgent.

        Args:
            output_path: Destination path for the .litertlm file.
        """
        super().__init__("packaging")
        self.output_path = Path(output_path)

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Build the .litertlm container file.

        Returns:
            AgentMessage containing the path to the final .litertlm file.
        """
        from litert_lm_builder import (
            LitertLmFileBuilder,
            Metadata,
            DType,
            TfLiteModelType,
            Backend,
        )

        # Gather inputs from upstream agents
        conversion_data = message.data
        metadata = conversion_data.get("metadata")
        tflite_path = conversion_data.get("tflite_path")
        tokenizer_path = conversion_data.get("tokenizer_path")

        # Validate required inputs
        if tflite_path is None:
            raise RuntimeError("No TFLite model path provided for packaging")

        tflite_file = Path(tflite_path)
        if not tflite_file.exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_file}")

        self.logger.info("Building .litertlm container...")
        self.logger.info("  TFLite model: %s", tflite_file)

        # Initialize the builder
        builder = LitertLmFileBuilder()

        # Add system metadata
        builder.add_system_metadata(
            Metadata(key="Authors", value="qarnux", dtype=DType.STRING)
        )
        builder.add_system_metadata(
            Metadata(
                key="SourceFormat",
                value="GGUF",
                dtype=DType.STRING,
            )
        )
        builder.add_system_metadata(
            Metadata(
                key="TargetBackend",
                value=Backend.CPU.name,
                dtype=DType.STRING,
            )
        )
        if metadata:
            builder.add_system_metadata(
                Metadata(
                    key="Architecture",
                    value=metadata.architecture,
                    dtype=DType.STRING,
                )
            )
            if metadata.block_count > 0:
                builder.add_system_metadata(
                    Metadata(
                        key="NumLayers",
                        value=str(metadata.block_count),
                        dtype=DType.STRING,
                    )
                )

        # Add the TFLite model
        builder.add_tflite_model(
            tflite_model_path=str(tflite_file),
            model_type=TfLiteModelType.PREFILL_DECODE,
        )

        # Add tokenizer if available
        if tokenizer_path and Path(tokenizer_path).exists():
            builder.add_sentencepiece_tokenizer(
                sp_tokenizer_path=str(tokenizer_path)
            )
            self.logger.info("  Tokenizer: %s", tokenizer_path)
        else:
            self.logger.warning("No tokenizer available, packaging without tokenizer")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the container
        with open(self.output_path, "wb") as f:
            builder.build(f)

        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "Successfully built .litertlm container: %s (%.1f MB)",
            self.output_path,
            file_size_mb,
        )

        return AgentMessage(
            source=self.agent_id,
            target="orchestrator",
            data={
                "litertlm_path": str(self.output_path),
                "file_size_bytes": self.output_path.stat().st_size,
            },
        )
