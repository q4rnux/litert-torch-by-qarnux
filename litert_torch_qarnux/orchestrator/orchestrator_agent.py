"""
OrchestratorAgent - Pipeline Coordination and Execution.

The OrchestratorAgent is the central coordinator of the GGUF-to-LiteRT-LM
conversion pipeline. It instantiates and manages all specialized agents,
passes data between them via AgentMessages, handles error propagation,
and provides progress reporting to the user.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, AgentStatus
from litert_torch_qarnux.orchestrator.parser_agent import ParserAgent
from litert_torch_qarnux.orchestrator.dequantization_agent import DequantizationAgent
from litert_torch_qarnux.orchestrator.model_authoring_agent import ModelAuthoringAgent
from litert_torch_qarnux.orchestrator.conversion_agent import ConversionAgent
from litert_torch_qarnux.orchestrator.tokenizer_agent import TokenizerAgent
from litert_torch_qarnux.orchestrator.packaging_agent import PackagingAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Coordinates the complete GGUF-to-LiteRT-LM conversion pipeline.

    The orchestrator manages the sequential execution of seven agents:
    1. ParserAgent: Parse GGUF file and extract metadata
    2. DequantizationAgent: Dequantize tensor weights
    3. ModelAuthoringAgent: Build PyTorch model and load weights
    4. TokenizerAgent: Extract and convert tokenizer
    5. ConversionAgent: Convert PyTorch model to TFLite
    6. PackagingAgent: Build .litertlm container

    The orchestrator also provides progress tracking, error handling,
    and logging throughout the pipeline.
    """

    # Pipeline stage names for progress reporting
    _PIPELINE_STAGES = [
        "Parsing GGUF file",
        "Dequantizing weights",
        "Authoring PyTorch model",
        "Converting tokenizer",
        "Converting to TFLite",
        "Packaging .litertlm container",
    ]

    def __init__(
        self,
        model_path: str | Path,
        output_dir: str | Path,
        quantize: bool = True,
        quantization_recipe: str = "dynamic_wi8_afp32",
    ):
        """
        Initialize the OrchestratorAgent.

        Args:
            model_path: Path to the input GGUF model file.
            output_dir: Directory for output files and intermediate artifacts.
            quantize: Whether to apply quantization during TFLite conversion.
            quantization_recipe: Name of the quantization recipe to use.
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.quantize = quantize
        self.quantization_recipe = quantization_recipe
        self.status = AgentStatus.IDLE
        self.start_time: Optional[float] = None
        self.current_stage = 0

        # Initialize agents
        self.parser = ParserAgent(self.model_path)
        self.dequantizer = DequantizationAgent()
        self.model_author = ModelAuthoringAgent()
        self.tokenizer = TokenizerAgent(self.output_dir)
        self.converter = ConversionAgent(
            self.output_dir, self.quantize, self.quantization_recipe
        )
        self.packager = PackagingAgent(
            self.output_dir / f"{self.model_path.stem}.litertlm"
        )

        logger.info("OrchestratorAgent initialized for %s", self.model_path)

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete conversion pipeline.

        Returns:
            Dictionary containing pipeline results and statistics.

        Raises:
            RuntimeError: If any pipeline stage fails.
        """
        self.status = AgentStatus.RUNNING
        self.start_time = time.time()
        self.current_stage = 0

        logger.info("=" * 60)
        logger.info("Starting GGUF-to-LiteRT-LM conversion pipeline")
        logger.info("  Input:  %s", self.model_path)
        logger.info("  Output: %s", self.output_dir)
        logger.info("=" * 60)

        results = {}

        try:
            # Stage 1: Parse GGUF file
            self._report_progress()
            message = AgentMessage(
                source="orchestrator",
                target="parser",
                data={"model_path": str(self.model_path)},
            )
            message = self.parser.start(message)
            if not message.success:
                raise RuntimeError(f"ParserAgent failed: {message.error_message}")
            results["metadata"] = message.data["metadata"]
            results["total_tensors"] = len(message.data["tensors"])

            # Stage 2: Dequantize weights
            self._report_progress()
            message = self.dequantizer.start(message)
            if not message.success:
                raise RuntimeError(
                    f"DequantizationAgent failed: {message.error_message}"
                )
            results["dequantized_count"] = len(message.data["dequantized_weights"])

            # Stage 3: Author PyTorch model
            self._report_progress()
            message = self.model_author.start(message)
            if not message.success:
                raise RuntimeError(
                    f"ModelAuthoringAgent failed: {message.error_message}"
                )
            results["architecture"] = message.data["architecture"]
            results["model_params"] = sum(
                p.numel() for p in message.data["model"].parameters()
            )

            # Stage 4: Convert tokenizer (runs in parallel with model)
            self._report_progress()
            tokenizer_message = AgentMessage(
                source="orchestrator",
                target="tokenizer",
                data={
                    "vocab": self.parser.parser.get_tokenizer_vocab(),
                    "scores": self.parser.parser.get_tokenizer_scores(),
                    "merges": self.parser.parser.get_tokenizer_merges(),
                    "tokenizer_model": self.parser.parser.get_tokenizer_model(),
                },
            )
            tokenizer_message = self.tokenizer.start(tokenizer_message)
            if not tokenizer_message.success:
                logger.warning(
                    "TokenizerAgent failed (non-fatal): %s",
                    tokenizer_message.error_message,
                )
                tokenizer_message = AgentMessage(
                    source="tokenizer",
                    target="packaging",
                    data={"tokenizer_path": None, "tokenizer_model": "unknown"},
                )

            # Merge tokenizer data into the main message
            message.data["tokenizer_path"] = tokenizer_message.data.get("tokenizer_path")

            # Stage 5: Convert to TFLite
            self._report_progress()
            message = self.converter.start(message)
            if not message.success:
                raise RuntimeError(
                    f"ConversionAgent failed: {message.error_message}"
                )
            results["tflite_path"] = message.data["tflite_path"]

            # Stage 6: Package .litertlm
            self._report_progress()
            message = self.packager.start(message)
            if not message.success:
                raise RuntimeError(
                    f"PackagingAgent failed: {message.error_message}"
                )
            results["litertlm_path"] = message.data["litertlm_path"]
            results["file_size_bytes"] = message.data["file_size_bytes"]

        except Exception as e:
            self.status = AgentStatus.FAILED
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.error("Pipeline failed at stage %d after %.1fs: %s", self.current_stage, elapsed, e)
            raise

        # Pipeline complete
        self.status = AgentStatus.COMPLETED
        elapsed = time.time() - self.start_time
        results["elapsed_seconds"] = elapsed

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully in %.1f seconds", elapsed)
        logger.info("  Output: %s", results.get("litertlm_path", "N/A"))
        logger.info("=" * 60)

        return results

    def get_pipeline_stages(self) -> list:
        """Return the list of pipeline stage names."""
        return self._PIPELINE_STAGES.copy()

    def get_current_stage(self) -> int:
        """Return the current pipeline stage index."""
        return self.current_stage

    def get_status(self) -> AgentStatus:
        """Return the current orchestrator status."""
        return self.status

    def _report_progress(self) -> None:
        """Log the current pipeline stage."""
        if self.current_stage < len(self._PIPELINE_STAGES):
            stage_name = self._PIPELINE_STAGES[self.current_stage]
            logger.info(
                "[%d/%d] %s...",
                self.current_stage + 1,
                len(self._PIPELINE_STAGES),
                stage_name,
            )
            self.current_stage += 1
