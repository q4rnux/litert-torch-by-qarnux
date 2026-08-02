"""
Multi-Agent Orchestration System.

This package implements the agent-based orchestration architecture for
GGUF-to-LiteRT-LM conversion. Each agent is responsible for a specific
stage of the pipeline, and the OrchestratorAgent coordinates the overall
workflow, managing data flow and error propagation between agents.

The pipeline consists of 8 agents:
1. ParserAgent: Parse GGUF file and extract metadata
2. DequantizationAgent: Dequantize tensor weights
3. ModelAuthoringAgent: Build PyTorch model and load weights
4. TokenizerAgent: Extract and convert tokenizer
5. MetadataAgent: Generate LLM metadata proto for runtime acceptance
6. ConversionAgent: Convert PyTorch model to TFLite
7. PackagingAgent: Build .litertlm container
"""

from litert_torch_qarnux.orchestrator.orchestrator_agent import OrchestratorAgent
from litert_torch_qarnux.orchestrator.parser_agent import ParserAgent
from litert_torch_qarnux.orchestrator.dequantization_agent import DequantizationAgent
from litert_torch_qarnux.orchestrator.model_authoring_agent import ModelAuthoringAgent
from litert_torch_qarnux.orchestrator.tokenizer_agent import TokenizerAgent
from litert_torch_qarnux.orchestrator.metadata_agent import MetadataAgent
from litert_torch_qarnux.orchestrator.conversion_agent import ConversionAgent
from litert_torch_qarnux.orchestrator.packaging_agent import PackagingAgent

__all__ = [
    "OrchestratorAgent",
    "ParserAgent",
    "DequantizationAgent",
    "ModelAuthoringAgent",
    "TokenizerAgent",
    "MetadataAgent",
    "ConversionAgent",
    "PackagingAgent",
]
