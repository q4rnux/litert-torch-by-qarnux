"""
MetadataAgent - LLM Metadata Proto Generation.

Generates the llm_metadata.pb protobuf file required by LiteRT-LM runtime
apps. This includes start/stop tokens, prompt templates, sampler parameters,
max token limits, and architecture-specific model type mapping.

This agent does NOT set uuid or creation_timestamp — those are auto-generated
by the litert-lm-builder. Instead, it generates and logs these values for
traceability.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent

logger = logging.getLogger(__name__)


# ── Prompt template definitions ──────────────────────────────────────────

@dataclass
class PromptAffixes:
    """Prefix and suffix for a prompt role."""
    prefix: str
    suffix: str


@dataclass
class PromptTemplates:
    """Prompt templates for user/model/system roles."""
    user: Optional[PromptAffixes] = None
    model: Optional[PromptAffixes] = None
    system: Optional[PromptAffixes] = None


@dataclass
class SamplerParameters:
    """Default sampler parameters."""
    type: str = "TOP_K"  # TOP_K, TOP_P, GREEDY
    k: int = 40
    p: float = 0.95
    temperature: float = 0.8
    seed: Optional[int] = None
    backend: str = "CPU"


@dataclass
class LlmMetadata:
    """Complete LLM metadata for the .litertlm container."""
    start_token: Optional[int] = None
    stop_tokens: List[int] = field(default_factory=list)
    prompt_templates: Optional[PromptTemplates] = None
    sampler_params: Optional[SamplerParameters] = None
    max_num_tokens: int = 4096
    llm_model_type: str = "GenericModel"
    jinja_prompt_template: Optional[str] = None
    system_metadata: Dict[str, str] = field(default_factory=dict)


# ── Architecture-to-LlmModelType mapping ─────────────────────────────────

_ARCH_TO_MODEL_TYPE: Dict[str, str] = {
    "llama": "GenericModel",
    "mistral": "GenericModel",
    "phi": "GenericModel",
    "phi2": "GenericModel",
    "phi3": "GenericModel",
    "smollm3": "GenericModel",
    "gemma": "Gemma3",
    "gemma2": "Gemma3",
    "gemma3": "Gemma3",
    "gemma3n": "Gemma3N",
    "gemma4": "Gemma4",
    "qwen2": "Qwen2p5",
    "qwen3": "Qwen3",
}


# ── Architecture-specific prompt templates ───────────────────────────────

def _llama_prompt_templates() -> PromptTemplates:
    """Llama/Mistral-style chat templates."""
    return PromptTemplates(
        user=PromptAffixes(
            prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
            suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        model=PromptAffixes(
            prefix="",
            suffix="",
        ),
        system=PromptAffixes(
            prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            suffix="<|eot_id|>",
        ),
    )


def _mistral_prompt_templates() -> PromptTemplates:
    """Mistral-style chat templates."""
    return PromptTemplates(
        user=PromptAffixes(
            prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
            suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        model=PromptAffixes(
            prefix="",
            suffix="",
        ),
        system=PromptAffixes(
            prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            suffix="<|eot_id|>",
        ),
    )


def _gemma_prompt_templates() -> PromptTemplates:
    """Gemma-style chat templates."""
    return PromptTemplates(
        user=PromptAffixes(
            prefix="<start_of_turn>user\n",
            suffix="<end_of_turn>\n",
        ),
        model=PromptAffixes(
            prefix="<start_of_turn>model\n",
            suffix="<end_of_turn>\n",
        ),
        system=PromptAffixes(
            prefix="<start_of_turn>system\n",
            suffix="<end_of_turn>\n",
        ),
    )


def _gemma3_prompt_templates() -> PromptTemplates:
    """Gemma3-style chat templates (same base as Gemma)."""
    return _gemma_prompt_templates()


def _qwen_prompt_templates() -> PromptTemplates:
    """Qwen-style chat templates."""
    return PromptTemplates(
        user=PromptAffixes(
            prefix="<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
            suffix="<|im_end|>\n<|im_start|>assistant\n",
        ),
        model=PromptAffixes(
            prefix="",
            suffix="<|im_end|>",
        ),
        system=PromptAffixes(
            prefix="<|im_start|>system\n",
            suffix="<|im_end|>\n",
        ),
    )


def _phi_prompt_templates() -> PromptTemplates:
    """Phi-style chat templates."""
    return PromptTemplates(
        user=PromptAffixes(
            prefix="<|system|>\n{system}<|end|>\n<|user|>\n",
            suffix="<|end|>\n<|assistant|>\n",
        ),
        model=PromptAffixes(
            prefix="",
            suffix="<|end|>",
        ),
        system=PromptAffixes(
            prefix="<|system|>\n",
            suffix="<|end|>\n",
        ),
    )


def _default_prompt_templates() -> PromptTemplates:
    """Generic fallback prompt templates."""
    return PromptTemplates(
        user=PromptAffixes(
            prefix="",
            suffix="",
        ),
        model=PromptAffixes(
            prefix="",
            suffix="",
        ),
        system=PromptAffixes(
            prefix="",
            suffix="",
        ),
    )


def get_prompt_templates(architecture: str) -> PromptTemplates:
    """
    Get architecture-specific prompt templates.

    Args:
        architecture: The GGUF architecture identifier.

    Returns:
        PromptTemplates for the given architecture.
    """
    arch_lower = architecture.lower()
    template_map = {
        "llama": _llama_prompt_templates,
        "mistral": _mistral_prompt_templates,
        "gemma": _gemma_prompt_templates,
        "gemma2": _gemma_prompt_templates,
        "gemma3": _gemma3_prompt_templates,
        "gemma3n": _gemma_prompt_templates,
        "gemma4": _gemma_prompt_templates,
        "qwen2": _qwen_prompt_templates,
        "qwen3": _qwen_prompt_templates,
        "phi": _phi_prompt_templates,
        "phi2": _phi_prompt_templates,
        "phi3": _phi_prompt_templates,
        "smollm3": _default_prompt_templates,
    }
    func = template_map.get(arch_lower, _default_prompt_templates)
    return func()


def get_llm_model_type(architecture: str) -> str:
    """
    Map architecture to LiteRT-LM LlmModelType.

    Args:
        architecture: The GGUF architecture identifier.

    Returns:
        String name of the LlmModelType enum value.
    """
    arch_lower = architecture.lower()
    return _ARCH_TO_MODEL_TYPE.get(arch_lower, "GenericModel")


class MetadataAgent(BaseAgent):
    """
    Generates LLM metadata proto information for the .litertlm container.

    This agent extracts BOS/EOS token IDs from the GGUF tokenizer data,
    selects architecture-appropriate prompt templates, configures sampler
    parameters, and builds the complete LlmMetadata structure. It also
    generates and logs uuid/creation_timestamp for traceability (without
    setting them in the builder, as those are auto-generated).
    """

    def __init__(
        self,
        output_dir: str | Path,
        target_backend: str = "CPU",
        quantization_recipe: str = "dynamic_wi8_afp32",
        conversion_tool_version: str = "1.0.0",
    ):
        """
        Initialize the MetadataAgent.

        Args:
            output_dir: Directory for output files.
            target_backend: Target backend (CPU, GPU, NPU).
            quantization_recipe: Quantization recipe used.
            conversion_tool_version: Version of the conversion tool.
        """
        super().__init__("metadata")
        self.output_dir = Path(output_dir)
        self.target_backend = target_backend
        self.quantization_recipe = quantization_recipe
        self.conversion_tool_version = conversion_tool_version

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Generate LLM metadata from GGUF parser data.

        Returns:
            AgentMessage containing the LlmMetadata structure and
            generated UUID/timestamp for traceability.
        """
        # Gather inputs
        metadata = message.data.get("metadata")
        special_tokens = message.data.get("special_tokens", {})
        tokenizer_model_type = message.data.get("tokenizer_model", "unknown")

        # Generate traceability UUID and timestamp
        generated_uuid = str(uuid_lib.uuid4())
        creation_timestamp = datetime.now(timezone.utc).isoformat()

        self.logger.info("Generated traceability UUID: %s", generated_uuid)
        self.logger.info("Generated creation timestamp: %s", creation_timestamp)

        if metadata is None:
            self.logger.warning("No metadata available for MetadataAgent")
            return AgentMessage(
                source=self.agent_id,
                target="packaging",
                data={
                    "llm_metadata": None,
                    "generated_uuid": generated_uuid,
                    "creation_timestamp": creation_timestamp,
                },
            )

        # Extract special token IDs
        start_token = self._extract_start_token(special_tokens, metadata)
        stop_tokens = self._extract_stop_tokens(special_tokens, metadata)

        # Get architecture
        architecture = metadata.architecture

        # Get prompt templates
        prompt_templates = get_prompt_templates(architecture)

        # Get LLM model type
        llm_model_type = get_llm_model_type(architecture)

        # Build sampler parameters
        sampler_params = SamplerParameters(
            type="TOP_K",
            k=40,
            p=0.95,
            temperature=0.8,
            seed=None,
            backend=self.target_backend,
        )

        # Build system metadata
        system_metadata = self._build_system_metadata(
            metadata, architecture, tokenizer_model_type
        )

        # Assemble LlmMetadata
        llm_metadata = LlmMetadata(
            start_token=start_token,
            stop_tokens=stop_tokens,
            prompt_templates=prompt_templates,
            sampler_params=sampler_params,
            max_num_tokens=metadata.context_length or 4096,
            llm_model_type=llm_model_type,
            system_metadata=system_metadata,
        )

        # Log summary
        self.logger.info("LLM Metadata generated:")
        self.logger.info("  Architecture: %s", architecture)
        self.logger.info("  LLM Model Type: %s", llm_model_type)
        self.logger.info("  Start Token: %s", start_token)
        self.logger.info("  Stop Tokens: %s", stop_tokens)
        self.logger.info("  Max Tokens: %d", llm_metadata.max_num_tokens)
        self.logger.info("  Backend: %s", self.target_backend)
        self.logger.info("  Prompt Templates: %s", architecture)

        return AgentMessage(
            source=self.agent_id,
            target="conversion",
            data={
                **message.data,
                "llm_metadata": llm_metadata,
                "generated_uuid": generated_uuid,
                "creation_timestamp": creation_timestamp,
            },
        )

    def _extract_start_token(
        self, special_tokens: Dict[str, int], metadata
    ) -> Optional[int]:
        """
        Extract BOS token ID from special tokens.

        Priority: <bos> > <s> > first special token > default 1.
        """
        # Check explicit BOS tokens
        for bos_name in ["<bos>", "<s>"]:
            if bos_name in special_tokens:
                return special_tokens[bos_name]

        # Check raw fields for explicit BOS token ID
        if hasattr(metadata, "raw_fields"):
            bos_id = metadata.raw_fields.get("tokenizer.ggml.bos_token_id")
            if bos_id is not None:
                try:
                    return int(bos_id)
                except (ValueError, TypeError):
                    pass

        # Default: try token ID 1 (common for SentencePiece)
        return 1

    def _extract_stop_tokens(
        self, special_tokens: Dict[str, int], metadata
    ) -> List[int]:
        """
        Extract EOS token IDs from special tokens.

        Priority: <eos>, </s>, then any remaining special tokens.
        """
        stop_tokens = []

        # Check explicit EOS tokens
        for eos_name in ["<eos>", "</s>"]:
            if eos_name in special_tokens:
                stop_tokens.append(special_tokens[eos_name])

        # Check raw fields for explicit EOS token ID
        if hasattr(metadata, "raw_fields"):
            eos_id = metadata.raw_fields.get("tokenizer.ggml.eos_token_id")
            if eos_id is not None:
                try:
                    eos_val = int(eos_id)
                    if eos_val not in stop_tokens:
                        stop_tokens.append(eos_val)
                except (ValueError, TypeError):
                    pass

        # If still no stop tokens, add common defaults
        if not stop_tokens:
            # Try to find from special tokens
            for name, tid in special_tokens.items():
                if tid not in stop_tokens and tid >= 0:
                    stop_tokens.append(tid)
                    break

        # Fallback
        if not stop_tokens:
            stop_tokens = [2]  # Common default EOS token ID

        return stop_tokens

    def _build_system_metadata(
        self, metadata, architecture: str, tokenizer_model: str
    ) -> Dict[str, str]:
        """
        Build the system metadata dictionary for the .litertlm container.

        Returns:
            Dictionary of metadata key-value pairs.
        """
        # Detect quantization type from tensor data
        source_quant = self._detect_source_quantization(metadata)

        system_metadata = {
            "Authors": "qarnux",
            "TargetBackend": self.target_backend,
            "Architecture": architecture,
            "ModelName": metadata.name or metadata.architecture,
            "SourceFormat": "GGUF",
            "QuantizationType": self.quantization_recipe,
            "ConversionTool": "litert-torch-by-qarnux",
            "ConversionToolVersion": self.conversion_tool_version,
            "SourceQuantization": source_quant,
        }

        # Add layer count if available
        if metadata.block_count > 0:
            system_metadata["NumLayers"] = str(metadata.block_count)

        # Add vocabulary size
        if metadata.vocab_size > 0:
            system_metadata["VocabSize"] = str(metadata.vocab_size)

        # Add context length
        if metadata.context_length > 0:
            system_metadata["ContextLength"] = str(metadata.context_length)

        return system_metadata

    def _detect_source_quantization(self, metadata) -> str:
        """
        Detect the source quantization type from GGUF file_type field.

        Returns:
            Human-readable quantization type string.
        """
        file_type_map = {
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
        }

        if metadata.file_type is not None:
            return file_type_map.get(metadata.file_type, f"F{metadata.file_type}")

        return "unknown"
