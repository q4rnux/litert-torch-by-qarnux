"""
Tests for the MetadataAgent and related metadata generation utilities.

Validates LLM metadata proto generation, prompt template mapping,
model type detection, UUID/timestamp generation, and system metadata.

These tests import directly from the module files to avoid requiring
heavy dependencies (torch, onnx2tf) that are not needed for metadata testing.
"""

import sys
import uuid as uuid_lib
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

import numpy as np

# Direct imports to avoid pulling in torch-dependent modules
from litert_torch_qarnux.utils.gguf_parser import GGUFMetadata
from litert_torch_qarnux.orchestrator.base_agent import AgentMessage

# Now import metadata_agent components
from litert_torch_qarnux.orchestrator.metadata_agent import (
    LlmMetadata,
    MetadataAgent,
    PromptAffixes,
    PromptTemplates,
    SamplerParameters,
    get_llm_model_type,
    get_prompt_templates,
)


class TestPromptTemplates:
    """Tests for architecture-specific prompt template generation."""

    def test_llama_prompt_templates(self):
        templates = get_prompt_templates("llama")
        assert templates is not None
        assert templates.user is not None
        assert templates.system is not None
        assert "<|begin_of_text|>" in templates.user.prefix
        assert "<|start_header_id|>system" in templates.user.prefix
        assert "<|eot_id|>" in templates.user.suffix
        assert "<|start_header_id|>assistant" in templates.user.suffix

    def test_mistral_prompt_templates(self):
        templates = get_prompt_templates("mistral")
        assert templates is not None
        assert templates.user is not None
        assert "<|begin_of_text|>" in templates.user.prefix

    def test_gemma_prompt_templates(self):
        templates = get_prompt_templates("gemma")
        assert templates is not None
        assert templates.user is not None
        assert "<start_of_turn>user" in templates.user.prefix
        assert "<end_of_turn>" in templates.user.suffix

    def test_gemma3_prompt_templates(self):
        templates = get_prompt_templates("gemma3")
        assert templates is not None
        assert templates.user is not None
        assert "<start_of_turn>user" in templates.user.prefix

    def test_gemma4_prompt_templates(self):
        templates = get_prompt_templates("gemma4")
        assert templates is not None
        assert templates.user is not None
        assert "<start_of_turn>user" in templates.user.prefix

    def test_qwen2_prompt_templates(self):
        templates = get_prompt_templates("qwen2")
        assert templates is not None
        assert templates.user is not None
        assert "<|im_start|>" in templates.user.prefix
        assert "<|im_end|>" in templates.user.suffix

    def test_qwen3_prompt_templates(self):
        templates = get_prompt_templates("qwen3")
        assert templates is not None
        assert templates.user is not None
        assert "<|im_start|>" in templates.user.prefix

    def test_phi_prompt_templates(self):
        templates = get_prompt_templates("phi")
        assert templates is not None
        assert templates.user is not None
        assert "<|system|>" in templates.user.prefix
        assert "<|end|>" in templates.user.prefix
        assert "<|assistant|>" in templates.user.suffix

    def test_phi3_prompt_templates(self):
        templates = get_prompt_templates("phi3")
        assert templates is not None
        assert templates.user is not None
        assert "<|system|>" in templates.user.prefix

    def test_default_prompt_templates_for_unknown(self):
        templates = get_prompt_templates("unknown_arch")
        assert templates is not None
        assert templates.user is not None

    def test_case_insensitive_architecture(self):
        templates_lower = get_prompt_templates("llama")
        templates_upper = get_prompt_templates("LLAMA")
        templates_mixed = get_prompt_templates("Llama")
        assert templates_lower.user.prefix == templates_upper.user.prefix
        assert templates_lower.user.prefix == templates_mixed.user.prefix


class TestLlmModelType:
    """Tests for architecture-to-LlmModelType mapping."""

    def test_generic_models(self):
        assert get_llm_model_type("llama") == "GenericModel"
        assert get_llm_model_type("mistral") == "GenericModel"
        assert get_llm_model_type("phi") == "GenericModel"
        assert get_llm_model_type("phi2") == "GenericModel"
        assert get_llm_model_type("phi3") == "GenericModel"
        assert get_llm_model_type("smollm3") == "GenericModel"

    def test_gemma_models(self):
        assert get_llm_model_type("gemma") == "Gemma3"
        assert get_llm_model_type("gemma2") == "Gemma3"
        assert get_llm_model_type("gemma3") == "Gemma3"

    def test_gemma3n(self):
        assert get_llm_model_type("gemma3n") == "Gemma3N"

    def test_gemma4(self):
        assert get_llm_model_type("gemma4") == "Gemma4"

    def test_qwen_models(self):
        assert get_llm_model_type("qwen2") == "Qwen2p5"
        assert get_llm_model_type("qwen3") == "Qwen3"

    def test_unknown_architecture_defaults_to_generic(self):
        assert get_llm_model_type("unknown") == "GenericModel"

    def test_case_insensitive(self):
        assert get_llm_model_type("LLAMA") == "GenericModel"
        assert get_llm_model_type("Gemma3") == "Gemma3"


class TestMetadataAgent:
    """Tests for the MetadataAgent."""

    @pytest.fixture
    def agent(self, tmp_path):
        return MetadataAgent(
            output_dir=tmp_path,
            target_backend="CPU",
            quantization_recipe="dynamic_wi8_afp32",
        )

    @pytest.fixture
    def gpu_agent(self, tmp_path):
        return MetadataAgent(
            output_dir=tmp_path,
            target_backend="GPU",
            quantization_recipe="dynamic_wi4_afp32",
        )

    @pytest.fixture
    def sample_metadata(self):
        return GGUFMetadata(
            architecture="llama",
            name="test-llama-model",
            embedding_length=4096,
            block_count=32,
            attention_head_count=32,
            context_length=8192,
            vocab_size=32000,
            file_type=15,  # Q4_K_M
        )

    @pytest.fixture
    def special_tokens(self):
        return {"<bos>": 1, "<eos>": 2, "<unk>": 0}

    def test_metadata_agent_generates_llm_metadata(self, agent, sample_metadata, special_tokens):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)

        assert result.success
        assert result.data["llm_metadata"] is not None

        llm_meta = result.data["llm_metadata"]
        assert isinstance(llm_meta, LlmMetadata)
        assert llm_meta.start_token == 1
        assert llm_meta.stop_tokens == [2]
        assert llm_meta.max_num_tokens == 8192
        assert llm_meta.llm_model_type == "GenericModel"

    def test_metadata_agent_generates_uuid_and_timestamp(self, agent, sample_metadata, special_tokens):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)

        assert "generated_uuid" in result.data
        assert "creation_timestamp" in result.data

        # Validate UUID format
        generated_uuid = result.data["generated_uuid"]
        assert uuid_lib.UUID(generated_uuid)

        # Validate timestamp is ISO 8601 format
        timestamp = result.data["creation_timestamp"]
        assert "T" in timestamp

    def test_metadata_agent_sets_correct_model_type(self, agent, special_tokens):
        # Test Gemma3 model type
        gemma_metadata = GGUFMetadata(
            architecture="gemma3",
            name="test-gemma3",
            context_length=4096,
        )
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": gemma_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "gemma",
            },
        )
        result = agent.execute(message)
        assert result.data["llm_metadata"].llm_model_type == "Gemma3"

        # Test Qwen3 model type
        qwen_metadata = GGUFMetadata(
            architecture="qwen3",
            name="test-qwen3",
            context_length=4096,
        )
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": qwen_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "bpe",
            },
        )
        result = agent.execute(message)
        assert result.data["llm_metadata"].llm_model_type == "Qwen3"

    def test_metadata_agent_sampler_params(self, agent, sample_metadata, special_tokens):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)

        sampler = result.data["llm_metadata"].sampler_params
        assert sampler is not None
        assert sampler.type == "TOP_K"
        assert sampler.k == 40
        assert sampler.p == 0.95
        assert sampler.temperature == 0.8
        assert sampler.backend == "CPU"

    def test_metadata_agent_sampler_params_gpu(self, gpu_agent, sample_metadata, special_tokens):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "llama",
            },
        )
        result = gpu_agent.execute(message)

        sampler = result.data["llm_metadata"].sampler_params
        assert sampler.backend == "GPU"

    def test_metadata_agent_system_metadata(self, agent, sample_metadata, special_tokens):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)

        system_meta = result.data["llm_metadata"].system_metadata
        assert system_meta["Authors"] == "qarnux"
        assert system_meta["TargetBackend"] == "CPU"
        assert system_meta["Architecture"] == "llama"
        assert system_meta["ModelName"] == "test-llama-model"
        assert system_meta["SourceFormat"] == "GGUF"
        assert system_meta["QuantizationType"] == "dynamic_wi8_afp32"
        assert system_meta["ConversionTool"] == "litert-torch-by-qarnux"
        assert system_meta["ConversionToolVersion"] == "1.0.0"
        assert system_meta["SourceQuantization"] == "Q4_K_M"
        assert system_meta["NumLayers"] == "32"
        assert system_meta["VocabSize"] == "32000"
        assert system_meta["ContextLength"] == "8192"

    def test_metadata_agent_stop_tokens_extraction(self, agent, special_tokens):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": GGUFMetadata(architecture="llama", context_length=4096),
                "special_tokens": {"<eos>": 2, "</s>": 3},
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)
        assert 2 in result.data["llm_metadata"].stop_tokens
        assert 3 in result.data["llm_metadata"].stop_tokens

    def test_metadata_agent_no_metadata_fallback(self, agent):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": None,
                "special_tokens": {},
                "tokenizer_model": "unknown",
            },
        )
        result = agent.execute(message)
        assert result.data["llm_metadata"] is None

    def test_metadata_agent_empty_special_tokens(self, agent, sample_metadata):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": {},
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)
        # Should fall back to default stop token
        assert result.data["llm_metadata"].stop_tokens == [2]

    def test_metadata_agent_prompt_templates_set(self, agent, sample_metadata, special_tokens):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)
        assert result.data["llm_metadata"].prompt_templates is not None
        assert result.data["llm_metadata"].prompt_templates.user is not None

    def test_metadata_agent_no_special_tokens_fallback(self, agent):
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": GGUFMetadata(architecture="llama", context_length=4096),
                "special_tokens": {},
                "tokenizer_model": "llama",
            },
        )
        result = agent.execute(message)
        # Should default start_token to 1
        assert result.data["llm_metadata"].start_token == 1

    def test_metadata_agent_data_passed_through(self, agent, sample_metadata, special_tokens):
        """Ensure MetadataAgent passes through existing data from upstream agents."""
        message = AgentMessage(
            source="orchestrator",
            target="metadata",
            data={
                "metadata": sample_metadata,
                "special_tokens": special_tokens,
                "tokenizer_model": "llama",
                "tflite_path": "/tmp/test.tflite",
                "model": "mock_model",
            },
        )
        result = agent.execute(message)
        # Ensure upstream data is preserved
        assert result.data["tflite_path"] == "/tmp/test.tflite"
        assert result.data["model"] == "mock_model"


class TestPromptAffixes:
    """Tests for PromptAffixes dataclass."""

    def test_creation(self):
        affixes = PromptAffixes(prefix="Hello ", suffix=" World")
        assert affixes.prefix == "Hello "
        assert affixes.suffix == " World"

    def test_empty(self):
        affixes = PromptAffixes(prefix="", suffix="")
        assert affixes.prefix == ""
        assert affixes.suffix == ""


class TestLlmMetadataDataclass:
    """Tests for the LlmMetadata dataclass."""

    def test_default_values(self):
        meta = LlmMetadata()
        assert meta.start_token is None
        assert meta.stop_tokens == []
        assert meta.prompt_templates is None
        assert meta.sampler_params is None
        assert meta.max_num_tokens == 4096
        assert meta.llm_model_type == "GenericModel"
        assert meta.system_metadata == {}

    def test_custom_values(self):
        meta = LlmMetadata(
            start_token=1,
            stop_tokens=[2, 3],
            max_num_tokens=8192,
            llm_model_type="Gemma3",
            system_metadata={"Authors": "qarnux"},
        )
        assert meta.start_token == 1
        assert meta.stop_tokens == [2, 3]
        assert meta.max_num_tokens == 8192
        assert meta.llm_model_type == "Gemma3"
        assert meta.system_metadata["Authors"] == "qarnux"


class TestSamplerParameters:
    """Tests for the SamplerParameters dataclass."""

    def test_default_values(self):
        params = SamplerParameters()
        assert params.type == "TOP_K"
        assert params.k == 40
        assert params.p == 0.95
        assert params.temperature == 0.8
        assert params.seed is None
        assert params.backend == "CPU"

    def test_custom_values(self):
        params = SamplerParameters(
            type="TOP_P",
            k=0,
            p=0.9,
            temperature=0.7,
            backend="GPU",
        )
        assert params.type == "TOP_P"
        assert params.backend == "GPU"
