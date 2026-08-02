"""
Tests for the tensor mapping and conversion utilities.

Validates the GGUF-to-PyTorch tensor name mapping for all supported
architectures and tests the quantization configuration system.
"""

import pytest
import json
from pathlib import Path

from litert_torch_qarnux.utils.tensor_mapping import TensorMapper
from litert_torch_qarnux.configs.quantization import QuantizationConfig, QuantizationMethod, QuantizationDType


class TestTensorMapper:
    """Tests for GGUF-to-PyTorch tensor name mapping."""

    def test_llama_attention_mapping(self):
        """Test Llama attention tensor name mapping."""
        mapper = TensorMapper("llama")

        # Attention layer tensors
        assert mapper.map_tensor("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
        assert mapper.map_tensor("blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight"
        assert mapper.map_tensor("blk.0.attn_k.weight") == "model.layers.0.self_attn.k_proj.weight"
        assert mapper.map_tensor("blk.0.attn_v.weight") == "model.layers.0.self_attn.v_proj.weight"
        assert mapper.map_tensor("blk.0.attn_output.weight") == "model.layers.0.self_attn.o_proj.weight"

    def test_llama_mlp_mapping(self):
        """Test Llama MLP tensor name mapping."""
        mapper = TensorMapper("llama")

        # MLP layer tensors
        assert mapper.map_tensor("blk.0.ffn_norm.weight") == "model.layers.0.post_attention_layernorm.weight"
        assert mapper.map_tensor("blk.0.ffn_gate.weight") == "model.layers.0.mlp.gate_proj.weight"
        assert mapper.map_tensor("blk.0.ffn_up.weight") == "model.layers.0.mlp.up_proj.weight"
        assert mapper.map_tensor("blk.0.ffn_down.weight") == "model.layers.0.mlp.down_proj.weight"

    def test_llama_model_level_mapping(self):
        """Test Llama model-level tensor name mapping."""
        mapper = TensorMapper("llama")

        assert mapper.map_tensor("token_embd.weight") == "model.embed_tokens.weight"
        assert mapper.map_tensor("output_norm.weight") == "model.norm.weight"
        assert mapper.map_tensor("output.weight") == "lm_head.weight"

    def test_llama_multi_layer(self):
        """Test mapping for different layer indices."""
        mapper = TensorMapper("llama")

        assert mapper.map_tensor("blk.5.attn_q.weight") == "model.layers.5.self_attn.q_proj.weight"
        assert mapper.map_tensor("blk.31.ffn_down.weight") == "model.layers.31.mlp.down_proj.weight"

    def test_gemma_mapping(self):
        """Test Gemma tensor name mapping."""
        mapper = TensorMapper("gemma")

        # Gemma has post-attention norms
        assert mapper.map_tensor("blk.0.attn_post_norm.weight") == "model.layers.0.post_attention_layernorm.weight"
        assert mapper.map_tensor("blk.0.ffn_post_norm.weight") == "model.layers.0.post_feedforward_layernorm.weight"

        # Gemma Q/K normalization
        assert mapper.map_tensor("blk.0.attn_q_norm.weight") == "model.layers.0.self_attn.q_norm.weight"
        assert mapper.map_tensor("blk.0.attn_k_norm.weight") == "model.layers.0.self_attn.k_norm.weight"

    def test_mistral_mapping(self):
        """Test Mistral tensor name mapping."""
        mapper = TensorMapper("mistral")

        assert mapper.map_tensor("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
        assert mapper.map_tensor("blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight"

    def test_qwen2_mapping(self):
        """Test Qwen2 tensor name mapping."""
        mapper = TensorMapper("qwen2")

        assert mapper.map_tensor("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
        assert mapper.map_tensor("blk.0.ffn_gate.weight") == "model.layers.0.mlp.gate_proj.weight"

    def test_unknown_tensor(self):
        """Test that unknown tensors return None."""
        mapper = TensorMapper("llama")
        assert mapper.map_tensor("some.unknown.tensor") is None

    def test_map_all_tensors(self):
        """Test batch tensor mapping."""
        mapper = TensorMapper("llama")
        names = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "unknown.tensor",
            "output.weight",
        ]
        result = mapper.map_all_tensors(names)

        assert "token_embd.weight" in result
        assert "blk.0.attn_q.weight" in result
        assert "output.weight" in result
        assert "unknown.tensor" not in result

    def test_llama_bias_mapping(self):
        """Test Llama bias tensor mapping."""
        mapper = TensorMapper("llama")

        assert mapper.map_tensor("blk.0.attn_q.bias") == "model.layers.0.self_attn.q_proj.bias"
        assert mapper.map_tensor("blk.0.attn_output.bias") == "model.layers.0.self_attn.o_proj.bias"

    def test_phi2_qkv_mapping(self):
        """Test Phi-2 combined QKV projection mapping."""
        mapper = TensorMapper("phi2")

        assert mapper.map_tensor("blk.0.attn_qkv.weight") == "model.layers.0.self_attn.qkv_proj.weight"


class TestQuantizationConfig:
    """Tests for the quantization configuration system."""

    def test_default_config(self):
        """Test default QuantizationConfig values."""
        config = QuantizationConfig()
        assert config.method == QuantizationMethod.WEIGHT_ONLY
        assert config.weight_dtype == "int8"
        assert config.activation_dtype == "fp32"
        assert config.per_channel is True

    def test_from_recipe_wi8(self):
        """Test loading WI8 recipe."""
        config = QuantizationConfig.from_recipe("dynamic_wi8_afp32")
        assert config.method == QuantizationMethod.WEIGHT_ONLY
        assert config.weight_dtype == "int8"
        assert config.activation_dtype == "fp32"

    def test_from_recipe_wi4(self):
        """Test loading WI4 recipe."""
        config = QuantizationConfig.from_recipe("dynamic_wi4_afp32")
        assert config.weight_dtype == "int4"

    def test_from_recipe_full_int8(self):
        """Test loading full INT8 recipe."""
        config = QuantizationConfig.from_recipe("full_int8")
        assert config.method == QuantizationMethod.FULL
        assert config.activation_dtype == "int8"

    def test_from_recipe_none(self):
        """Test loading no-quantization recipe."""
        config = QuantizationConfig.from_recipe("none")
        assert config.method == QuantizationMethod.NONE

    def test_from_recipe_invalid(self):
        """Test that invalid recipe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown quantization recipe"):
            QuantizationConfig.from_recipe("nonexistent_recipe")

    def test_from_file(self, tmp_path):
        """Test loading recipe from JSON file."""
        recipe = {
            "method": QuantizationMethod.WEIGHT_ONLY,
            "weight_dtype": QuantizationDType.INT8,
            "activation_dtype": "fp32",
            "per_channel": True,
            "group_size": 64,
        }

        recipe_file = tmp_path / "recipe.json"
        with open(recipe_file, "w") as f:
            json.dump(recipe, f)

        config = QuantizationConfig.from_file(str(recipe_file))
        assert config.method == QuantizationMethod.WEIGHT_ONLY
        assert config.group_size == 64

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = QuantizationConfig()
        d = config.to_dict()
        assert d["method"] == "weight_only"
        assert d["weight_dtype"] == "int8"
        assert d["activation_dtype"] == "fp32"

    def test_list_recipes(self):
        """Test listing available recipes."""
        recipes = QuantizationConfig.list_recipes()
        assert "dynamic_wi8_afp32" in recipes
        assert "dynamic_wi4_afp32" in recipes
        assert "full_int8" in recipes
        assert "none" in recipes
