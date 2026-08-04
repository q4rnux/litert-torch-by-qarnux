"""
Tests for the model quantization module.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from litert_torch_qarnux.model_quantization.config import (
    BehaviorCategory,
    BehaviorProfile,
    QuantDType,
    QuantMethod,
    QuantizationProfile,
    SourceFormat,
    TemplateConfig,
    DEFAULT_CATEGORIES,
)
from litert_torch_qarnux.model_quantization.formatters.registry import FormatRegistry


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestBehaviorCategory:
    """Tests for BehaviorCategory."""

    def test_default_emphasis(self):
        cat = BehaviorCategory(name="coding")
        assert cat.name == "coding"
        assert cat.emphasis == 0.0

    def test_custom_emphasis(self):
        cat = BehaviorCategory(name="reasoning", emphasis=0.7)
        assert cat.emphasis == 0.7

    def test_negative_emphasis(self):
        cat = BehaviorCategory(name="hallucination", emphasis=-0.8)
        assert cat.emphasis == -0.8

    def test_boundary_values(self):
        cat_pos = BehaviorCategory(name="test", emphasis=1.0)
        assert cat_pos.emphasis == 1.0
        cat_neg = BehaviorCategory(name="test", emphasis=-1.0)
        assert cat_neg.emphasis == -1.0

    def test_invalid_emphasis_raises(self):
        with pytest.raises(ValueError, match="between -1.0 and 1.0"):
            BehaviorCategory(name="test", emphasis=1.5)
        with pytest.raises(ValueError, match="between -1.0 and 1.0"):
            BehaviorCategory(name="test", emphasis=-1.1)

    def test_to_dict(self):
        cat = BehaviorCategory(name="coding", emphasis=0.5, description="Code quality")
        d = cat.to_dict()
        assert d == {"name": "coding", "emphasis": 0.5, "description": "Code quality"}

    def test_from_dict(self):
        cat = BehaviorCategory.from_dict({
            "name": "reasoning",
            "emphasis": 0.7,
            "description": "Logical depth",
        })
        assert cat.name == "reasoning"
        assert cat.emphasis == 0.7

    def test_name_normalization(self):
        cat = BehaviorCategory(name="Over-Reacting")
        assert cat.name == "over_reacting"

    def test_repr(self):
        cat = BehaviorCategory(name="coding", emphasis=0.5)
        r = repr(cat)
        assert "coding" in r
        assert "0.5" in r


class TestBehaviorProfile:
    """Tests for BehaviorProfile."""

    def test_default_categories(self):
        bp = BehaviorProfile()
        assert len(bp.categories) == len(DEFAULT_CATEGORIES)
        assert "coding" in bp.categories
        assert "hallucination" in bp.categories

    def test_set_emphasis(self):
        bp = BehaviorProfile()
        bp.set_emphasis("coding", 0.8)
        assert bp.get_emphasis("coding") == 0.8

    def test_set_emphasis_custom_category(self):
        bp = BehaviorProfile()
        bp.set_emphasis("my_custom_cat", 0.5)
        assert "my_custom_cat" in bp.categories
        assert bp.get_emphasis("my_custom_cat") == 0.5

    def test_add_category(self):
        bp = BehaviorProfile()
        bp.add_category("new_cat", "A new category", emphasis=0.3)
        assert "new_cat" in bp.categories

    def test_add_duplicate_category_raises(self):
        bp = BehaviorProfile()
        # "coding" already exists in default categories
        with pytest.raises(ValueError, match="already exists"):
            bp.add_category("coding", "Duplicate")

    def test_remove_category(self):
        bp = BehaviorProfile()
        bp.remove_category("humor")
        assert "humor" not in bp.categories

    def test_remove_nonexistent_no_error(self):
        bp = BehaviorProfile()
        bp.remove_category("nonexistent")  # Should not raise

    def test_summary(self):
        bp = BehaviorProfile()
        bp.set_emphasis("coding", 0.8)
        bp.set_emphasis("hallucination", -0.9)
        summary = bp.summary()
        # coding should be before hallucination (higher emphasis first)
        keys = list(summary.keys())
        assert keys.index("coding") < keys.index("hallucination")

    def test_to_dict_and_from_dict_roundtrip(self):
        bp = BehaviorProfile()
        bp.set_emphasis("coding", 0.5)
        bp.set_emphasis("hallucination", -0.8)
        d = bp.to_dict()
        bp2 = BehaviorProfile.from_dict(d)
        assert bp2.get_emphasis("coding") == 0.5
        assert bp2.get_emphasis("hallucination") == -0.8


class TestTemplateConfig:
    """Tests for TemplateConfig."""

    def test_default_empty(self):
        tc = TemplateConfig()
        assert tc.chat_template is None
        assert tc.system_prompt is None
        assert tc.skill_md_content is None

    def test_set_chat_template(self):
        tc = TemplateConfig()
        tc.set_chat_template("{{ user }}{{ assistant }}")
        assert tc.chat_template == "{{ user }}{{ assistant }}"

    def test_load_skill_md(self, tmp_path):
        skill_file = tmp_path / "skill.md"
        skill_file.write_text("# My Skill\n\nThis is a test skill.")
        tc = TemplateConfig()
        tc.load_skill_md(str(skill_file))
        assert tc.skill_md_content == "# My Skill\n\nThis is a test skill."

    def test_load_skill_md_not_found(self, tmp_path):
        tc = TemplateConfig()
        with pytest.raises(FileNotFoundError):
            tc.load_skill_md(str(tmp_path / "nonexistent.md"))

    def test_set_skill_md_content(self):
        tc = TemplateConfig()
        tc.set_skill_md_content("# Direct Content")
        assert tc.skill_md_content == "# Direct Content"

    def test_to_dict(self):
        tc = TemplateConfig()
        tc.chat_template = "template"
        tc.system_prompt = "You are helpful"
        tc.role = "assistant"
        d = tc.to_dict()
        assert d["chat_template"] == "template"
        assert d["system_prompt"] == "You are helpful"
        assert d["role"] == "assistant"

    def test_from_dict(self):
        tc = TemplateConfig.from_dict({
            "chat_template": "tmpl",
            "personality": "friendly",
        })
        assert tc.chat_template == "tmpl"
        assert tc.personality == "friendly"

    def test_to_dict_excludes_none(self):
        tc = TemplateConfig()
        tc.chat_template = "test"
        d = tc.to_dict()
        assert "system_prompt" not in d
        assert "role" not in d


class TestQuantizationProfile:
    """Tests for QuantizationProfile."""

    def test_default_values(self):
        profile = QuantizationProfile()
        assert profile.source_format == SourceFormat.AUTO
        assert profile.method == QuantMethod.POST_TRAINING_STATIC
        assert profile.output_dtype == QuantDType.INT8
        assert profile.group_size == 32
        assert profile.per_channel is True

    def test_to_dict(self):
        profile = QuantizationProfile(
            input_path="model.gguf",
            output_dtype=QuantDType.INT8,
        )
        d = profile.to_dict()
        assert d["input_path"] == "model.gguf"
        assert d["output_dtype"] == "int8"
        assert "behavior" in d
        assert "template" in d

    def test_from_dict_roundtrip(self):
        profile = QuantizationProfile(
            input_path="test.gguf",
            output_dtype=QuantDType.FP16,
            method=QuantMethod.GPTQ,
        )
        d = profile.to_dict()
        profile2 = QuantizationProfile.from_dict(d)
        assert profile2.input_path == "test.gguf"
        assert profile2.output_dtype == QuantDType.FP16
        assert profile2.method == QuantMethod.GPTQ

    def test_save_and_load_yaml(self, tmp_path):
        profile = QuantizationProfile(input_path="model.gguf")
        profile.behavior.set_emphasis("coding", 0.6)
        profile_path = str(tmp_path / "config.yaml")
        profile.save_yaml(profile_path)
        assert Path(profile_path).exists()

        loaded = QuantizationProfile.from_yaml(profile_path)
        assert loaded.input_path == "model.gguf"
        assert loaded.behavior.get_emphasis("coding") == 0.6

    def test_save_and_load_json(self, tmp_path):
        profile = QuantizationProfile(input_path="model.gguf")
        profile.behavior.set_emphasis("hallucination", -0.9)
        profile_path = str(tmp_path / "config.json")
        profile.save_json(profile_path)
        assert Path(profile_path).exists()

        loaded = QuantizationProfile.from_json(profile_path)
        assert loaded.input_path == "model.gguf"
        assert loaded.behavior.get_emphasis("hallucination") == -0.9

    def test_from_file_yaml(self, tmp_path):
        profile = QuantizationProfile(input_path="test.gguf")
        path = str(tmp_path / "config.yaml")
        profile.save_yaml(path)
        loaded = QuantizationProfile.from_file(path)
        assert loaded.input_path == "test.gguf"

    def test_from_file_json(self, tmp_path):
        profile = QuantizationProfile(input_path="test.gguf")
        path = str(tmp_path / "config.json")
        profile.save_json(path)
        loaded = QuantizationProfile.from_file(path)
        assert loaded.input_path == "test.gguf"

    def test_from_file_unsupported_extension(self, tmp_path):
        path = str(tmp_path / "config.toml")
        with pytest.raises(ValueError, match="Unsupported"):
            QuantizationProfile.from_file(path)


# ---------------------------------------------------------------------------
# Format Registry Tests
# ---------------------------------------------------------------------------

class TestFormatRegistry:
    """Tests for FormatRegistry."""

    def test_detect_gguf(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.touch()
        fmt = FormatRegistry.detect_format(f)
        assert fmt == SourceFormat.GGUF

    def test_detect_onnx(self, tmp_path):
        f = tmp_path / "model.onnx"
        f.touch()
        fmt = FormatRegistry.detect_format(f)
        assert fmt == SourceFormat.ONNX

    def test_detect_pytorch_pt(self, tmp_path):
        f = tmp_path / "model.pt"
        f.touch()
        fmt = FormatRegistry.detect_format(f)
        assert fmt == SourceFormat.PYTORCH

    def test_detect_pytorch_pth(self, tmp_path):
        f = tmp_path / "model.pth"
        f.touch()
        fmt = FormatRegistry.detect_format(f)
        assert fmt == SourceFormat.PYTORCH

    def test_detect_safetensors(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.touch()
        fmt = FormatRegistry.detect_format(f)
        assert fmt == SourceFormat.SAFETENSORS

    def test_detect_unknown(self, tmp_path):
        f = tmp_path / "model.xyz"
        f.touch()
        fmt = FormatRegistry.detect_format(f)
        assert fmt == SourceFormat.AUTO

    def test_supported_formats(self):
        formats = FormatRegistry.supported_formats()
        assert "gguf" in formats
        assert "onnx" in formats
        assert "pytorch" in formats
        assert "safetensors" in formats

    def test_supported_extensions(self):
        exts = FormatRegistry.supported_extensions()
        assert ".gguf" in exts
        assert ".onnx" in exts
        assert ".pt" in exts
        assert ".safetensors" in exts

    def test_get_handler_gguf(self):
        profile = QuantizationProfile()
        handler = FormatRegistry.get_handler(SourceFormat.GGUF, profile)
        assert handler is not None

    def test_get_handler_unknown_raises(self):
        profile = QuantizationProfile()
        with pytest.raises(ValueError, match="No handler"):
            FormatRegistry.get_handler(SourceFormat.AUTO, profile)


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for the CLI argument parser."""

    def test_create_parser(self):
        from litert_torch_qarnux.model_quantization.cli import create_parser
        parser = create_parser()
        assert parser is not None

    def test_parse_quantize_args(self):
        from litert_torch_qarnux.model_quantization.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            "quantize",
            "--input", "model.gguf",
            "--dtype", "int8",
            "--method", "post_training_dynamic",
        ])
        assert args.command == "quantize"
        assert args.input == "model.gguf"
        assert args.dtype == "int8"
        assert args.method == "post_training_dynamic"

    def test_parse_behavior_list(self):
        from litert_torch_qarnux.model_quantization.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["behavior", "list"])
        assert args.command == "behavior"
        assert args.behavior_command == "list"

    def test_parse_template_list(self):
        from litert_torch_qarnux.model_quantization.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["template", "list"])
        assert args.command == "template"
        assert args.template_command == "list"

    def test_parse_info(self):
        from litert_torch_qarnux.model_quantization.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["info", "--input", "model.gguf"])
        assert args.command == "info"
        assert args.input == "model.gguf"


# ---------------------------------------------------------------------------
# Templates Tests
# ---------------------------------------------------------------------------

class TestTemplates:
    """Tests for built-in templates."""

    def test_chat_templates_exist(self):
        from litert_torch_qarnux.model_quantization.templates import CHAT_TEMPLATES
        assert len(CHAT_TEMPLATES) > 0
        assert "default" in CHAT_TEMPLATES
        assert "llama3" in CHAT_TEMPLATES

    def test_skill_templates_exist(self):
        from litert_torch_qarnux.model_quantization.templates import SKILL_MD_TEMPLATES
        assert len(SKILL_MD_TEMPLATES) > 0
        assert "general_assistant" in SKILL_MD_TEMPLATES
        assert "code_assistant" in SKILL_MD_TEMPLATES

    def test_chat_templates_are_strings(self):
        from litert_torch_qarnux.model_quantization.templates import CHAT_TEMPLATES
        for name, tmpl in CHAT_TEMPLATES.items():
            assert isinstance(tmpl, str)
            assert len(tmpl) > 0

    def test_skill_templates_are_strings(self):
        from litert_torch_qarnux.model_quantization.templates import SKILL_MD_TEMPLATES
        for name, tmpl in SKILL_MD_TEMPLATES.items():
            assert isinstance(tmpl, str)
            assert len(tmpl) > 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the full pipeline (without actual model files)."""

    def test_profile_to_dict_with_behavior_and_template(self):
        profile = QuantizationProfile(
            input_path="test.gguf",
            output_dtype=QuantDType.INT8,
        )
        profile.behavior.set_emphasis("coding", 0.7)
        profile.behavior.set_emphasis("hallucination", -0.9)
        profile.template.system_prompt = "Test prompt"
        profile.template.chat_template = "Test template"

        d = profile.to_dict()
        assert d["behavior"]["categories"]["coding"]["emphasis"] == 0.7
        assert d["behavior"]["categories"]["hallucination"]["emphasis"] == -0.9
        assert d["template"]["system_prompt"] == "Test prompt"

    def test_full_roundtrip_yaml(self, tmp_path):
        """Test creating a profile, saving to YAML, loading, and verifying."""
        profile = QuantizationProfile(
            input_path="model.gguf",
            output_path="model-q.gguf",
            output_dtype=QuantDType.INT8,
        )
        profile.behavior.set_emphasis("coding", 0.5)
        profile.behavior.set_emphasis("reasoning", 0.7)
        profile.behavior.set_emphasis("hallucination", -0.8)
        profile.behavior.set_emphasis("fabricating", -0.9)
        profile.behavior.set_emphasis("gaslighting", -0.9)
        profile.behavior.set_emphasis("over_reacting", -0.5)
        profile.behavior.set_emphasis("helpfulness", 0.6)
        profile.behavior.set_emphasis("conciseness", 0.4)
        profile.template.system_prompt = "You are helpful."
        profile.template.chat_template = "chatml"

        path = str(tmp_path / "full_config.yaml")
        profile.save_yaml(path)

        loaded = QuantizationProfile.from_yaml(path)
        assert loaded.input_path == "model.gguf"
        assert loaded.behavior.get_emphasis("coding") == 0.5
        assert loaded.behavior.get_emphasis("hallucination") == -0.8
        assert loaded.template.system_prompt == "You are helpful."

    def test_behavior_summary_ordering(self):
        profile = BehaviorProfile()
        profile.set_emphasis("coding", 0.8)
        profile.set_emphasis("hallucination", -0.9)
        profile.set_emphasis("reasoning", 0.5)
        profile.set_emphasis("helpfulness", 0.3)

        summary = profile.summary()
        # Should be sorted by emphasis descending
        emphases = list(summary.values())
        assert emphases == sorted(emphases, reverse=True)

    def test_config_yaml_example_loads(self):
        """Test that the example YAML config file can be loaded."""
        example_path = Path(__file__).parent.parent / "examples" / "quantization_examples" / "sample_config.yaml"
        if example_path.exists():
            profile = QuantizationProfile.from_yaml(str(example_path))
            assert profile.input_path != ""
            assert len(profile.behavior.categories) > 0

    def test_config_json_example_loads(self):
        """Test that the example JSON config file can be loaded."""
        example_path = Path(__file__).parent.parent / "examples" / "quantization_examples" / "sample_config.json"
        if example_path.exists():
            profile = QuantizationProfile.from_json(str(example_path))
            assert profile.input_path != ""
            assert len(profile.behavior.categories) > 0
