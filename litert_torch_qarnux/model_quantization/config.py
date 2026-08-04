"""
Quantization Configuration.

Defines configuration dataclasses for the model quantization pipeline,
including quantization profiles, behavior categories, and template configs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QuantDType(str, Enum):
    """Supported quantization data types."""
    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class QuantMethod(str, Enum):
    """Quantization methods."""
    POST_TRAINING_STATIC = "post_training_static"
    POST_TRAINING_DYNAMIC = "post_training_dynamic"
    QUANTIZATION_AWARE_TRAINING = "quantization_aware_training"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF_Q4_0 = "gguf_q4_0"
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q5_K_M = "gguf_q5_k_m"
    GGUF_Q6_K = "gguf_q6_k"
    GGUF_Q8_0 = "gguf_q8_0"
    GGUF_F16 = "gguf_f16"
    ONNX_QUANTIZE_DYNAMIC = "onnx_quantize_dynamic"
    ONNX_QUANTIZE_STATIC = "onnx_quantize_static"


class SourceFormat(str, Enum):
    """Supported source model formats."""
    GGUF = "gguf"
    ONNX = "onnx"
    PYTORCH = "pytorch"          # .pt / .pth
    SAFETENSORS = "safetensors"
    HUGGINGFACE = "huggingface"  # transformers format
    BINARY = "binary"            # .bin (legacy)
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Behavior Categories
# ---------------------------------------------------------------------------

# Default categories that users can assign emphasis to.
# Each category has a default emphasis of 0.0 (neutral).
# Positive values increase the behavior; negative values suppress it.
DEFAULT_CATEGORIES: Dict[str, str] = {
    "coding": "Code generation, debugging, and programming assistance quality.",
    "reasoning": "Logical reasoning, chain-of-thought, and analytical depth.",
    "drawing": "Visual / diagram / art description and generation capability.",
    "brainstorming": "Creative ideation, divergent thinking, and idea generation.",
    "hallucination": "Tendency to fabricate or confabulate information (suppress).",
    "fabricating": "Inventing facts, citations, or data (suppress).",
    "gaslighting": "Contradicting user or denying established facts (suppress).",
    "over_reacting": "Excessive emotional or verbose responses (suppress).",
    "examples": "Providing concrete examples and demonstrations.",
    "conciseness": "Brevity and to-the-point communication.",
    "creativity": "Original and novel thinking approaches.",
    "safety": "Refusal of harmful, dangerous, or unethical requests.",
    "helpfulness": "Overall willingness and eagerness to assist.",
    "honesty": "Admitting uncertainty and lack of knowledge.",
    "empathy": "Understanding and acknowledging user emotions.",
    "humor": "Appropriate use of wit and humor.",
    "detail": "Depth and thoroughness of explanations.",
    "formatting": "Quality of output formatting (markdown, code blocks, etc.).",
}


class BehaviorCategory:
    """
    Represents a single behavior category with an emphasis level.

    Emphasis levels range from -1.0 (strongly suppress) to +1.0 (strongly
    emphasize), with 0.0 as neutral/default.
    """

    def __init__(self, name: str, emphasis: float = 0.0, description: str = ""):
        self.name = name.lower().replace("-", "_")
        self._validate_emphasis(emphasis)
        self.emphasis = emphasis
        self.description = description

    @staticmethod
    def _validate_emphasis(value: float) -> None:
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"Emphasis must be between -1.0 and 1.0, got {value}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "emphasis": self.emphasis,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorCategory":
        return cls(
            name=data["name"],
            emphasis=data.get("emphasis", 0.0),
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        return f"BehaviorCategory(name={self.name!r}, emphasis={self.emphasis})"


# ---------------------------------------------------------------------------
# Behavior Profile
# ---------------------------------------------------------------------------

@dataclass
class BehaviorProfile:
    """
    A collection of behavior categories with emphasis levels that shape
    the quantized model's behavioral profile.
    """

    categories: Dict[str, BehaviorCategory] = field(default_factory=dict)

    def __post_init__(self):
        if not self.categories:
            self._init_defaults()

    def _init_defaults(self) -> None:
        for name, desc in DEFAULT_CATEGORIES.items():
            self.categories[name] = BehaviorCategory(name=name, description=desc)

    def set_emphasis(self, name: str, emphasis: float) -> "BehaviorProfile":
        name = name.lower().replace("-", "_")
        if name not in self.categories:
            self.categories[name] = BehaviorCategory(
                name=name, emphasis=emphasis,
                description=f"User-defined category: {name}",
            )
        else:
            self.categories[name].emphasis = emphasis
        return self

    def get_emphasis(self, name: str) -> float:
        return self.categories.get(name.lower().replace("-", "_"), BehaviorCategory(name)).emphasis

    def add_category(self, name: str, description: str = "", emphasis: float = 0.0) -> "BehaviorProfile":
        name = name.lower().replace("-", "_")
        if name in self.categories:
            raise ValueError(f"Category '{name}' already exists")
        self.categories[name] = BehaviorCategory(
            name=name, emphasis=emphasis, description=description,
        )
        return self

    def remove_category(self, name: str) -> "BehaviorProfile":
        name = name.lower().replace("-", "_")
        self.categories.pop(name, None)
        return self

    def summary(self) -> Dict[str, float]:
        """Return a sorted dict of category -> emphasis."""
        return {
            k: v.emphasis
            for k, v in sorted(self.categories.items(), key=lambda x: x[1].emphasis, reverse=True)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categories": {k: v.to_dict() for k, v in self.categories.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorProfile":
        bp = cls(categories={})
        for k, v in data.get("categories", {}).items():
            bp.categories[k] = BehaviorCategory.from_dict(v)
        if not bp.categories:
            bp._init_defaults()
        return bp


# ---------------------------------------------------------------------------
# Template Config
# ---------------------------------------------------------------------------

@dataclass
class TemplateConfig:
    """
    Configuration for embedding chat templates or skill.md files into
    the quantized model.
    """

    chat_template: Optional[str] = None       # Jinja2-style chat template string
    system_prompt: Optional[str] = None        # System prompt text
    skill_md_path: Optional[str] = None        # Path to a skill.md file
    skill_md_content: Optional[str] = None     # Raw skill.md content (alternative)
    personality: Optional[str] = None          # Personality description
    role: Optional[str] = None                 # Model role (e.g., "assistant", "code-reviewer")

    def load_skill_md(self, path: str) -> "TemplateConfig":
        """Load a skill.md file and store its content."""
        skill_path = Path(path)
        if not skill_path.exists():
            raise FileNotFoundError(f"skill.md not found: {path}")
        self.skill_md_path = str(skill_path)
        self.skill_md_content = skill_path.read_text(encoding="utf-8")
        logger.info("Loaded skill.md from %s (%d bytes)", path, len(self.skill_md_content))
        return self

    def set_skill_md_content(self, content: str) -> "TemplateConfig":
        """Set skill.md content directly."""
        self.skill_md_content = content
        return self

    def set_chat_template(self, template: str) -> "TemplateConfig":
        """Set a chat template string."""
        self.chat_template = template
        return self

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.chat_template:
            result["chat_template"] = self.chat_template
        if self.system_prompt:
            result["system_prompt"] = self.system_prompt
        if self.skill_md_path:
            result["skill_md_path"] = self.skill_md_path
        if self.skill_md_content:
            result["skill_md_content"] = self.skill_md_content
        if self.personality:
            result["personality"] = self.personality
        if self.role:
            result["role"] = self.role
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateConfig":
        tc = cls()
        tc.chat_template = data.get("chat_template")
        tc.system_prompt = data.get("system_prompt")
        tc.skill_md_path = data.get("skill_md_path")
        tc.skill_md_content = data.get("skill_md_content")
        tc.personality = data.get("personality")
        tc.role = data.get("role")
        return tc


# ---------------------------------------------------------------------------
# Quantization Profile (top-level config)
# ---------------------------------------------------------------------------

@dataclass
class QuantizationProfile:
    """
    Complete quantization profile combining format, method, behavior,
    and template settings.
    """

    # Source model
    input_path: str = ""
    source_format: SourceFormat = SourceFormat.AUTO

    # Quantization settings
    method: QuantMethod = QuantMethod.POST_TRAINING_STATIC
    output_dtype: QuantDType = QuantDType.INT8
    group_size: int = 32
    per_channel: bool = True
    symmetric: bool = True

    # Output
    output_path: str = ""
    output_format: SourceFormat = SourceFormat.GGUF

    # Behavior
    behavior: BehaviorProfile = field(default_factory=BehaviorProfile)

    # Templates
    template: TemplateConfig = field(default_factory=TemplateConfig)

    # Misc
    calib_data_path: Optional[str] = None
    calib_samples: int = 128
    max_seq_len: int = 2048
    metadata: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "source_format": self.source_format.value,
            "method": self.method.value,
            "output_dtype": self.output_dtype.value,
            "group_size": self.group_size,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
            "output_path": self.output_path,
            "output_format": self.output_format.value,
            "behavior": self.behavior.to_dict(),
            "template": self.template.to_dict(),
            "calib_data_path": self.calib_data_path,
            "calib_samples": self.calib_samples,
            "max_seq_len": self.max_seq_len,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationProfile":
        profile = cls()
        profile.input_path = data.get("input_path", "")
        profile.source_format = SourceFormat(data.get("source_format", "auto"))
        profile.method = QuantMethod(data.get("method", "post_training_static"))
        profile.output_dtype = QuantDType(data.get("output_dtype", "int8"))
        profile.group_size = data.get("group_size", 32)
        profile.per_channel = data.get("per_channel", True)
        profile.symmetric = data.get("symmetric", True)
        profile.output_path = data.get("output_path", "")
        profile.output_format = SourceFormat(data.get("output_format", "gguf"))
        profile.behavior = BehaviorProfile.from_dict(data.get("behavior", {}))
        profile.template = TemplateConfig.from_dict(data.get("template", {}))
        profile.calib_data_path = data.get("calib_data_path")
        profile.calib_samples = data.get("calib_samples", 128)
        profile.max_seq_len = data.get("max_seq_len", 2048)
        profile.metadata = data.get("metadata", {})
        return profile

    def save_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved profile to %s", path)

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved profile to %s", path)

    @classmethod
    def from_yaml(cls, path: str) -> "QuantizationProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "QuantizationProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str) -> "QuantizationProfile":
        ext = Path(path).suffix.lower()
        if ext in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif ext == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")
