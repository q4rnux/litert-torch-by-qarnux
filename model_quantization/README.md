# Model Quantization Module

A production-quality Python module for quantizing AI model files from various formats into optimized quantized versions. Supports embedding chat templates, skill.md files, and categorized behavior emphasis fields.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Formats](#supported-formats)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
  - [Quantization Profiles](#quantization-profiles)
  - [Behavior Categories](#behavior-categories)
  - [Template Embedding](#template-embedding)
- [Python API](#python-api)
- [Examples](#examples)
- [Architecture](#architecture)
- [Extensibility](#extensibility)

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-format Support** | Quantize models from GGUF, ONNX, PyTorch (.pt/.pth), SafeTensors, and HuggingFace formats |
| **Multiple Quantization Methods** | Post-training static/dynamic, GPTQ, AWQ, GGUF-native quantization |
| **Behavior Profiling** | 18+ categorized emphasis fields to shape model behavior during quantization |
| **Template Embedding** | Embed chat templates (Jinja2), skill.md files, system prompts, and personality descriptions directly into quantized models |
| **YAML/JSON Config** | Full configuration via YAML or JSON files, or CLI arguments |
| **Orchestrator Integration** | Plug-and-play agent for the multi-agent orchestration pipeline |
| **Sidecar Metadata** | JSON sidecar files for formats that don't support embedded metadata |

---

## Installation

This module is part of the `litert-torch-qarnux` package. Install the full dependencies:

```bash
# From the repository root
pip install -e ".[full]"

# Or install optional dependencies for specific formats
pip install gguf onnx onnxruntime safetensors torch
```

---

## Quick Start

```bash
# Quantize a GGUF model to INT8 with default settings
python -m litert_torch_qarnux.model_quantization quantize \
    --input models/llama-7b.gguf \
    --dtype int8 \
    --output models/llama-7b-quantized.gguf

# Quantize with a config file
python -m litert_torch_qarnux.model_quantization quantize \
    --config examples/quantization_examples/sample_config.yaml

# Generate a sample config
python -m litert_torch_qarnux.model_quantization init-config \
    --output my_config.yaml --format yaml
```

---

## Supported Formats

| Input Format | Extensions | Handler | Quantization Support |
|--------------|------------|---------|---------------------|
| GGUF | `.gguf` | `GGUFFormatHandler` | Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16 |
| ONNX | `.onnx` | `ONNXFormatHandler` | Dynamic INT8, Dynamic UINT8 |
| PyTorch | `.pt`, `.pth`, `.ckpt` | `PyTorchFormatHandler` | Weight-only INT8, FP16, BF16 |
| SafeTensors | `.safetensors` | `SafeTensorsFormatHandler` | Per-channel INT8, FP16 |
| HuggingFace | Directory | Auto-detected | Via PyTorch handler |

---

## CLI Reference

### `quantize`

Quantize a model file.

```
litert-quantize quantize [OPTIONS]

Options:
  -i, --input PATH          Path to the input model file (required)
  -o, --output PATH          Path for the quantized output file
  -m, --method METHOD        Quantization method (default: post_training_static)
  -d, --dtype DTYPE          Target data type: int4, int8, fp16, bf16, fp32
  -f, --format FORMAT        Source format: auto, gguf, onnx, pytorch, safetensors
  --output-format FORMAT     Output format (default: gguf)
  --group-size INT           Quantization group size (default: 32)
  -c, --config PATH          YAML/JSON configuration file
  --skill-md PATH            Path to skill.md file to embed
  --chat-template NAME       Built-in chat template (default, chatml, llama2, llama3, mistral, qwen)
  --system-prompt TEXT       System prompt to embed
  --personality TEXT         Personality description
  --role TEXT                Model role (e.g., assistant, code-reviewer)
  --calib-data PATH          Calibration data for static quantization
  -v, --verbose              Enable debug logging
  --json-output              Output results as JSON
```

### `behavior`

Manage behavior categories.

```
litert-quantize behavior list          # List all categories
litert-quantize behavior set NAME VALUE -o OUTPUT  # Set emphasis
litert-quantize behavior profile --preset NAME -o OUTPUT  # Create preset profile
```

Available presets: `safe_coder`, `creative_writer`, `research_analyst`, `friendly_assistant`

### `template`

Manage chat templates and skill.md files.

```
litert-quantize template list          # List built-in templates
litert-quantize template show NAME --type chat|skill  # Show template content
```

### `info`

Display model file information.

```
litert-quantize info -i models/llama-7b.gguf
```

### `init-config`

Generate a sample configuration file.

```
litert-quantize init-config -o config.yaml --format yaml|json
```

---

## Configuration

### Quantization Profiles

A complete profile combines source/target format, quantization method, behavior, and template settings.

**YAML format:**
```yaml
input_path: "model.gguf"
source_format: "auto"
method: "post_training_static"
output_dtype: "int8"
group_size: 32
per_channel: true
output_path: "model-quantized.gguf"
output_format: "gguf"

behavior:
  categories:
    coding: {name: "coding", emphasis: 0.5}
    reasoning: {name: "reasoning", emphasis: 0.7}
    hallucination: {name: "hallucination", emphasis: -0.8}

template:
  chat_template: "{% for message in messages %}..."
  system_prompt: "You are a helpful assistant."
  role: "assistant"
```

**JSON format:**
```json
{
  "input_path": "model.gguf",
  "method": "post_training_static",
  "output_dtype": "int8",
  "behavior": {
    "categories": {
      "coding": {"name": "coding", "emphasis": 0.5}
    }
  },
  "template": {
    "system_prompt": "You are a helpful assistant."
  }
}
```

### Behavior Categories

Behavior categories allow you to shape the quantized model's tendencies. Each category has an **emphasis level** from `-1.0` (strongly suppress) to `+1.0` (strongly emphasize).

| Category | Description | Typical Use |
|----------|-------------|-------------|
| `coding` | Code generation and debugging quality | Set high for coding assistants |
| `reasoning` | Logical reasoning and chain-of-thought | Set high for analytical models |
| `drawing` | Visual/diagram description capability | Set for art-focused models |
| `brainstorming` | Creative ideation | Set high for creative tasks |
| `hallucination` | Tendency to fabricate info | **Always set negative** |
| `fabricating` | Inventing facts/citations | **Always set negative** |
| `gaslighting` | Contradicting user about facts | **Always set negative** |
| `over_reacting` | Excessive emotional responses | Set negative for professional tone |
| `examples` | Providing concrete examples | Set high for educational models |
| `conciseness` | Brevity and directness | Set based on use case |
| `creativity` | Original thinking | Set high for creative tasks |
| `safety` | Refusal of harmful requests | Set high for production models |
| `helpfulness` | Willingness to assist | Generally set positive |
| `honesty` | Admitting uncertainty | Set high for factual models |
| `empathy` | Understanding emotions | Set for chat/counseling models |
| `humor` | Appropriate wit | Set based on tone preference |
| `detail` | Depth of explanations | Set based on audience |
| `formatting` | Output formatting quality | Set high for structured output |

**Custom categories** can be added:
```python
from litert_torch_qarnux.model_quantization.config import BehaviorProfile

profile = BehaviorProfile()
profile.add_category(
    name="mathematical_rigor",
    description="Precision in mathematical reasoning",
    emphasis=0.8,
)
```

### Template Embedding

The module supports embedding four types of content into quantized models:

1. **Chat Templates** (Jinja2 format) - Define conversation turn formatting
2. **System Prompts** - Set the default system-level instructions
3. **Skill.md Files** - Embed detailed capability definitions
4. **Personality/Role** - Set behavioral identity

Built-in chat templates: `default`, `chatml`, `llama2`, `llama3`, `mistral`, `qwen`

---

## Python API

```python
from litert_torch_qarnux.model_quantization.config import (
    QuantizationProfile, BehaviorProfile, TemplateConfig,
)
from litert_torch_qarnux.model_quantization.quantizer import ModelQuantizer

# 1. Create a profile
profile = QuantizationProfile(
    input_path="model.gguf",
    output_dtype="int8",
    method="post_training_static",
)

# 2. Configure behavior
profile.behavior.set_emphasis("coding", 0.7)
profile.behavior.set_emphasis("hallucination", -0.9)

# 3. Configure template
profile.template.system_prompt = "You are a helpful coding assistant."
profile.template.chat_template = "<built-in or custom Jinja2 template>"
profile.template.load_skill_md("path/to/skill.md")

# 4. Run quantization
quantizer = ModelQuantizer(profile)
result = quantizer.quantize()

print(f"Output: {result['output_path']}")
print(f"Size: {result['file_size_bytes'] / (1024*1024):.1f} MB")
```

**Loading from file:**
```python
profile = QuantizationProfile.from_yaml("config.yaml")
profile = QuantizationProfile.from_json("config.json")
```

**Using with Orchestrator:**
```python
from litert_torch_qarnux.model_quantization.agent import QuantizationAgent
from litert_torch_qarnux.orchestrator.base_agent import AgentMessage

agent = QuantizationAgent(profile)
message = AgentMessage(
    source="conversion_agent",
    target="quantization_agent",
    data={"model_path": "model.gguf"},
)
result = agent.start(message)
```

---

## Examples

### Example 1: Simple INT8 Quantization
```bash
python -m litert_torch_qarnux.model_quantization quantize \
    -i models/mistral-7b.gguf \
    -d int8 \
    -o models/mistral-7b-int8.gguf
```

### Example 2: Quantize with Behavior Profile
```bash
# Create a behavior config
python -m litert_torch_qarnux.model_quantization behavior set coding 0.8 -o behavior.yaml

# Quantize with it
python -m litert_torch_qarnux.model_quantization quantize \
    -i model.gguf -d int8 \
    -c examples/quantization_examples/sample_config.yaml
```

### Example 3: Quantize with Embedded Templates
```bash
python -m litert_torch_qarnux.model_quantization quantize \
    -i model.gguf \
    -d int8 \
    --chat-template llama3 \
    --system-prompt "You are a research assistant." \
    --role assistant \
    --skill-md examples/quantization_examples/skill_code_reviewer.md
```

### Example 4: Convert ONNX to GGUF
```bash
python -m litert_torch_qarnux.model_quantization quantize \
    -i model.onnx \
    -f onnx \
    --output-format gguf \
    -d int8
```

---

## Architecture

```
model_quantization/
├── __init__.py          # Package exports
├── __main__.py          # Module entry point
├── cli.py               # CLI interface (argparse)
├── config.py            # Configuration dataclasses
├── quantizer.py         # Main quantization orchestrator
├── agent.py             # Orchestrator pipeline agent
├── formatters/
│   ├── __init__.py
│   ├── base.py          # Abstract format handler
│   ├── registry.py      # Format auto-detection
│   ├── gguf_handler.py  # GGUF format handler
│   ├── onnx_handler.py  # ONNX format handler
│   ├── pytorch_handler.py # PyTorch format handler
│   └── safetensors_handler.py # SafeTensors handler
├── agents/
│   └── __init__.py
└── templates/
    └── __init__.py      # Built-in templates
```

---

## Extensibility

### Adding a New Format Handler

1. Create a new handler class inheriting from `BaseFormatHandler`
2. Implement the required abstract methods
3. Register it in `FormatRegistry`

```python
from litert_torch_qarnux.model_quantization.formatters.base import BaseFormatHandler
from litert_torch_qarnux.model_quantization.formatters.registry import FormatRegistry

class MyFormatHandler(BaseFormatHandler):
    FORMAT_NAME = "MyFormat"
    SUPPORTED_EXTENSIONS = (".myext",)

    def detect_format(self, path):
        ...

    def load_metadata(self, path):
        ...

    def quantize(self, path):
        ...

    def embed_template(self, quantized_path):
        ...

    def write_behavior_metadata(self, quantized_path):
        ...

FormatRegistry.register(SourceFormat("myformat"), MyFormatHandler)
```

### Adding Custom Behavior Categories

```python
profile = QuantizationProfile()
profile.behavior.add_category(
    name="scientific_accuracy",
    description="Precision in scientific claims",
    emphasis=0.9,
)
```

### Adding Custom Chat Templates

```python
from litert_torch_qarnux.model_quantization.config import TemplateConfig

tc = TemplateConfig()
tc.set_chat_template("{{ bos_token }}{% for msg in messages %}...{% endfor %}")
```

---

## Logging

The module uses Python's standard `logging` module. All log messages are prefixed with `[FORMAT]` for easy identification. Set the log level via `--verbose` or by configuring the root logger.

---

## Testing

```bash
# Run tests
pytest tests/test_quantization.py -v

# Run with coverage
pytest tests/test_quantization.py --cov=model_quantization
```

---

## License

Apache-2.0. See the parent project LICENSE file for details.
