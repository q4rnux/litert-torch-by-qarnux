# litert-torch-by-qarnux

A CLI tool that converts GGUF model files to Google's LiteRT-LM format (`.litertlm`) using a multi-agent orchestration architecture. This tool bridges the gap between the quantized GGUF format used by llama.cpp and Google's on-device inference framework LiteRT-LM.

**Version 1.0.0** — Production-ready with full LiteRT-LM runtime compatibility.

## Overview

The LiteRT-LM format (`.litertlm`) is a unified container that packages TFLite models, tokenizer files, external weights, and model metadata for on-device inference. This tool provides a complete pipeline that parses GGUF files, dequantizes weights, constructs PyTorch model representations, converts to TFLite, generates LLM metadata protos, and packages everything into a `.litertlm` container.

The conversion pipeline uses a **multi-agent orchestration architecture** where each specialized agent handles a distinct stage of the process, coordinated by an OrchestratorAgent.

## Key Features (v1.0.0)

- **Full LiteRT-LM Runtime Compatibility**: Generates valid `.litertlm` files accepted by any LiteRT-LM runtime app
- **LLM Metadata Proto Generation**: Automatic generation of `llm_metadata.pb` with start/stop tokens, prompt templates, sampler parameters, and model type mapping
- **Architecture-Specific Prompt Templates**: Built-in templates for Llama, Mistral, Gemma, Qwen, and Phi architectures
- **Dual Tokenizer Support**: Handles both SentencePiece (`.model`) and HuggingFace (`tokenizer.json`) tokenizer formats
- **Configurable Target Backend**: Support for CPU, GPU, and NPU targets
- **Comprehensive System Metadata**: Rich metadata including architecture, quantization info, and conversion tool details
- **UUID and Timestamp Traceability**: Auto-generated unique IDs for build traceability

## Supported Architectures

| GGUF Architecture | PyTorch Class | LlmModelType | Prompt Template |
|---|---|---|---|
| `llama` | LlamaForCausalLM | GenericModel | Llama-style |
| `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4` | GemmaForCausalLM | Gemma3/Gemma3N/Gemma4 | Gemma-style |
| `mistral` | MistralForCausalLM | GenericModel | Llama-style |
| `qwen2`, `qwen3` | Qwen2ForCausalLM | Qwen2p5/Qwen3 | Qwen-style |
| `phi`, `phi2`, `phi3` | PhiForCausalLM | GenericModel | Phi-style |
| `smollm3` | SmolLM3ForCausalLM | GenericModel | Generic |

## Installation

### From Source

```bash
git clone https://github.com/qarnux/litert-torch-by-qarnux.git
cd litert-torch-by-qarnux
pip install -e ".[dev,full]"
```

### Quick Install

```bash
pip install litert-torch-qarnux
```

### Prerequisites

The full conversion pipeline requires the following packages:

| Package | Purpose | Installation |
|---|---|---|
| `gguf` | GGUF file parsing | `pip install gguf` |
| `litert-lm-builder` | .litertlm container building | `pip install litert-lm-builder` |
| `sentencepiece` | Tokenizer conversion | `pip install sentencepiece` |
| `onnx2tf` | ONNX-to-TFLite conversion | `pip install onnx2tf tensorflow` |
| `numpy`, `tqdm`, `rich` | Utilities and progress bars | Installed automatically |

## Usage

### Command-Line Interface

The primary command converts a local GGUF file to `.litertlm` format:

```bash
litert-torch export_hf --model=Any_Local_GGUF --output_dir=./litert_output
```

### Command Reference

```
litert-torch export_hf \
    --model=PATH_TO_GGUF \
    --output_dir=DIRECTORY \
    [--quantize] \
    [--no-quantize] \
    [--quantization_recipe=RECIPE] \
    [--backend=BACKEND] \
    [--verbose]
```

| Argument | Description | Default |
|---|---|---|
| `--model` | Path to the local GGUF model file | Required |
| `--output_dir` | Output directory for .litertlm and artifacts | Required |
| `--quantize` | Enable quantization during conversion | Enabled |
| `--no-quantize` | Disable quantization | Disabled |
| `--quantization_recipe` | Quantization recipe name or path to JSON | `dynamic_wi8_afp32` |
| `--backend` | Target backend: `CPU`, `GPU`, or `NPU` | `CPU` |
| `--verbose` | Enable debug-level logging | Disabled |

### Examples

Convert a Llama model:

```bash
litert-torch export_hf \
    --model=TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
    --output_dir=./litert_output
```

Convert a Gemma3 model for GPU inference:

```bash
litert-torch export_hf \
    --model=gemma3.gguf \
    --output_dir=./litert_output \
    --backend=GPU
```

Convert with INT4 weight quantization:

```bash
litert-torch export_hf \
    --model=model.gguf \
    --output_dir=./output \
    --quantization_recipe=dynamic_wi4_afp32
```

Convert without quantization (full FP32):

```bash
litert-torch export_hf \
    --model=model.gguf \
    --output_dir=./output \
    --no-quantize
```

List supported architectures:

```bash
litert-torch list_architectures
```

Run as a Python module:

```bash
python -m litert_torch_qarnux export_hf \
    --model=model.gguf \
    --output_dir=./output
```

### Using Shell Scripts

Pre-built example scripts are available in the `examples/` directory:

```bash
./examples/convert_tinyllama.sh [path_to_gguf]
./examples/convert_gemma.sh [path_to_gguf]
```

## Metadata Generation

### LLM Metadata Proto

The MetadataAgent automatically generates the following fields in the `.litertlm` container:

| Field | Description | Source |
|---|---|---|
| `start_token` | BOS token prepended to input sequences | GGUF special tokens or default |
| `stop_tokens` | EOS tokens for output termination | GGUF special tokens |
| `prompt_templates` | Architecture-specific chat templates | Built-in mapping |
| `sampler_params` | Default sampling: temp=0.8, top_k=40, top_p=0.95 | Configuration |
| `max_num_tokens` | Maximum processable tokens | GGUF context_length |
| `llm_model_type` | Runtime-specific model type | Architecture mapping |

### System Metadata

| Key | Value |
|---|---|
| `Authors` | `qarnux` |
| `TargetBackend` | `CPU` / `GPU` / `NPU` (configurable) |
| `Architecture` | Detected architecture |
| `ModelName` | From GGUF `general.name` |
| `SourceFormat` | `GGUF` |
| `QuantizationType` | Applied quantization recipe |
| `ConversionTool` | `litert-torch-by-qarnux` |
| `ConversionToolVersion` | `1.0.0` |
| `SourceQuantization` | Original GGUF quantization type |

### Prompt Templates

Architecture-specific prompt templates are automatically applied:

**Llama/Mistral:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

**Gemma:**
```
<start_of_turn>user
{user}<end_of_turn>
<start_of_turn>model
```

**Qwen:**
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

**Phi:**
```
<|system|>
{system}<|end|>
<|user|>
{user}<|end|>
<|assistant|>
```

## Architecture

The conversion pipeline consists of seven specialized agents:

```
GGUF File
    │
    ▼
┌─────────────────┐
│  ParserAgent     │  Extract metadata, tensors, tokenizer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dequantization  │  Convert Q4/Q8/etc. to FP32
│    Agent        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ ModelAuthoring  │     │  TokenizerAgent  │
│    Agent        │     │  (parallel)      │
└────────┬────────┘     └────────┬────────┘
         │                        │
         │         ┌──────────────┘
         │         │
         ▼         ▼
┌─────────────────┐
│  MetadataAgent   │  Generate LLM metadata proto
│                  │  (prompt templates, tokens, sampler)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ConversionAgent  │  PyTorch → TFLite
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PackagingAgent   │  Build .litertlm
│                  │  (auto UUID, auto timestamp)
└────────┬────────┘
         │
         ▼
    .litertlm
```

### Agent Descriptions

The **OrchestratorAgent** coordinates the entire pipeline, managing execution order, error propagation, and progress reporting. It instantiates all specialized agents and passes data between them through structured AgentMessage objects.

The **ParserAgent** opens the GGUF binary file and extracts model architecture identification, hyperparameters (hidden size, layer count, attention heads), tensor descriptors, and tokenizer vocabulary data.

The **DequantizationAgent** processes each quantized tensor in the GGUF file, converting from formats such as Q4_0, Q5_0, Q8_0, and others into full-precision float32 arrays. It also applies the tensor name mapping to produce PyTorch-compatible parameter paths.

The **ModelAuthoringAgent** selects the appropriate model class from the registry based on the detected architecture, constructs the full PyTorch nn.Module, and loads the dequantized weights.

The **TokenizerAgent** runs in parallel with model authoring, extracting the tokenizer vocabulary from GGUF metadata and converting it to the appropriate format (SentencePiece `.model` or HuggingFace `tokenizer.json`) based on the detected tokenizer type.

The **MetadataAgent** (NEW in v1.0.0) generates the LLM metadata proto required for runtime acceptance, including start/stop token IDs, architecture-specific prompt templates, sampler parameters, model type mapping, and comprehensive system metadata. It also generates a unique UUID and timestamp for build traceability.

The **ConversionAgent** converts the PyTorch model to TFLite format using either direct litert-torch conversion or ONNX as an intermediate representation.

The **PackagingAgent** assembles all artifacts into the final `.litertlm` container using the `litert-lm-builder` library. It adds the TFLite model with `PREFILL_DECODE` model type, tokenizer (auto-detecting SentencePiece or HuggingFace format), LLM metadata proto, and system metadata. The `uuid` and `creation_timestamp` fields are auto-generated by the builder.

### Quantization Recipes

The tool supports several built-in quantization recipes:

| Recipe Name | Method | Weight Type | Activation Type |
|---|---|---|---|
| `none` | None | FP32 | FP32 |
| `dynamic_wi8_afp32` | Weight-only | INT8 | FP32 |
| `dynamic_wi4_afp32` | Weight-only | INT4 | FP32 |
| `full_int8` | Full | INT8 | INT8 |
| `float8` | Float8 | FP8 | FP16 |

Custom quantization recipes can be provided as JSON files via the `--quantization_recipe` argument.

## Project Structure

```
litert-torch-by-qarnux/
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── setup.py
├── requirements.txt
├── litert_torch_qarnux/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                    # CLI entry point
│   ├── orchestrator/             # Multi-agent system
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Base agent class
│   │   ├── orchestrator_agent.py # Pipeline coordinator
│   │   ├── parser_agent.py       # GGUF parsing
│   │   ├── dequantization_agent.py
│   │   ├── model_authoring_agent.py
│   │   ├── conversion_agent.py
│   │   ├── tokenizer_agent.py    # Dual tokenizer support
│   │   ├── metadata_agent.py     # LLM metadata proto generation (NEW)
│   │   └── packaging_agent.py    # Container building
│   ├── models/                   # PyTorch architectures
│   │   ├── __init__.py
│   │   ├── base.py               # Model registry
│   │   ├── llama.py
│   │   ├── gemma.py
│   │   ├── mistral.py
│   │   └── qwen2.py
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── gguf_parser.py
│   │   ├── tensor_mapping.py
│   │   ├── tokenizer_converter.py
│   │   └── tflite_converter.py
│   └── configs/                  # Configurations
│       ├── __init__.py
│       ├── quantization.py
│       └── model_configs.py
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_conversion.py
│   └── test_metadata_agent.py    # NEW
└── examples/
    ├── convert_tinyllama.sh
    └── convert_gemma.sh
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
ruff check .
ruff format .
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## References

[1]: https://github.com/google-ai-edge/litert-lm "google-ai-edge/litert-lm — LiteRT-LM Builder Documentation"
[2]: https://developers.google.com/edge/litert-lm/file_builder "LiteRT-LM File Builder — Google AI Edge"
[3]: https://github.com/google-ai-edge/litert-torch "google-ai-edge/litert-torch — PyTorch to TFLite Conversion"
[4]: https://developers.google.com/edge/litert/conversion/pytorch/genai "Convert PyTorch GenAI Models — Google AI Edge"
[5]: https://pypi.org/project/gguf/ "gguf — Python Package for GGUF Files"
[6]: https://github.com/ggml-org/llama.cpp "llama.cpp — GGUF Format Implementation"
