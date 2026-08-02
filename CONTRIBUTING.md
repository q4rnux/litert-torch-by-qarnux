# Contributing to litert-torch-qarnux

Thank you for your interest in contributing to this project. This document
provides guidelines for contributing code, reporting issues, and submitting
pull requests.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with the following information:

1. A clear description of the bug and its expected behavior.
2. Steps to reproduce the issue, including the GGUF file architecture and
   quantization type.
3. The full error traceback and relevant log output.
4. Your environment details (Python version, OS, package versions).

### Suggesting New Features

Feature requests are welcome. Please open an issue describing:

1. The proposed feature and its use case.
2. Which GGUF architectures or model families it would support.
3. Any relevant technical details or references.

### Submitting Code Changes

1. Fork the repository and create a feature branch from `main`.
2. Write clear, well-documented code following the existing style.
3. Add tests for any new functionality.
4. Ensure all existing tests pass (`pytest`).
5. Submit a pull request with a descriptive title and detailed description.

## Development Setup

```bash
# Clone and set up the development environment
git clone https://github.com/qarnux/litert-torch-by-qarnux.git
cd litert-torch-by-qarnux
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,full]"
```

## Code Style

This project uses `ruff` for linting and formatting. The configuration is
in `pyproject.toml`. Before submitting a PR, run:

```bash
ruff check .
ruff format .
```

## Adding Support for New Architectures

To add support for a new GGUF architecture:

1. **Tensor Mapping**: Add the architecture-specific tensor name mapping
   in `utils/tensor_mapping.py` under the `TensorMapper` class.

2. **Model Definition**: Create a new module in `models/` with a class
   that extends `BaseModel`. Implement `build()` and `load_weights()`.

3. **Registration**: Register the new model class in the `ModelRegistry`
   at the bottom of the model file.

4. **Config**: Add a `ModelConfig` entry in `configs/model_configs.py`.

5. **Tests**: Add tests for the new architecture in `tests/`.

## Architecture Overview

The conversion pipeline uses a multi-agent orchestration pattern:

```
GGUF File
  └── ParserAgent          (extract metadata & tensors)
       └── DequantizationAgent  (dequantize weights to FP32)
            └── ModelAuthoringAgent (build PyTorch model, load weights)
                 └── ConversionAgent  (PyTorch → TFLite)
                      └── PackagingAgent (build .litertlm container)
                           └── .litertlm Output
```

The `TokenizerAgent` runs in parallel with `ModelAuthoringAgent` to
extract and convert tokenizer data independently.

## License

By contributing, you agree that your contributions will be licensed under
the Apache 2.0 License.
