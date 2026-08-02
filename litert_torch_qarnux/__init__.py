"""
litert-torch-qarnux: GGUF to LiteRT-LM Converter.

A CLI tool that converts GGUF model files to Google's LiteRT-LM format
(.litertlm) using a multi-agent orchestration architecture.

Usage:
    litert-torch export_hf --model=model.gguf --output_dir=./output

For more information, see the README at:
https://github.com/qarnux/litert-torch-by-qarnux
"""

__version__ = "0.3.0"
__author__ = "qarnux"
__license__ = "Apache-2.0"

from litert_torch_qarnux.cli import main

__all__ = ["main", "__version__"]
