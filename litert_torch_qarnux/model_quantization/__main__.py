"""
Package entry point for `python -m litert_torch_qarnux.model_quantization`.

Usage:
    python -m litert_torch_qarnux.model_quantization quantize --input model.gguf --dtype int8
    python -m litert_torch_qarnux.model_quantization behavior list
    python -m litert_torch_qarnux.model_quantization template list
"""
from litert_torch_qarnux.model_quantization.cli import main

if __name__ == "__main__":
    main()
