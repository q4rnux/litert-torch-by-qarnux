"""
Package entry point for `python -m litert_torch_qarnux`.

Allows running the CLI as a Python module:
    python -m litert_torch_qarnux export_hf --model=model.gguf --output_dir=./output
"""

from litert_torch_qarnux.cli import main

if __name__ == "__main__":
    main()
