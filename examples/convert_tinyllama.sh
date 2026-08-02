#!/bin/bash
# convert_tinyllama.sh
#
# Example: Convert a TinyLlama GGUF model to LiteRT-LM format.
#
# Prerequisites:
#   - A GGUF file for TinyLlama (e.g., TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf)
#   - pip install litert-torch-qarnux
#
# Usage:
#   ./examples/convert_tinyllama.sh [path_to_gguf]

set -euo pipefail

GGUF_PATH="${1:-./TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf}"
OUTPUT_DIR="./litert_output_tinyllama"

echo "=============================================="
echo "  TinyLlama GGUF to LiteRT-LM Converter"
echo "=============================================="
echo ""
echo "  Input:  ${GGUF_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

if [ ! -f "${GGUF_PATH}" ]; then
    echo "Error: GGUF file not found: ${GGUF_PATH}"
    echo ""
    echo "Download a TinyLlama GGUF file first:"
    echo "  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    exit 1
fi

echo "Starting conversion..."
echo ""

litert-torch export_hf \
    --model="${GGUF_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --quantization_recipe=dynamic_wi8_afp32 \
    --verbose

echo ""
echo "Conversion complete!"
echo "Output file: ${OUTPUT_DIR}/*.litertlm"
