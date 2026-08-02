#!/bin/bash
# convert_gemma.sh
#
# Example: Convert a Gemma GGUF model to LiteRT-LM format.
#
# Prerequisites:
#   - A GGUF file for Gemma (e.g., gemma-2b-it.Q4_K_M.gguf)
#   - pip install litert-torch-qarnux
#
# Usage:
#   ./examples/convert_gemma.sh [path_to_gguf]

set -euo pipefail

GGUF_PATH="${1:-./gemma-2b-it.Q4_K_M.gguf}"
OUTPUT_DIR="./litert_output_gemma"

echo "=============================================="
echo "  Gemma GGUF to LiteRT-LM Converter"
echo "=============================================="
echo ""
echo "  Input:  ${GGUF_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

if [ ! -f "${GGUF_PATH}" ]; then
    echo "Error: GGUF file not found: ${GGUF_PATH}"
    echo ""
    echo "Download a Gemma GGUF file first:"
    echo "  wget https://huggingface.co/google/gemma-2b-it-gguf/resolve/main/gemma-2b-it.q4_k_m.gguf"
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
