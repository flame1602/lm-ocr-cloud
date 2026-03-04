#!/bin/bash
set -e

echo "═══════════════════════════════════════"
echo "  GLM-OCR Server Startup"
echo "═══════════════════════════════════════"

# Config
VLLM_PORT=${VLLM_PORT:-8899}
PORT=${PORT:-8080}
MODEL="zai-org/GLM-OCR"

# Create data dirs
mkdir -p /data/input_pdfs /data/output_markdown

echo ""
echo "🚀 Starting vLLM server (GLM-OCR)..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --allowed-local-media-path / \
    --port "$VLLM_PORT" \
    --served-model-name glm-ocr \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 &

echo "🌐 Starting Web App on port ${PORT}..."
echo "🔐 Password: ${APP_PASSWORD:-glm-ocr-2024}"
echo ""

cd /app
exec python app.py
