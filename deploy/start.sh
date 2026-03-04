#!/bin/bash
set -e

echo "═══════════════════════════════════════"
echo "  GLM-OCR Server Startup"
echo "═══════════════════════════════════════"

# Config
VLLM_PORT=${VLLM_PORT:-8899}
WEB_PORT=${WEB_PORT:-5000}
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

VLLM_PID=$!

echo "⏳ Waiting for vLLM..."
SECONDS=0
while [ $SECONDS -lt 600 ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM ready! (${SECONDS}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM process died!"
        exit 1
    fi
    sleep 5
done

echo ""
echo "🌐 Starting Web App on port ${WEB_PORT}..."
echo "🔐 Password: ${APP_PASSWORD:-glm-ocr-2024}"
echo ""

cd /app
exec python app.py
