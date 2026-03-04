#!/bin/bash
echo "═══════════════════════════════════════"
echo "  GLM-OCR Server Startup"
echo "═══════════════════════════════════════"

# Config
VLLM_PORT=${VLLM_PORT:-8899}
PORT=${PORT:-8080}
MODEL="zai-org/GLM-OCR"

# Create data dirs
mkdir -p /data/input_pdfs /data/output_markdown

# STEP 1: Start Flask FIRST so Cloud Run sees port immediately
echo "🌐 Starting Web App on port ${PORT}..."
echo "🔐 Password: ${APP_PASSWORD:-glm-ocr-2024}"
cd /app
python3 app.py &
FLASK_PID=$!

# Wait for Flask to bind
sleep 3
echo "✅ Flask ready on port ${PORT}"

# STEP 2: Start vLLM in background (will take 3-5 minutes to load model)
echo ""
echo "🚀 Starting vLLM server (GLM-OCR)... (this takes 3-5 minutes)"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --allowed-local-media-path / \
    --port "$VLLM_PORT" \
    --served-model-name glm-ocr \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 &
VLLM_PID=$!

# Wait for either process to exit
wait -n $FLASK_PID $VLLM_PID
echo "❌ A process exited unexpectedly"
kill $FLASK_PID $VLLM_PID 2>/dev/null
exit 1
