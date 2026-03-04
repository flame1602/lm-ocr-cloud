FROM vllm/vllm-openai:latest

# CRITICAL: Clear base image ENTRYPOINT so our start.sh runs directly
ENTRYPOINT []

# Install additional dependencies
RUN pip install --ignore-installed --no-cache-dir pymupdf openai flask

# Create app directory
WORKDIR /app
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

# Create data directories
RUN mkdir -p /data/input_pdfs /data/output_markdown

# Environment
ENV VLLM_PORT=8899
ENV PORT=8080
ENV APP_PASSWORD=glm-ocr-2024
ENV INPUT_DIR=/data/input_pdfs
ENV OUTPUT_DIR=/data/output_markdown

# Expose web port
EXPOSE 8080

CMD ["/bin/bash", "/app/start.sh"]
