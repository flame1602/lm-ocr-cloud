FROM vllm/vllm-openai:latest

# Install additional dependencies
RUN pip install --no-cache-dir pymupdf openai flask PyMuPDF

# Create app directory
WORKDIR /app
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

# Create data directories
RUN mkdir -p /data/input_pdfs /data/output_markdown

# Environment
ENV VLLM_PORT=8899
ENV WEB_PORT=5000
ENV APP_PASSWORD=glm-ocr-2024
ENV INPUT_DIR=/data/input_pdfs
ENV OUTPUT_DIR=/data/output_markdown

# Expose web port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["/app/start.sh"]
