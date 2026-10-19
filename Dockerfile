# CUDA image (also runs CPU-only if no GPU is passed through).
# Build a lighter CPU-only image with:  --build-arg BASE_IMAGE=python:3.12-slim --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG TORCH_INDEX=""

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    COUNCIL_DATA_DIR=/data \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# When TORCH_INDEX is set (CPU build) torch comes from that index; on the
# pytorch base image the preinstalled CUDA torch already satisfies the pin.
RUN if [ -n "$TORCH_INDEX" ]; then pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url "$TORCH_INDEX"; fi \
    && pip install --no-cache-dir -r requirements.txt

COPY omnivoice/ ./omnivoice/
COPY app/ ./app/

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]
