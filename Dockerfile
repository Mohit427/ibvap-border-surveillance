# IBVAP - container image for the hosted sample-clip demo (Hugging Face Spaces).
#
# There is no camera in a container, so this build only serves the bundled
# sample clips / uploaded video (see app/video_source.py auto-fallback) -
# the live-webcam pitch demo runs locally instead (see README).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    IBVAP_HOST=0.0.0.0 \
    IBVAP_OPEN_BROWSER=0 \
    PORT=7860

# libgl1/libglib2.0-0: opencv needs these even in headless/no-GUI use.
# libgomp1: torch's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch/torchvision first (the default PyPI wheel pulls in ~2GB of
# CUDA libraries that are useless here and blow the image size/build time).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake model weights and EasyOCR's own models into the image at build time
# so the running container never has a slow/flaky first-request download.
RUN python scripts/download_models.py \
    && python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

EXPOSE 7860

CMD ["python", "run.py"]
