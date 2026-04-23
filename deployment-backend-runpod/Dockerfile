FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deployment-backend-runpod/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY stylegan3 /app/stylegan3
COPY deployment-backend-runpod /app

# Precompile StyleGAN CUDA extensions during image build.
RUN python3 -c "import sys; sys.path.insert(0, '/app/stylegan3'); from torch_utils.ops import bias_act, upfirdn2d; bias_act._init(); upfirdn2d._init(); print('Extensions OK')"

CMD ["python3", "-u", "handler.py"]
