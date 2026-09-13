FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG GALAXEA_REVISION=89f2322b4ad016e192437adc1a2c253b05bab246
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/GalaxeaVLA/src:/opt/GalaxeaVLA:/app \
    HF_HUB_OFFLINE=1 WANDB_MODE=disabled DS_BUILD_OPS=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip git ca-certificates build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir uv==0.8.15 \
    && uv python install 3.10.16 \
    && uv venv /opt/venv --python 3.10.16
RUN uv pip install --no-cache torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
COPY docker/galaxea-policy/requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache -r /tmp/requirements.txt
RUN git clone https://github.com/OpenGalaxea/GalaxeaVLA.git /opt/GalaxeaVLA \
    && cd /opt/GalaxeaVLA && git checkout --detach "$GALAXEA_REVISION" \
    && test "$(git rev-parse HEAD)" = "$GALAXEA_REVISION"
COPY docker/galaxea-policy/harden_loads.py /tmp/harden_loads.py
RUN python /tmp/harden_loads.py /opt/GalaxeaVLA \
    && cd /opt/GalaxeaVLA && python -c "import scripts.serve_policy"
WORKDIR /opt/GalaxeaVLA
RUN ln -s /checkpoints checkpoints
# Runtime wrapper is copied after dependencies so iteration reuses heavy layers.
COPY policy/galaxea/ /app/galaxea/
COPY docker/galaxea-policy/checkpoint-manifest.json /app/checkpoint-manifest.json
COPY docker/galaxea-policy/server.py /app/galaxea_server.py
ENTRYPOINT ["python", "/app/galaxea_server.py"]
