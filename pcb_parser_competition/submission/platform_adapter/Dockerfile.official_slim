# Competition A800 runtime. The official base already includes CUDA PyTorch;
# do not install a second CUDA stack over its immutable layers.
FROM edarace-agent-base:ubuntu24
WORKDIR /submission
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg YOLO_AUTOINSTALL=false PCB_DEVICE=auto PCB_OUTPUT_FORMAT=target
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=8 PIP_RESUME_RETRIES=12 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /submission/
RUN --mount=type=cache,id=pcb-pip-linux-py312,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages --ignore-installed pip==26.2.1
RUN --mount=type=cache,id=pcb-pip-linux-py312,target=/root/.cache/pip \
    printf 'torch==2.5.1\ntorchvision==0.20.1\n' > /tmp/pcb-runtime-constraints.txt \
    && python3 -m pip install --break-system-packages -c /tmp/pcb-runtime-constraints.txt -r /submission/requirements.txt \
    && python3 -m pip check
COPY . /submission/
RUN chmod +x /submission/run.sh \
    && python3 -c "import torch, torchvision, rapidocr, sahi, skimage, networkx; assert torch.__version__.startswith('2.5.1'); assert torch.version.cuda == '12.1'; from PIL import ImageFont; ImageFont.truetype('/submission/assets/fonts/DejaVuSansMono.ttf',12)"
ENTRYPOINT ["/submission/run.sh"]
