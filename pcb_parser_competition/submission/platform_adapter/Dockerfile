FROM edarace-agent-base:ubuntu24

WORKDIR /submission
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg YOLO_AUTOINSTALL=false PCB_DEVICE=auto PCB_OUTPUT_FORMAT=target

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /submission/
RUN pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 torchvision==0.25.0 \
    && pip3 install --break-system-packages --no-cache-dir -r /submission/requirements.txt

COPY . /submission/
RUN chmod +x /submission/run.sh \
    && python3 -c "from pathlib import Path; import rapidocr, torch, torchvision, sahi, skimage, networkx; from PIL import ImageFont; ImageFont.truetype('/submission/assets/fonts/DejaVuSansMono.ttf',12); assert len(list(Path('/submission/assets/ocr').glob('*.onnx')))==3"
ENTRYPOINT ["/submission/run.sh"]
