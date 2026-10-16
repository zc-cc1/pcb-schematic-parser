# 提交方式 1：平台自行构建镜像（不再上传几 GB 的预构建包）。
#
# 这个基础镜像已经自带 torch 2.5.1+cu121 / torchvision 0.20.1+cu121 及全部
# nvidia-*-cu12 运行库（约 5.3 GB 的层）。下面那份 constraints 文件是刻意保留的
# 护栏：它让 pip 认为 torch/torchvision 已满足，从而不会再去下载多 GB 的 CUDA
# wheel —— 第一次评测 0 分就是卡在平台 30 分钟构建上限内装不完 cu128 torch。
# 不要在这里加 --index-url，也不要改这两个版本号，否则会触发重新下载。
#
# 与 Dockerfile.official_slim 的唯一区别：去掉了 `--mount=type=cache` 这类
# BuildKit 专属语法，保证传统 docker build 与 BuildKit 都能构建；构建结果等价。
FROM edarace-agent-base:ubuntu24

WORKDIR /submission

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    YOLO_AUTOINSTALL=false \
    PCB_DEVICE=auto \
    PCB_OUTPUT_FORMAT=target

# opencv-python 需要 GL/GLib；改为 headless 版可以连这一层一起省掉，但那会换掉
# cv2 版本号，需先确认输出逐字节一致。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /submission/

RUN python3 -m pip install --break-system-packages --no-cache-dir --ignore-installed pip==26.2.1

RUN printf 'torch==2.5.1\ntorchvision==0.20.1\n' > /tmp/pcb-runtime-constraints.txt \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        -c /tmp/pcb-runtime-constraints.txt -r /submission/requirements.txt \
    && python3 -m pip check

COPY . /submission/

RUN chmod +x /submission/run.sh \
    && python3 -c "import torch, torchvision, rapidocr, sahi, skimage, networkx; assert torch.__version__.startswith('2.5.1'); assert torch.version.cuda == '12.1'; from PIL import ImageFont; ImageFont.truetype('/submission/assets/fonts/DejaVuSansMono.ttf',12)"

ENTRYPOINT ["/submission/run.sh"]
