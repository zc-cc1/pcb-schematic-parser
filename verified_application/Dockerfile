# Reference Dockerfile. Build from the verified_application/ directory:
#   docker build -t pcb-parser:test .
# The originally tested runtime was a CUDA 12.1 preinstalled base image targeting
# sm_80 GPUs; on unsupported architectures run with PCB_DEVICE=cpu.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENTRYPOINT ["python", "-u", "agent.py"]
