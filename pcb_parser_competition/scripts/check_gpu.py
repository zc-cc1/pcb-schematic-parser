"""Verify actual CUDA kernels and torchvision NMS, not only driver visibility."""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torchvision
from torchvision.ops import nms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/setup/gpu_check.json"))
    args = parser.parse_args()
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not report["cuda_available"]:
        raise RuntimeError("CUDA unavailable; refusing to silently fall back to CPU")
    props = torch.cuda.get_device_properties(0)
    report.update(gpu_name=props.name, compute_capability=list(torch.cuda.get_device_capability(0)),
                  total_memory_mib=props.total_memory / 2**20, compiled_arches=torch.cuda.get_arch_list())
    torch.manual_seed(42)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    expected = a @ b
    actual = (a.cuda() @ b.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    report["matmul_max_absolute_error"] = float((actual - expected).abs().max())
    image = torch.randn(1, 3, 64, 64, device="cuda")
    layer = torch.nn.Conv2d(3, 16, 3).cuda().eval()
    with torch.inference_mode():
        result = layer(image)
    assert torch.isfinite(result).all()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]], device="cuda")
    kept = nms(boxes, torch.tensor([.9, .8, .7], device="cuda"), .5).cpu().tolist()
    assert kept == [0, 2], kept
    report["torchvision_cuda_nms"] = "passed"
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")
    with torch.inference_mode():
        for _ in range(3):
            x @ y
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            z = x @ y
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
    assert torch.isfinite(z).all()
    report["warm_2048_matmul_median_ms"] = statistics.median(elapsed)
    report["status"] = "passed_real_gpu_kernels"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
