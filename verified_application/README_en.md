# Inference Package

Source package of the complete inference pipeline for PCB schematic parsing:
YOLO component detection → RapidOCR → frozen dual-head U-Net (wire mask + pin tips) → pin attribution, common-pin library and field completion → engineering whole-net export.

Default weights are bundled: `detector_dataset300_v3_nano.pt` (detector) and `unet_wire_pin_v1.pt` (wire/pin U-Net). No LLM/VLM API is used; the pipeline runs fully offline with zero tokens.

## Runtime environment

| Component | Version |
| --- | --- |
| OS | Ubuntu 24.04 LTS |
| Python | 3.12 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| torchvision | 0.20.1 |
| OpenCV | opencv-python-headless 4.10.0 |
| OCR | RapidOCR (bundled ONNX models) |
| Detection | Ultralytics YOLO |
| Other | NumPy / Pillow / PyYAML / tqdm / scikit-image / networkx |

Base image: `edarace-agent-base:ubuntu24` (CUDA 12.1). All weights required for inference ship with the package; no second CUDA stack is installed on top.

## Run

Docker (absolute input/output paths):

    docker build -t pcb-parser:test .
    docker run --rm --gpus all -v <abs-input>:/input:ro -v <abs-output>:/output:rw pcb-parser:test -input /input -output /output

CPU-only: drop `--gpus all` and add `-e PCB_DEVICE=cpu` before the image name. With matching Python dependencies:

    python agent.py -input <input-dir> -output <output-dir> --device cpu

Notes:

- GPU is auto-selected after a real CUDA matmul probe; on unsupported architectures the exact error is logged and the run falls back to CPU.
- PNG inputs may live in subdirectories; leading zeros in filenames are preserved. Duplicate PNG names across directories raise an error to avoid overwriting results.
- Per-case logs isolate failures: a failing page still yields an empty JSON and the run continues; the reason is recorded in `decision_log.json`. Total failure or failed precheck returns a non-zero exit code.
- Up to 3 cases run in parallel when GPU memory >= 24 GiB; small-memory GPUs and CPU default to serial. Override with `--workers 1/2/3`.

## Output contract

- `/output/final_results/<case_id>_Pred.json` — default export (nested `pins`, `hyperGraph`, `edges`)
- `/output/project_results/<case_id>_Pred.json` — original model/OCR results preserved for debugging
- `/output/decision_log.json` (also mirrored under `final_results/`) — per-case status and failure reasons
- `/output/logs/` — per-case evidence: OCR texts, LED/IC geometry before/after, power export evidence, runtime environment

See `EXPORT_SEMANTICS.md` for export semantics, `OPTIMIZATION.md` for recent changes, `docs/` for per-stage notes, and `package_manifest.json` for per-file hashes, models and configs.

## Known limits

- Whole-net coverage is still limited; engineering designator rules are not yet confirmed against reference annotations.
- A successful package build does not imply any particular evaluation result; scores mentioned in docs are local-scope numbers.
