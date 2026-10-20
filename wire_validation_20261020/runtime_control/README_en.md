# EDARace Agent

Image: `edarace-agent-base:ubuntu24`  
Workdir inside container: `/submission`

---

## What's inside the image

| Component | Version |
| --- | --- |
| OS | Ubuntu 24.04 LTS |
| Python | 3.12 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| torchvision | 0.20.1 |
| OpenCV | opencv-python-headless 4.10.0 |
| Transformers | 4.46.3 (+ accelerate / safetensors / tokenizers / sentencepiece) |
| Other | NumPy / Pillow / PyYAML / tqdm / requests / huggingface-hub |
| Editors | vim, nano |

GPU is available when the container is launched with `--gpus all`.

---

## What your program must do

The container entrypoint is `/submission/agent_system`. The CLI contract is fixed:

```bash
/submission/agent_system -input /input -output /output
```

You must replace `/submission/agent_system` with your own agent implementation (keep the same CLI flags and output layout).

### Input directory `/input`

The evaluator mounts a directory that contains **multiple schematic images**:

```text
/input/<case_name>.png
/input/<case_name>.png
...
```

Each image is one case. `<case_name>` is the image filename without extension, e.g. for `0025-Altium Designer.png`, `case_name = 0025-Altium Designer`.

### Output directory `/output`

Your agent must write results into `/output`, with the following **mandatory layout**:

```text
/output/decision_log.json
/output/final_results/<case_name>_Pred.json   # one per input image
/output/final_results/run.log                 # optional: run log
/output/runs/                                 # optional: intermediate artifacts
```

The exact JSON schema for `decision_log.json` and `<case_name>_Pred.json` is specified in the published EDARace handbook. Please refer to that document for details.

### Environment variables (injected at runtime)

| Variable | Description |
| --- | --- |
| `LLM_BASE_URL` | LLM API base URL |
| `LLM_API_KEY` | API key |
| `LLM_MODEL` | model name |
| `NVIDIA_VISIBLE_DEVICES` | GPU list, usually `all` |

---

## Submission

Build and submit your final image directly. The image must:

- Be based on `edarace-agent-base:ubuntu24`.
- Provide `/submission/agent_system` as the entrypoint, accepting `-input <dir> -output <dir>`.
- Use `/input` as the read-only input directory and `/output` as the writable output directory.
- Write `/output/decision_log.json` and `/output/final_results/<case_name>_Pred.json` per the contract above.