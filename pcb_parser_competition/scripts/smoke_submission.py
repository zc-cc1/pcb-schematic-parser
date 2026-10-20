"""Local smoke test for the submission entry point (no Docker needed).

Runs ``submission/agent.py`` against a temporary ``-input``/``-output`` pair using
the local GPU venv, then asserts the things the scorer cares about:

* every case gets ``<case>_Pred.json`` and there is a ``decision_log.json``;
* the JSON has exactly the three accepted top-level sections;
* component types are the official raw vocabulary (``r``/``c``/``box``/``gnd``/``v``),
  not the internal names - this is the regression that silently cost ~40 F1 points;
* component boxes are the shipped canonical sizes for the fixed-size classes;
* bboxes are ordered, finite and inside the image.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "submission" / "agent.py"
RAW_TYPES = {"r", "c", "l", "d", "box", "gnd", "v", "led", "pin", "testpoint",
             "net_input", "net_output", "crystal", "mosfet", "switch", "amp", "bjt"}
CANONICAL = {"c": {(16.0, 20.0), (20.0, 16.0)}, "r": {(8.0, 20.0), (20.0, 8.0)},
             "gnd": {(20.0, 19.0), (19.0, 20.0)}, "v": {(10.0, 10.0)}}


def run_agent(input_dir, output_dir, limit):
    venv = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
    python = str(venv) if venv.exists() else sys.executable
    env = dict(os.environ, PCB_SUBMISSION_ROOT=str(ROOT), PYTHONIOENCODING="utf-8")
    cmd = [python, str(AGENT), "-input", str(input_dir), "-output", str(output_dir),
           "-limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["0002", "0018"])
    ap.add_argument("--limit", type=int, default=2)
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="pcb_smoke_"))
    input_dir, output_dir = work / "input", work / "output"
    input_dir.mkdir()
    for cid in args.cases:
        # mirror the harness layout: /input/<case>/<case>.png
        target = input_dir / cid
        target.mkdir()
        shutil.copy(ROOT / "data" / "standardized" / cid / f"{cid}.png", target / f"{cid}.png")
    print(f"input staged at {input_dir}")

    proc = run_agent(input_dir, output_dir, args.limit)
    print(proc.stdout.strip()[-2000:] or "(no stdout)")
    if proc.returncode != 0:
        print(proc.stderr.strip()[-3000:], file=sys.stderr)
        print("FAIL: agent exited non-zero")
        return 1

    results = output_dir / "final_results"
    failures = []
    produced = sorted(results.glob("*_Pred.json"))
    if not produced:
        failures.append("no prediction files written")
    if not (results / "decision_log.json").is_file():
        failures.append("decision_log.json missing")

    log = json.loads((results / "decision_log.json").read_text(encoding="utf-8"))
    print(f"decision_log: method={log['method']!r} cases={log['processed_cases']} "
          f"succeeded={log['succeeded_cases']} tokens={log['token_usage']['total_tokens']}")

    for path in produced:
        case = path.name[: -len("_Pred.json")]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"components", "pins", "nets"}:
            failures.append(f"{case}: top-level keys {sorted(payload)}")
            continue
        comps = payload["components"]
        if not comps:
            failures.append(f"{case}: no components")
            continue
        name = ROOT / "data" / "standardized" / case / f"{case}.png"
        width = height = None
        if name.exists():
            from PIL import Image
            width, height = Image.open(name).size
        types = set()
        for key, comp in comps.items():
            types.add(comp["type"])
            bbox = comp["bbox"]
            if len(bbox) != 4 or any(not isinstance(v, float) for v in bbox):
                failures.append(f"{case}/{key}: bad bbox {bbox}")
                continue
            if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                failures.append(f"{case}/{key}: unordered bbox {bbox}")
            if width and height:
                if not (0 <= bbox[0] and 0 <= bbox[1] and bbox[2] <= width and bbox[3] <= height):
                    failures.append(f"{case}/{key}: bbox outside image {bbox}")
            canonical = CANONICAL.get(comp["type"])
            if canonical:
                size = (round(bbox[2] - bbox[0], 1), round(bbox[3] - bbox[1], 1))
                if size not in canonical:
                    failures.append(f"{case}/{key}: {comp['type']} size {size} not canonical "
                                    f"{sorted(canonical)}")
        internal = types - RAW_TYPES - {"other", "fuse", "buzzer", "motor", "battary",
                                        "transfomer", "jumper", "block", "esd", "m", "net_short",
                                        "opt", "dc", "circle_header", "ant", "current_source"}
        if internal:
            failures.append(f"{case}: internal-style types exported: {sorted(internal)}")
        print(f"  {case}: {len(comps)} components, types={sorted(types)}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"\n{len(failures)} problem(s); artefacts kept at {work}")
        return 1
    print(f"PASS: {len(produced)} prediction file(s) + decision_log.json, schema/types/geometry OK")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
