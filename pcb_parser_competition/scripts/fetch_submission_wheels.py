"""Fetch Linux cp312 wheels for the extras the base image lacks, offline-buildable.

Why this exists: `docker build` cannot reach PyPI from this workstation's Docker VM
(TCP connects, then the TLS ClientHello is dropped - the host reaches PyPI fine).
The base image already ships torch/torchvision/numpy/cv2/PyYAML/networkx/Pillow, so
only a handful of packages have to be added, and they can be fetched on the host.

`pip download -r ...` with `--platform` hangs in the resolver, so resolution is done
here instead: fetch each requirement with `--no-deps`, read `Requires-Dist` out of the
wheel METADATA, and repeat until the closure is covered. Anything the base image
already provides is skipped.

Usage:
    .venv-gpu\\Scripts\\python.exe scripts\\fetch_submission_wheels.py
"""
import argparse
import re
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Installed in edarace-agent-base:ubuntu24 already (probed, not assumed).
PROVIDED = {"torch", "torchvision", "numpy", "opencv-python", "pyyaml", "networkx", "pillow"}
# Never needed on Linux / always stdlib-adjacent in this stack.
SKIP = {"pyreadline3", "pywin32", "colorama", "tqdm-linux", "setuptools", "pip", "wheel"}

TARGET_ARGS = [
    "--only-binary=:all:",
    "--platform", "manylinux2014_x86_64",
    "--platform", "manylinux_2_28_x86_64",
    "--python-version", "3.12",
    "--implementation", "cp",
    "--abi", "cp312",
]


def normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[=<>!~\[ ]", line, 1)[0]
        out[normalise(name)] = line
    return out


def marker_applies(marker):
    """Keep Requires-Dist entries that apply to linux/x86_64/cp312."""
    if not marker:
        return True
    m = marker.lower().replace('"', "'")
    for bad in ("sys_platform == 'win32'", "platform_system == 'windows'", "os_name == 'nt'"):
        if bad in m:
            return False
    if "python_version < '3.12'" in m or "python_version < '3.9'" in m:
        return False
    return True


def wheel_requires(wheel):
    with zipfile.ZipFile(wheel) as zf:
        meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        msg = Parser().parsestr(zf.read(meta).decode("utf-8", "replace"))
    reqs = []
    for raw in msg.get_all("Requires-Dist") or []:
        if "extra ==" in raw:
            continue
        name_part, _, marker = raw.partition(";")
        if not marker_applies(marker.strip()):
            continue
        name = re.split(r"[=<>!~\[ (]", name_part.strip(), 1)[0]
        spec = name_part[len(name):].strip()
        reqs.append((normalise(name), spec))
    return reqs


def run_download(requirement, dest):
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(dest),
           *TARGET_ARGS, requirement]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default=str(ROOT / "submission/out/requirements-extras.txt"))
    ap.add_argument("--dest", default=str(ROOT / "submission/out/wheels"))
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = parse_requirements(args.requirements)
    queue = [(name, spec, "explicit") for name, spec in sorted(wanted.items())]
    done, downloaded, problems = {}, [], []

    while queue:
        name, spec, origin = queue.pop(0)
        if name in done:
            continue
        if name in PROVIDED or name in SKIP:
            done[name] = "provided-by-base"
            print(f"  skip  {name:28} ({origin}; base image provides it)")
            continue
        requirement = f"{name}{spec}" if spec else name
        proc = run_download(requirement, dest)
        if proc.returncode != 0:
            # a bad pin is not fatal: retry unpinned so the closure still closes
            proc = run_download(name, dest)
        if proc.returncode != 0:
            problems.append((requirement, proc.stderr.strip().splitlines()[-1:] or [""]))
            done[name] = "FAILED"
            print(f"  FAIL  {requirement}")
            continue
        wheels = sorted(dest.glob(f"{name.replace('-', '_')}*.whl")) or sorted(dest.glob(f"{name}*.whl"))
        if not wheels:
            problems.append((requirement, ["no wheel landed"]))
            done[name] = "FAILED"
            continue
        wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        done[name] = wheel.name
        downloaded.append(wheel.name)
        print(f"  ok    {wheel.name:44} ({origin})")
        for dep_name, dep_spec in wheel_requires(wheel):
            if dep_name not in done:
                queue.append((dep_name, dep_spec, f"needed by {name}"))

    total = sum(p.stat().st_size for p in dest.glob("*.whl"))
    print(f"\n{len(downloaded)} wheels, {total/1e6:.1f} MB in {dest}")
    if problems:
        print("\nproblems:")
        for requirement, why in problems:
            print(f"  {requirement}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
