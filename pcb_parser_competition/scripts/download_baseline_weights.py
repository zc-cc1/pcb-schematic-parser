"""Fetch the chosen public circuit detector and record its exact content hash."""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "HR-DONK/circuit-recognition"
MODEL = "circuit_yolov8n_v3.pt"


def fetch(url, destination):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PCB-competition-baseline/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    destination = ROOT / "weights" / MODEL
    provenance_path = ROOT / "weights" / "baseline_provenance.json"
    if destination.exists() and provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if hashlib.sha256(destination.read_bytes()).hexdigest() == previous["sha256"]:
            print(json.dumps({"status": "existing_hash_verified", "path": str(destination)}, ensure_ascii=False))
            return
        raise RuntimeError("Existing detector hash does not match provenance")
    commit = None
    revision_error = None
    try:
        request = urllib.request.Request(f"https://api.github.com/repos/{REPO}/commits/main",
                                         headers={"User-Agent": "PCB-competition-baseline/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            commit = json.load(response)["sha"]
    except Exception as exc:
        revision_error = str(exc)
    ref = commit or "main"
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/models/{MODEL}"
    temporary = destination.with_suffix(".pt.download")
    fetch(url, temporary)
    with temporary.open("rb") as stream:
        header = stream.read(200)
    if temporary.stat().st_size < 1024 * 1024 or header.startswith(b"version https://git-lfs"):
        raise RuntimeError("Expected a real detector checkpoint; received a small file or LFS pointer")
    temporary.replace(destination)
    license_path = ROOT / "weights" / "HR-DONK_LICENSE.txt"
    fetch(f"https://raw.githubusercontent.com/{REPO}/{ref}/LICENSE", license_path)
    provenance = {
        "repository": f"https://github.com/{REPO}", "revision": commit,
        "revision_lookup_error": revision_error, "download_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(destination), "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "repository_declared_license": "MIT", "license_copy": str(license_path),
        "framework_license": "Ultralytics AGPL-3.0; separate from repository license",
        "intended_use": "Initial detector baseline only; PCB-domain accuracy is unverified",
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
