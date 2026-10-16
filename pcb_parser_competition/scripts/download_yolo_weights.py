"""Download Ultralytics COCO release weights and record their provenance."""
import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def fetch(url, destination):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PCB-competition-baseline/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11s.pt", "yolo11m.pt"])
    args = parser.parse_args()
    destination = ROOT / "weights" / args.model
    provenance_path = ROOT / "weights" / f"{args.model}_provenance.json"
    if destination.exists() and provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if hashlib.sha256(destination.read_bytes()).hexdigest() == previous["sha256"]:
            print(json.dumps({"status": "existing_hash_verified", "path": str(destination)}, ensure_ascii=False))
            return
        raise RuntimeError("Existing weight hash does not match provenance")
    url = f"{BASE_URL}/{args.model}"
    fetch(url, destination)
    sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps({
        "model": args.model, "url": url, "sha256": sha256,
        "bytes": destination.stat().st_size,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "COCO-pretrained initialization for detector_dataset300_v3_s_p2 (architecture transfer via model.load)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "downloaded", "path": str(destination), "sha256": sha256}, ensure_ascii=False))


if __name__ == "__main__":
    main()
