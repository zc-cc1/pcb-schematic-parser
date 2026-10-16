"""Resume an official CUDA wheel download when the default CDN stalls."""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-win_amd64.whl"
OFFICIAL_INDEX = "https://download.pytorch.org/whl/cu128/torch/"
EXPECTED_SHA256 = "fbde8f6a9ec8c76979a0d14df21c10b9e5cab6f0d106a73ca73e2179bc597cae"
TOTAL_BYTES = 2867409626
BLOCK_BYTES = 8 * 1024 * 1024


def main():
    folder = ROOT / "artifacts/setup/wheels"
    parts = folder / "torch_parts"
    parts.mkdir(parents=True, exist_ok=True)
    target = folder / "torch-2.10.0+cu128-cp312-cp312-win_amd64.whl"
    if target.exists() and file_hash(target) == EXPECTED_SHA256:
        print("Existing official wheel SHA256 verified", flush=True)
        return
    count = (TOTAL_BYTES + BLOCK_BYTES - 1) // BLOCK_BYTES
    start_time = time.perf_counter()

    def download(index):
        start = index * BLOCK_BYTES
        end = min(start + BLOCK_BYTES, TOTAL_BYTES) - 1
        part = parts / f"{index:04d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(5):
            try:
                request = urllib.request.Request(URL + f"?pcb_part={index}", headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{TOTAL_BYTES}":
                        raise RuntimeError("Unexpected response range")
                    payload = response.read(end - start + 2)
                if len(payload) != end - start + 1:
                    raise RuntimeError("Incomplete chunk")
                part.write_bytes(payload)
                return part
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, index) for index in range(count)]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if completed % 16 == 0 or completed == count:
                print(json.dumps({"chunks_done": completed, "chunks_total": count,
                                  "percent": round(completed / count * 100, 1),
                                  "elapsed_sec": round(time.perf_counter() - start_time, 1)}), flush=True)
    staging = target.with_suffix(".whl.download")
    with staging.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:04d}.part").open("rb") as part:
                shutil.copyfileobj(part, output)
    actual_hash = file_hash(staging)
    if staging.stat().st_size != TOTAL_BYTES or actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Official wheel SHA256 mismatch: {actual_hash}")
    staging.replace(target)
    for part in parts.glob("*.part"):
        part.unlink()
    record = {"path": str(target), "official_index": OFFICIAL_INDEX, "url": URL,
              "bytes": TOTAL_BYTES, "sha256": actual_hash, "hash_matches_official_index": True,
              "elapsed_sec": time.perf_counter() - start_time}
    (folder / "torch_download.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
