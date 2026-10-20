"""Refresh package_manifest.json and build the uploadable source ZIP.

The runtime validates the 12 `runtime_assets` hashes at start-up, so they must stay
exact; everything else in `files` is provenance and is recomputed from disk.  The ZIP
leaves out local backups, caches and run outputs.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"D:/pcb5_submission")
OUT = Path(r"D:/pcb_submission_source_7966.zip")

SKIP_DIRS = {"runs", "work", "logs", "final_results", "project_results", "component_results",
             "intermediate_files", "__pycache__", ".pytest_cache", ".venv", "artifacts"}
SKIP_SUFFIX = {".pyc", ".bak", ".tar", ".zip", ".sha256"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wanted(relative: Path) -> bool:
    if any(part in SKIP_DIRS for part in relative.parts):
        return False
    if relative.suffix in SKIP_SUFFIX:
        return False
    return True


manifest_path = ROOT / "package_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

files = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT).as_posix()
    if not wanted(Path(relative)):
        continue
    if relative == "package_manifest.json":
        continue
    files[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}

manifest["files"] = files
manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# verify the runtime guard still passes on the refreshed manifest
for relative in manifest["runtime_assets"]:
    actual = digest(ROOT / relative)
    expected = files[relative]["sha256"]
    if actual != expected:
        raise SystemExit(f"runtime asset hash mismatch: {relative}")

entries = sorted([p for p in ROOT.rglob("*") if p.is_file() and wanted(p.relative_to(ROOT))],
                 key=lambda p: p.relative_to(ROOT).as_posix())
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in entries:
        archive.write(path, path.relative_to(ROOT).as_posix())

print(f"manifest files: {len(files)}")
print(f"zip: {OUT}  {OUT.stat().st_size/1e6:.1f} MB  ({len(entries)} files)")
for required in ("Dockerfile", "run.sh", "requirements.txt", "agent.py", "package_manifest.json",
                 "weights/unet_wire_pin_v1.pt", "configs/wire_graph_v1.json",
                 "src/pcb_parser/wire_graph.py"):
    print(f"   {'OK ' if (ROOT/required).is_file() else 'MISSING'} {required}")
