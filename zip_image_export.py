"""Wrap the exported image tar in the ZIP the platform uploads, plus a sha256 sidecar."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

OUT = Path(r"D:/pcb_submission_out")
tar = OUT / "pcb-parser-v80.tar"
zip_path = OUT / "pcb-parser-v80.zip"

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    archive.write(tar, tar.name)

with zipfile.ZipFile(zip_path) as archive:
    bad = archive.testzip()
    names = archive.namelist()

digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
(OUT / "pcb-parser-v80.zip.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")

print(f"zip: {zip_path}  {zip_path.stat().st_size/1e9:.2f} GB")
print(f"entries: {names}  integrity: {'OK' if bad is None else bad}")
print(f"sha256: {digest}")
