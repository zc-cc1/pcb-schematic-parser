"""Read-only material discovery; write inventory reports inside this project."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYWORDS = re.compile(r"pcb|schematic|circuit|golden|netlist|赛题|评测|原理图|元件库|(?:^|[_ -])qa(?:[_. -]|$)", re.I)
PACKAGES = ["ultralytics", "sahi", "rapidocr", "rapidocr-onnxruntime", "onnxruntime", "opencv-python", "opencv-python-headless", "scikit-image", "skan", "networkx", "torch", "jsonschema", "Pillow"]
SKIP_DIRS = {".git", ".venv", ".venv-gpu", "venv", "node_modules", ".agents", ".codex", "__pycache__"}


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_record(path: Path, with_hash: bool = False) -> dict:
    st = path.stat()
    record = {"path": str(path.resolve()), "bytes": st.st_size, "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    if with_hash:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
    return record


def list_files(root: Path) -> tuple[list[Path], str]:
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--files", "--hidden"]
        for name in sorted(SKIP_DIRS):
            cmd += ["-g", f"!**/{name}/**"]
        cmd += [str(root)]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.strip() or f"rg exit {result.returncode}")
        return [Path(line) for line in result.stdout.splitlines() if line], "rg --files --hidden"
    return [p for p in root.rglob("*") if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(root).parts)], "Path.rglob fallback"


def scan_roots(roots: list[Path]) -> dict:
    records, errors, files = [], [], {}
    for root in roots:
        if not root.is_dir():
            errors.append({"path": str(root), "reason": "not an accessible directory"})
            continue
        try:
            found, method = list_files(root)
            count = 0
            for p in found:
                resolved = p.resolve()
                # Generated examples/reports must never become apparent Golden data.
                if resolved.is_relative_to(PROJECT_ROOT):
                    continue
                files[str(resolved).casefold()] = resolved
                count += 1
            records.append({"root": str(root), "method": method, "files_after_project_exclusion": count})
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append({"path": str(root), "reason": str(exc)})
    golden_candidates, keyword_files = [], []
    json_checked = 0
    json_skipped = []
    for path in sorted(files.values()):
        if KEYWORDS.search(path.name) or path.suffix.lower() in {".kicad_sym", ".kicad_sch"}:
            keyword_files.append(file_record(path))
        if path.suffix.lower() == ".json":
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    json_skipped.append({"path": str(path), "reason": "larger than 16 MiB"})
                    continue
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                json_checked += 1
                if isinstance(value, dict) and all(isinstance(value.get(k), dict) for k in ("components", "pins", "nets")):
                    missing_refs = []
                    for net_id, net in value["nets"].items():
                        graph = net.get("hyperGraph") if isinstance(net, dict) else None
                        if not isinstance(graph, str):
                            missing_refs.append({"net": net_id, "reason": "missing hyperGraph string"})
                            continue
                        for token in graph.strip().removeprefix("(").removesuffix(")").split(","):
                            token = token.strip()
                            if not token:
                                continue
                            parts = token.rsplit(".", 1)
                            if len(parts) != 2 or parts[0] not in value["components"] or not isinstance(value["pins"].get(parts[0]), dict) or "pin_" + parts[1] not in value["pins"].get(parts[0], {}):
                                missing_refs.append({"net": net_id, "reference": token})
                    golden_candidates.append({**file_record(path, with_hash=True), "status": "example_with_unresolved_references" if missing_refs else "candidate_not_verified", "top_level_keys": sorted(value), "counts": {k: len(value[k]) for k in ("components", "pins", "nets")}, "reference_warning_count": len(missing_refs), "reference_warning_examples": missing_refs[:8]})
            except (OSError, ValueError, UnicodeError) as exc:
                json_skipped.append({"path": str(path), "reason": type(exc).__name__})
    return {
        "roots": records, "scan_errors": errors, "file_count": len(files),
        "extension_counts": dict(Counter(p.suffix.lower() or "[no extension]" for p in files.values()).most_common()),
        "png_parent_counts": dict(Counter(str(p.parent) for p in files.values() if p.suffix.lower() == ".png").most_common()),
        "keyword_file_matches": keyword_files,
        "json_structure_checked": json_checked, "json_skipped": json_skipped,
        "submission_shaped_json_candidates": golden_candidates,
        "limitations": ["Search is limited to the explicit roots; not a whole-device search.", "Project-generated files are excluded.", "Archives and PDF screenshots are not dataset cases.", "PNG files require source/annotation verification before use.", "JSON candidate shape does not establish Golden provenance or correctness.", "Files above 16 MiB and unreadable JSON are reported separately."],
    }


def probe_environment() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = {"status": "installed", "version": version(name)}
        except PackageNotFoundError:
            packages[name] = {"status": "not_installed_in_this_interpreter"}
    gpu = {"status": "not_checked"}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run([executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
            gpu = {"status": "query_completed" if result.returncode == 0 else "query_failed", "exit_code": result.returncode, "output": result.stdout.strip(), "error": result.stderr.strip(), "memory_unit": "MiB"}
        except (OSError, subprocess.SubprocessError) as exc:
            gpu = {"status": "query_failed", "error": str(exc)}
    return {"python_executable": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(), "packages": packages, "gpu": gpu, "notes": ["This checks only the current interpreter, not every Python environment.", "Detected GPU/driver does not establish PyTorch/CUDA compatibility.", "Official runtime versions and entrypoint remain unverified."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", action="append", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.search_root]
    spec = Path(args.spec).expanduser().resolve()
    if not spec.is_file():
        parser.error(f"Specification not found: {spec}")
    generated_at = datetime.now(timezone.utc).isoformat()
    scan = scan_roots(roots)
    report = {"inventory_version": "0.2", "generated_at_utc": generated_at, "specification": {"status": "present", **file_record(spec, with_hash=True)}, "discovery": scan, "confirmed_official_cases": [], "official_evaluator": {"status": "not_identified_in_scanned_materials"}, "symbol_library": {"status": "not_identified_in_scanned_materials"}, "qa": {"status": "not_identified_in_scanned_materials"}}
    audit_path = PROJECT_ROOT / "data/dataset_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        report["raw_dataset_audit"] = {"path": str(audit_path), "dataset_root": audit["dataset_root"], "audit_time_utc": audit["generated_at_utc"], "paired_readable_cases": audit["paired_readable_cases"], "official_submission_format_cases": audit["official_submission_format_cases"]}
    write_json(PROJECT_ROOT / "data/material_inventory.json", report)
    environment = {"generated_at_utc": generated_at, **probe_environment()}
    write_json(PROJECT_ROOT / "docs/environment.json", environment)
    print(json.dumps({"inventory": str(PROJECT_ROOT / "data/material_inventory.json"), "environment": str(PROJECT_ROOT / "docs/environment.json"), "scanned_files": scan["file_count"], "json_checked": scan["json_structure_checked"], "candidate_json_count": len(scan["submission_shaped_json_candidates"]), "confirmed_official_cases": 0, "scan_errors": len(scan["scan_errors"]), "gpu": environment["gpu"]}, ensure_ascii=False, indent=2))
    return 1 if scan["scan_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
