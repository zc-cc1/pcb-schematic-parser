"""Run tests/test_export_geometry.py without pytest (it is not installed here)."""
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

spec = importlib.util.spec_from_file_location("test_export_geometry", ROOT / "tests/test_export_geometry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

failures = 0
for name in sorted(dir(module)):
    if not name.startswith("test_"):
        continue
    fn = getattr(module, name)
    if not callable(fn):
        continue
    try:
        fn()
        print(f"PASS {name}")
    except Exception:
        failures += 1
        print(f"FAIL {name}")
        traceback.print_exc()
print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
