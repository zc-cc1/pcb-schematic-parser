"""Run unittest suites and imported plain test functions without installing pytest."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"tests"))
suite = unittest.TestSuite()
for test_path in sorted((ROOT/"tests").glob("test_*.py")):
    if test_path.name in {"test_external_cad_inventory.py", "test_fusion.py"}:
        continue
    spec = importlib.util.spec_from_file_location(test_path.stem, test_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_module))
print("CAD/PDF inventory tests run separately in the bundled document Python environment.", flush=True)
path=ROOT/"tests/test_fusion.py"
spec=importlib.util.spec_from_file_location("imported_fusion_tests",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in sorted(dir(module)):
    if name.startswith("test_") and callable(getattr(module,name)):
        suite.addTest(unittest.FunctionTestCase(getattr(module,name),description=name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
