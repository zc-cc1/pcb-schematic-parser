"""Exercise the patched adapter: part numbers survive, misread designators are repaired."""
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, r"D:\pcb5_submission")

import adapter  # noqa: E402

print("near-miss checks")
for text, key, expected in [("ED1", "LED1", True), ("led1", "LED1", True), ("SS34", "D1", False),
                            ("1N4007", "D1", False), ("NCD0603Y2", "LED1", False),
                            ("GL0603UG01", "LED2", False), ("R1O", "R10", True), ("C7", "C7", True)]:
    got = adapter._near_miss(text, key)
    print(f"  {text:<12} vs {key:<8} -> {got!s:<6} expected {expected}  {'OK' if got == expected else 'MISMATCH'}")

src = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第8次测评-54.92/intermediate_files/final_results/12_Pred.json")
doc = json.loads(src.read_text(encoding="utf-8"))
print("\ncomponents before:", {k: v.get("Name") for k, v in doc["components"].items() if v.get("type") in {"d", "led", "r", "c"}})

doc = deepcopy(doc)
# inject the three shapes the 8th run mis-handled
doc["components"]["LED1"] = {"type": "led", "Name": "ED1", "value": None, "bbox": [10, 10, 40, 40]}
doc["components"]["LED2"] = {"type": "led", "Name": "NCD0603Y2", "value": None, "bbox": [50, 50, 80, 80]}
doc["components"]["D1"] = {"type": "d", "Name": "SS34", "value": None, "bbox": [90, 90, 110, 120]}
doc["components"]["LED3"] = {"type": "led", "Name": None, "value": None, "bbox": [120, 120, 150, 150]}
try:
    out = adapter.to_target(deepcopy(doc))
    names = {k: v.get("Name") for k, v in out["components"].items() if v.get("type") in {"d", "led"}}
    print("after to_target:", names)
    expected = {"LED1": "LED1", "LED2": "NCD0603Y2", "D1": "SS34", "LED3": "LED3"}
    print("expected      :", expected)
    print("MATCH" if names == expected else "MISMATCH")
except Exception as error:  # noqa: BLE001
    print("to_target raised:", type(error).__name__, error)
