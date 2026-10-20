"""Set wire_threshold consistently in the two deployment configs that carry it."""
import json
import sys
from pathlib import Path

ROOT = Path(r"D:/pcb5_submission/configs")
TARGETS = ("wire_graph_v1.json", "unet_inference_v2.json")


def main() -> None:
    value = float(sys.argv[1])
    for name in TARGETS:
        path = ROOT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        if "wire_threshold" not in data:
            print(f"{name}: no wire_threshold key, skipped")
            continue
        data["wire_threshold"] = value
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: wire_threshold = {value}")


if __name__ == "__main__":
    main()
