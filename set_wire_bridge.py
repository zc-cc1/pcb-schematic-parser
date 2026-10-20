"""Set the wire-graph gap-bridge parameters (0 = off)."""
import json
import sys
from pathlib import Path

PATH = Path(r"D:/pcb5_submission/configs/wire_graph_v1.json")


def main() -> None:
    gap = float(sys.argv[1])
    minimum = float(sys.argv[2]) if len(sys.argv) > 2 else None
    data = json.loads(PATH.read_text(encoding="utf-8"))
    data["bridge_max_gap_px"] = gap
    if minimum is not None:
        data["bridge_min_probability"] = minimum
    if gap > float(data.get("gap_max_distance_px") or 0):
        data["gap_max_distance_px"] = gap          # candidates are searched within this radius
    PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"bridge_max_gap_px={data['bridge_max_gap_px']}  bridge_min_probability={data['bridge_min_probability']}  "
          f"gap_max_distance_px={data['gap_max_distance_px']}")


if __name__ == "__main__":
    main()
