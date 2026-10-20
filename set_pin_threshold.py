"""Set the pin peak-decoder threshold (and optionally the peak kernel) in the live config."""
import json
import sys
from pathlib import Path

PATH = Path(r"D:/pcb5_submission/configs/unet_inference_v2.json")


def main() -> None:
    threshold = float(sys.argv[1])
    kernel = int(sys.argv[2]) if len(sys.argv) > 2 else None
    data = json.loads(PATH.read_text(encoding="utf-8"))
    data["pin_threshold"] = threshold
    if kernel is not None:
        data["pin_peak_kernel"] = kernel
    PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pin_threshold={data['pin_threshold']}  pin_peak_kernel={data['pin_peak_kernel']}")


if __name__ == "__main__":
    main()
