"""Install a U-Net checkpoint into the deployment copy and keep every hash guard in sync.

Usage:  python set_unet.py <checkpoint.pt>
        python set_unet.py v1        (restores weights/unet_wire_pin_v1.pt.bak)
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:/pcb5_submission")
WEIGHTS = ROOT / "weights/unet_wire_pin_v1.pt"
BACKUP = ROOT / "weights/unet_wire_pin_v1.pt.bak"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    arg = sys.argv[1]
    source = BACKUP if arg == "v1" else Path(arg)
    if not source.is_file():
        raise SystemExit(f"missing checkpoint: {source}")
    if not BACKUP.exists() or sha(BACKUP) == sha(WEIGHTS):
        shutil.copy2(WEIGHTS, BACKUP)          # keep the as-shipped v1 checkpoint once
    shutil.copy2(source, WEIGHTS)
    digest = sha(WEIGHTS)

    cfg_path = ROOT / "configs/unet_inference_v2.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["checkpoint_sha256"] = digest
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    man_path = ROOT / "package_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for value in man.values():
        if isinstance(value, dict) and "weights/unet_wire_pin_v1.pt" in value:
            value["weights/unet_wire_pin_v1.pt"] = {"bytes": WEIGHTS.stat().st_size, "sha256": digest}
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"installed {source} -> {WEIGHTS}")
    print(f"sha256 {digest}")
    print("config  ", json.loads(cfg_path.read_text(encoding='utf-8'))["checkpoint_sha256"] == digest)
    print("manifest", any(isinstance(v, dict) and v.get("weights/unet_wire_pin_v1.pt", {}).get("sha256") == digest
                         for v in man.values()))


if __name__ == "__main__":
    main()
