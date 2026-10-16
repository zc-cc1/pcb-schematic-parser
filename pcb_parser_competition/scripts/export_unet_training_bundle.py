"""Merge only U-Net files into the user-named YOLO project, with backups/hashes."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    "src/pcb_parser/unet_model.py","src/pcb_parser/unet_data.py",
    "src/pcb_parser/vendor/pytorch_unet/__init__.py","src/pcb_parser/vendor/pytorch_unet/unet_parts.py",
    "src/pcb_parser/vendor/pytorch_unet/unet_model.py","src/pcb_parser/vendor/pytorch_unet/LICENSE",
    "src/pcb_parser/vendor/pytorch_unet/provenance.json",
    "scripts/train_unet.py","scripts/run_unet.py","scripts/prepare_unet_dataset.py",
    "scripts/import_unet_source.py","scripts/verify_unet_runtime.py","scripts/run_unet_training.ps1",
    "configs/unet_wire_pin.json","configs/unet_label_reviews.json","tests/test_unet.py",
    "docs/unet_wire_pin_plan.md","docs/unet_training_handoff.md"
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--target",type=Path,required=True)
    args=p.parse_args()
    target=args.target.resolve(strict=True)
    if target==ROOT or not (target/"scripts/train_detector.py").is_file():
        raise ValueError("Expected an existing separate YOLO project")
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    archive=target/f"data/local_imports/unet_{stamp}"
    archive.mkdir(parents=True)
    protected={str(path.relative_to(target)):sha(path) for directory,pattern in
               [("src","*.py"),("scripts","*.py"),("configs","*.json")]
               for path in (target/directory).rglob(pattern) if str(path.relative_to(target)).replace("\\","/") not in FILES}
    manifest={"source_root":str(ROOT),"target_root":str(target),"formal_training_started":False,"files":[]}
    for relative in FILES:
        source=ROOT/relative
        destination=target/relative
        if not destination.resolve().is_relative_to(target):
            raise ValueError(f"Destination resolves outside target: {destination}")
        digest=sha(source)
        action="new"
        backup=None
        if destination.exists():
            if sha(destination)==digest:
                action="unchanged"
            else:
                action="replaced_with_backup"
                backup=archive/"prior_workspace_files"/relative
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(destination,backup)
        if action!="unchanged":
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
        assert sha(destination)==digest
        manifest["files"].append({"path":relative,"sha256":digest,"action":action,"backup":str(backup) if backup else None})
    readme=target/"README.md"
    text=readme.read_text(encoding="utf-8")
    if "## U-Net训练模块（合并交接）" not in text:
        backup=archive/"prior_workspace_files/README.md"
        backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(readme,backup)
        addition="""

## U-Net训练模块（合并交接）

已合并双输出U-Net导线分割/引脚tip热图训练模块，与现有YOLO源码、配置和权重并存。运行入口为 `scripts/run_unet_training.ps1`，接手agent先阅读 [U-Net训练交接说明](docs/unet_training_handoff.md)。

在本工程根目录执行 `./scripts/run_unet_training.ps1 -Mode Check` 检查路径与环境；显卡空闲后，可使用 `-Mode Train -AllowDraft -Device cuda:0` 启动草稿标签实验。未人工复核的标签不称为正式Golden；检查和CPU短跑不代表训练质量。详细配置与数据状态见交接说明。
"""
        readme.write_text(text+addition,encoding="utf-8")
        manifest["readme_backup"]=str(backup)
    changes=[relative for relative,digest in protected.items() if sha(target/relative)!=digest]
    if changes:
        raise ValueError(f"Existing YOLO/shared files changed during merge: {changes}")
    manifest["existing_yolo_and_shared_code_configs_preserved"]=True
    manifest["protected_existing_file_count"]=len(protected)
    result=archive/"import_manifest.json"
    result.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"artifacts/unet_latest_transfer.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"merged","files":len(FILES),"target":str(target),"manifest":str(result),
                      "protected_existing_files":len(protected)},ensure_ascii=False))


if __name__=="__main__":
    main()
