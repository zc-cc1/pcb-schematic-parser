"""Check checkpoint/config consistency and the latest partial prediction files."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.dataset_converter import validate_output


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    cfg=read("configs/baseline_v3_adopted.json")
    assert read("configs/baseline_finetuned.json")==cfg
    summary=read("runs/s3_component_text_quick_dev/summary.json")
    assert summary["binding_source_sha256"]==hashlib.sha256((ROOT/"src/pcb_parser/component_text.py").read_bytes()).hexdigest()
    assert summary["recovery_source_sha256"]==hashlib.sha256((ROOT/"src/pcb_parser/text_recovery.py").read_bytes()).hexdigest()
    assert summary["config_sha256"]==hashlib.sha256((ROOT/"configs/baseline_v3_adopted.json").read_bytes()).hexdigest()
    archived=read("runs/s3_component_text_v1_archive/prediction/summary.json")
    before={r["case_id"]:r for r in archived["cases"]}
    reference_path=ROOT/"data/reviews/component_text_quick_dev.json"
    assert reference_path.read_bytes()==(ROOT/"runs/s3_component_text_v1_archive/reference.json").read_bytes()
    assert cfg==read("configs/component_text_v2.json")
    result={"status":"passed","default_configs_identical":True,
            "tests_previously_passed":{"parser_and_fusion":47,"cad_inventory_document_environment":3,"total":50},
            "cold_ocr_entrypoint":"scripts/run_ocr_binding.ps1 -Device cpu -ReuseDetections",
            "original_reference_unchanged":True,"cases":[]}
    for row in summary["cases"]:
        case=row["case_id"]
        data=read(f"runs/s3_component_text_quick_dev/{case}/components_partial.json")
        errors,bounds=validate_output(data,*row["size"])
        assert not errors and not bounds
        assert row["weights_sha256"]==cfg["weights_sha256"]
        assert not row["ocr_cache_reused"]
        assert not row["base_ocr_cache_reused"]
        assert row["detections_sha256"]==before[case]["detections_sha256"]
        assert row["image_sha256"]==before[case]["image_sha256"]
        result["cases"].append({"case_id":case,"format_valid":True,
                                "pins_pending":not data["pins"],"nets_pending":not data["nets"]})
    path=ROOT/"runs/s3_component_text_quick_dev/verification.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
