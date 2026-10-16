"""Read-only cached path checks and final report, never select a checkpoint."""
import json
import re
import numpy as np
from PIL import Image
from unet_audit_common import ROOT,OUT,read,write,sha,assert_protected
from pcb_parser.dataset_converter import load_source_annotation
from verify_unet_contrast import path_specs,path_records
from report_unet_negative_sampling_v4 import write_report


def main():
    run=ROOT/"runs/unet_negative_sampling_pilot_v4"
    result=read(run/"evaluation.json");assert result["status"]=="fixed16_paired_pilot_evaluation_complete"
    direct=read(ROOT/"runs/unet_v1_BN_diagnostic_v4/evaluation.json");assert direct["status"]=="original_v1_BN_diagnostic_complete"
    frozen=read(run/"protected_inputs.json")|read(run/"additional_source_attestation.json")
    assert all(sha(p)==v for p,v in frozen.items())
    protocol=read(run/"endpoint_protocol.json")
    assert sha(ROOT/"scripts/evaluate_unet_negative_sampling_v4.py")==protocol["scripts"]["evaluate_unet_negative_sampling_v4.py"]
    for name,v in result["weights_sha256"].items():assert sha(run/name/"last.pt")==v
    raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    gt=np.asarray(Image.open(ROOT/"data/segmentation_unet_v1/0036/wire_mask.png"))>0
    annotation,_=load_source_annotation(raw["0036"]["raw_annotation_path"])
    specs=[s for s in path_specs(annotation,gt.shape) if s["edge_id"] in ("E_1","E_8")]
    assert len(specs)==2 and {s["length"] for s in specs}=={90.,46.}
    masks={name:run/"evaluation/0036"/name/"wire_mask.png" for name in result["weights_sha256"]}
    masks["v1_original"]=ROOT/"runs/unet_v1_BN_diagnostic_v4/0036/v1_original_wire.png"
    masks["v1_BN_reestimated"]=ROOT/"runs/unet_v1_BN_diagnostic_v4/0036/v1_BN_reestimated_wire.png"
    checks={name:path_records(np.asarray(Image.open(path))>0,specs) for name,path in masks.items()}
    write(run/"local_path_checks.json",dict(case_id="0036",fixed_reference_edge_ids=["E_1","E_8"],models=checks,
        no_prediction_repair=True,predictions_reused=True,path_radius=2,min_length=10,reference_labels_certified=False,script_sha256=sha(__file__)))
    write_report(run,read(run/"configuration.json"),result)
    report=(run/"report.md").read_text(encoding="utf-8");paths=re.findall(r"\]\((C:/[^)]+)\)",report)
    from pathlib import Path
    assert all(Path(p).exists() for p in paths)
    assert len(re.findall(r"^## \d+\.",report,re.M))==8
    check=read(run/"final_checks.json")
    check.update(report_links_checked=len(paths),eight_sections=True,local_X1_paths_checked=True,
        original_v1_BN_diagnostic_complete=True,post_scoring_report_revision=True,
        prediction_script_unchanged_since_endpoint_protocol=True,
        original_BN_diagnostic_script_sha256=sha(ROOT/"scripts/diagnose_original_unet_BN_v4.py"),
        finalize_script_sha256=sha(__file__),protected_original_files=assert_protected())
    write(run/"final_checks.json",check)
    write(run/"report_revision.json",dict(reason="Post-scoring reporting includes common-BN recall/connectivity tradeoffs and completed original-v1 BN diagnostic; all numeric evaluation files and prediction scripts unchanged.",
        before_report_script_sha256=protocol["scripts"]["report_unet_negative_sampling_v4.py"],after_report_script_sha256=sha(ROOT/"scripts/report_unet_negative_sampling_v4.py"),
        endpoint_evaluation_sha256=sha(run/"evaluation.json"),original_BN_evaluation_sha256=sha(ROOT/"runs/unet_v1_BN_diagnostic_v4/evaluation.json")))
    print(json.dumps(check,ensure_ascii=False))


if __name__=="__main__":main()
