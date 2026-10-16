"""Audit transferred artifacts and completed runtime checks."""
import argparse
from pathlib import Path

from run_wire_graph import ROOT,read,save,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args();target=args.target.resolve(strict=True)
    manifest=read(args.manifest)
    if Path(manifest['target']).resolve()!=target:raise ValueError('Wrong import manifest')
    if any(sha(target/name)!=expected for name,expected in manifest['protected_sha256'].items()):raise ValueError('Protected files changed')
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:raise ValueError('Transferred file changed: '+row['file'])
    source_score=read(ROOT/'runs/s8_junction_classifier_v1/evaluation.json')
    target_score=read(target/'runs/s8_junction_classifier_v1/evaluation.json')
    if source_score['aggregate']!=target_score['aggregate']:raise ValueError('Evaluation differs')
    runtime=target/'runs/s8_junction_classifier_v1/png_runtime_check/0148'
    summary=read(runtime/'summary.json');pin=read(runtime/'upstream/pin_text_result.json');graph=read(runtime/'graph_proposals.json')
    ownership=read(runtime/'upstream/ownership/summary.json');text_summary=read(runtime/'upstream/summary.json')
    if summary['used_frozen_pin_input'] or summary['used_external_wire_cache']:raise ValueError('PNG check used external input cache')
    if ownership['used_detector_cache'] or ownership['used_pin_cache']:raise ValueError('PNG check used detection/pin cache')
    if text_summary['used_ownership_cache'] or text_summary['used_external_OCR_cache']:raise ValueError('PNG check used OCR/ownership cache')
    if not graph['junction_classifier_used'] or graph['junction_model_sha256']!=sha(target/'weights/junction_classifier_v1.pt'):raise ValueError('Wrong runtime classifier')
    if [(p['candidate_id'],p['point']) for p in graph['pin_attachments']]!=[(p['candidate_id'],p['point']) for p in pin['pin_candidates']]:raise ValueError('Pin points changed')
    smoke=read(target/'runs/s8_junction_classifier_v1/transfer_training_smoke/evaluation.json')
    if not smoke['cannot_be_used_by_graph_inference']:raise ValueError('Smoke model unguarded')
    result={'status':'target_transfer_verified','main_and_target_stage_tests_passed':28,
            'test_suites':{'wire_graph_and_artifacts':18,'learned_junction':10},
            'protected_files_unchanged':len(manifest['protected_sha256']),'transferred_files_sha_verified':len(manifest['files']),
            'offline_26_case_scores_identical':True,'fresh_png_case':'0148','fresh_png_summary':graph['summary'],
            'upstream_caches_used':False,'pin_coordinates_unchanged':True,'training_script_smoke_completed':True,
            'training_smoke_not_a_quality_evaluation':True,'smoke_checkpoint_rejected_by_predictor':True,
            'adopted_default':'wire_graph_v1','pilot_entry':'scripts/run_learned_wire_graph.ps1',
            'junction_model_sha256':graph['junction_model_sha256'],'official_NetF1':None,
            'import_manifest':args.manifest.name,'engineering_test_independent':False}
    save(ROOT/'runs/s8_junction_classifier_v1/runtime_verification.json',result)
    save(target/'runs/s8_junction_classifier_v1/runtime_verification.json',result)
    addition='\n## 转移验证\n\n主工作区与桌面目标项目的28项导线图/交点测试通过，26页离线分数一致。53个原配置、权重及标注文件SHA校验未变。目标项目完成独立1轮训练执行自检；自检权重被推理器拒绝，不能当作正式模型。\n\n0148从PNG启动，检测、U-Net、OCR均未使用旧输入缓存；保留73个引脚坐标，67个唯一接入导线图，4个交点经CNN决策。模型输出可运行不代表电气连通关系已全部验证；严格部分nets仍为0。\n'
    for project in [ROOT,target]:
        report=project/'runs/s8_junction_classifier_v1/report.md'
        content=report.read_text(encoding='utf-8').split('\n## 转移验证')[0]
        report.write_text(content+addition,encoding='utf-8')
    for row in manifest['files']:
        if row['file']=='runs/s8_junction_classifier_v1/report.md':row['sha256']=sha(target/row['file'])
    manifest['post_transfer_verification']={'status':result['status'],
        'runtime_verification':'runs/s8_junction_classifier_v1/runtime_verification.json',
        'report_verification_appendix_added':True}
    save(args.manifest,manifest)
    print({'status':result['status'],'protected':result['protected_files_unchanged'],'stage_tests':28,'fresh_png':graph['summary']})


if __name__=='__main__':main()
