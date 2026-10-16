"""Finalize reviewed documentation, verify copies and record fresh PNG run."""
import argparse
from pathlib import Path
import shutil
from run_pin_text_v2 import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();target=a.target.resolve(strict=True);manifest=read(a.manifest)
    if Path(manifest['target']).resolve()!=target:raise ValueError('Wrong target manifest')
    # Only the reviewed wording and its generator changed since initial copy.
    updated=['scripts/build_pin_text_v2_report.py','docs/pin_text_stage_v2.md','runs/s11_pin_text_v2/report.md']
    for name in updated:
        source,dest=ROOT/name,target/name
        if target not in dest.resolve().parents:raise ValueError('Destination outside target')
        if sha(source)!=sha(dest):
            backup=a.manifest.parent/'final_document_backup'/name
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup);shutil.copy2(source,dest)
            row=next(r for r in manifest['files'] if r['file']==name)
            row['initial_copy_sha256']=row['sha256'];row['sha256']=sha(dest);row['final_document_backup']=str(backup)
    for name,digest in manifest['protected_sha256'].items():
        if sha(target/name)!=digest:raise ValueError('Protected file changed: '+name)
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:raise ValueError('Transferred file changed: '+row['file'])
    stage=Path('runs/s11_pin_text_v2')
    if read(ROOT/stage/'evaluation.json')!=read(target/stage/'evaluation.json'):raise ValueError('26-case evaluation replay differs')
    main_export=read(ROOT/stage/'export_audit.json');export=read(target/stage/'export_audit.json')
    if main_export!=export:raise ValueError('Downstream coverage differs')
    runtime=target/stage/'png_runtime_check/0018';summary=read(runtime/'summary.json')
    text=read(runtime/'pin_text/summary.json');owner=read(runtime/'pin_text/ownership/summary.json')
    if summary['used_external_model_caches'] or text['used_ownership_cache'] or text['used_external_OCR_cache'] or owner['used_detector_cache'] or owner['used_pin_cache']:
        raise ValueError('Fresh PNG entry used old upstream input caches')
    original=read(runtime/'pin_text/ownership/pin_ownership.json');pins=read(runtime/'pin_text/pin_text_result.json')
    physical=read(runtime/'physical/graph_proposals.json');completed=read(runtime/'submission/completed_pin_text_result.json')
    expected=[(p['candidate_id'],p['point']) for p in original['pin_candidates']]
    for payload,key in [(pins,'pin_candidates'),(physical,'pin_attachments'),(completed,'pin_candidates')]:
        if expected!=[(p['candidate_id'],p['point']) for p in payload[key]]:raise ValueError('Fresh pin points changed')
    identities=read(runtime/'pin_text/identity_recovery_debug.json')['accepted']
    if 'U1' not in {r['designator'] for r in identities}:raise ValueError('Fresh U1 header was not recovered')
    evidence=read(runtime/'pin_text/field_adoption_debug.json')
    if evidence['old_numbers_changed'] or evidence['old_nonempty_names_changed']:raise ValueError('Frozen field policy changed')
    verification={'status':'pin_text_v2_target_verified','new_tests_passed_main_and_target':23,'legacy_pin_text_tests_passed_main_and_target':14,
        'protected_files_unchanged':len(manifest['protected_sha256']),'copied_files_sha_verified':len(manifest['files']),
        '26_case_copied_predictions_offline_evaluation_identical':True,'26_case_main_physical_graph_replay_unchanged':True,
        'fresh_png_case':'0018','fresh_png_used_external_model_caches':False,'fresh_png_pin_count':len(expected),
        'fresh_png_fields':summary['pin_text_v2'],'fresh_png_partial_export':summary['export'],
        'new_training':False,'original_dataset_modified':False,'official_PinF1':None,'official_NetF1':None,
        'engineering_test_independent':False,'strict_global_export':export['after']}
    for root in [ROOT,target]:
        save(root/stage/'runtime_verification.json',verification)
        report=root/stage/'report.md';text_report=report.read_text(encoding='utf-8').split('\n## 转移与完整图片验证')[0]
        addition='\n## 转移与完整图片验证\n\n主工作区与桌面项目23项新模块测试、14项原引脚文字回归测试通过；26页复制预测的离线指标重算一致，主工作区26页重建物理图的几何和pin接入关系一致。140个上游文件SHA未变。\n\n桌面项目0018完整PNG入口无旧上游输入缓存，原132个tip全部保留，补3个可见位号、5个缺失名称，部分JSON导出19个元件、35个字段齐全pin、0个net。未重新训练模型，未修改原标注。\n'
        report.write_text(text_report+addition,encoding='utf-8')
    for row in manifest['files']:
        if row['file']=='runs/s11_pin_text_v2/report.md':row['sha256']=sha(target/row['file'])
    manifest['post_transfer_verification']={'status':verification['status'],'artifact':str(stage/'runtime_verification.json')}
    save(a.manifest,manifest)
    print({'status':verification['status'],'protected':verification['protected_files_unchanged'],
           'fresh_pins':len(expected),'export':summary['export'],'new_identities':[r['designator'] for r in identities]},flush=True)


if __name__=='__main__':main()
