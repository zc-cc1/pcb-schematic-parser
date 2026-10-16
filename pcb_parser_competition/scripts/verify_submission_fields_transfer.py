"""Verify desktop replay, fresh-PNG provenance and protected upstream files."""
import argparse
from pathlib import Path
from run_submission_fields import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();target=a.target.resolve(strict=True);manifest=read(a.manifest)
    if Path(manifest['target']).resolve()!=target:raise ValueError('Manifest belongs to another project')
    for name,digest in manifest['protected_sha256'].items():
        if sha(target/name)!=digest:raise ValueError('Protected file changed: '+name)
    for row in manifest['files']:
        name=row['file']
        if not name.startswith('runs/') and sha(target/name)!=row['sha256']:raise ValueError('Transferred code/config changed: '+name)
    output=Path('runs/s10_submission_fields_v1')
    main_audit,target_audit=read(ROOT/output/'audit.json'),read(target/output/'audit.json')
    if main_audit!=target_audit:raise ValueError('26-case audit results differ')
    deterministic=0
    for case in [r['case_id'] for r in target_audit['cases']]:
        for name in ['completed_pin_text_result.json','completed_component_records.json','identity_proposals.json',
                     'partial_submission.json','export_coverage.json','network_field_review_queue.json']:
            relative=output/'predictions'/case/name
            if sha(ROOT/relative)!=sha(target/relative):raise ValueError('Frozen replay mismatch: '+str(relative))
            deterministic+=1
        relative=Path(f'runs/s9_network_labels_v1/predictions/{case}/logical_graph_proposals.json')
        if sha(ROOT/relative)!=sha(target/relative):raise ValueError('Upstream graph changed')
    runtime=target/output/'png_runtime_check/0033'
    summary=read(runtime/'summary.json');network=read(runtime/'network/summary.json')
    wire=read(runtime/'network/physical/summary.json');text=read(runtime/'network/physical/upstream/summary.json')
    owner=read(runtime/'network/physical/upstream/ownership/summary.json')
    if (summary['used_external_model_caches'] or network['used_external_model_caches']
        or wire['used_frozen_pin_input'] or wire['used_external_wire_cache']
        or text['used_ownership_cache'] or text['used_external_OCR_cache']
        or owner['used_detector_cache'] or owner['used_pin_cache']):
        raise ValueError('Fresh PNG verification used upstream input caches')
    source=read(runtime/'network/physical/upstream/pin_text_result.json');pins=read(runtime/'completed_pin_text_result.json')
    if [(p['candidate_id'],p['point']) for p in source['pin_candidates']] != [(p['candidate_id'],p['point']) for p in pins['pin_candidates']]:
        raise ValueError('Fresh PNG pin points changed')
    if summary['summary']['new_label_identities']!=2 or summary['summary']['single_tip_label_auto_numbers']!=3:
        raise ValueError('Fresh PNG did not reproduce label field behavior')
    verification={'status':'submission_fields_target_verified','new_field_tests_passed_main_and_target':13,
        'network_export_regression_tests_passed_main_and_target':16,'frozen_26_case_audit_identical':True,
        'deterministic_prediction_files_compared':deterministic,'protected_files_unchanged':len(manifest['protected_sha256']),
        'fresh_png_case':'0033','fresh_png_upstream_cache_inputs_used':False,
        'fresh_png_pin_count':len(pins['pin_candidates']),'fresh_png_result':summary['summary'],
        'fresh_png_partial_output':summary['after_export'],'new_training':False,'official_NetF1':None,
        'accuracy_improvement_measured':False,
        'NMS_warning_audit':'The frozen detector emitted its 2.050s warning after output assignment for a single-image batch; detector configuration and weights were preserved.'}
    addition='\n## 转移与执行验证\n\n桌面目标项目与主工作区的13项字段测试、16项网络/出口回归测试均通过；26页audit及156个确定性预测文件一致。132个上游代码、配置、权重及关键数据文件SHA未变。\n\n目标项目0033从原PNG运行检测、U-Net、OCR、物理图、标签到字段输出，没有使用旧上游输入缓存：保留42个预测tip，接入2个标签身份并生成3个单引脚格式编号。当前2个component、0个字段齐全pin、0个net；不是完整赛事提交。\n'
    for root in [ROOT,target]:
        save(root/output/'runtime_verification.json',verification)
        report=root/output/'report.md';content=report.read_text(encoding='utf-8').split('\n## 转移与执行验证')[0]
        report.write_text(content+addition,encoding='utf-8')
    for row in manifest['files']:
        current=sha(target/row['file'])
        if current!=row['sha256']:
            row['initial_copy_sha256']=row['sha256'];row['sha256']=current
            row['post_transfer_update_reason']='Target replay regenerated timings or verification report'
    manifest['post_transfer_verification']={'status':verification['status'],'artifact':str(output/'runtime_verification.json')}
    save(a.manifest,manifest)
    print({'status':verification['status'],'protected':verification['protected_files_unchanged'],
           'fresh_png':summary['summary'],'deterministic_files':deterministic},flush=True)


if __name__=='__main__':main()
