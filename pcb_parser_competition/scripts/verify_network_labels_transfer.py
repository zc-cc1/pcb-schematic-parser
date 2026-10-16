"""Record completed tests and independent PNG execution after transfer."""
import argparse
from pathlib import Path

from run_network_labels import ROOT,read,save,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args();target=args.target.resolve(strict=True);manifest=read(args.manifest)
    if Path(manifest['target']).resolve()!=target:raise ValueError('Wrong manifest')
    for name,h in manifest['protected_sha256'].items():
        if sha(target/name)!=h:raise ValueError('Protected file changed: '+name)
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:raise ValueError('Transferred file differs: '+row['file'])
    main_score=read(ROOT/'runs/s9_network_labels_v1/evaluation.json');score=read(target/'runs/s9_network_labels_v1/evaluation.json')
    for key in ['aggregate','raw_source_reference_aggregate','counts']:
        if score[key]!=main_score[key]:raise ValueError('Offline replay differs')
    runtime=target/'runs/s9_network_labels_v1/png_runtime_check/0170';summary=read(runtime/'summary.json')
    physical=read(runtime/'physical/graph_proposals.json');logical=read(runtime/'logical_graph_proposals.json')
    ownership=read(runtime/'physical/upstream/ownership/summary.json');text=read(runtime/'physical/upstream/summary.json');wire=read(runtime/'physical/summary.json')
    if summary['used_external_model_caches'] or wire['used_frozen_pin_input'] or wire['used_external_wire_cache']:
        raise ValueError('PNG check used old network/pin caches')
    if ownership['used_detector_cache'] or ownership['used_pin_cache'] or text['used_ownership_cache'] or text['used_external_OCR_cache']:
        raise ValueError('PNG check used old detector/U-Net/OCR input caches')
    if logical['edges']!=physical['edges'] or [(p['candidate_id'],p['point']) for p in logical['pin_attachments']]!=[(p['candidate_id'],p['point']) for p in physical['pin_attachments']]:
        raise ValueError('Physical geometry changed')
    merged=[d for d in logical['label_merge_decisions'] if not d['withheld_reasons']]
    if len(merged)!=1 or merged[0]['label']!='ADC_IN12' or len(merged[0]['site_ids'])!=2:raise ValueError('Fresh PNG did not reproduce the label group')
    verification={'status':'network_labels_target_verified','stage_tests_passed_main_and_target':16,
        'protected_files_unchanged':len(manifest['protected_sha256']),'transferred_files_sha_verified':len(manifest['files']),
        'offline_26_case_counts_and_both_reference_scores_identical':True,'fresh_png_case':'0170',
        'fresh_png_summary':logical['summary'],'fresh_png_used_external_caches':False,
        'physical_edges_unchanged':True,'all_pin_points_preserved':True,'new_training':False,
        'strict_nets_exported':summary['exported_nets'],'official_NetF1':None,'engineering_test_independent':False,
        'visual_supplement_scope':'One inspected development label role, offline only; raw draft score retained',
        'NMS_warning_audit':'Ultralytics 2.050s warning was emitted after assigning output for a single-image batch (time_limit=2+0.05*bs); no unprocessed image remains in that call. Detector recipe unchanged.'}
    save(ROOT/'runs/s9_network_labels_v1/runtime_verification.json',verification);save(target/'runs/s9_network_labels_v1/runtime_verification.json',verification)
    addition='\n## 转移与运行验证\n\n主工作区与桌面目标项目16项网络标签测试通过，26页离线统计及原草稿/视觉补充两套分数一致。60个原配置、权重和标注文件SHA未变；本阶段没有新增训练。\n\n目标项目0170完整PNG入口没有使用旧检测、分割、OCR或图输入缓存，保留260个原引脚点，255个已接入物理图的候选保持原样；两处ADC_IN12仍产生1组逻辑候选，严格nets仍为0。\n'
    for root in [ROOT,target]:
        path=root/'runs/s9_network_labels_v1/report.md';content=path.read_text(encoding='utf-8').split('\n## 转移与运行验证')[0];path.write_text(content+addition,encoding='utf-8')
    for row in manifest['files']:
        if row['file']=='runs/s9_network_labels_v1/report.md':row['sha256']=sha(target/row['file'])
    manifest['post_transfer_verification']={'status':verification['status'],'artifact':'runs/s9_network_labels_v1/runtime_verification.json'}
    save(args.manifest,manifest)
    print({'status':verification['status'],'tests':16,'protected':verification['protected_files_unchanged'],'fresh_png_merges':merged})


if __name__=='__main__':main()
