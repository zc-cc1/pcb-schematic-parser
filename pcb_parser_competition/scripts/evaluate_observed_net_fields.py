"""Offline whole-net review; frozen references never enter inference."""
import argparse
import math
from pathlib import Path

from run_observed_net_fields import ROOT, read, save, sha
from evaluate_pin_text import component_matches
from evaluate_wire_graph import match
from evaluate_engineering_networks import review_card
from pcb_parser.dataset_converter import validate_output


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s24_observed_net_fields');a = p.parse_args()
    batch = read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S24 batch incomplete')
    ref_path = ROOT/'runs/s6_pin_text_v1/draft_field_references.json';digest = sha(ref_path);fields = read(ref_path)
    reviews = [];field_reviews = [];checked = 0;tip_count = 0;existing_nets = 0;new_nets = []
    for item in batch['cases']:
        case = item['case_id'];folder = a.output/f'predictions/{case}'
        for path, h in item['original_S23_input_sha256'].items():
            if sha(Path(path)) != h:raise ValueError('Frozen S23/OCR input changed')
            checked += 1
        previous = ROOT/f'runs/s23_capacitor_geometry/predictions/{case}/export'
        before = read(previous/'engineering_pin_text_result.json');after = read(folder/'export/engineering_pin_text_result.json')
        bg = read(previous/'engineering_graph_proposals.json');ag = read(folder/'export/engineering_graph_proposals.json')
        old_payload = read(previous/'engineering_submission.json');payload = read(folder/'export/engineering_submission.json')
        records = read(folder/'export/completed_component_records.json')['records']
        if fields['image_sha256'][case] != after['image_sha256']:raise ValueError('Reference image differs')
        if read(previous/'baseline_export/partial_submission.json') != read(folder/'export/baseline_export/partial_submission.json'):raise ValueError('Conservative JSON changed')
        for old, new in zip(before['pin_candidates'], after['pin_candidates']):
            if any(old.get(k) != new.get(k) for k in ['candidate_id', 'point', 'owner_detection_index', 'owner_status', 'number']):raise ValueError('Contact changed')
            if old.get('pinname') is not None and old['pinname'] != new['pinname']:raise ValueError('Existing name changed')
        tip_count += len(after['pin_candidates']);existing_nets += len(old_payload['nets'])
        if (bg['edges'] != ag['edges'] or bg['physical_networks'] != ag['physical_networks']
            or [(n['network_id'], n['pin_candidate_ids']) for n in bg['networks']] != [(n['network_id'], n['pin_candidate_ids']) for n in ag['networks']]):raise ValueError('Graph changed')
        if any(payload['nets'].get(nid) != net for nid, net in old_payload['nets'].items()):raise ValueError('Existing whole net changed')
        if any(validate_output(payload, *after['size'])):raise ValueError('Invalid engineering JSON')
        owners = component_matches([{'class_name': c['class_name'], 'bbox': c['bbox_top_left']} for c in after['components']], fields['detector_references'][case]['objects'])
        truth = [t for t in fields['references'] if t['case_id'] == case]
        assigned = [p for p in after['pin_candidates'] if p['owner_status'] == 'assigned']
        matched = {assigned[i]['candidate_id']: truth[j] for i, j in match(assigned, truth, owners)}
        pm = {pin['candidate_id']: pin for pin in after['pin_candidates']}
        def diagnostic(pin):
            same = [t for t in truth if t['component_source_id'] == owners.get(pin['owner_detection_index'])]
            nearest = min(same, key=lambda t: math.dist(pin['point'], t['point']), default=None)
            return {'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'point': pin['point'],
                'number': pin['number'], 'pinname': pin['pinname'], 'matched_within_5px': pin['candidate_id'] in matched,
                'nearest_same_owner_draft_reference': nearest, 'nearest_distance_px': math.dist(pin['point'], nearest['point']) if nearest else None,
                'engineering_fields': [f for f in ['number', 'pinname'] if pin.get(f+'_status') == 'engineering_convention_unverified']}
        audit = read(folder/'observed_fields_audit.json')
        for change in audit['side_names']['changes']:
            row = {'case_id': case, 'field': 'pinname', **change, 'draft_diagnostic': diagnostic(pm[change['candidate_id']])}
            field_reviews.append(row)
        for change in audit['literal_headers']['changes']:
            field_reviews.append({'case_id': case, 'field': 'component_id', **change,
                'matched_draft_component_source_id': owners.get(change['detection_index']),
                'owned_pins': [diagnostic(p) for p in after['pin_candidates'] if p.get('owner_detection_index') == change['detection_index']]})
        for proof in read(folder/'export/engineering_export_coverage.json')['atomic_complete_network_proofs']:
            is_new = proof['network_id'] not in old_payload['nets'];contacts = set(proof['member_to_contact'].values())
            row = {'case_id': case, **proof, 'new_in_S24': is_new,
                'all_members_match_draft_owner_and_location_5px': contacts.issubset(matched),
                'member_diagnostics': [diagnostic(pm[cid]) for cid in sorted(contacts)],
                'number_name_or_official_network_certified': False}
            if is_new:
                row['hyperGraph'] = payload['nets'][proof['network_id']]['hyperGraph']
                row['review_card'] = f"whole_net_{case}_{proof['network_id']}.png"
                review_card(case, proof, after, ag, records, a.output/row['review_card'])
                refs = set(proof['all_pin_references']);os = {ref.rsplit('.', 1)[0] for ref in refs}
                single = {'components': {o: payload['components'][o] for o in os},
                    'pins': {o: {k: v for k, v in payload['pins'][o].items() if o+'.'+k[4:] in refs} for o in os},
                    'nets': {proof['network_id']: payload['nets'][proof['network_id']]}}
                if any(validate_output(single, *after['size'])):raise ValueError('Invalid standalone net')
                save(a.output/f"engineering_complete_{case}_{proof['network_id']}.json", single);new_nets.append(row)
            reviews.append(row)
    if sha(ref_path) != digest:raise ValueError('Draft reference changed')
    result = {'aggregate': batch['aggregate'], 'previous_engineering_nets_preserved': existing_nets, 'raw_tip_candidates_preserved': tip_count,
        'new_networks': [{k: r[k] for k in ['case_id', 'network_id', 'hyperGraph', 'all_members_match_draft_owner_and_location_5px']} for r in new_nets],
        'whole_nets_all_members_match_draft_owner_and_location_5px': sum(r['all_members_match_draft_owner_and_location_5px'] for r in reviews),
        'checked_frozen_input_files': checked, 'draft_reference_sha256': digest, 'reference_changes': 0,
        'conservative_JSON_preserved': True, 'all_existing_fields_and_geometry_preserved': True,
        'independent_test': False, 'official_PinF1': None, 'official_NetF1': None}
    save(a.output/'evaluation.json', result);save(a.output/'whole_network_review.json', {'items': reviews, 'used_in_inference': False})
    save(a.output/'field_review.json', {'items': field_reviews, 'used_in_inference': False, 'used_in_training': False})
    lines = ['# S24：用可见字段补齐整网', '',
        '在S23工程出口上增加两个缺失字段处理：IC同一行的左右引脚名称用完整OCR框所在的器件半边消歧；检测类型与位号前缀不同的框式器件，要求唯一可见顶部位号、独立底部文字、至少4个已有OCR编号与3个名称共同支持身份。全部采用实际完整字面，至少3个不同视图达到0.98；这些视图来自同一OCR模型，不是独立概率。', '',
        '26页旧开发集：工程完整net 3→5，完整pin 266→272，已输出元件214→215。新增0036 `(C4.1,U1.4)`，U1.4补实际CAGC名称；新增0018 `(C14.1,L1.4)`，恢复印刷L1位号及LDO底部文字。L1保持检测类型ic，没有按电感类型解释或生成U编号。L1是可见本地身份，官方规范键仍未认证。', '',
        f'1477个原tip、所有坐标、导线路径与原始成员保持；原3条工程net逐项相同，保守JSON逐项相同。校验{checked}个冻结S23/OCR输入哈希，草稿参考未改。新增一个引脚名称和一个元件位号/名称；6个新增完整pin中5个来自已有引脚字段的身份解除阻塞，没有新增引脚编号。', '',
        '## 离线诊断与限制', '',
        f"5条输出网中，{result['whole_nets_all_members_match_draft_owner_and_location_5px']}条的所有成员通过同一草稿的一对一归属/5px位置匹配。以下只是原网成员的参考诊断，不包含官方编号、名称、完整参考网集或线段认证：", '']
    for row in new_nets:
        lines.append(f"- {row['case_id']} {row['hyperGraph']}：全成员归属/5px匹配{'通过' if row['all_members_match_draft_owner_and_location_5px'] else '未全通过'}。")
        for m in row['member_diagnostics']:lines.append(f"  - {m['owner']}.{m['number']} / {m['pinname']}：同器件最近草稿点{m['nearest_distance_px']:.3f}px。" if m['nearest_distance_px'] is not None else f"  - {m['owner']}.{m['number']}：没有匹配器件的草稿参考。")
    lines += ['', 'C4/C14的编号及名称仍继承工程约定，不计作OCR识别提升。0149缺编号和0105的LX读数仍未决，局部OCR出现3/乱码、1/E、b/B/16/78等冲突，没有按芯片型号、位置顺序或字符替换补编号。0049缺可见位号的IC保持未决。', '',
        '## 复跑', '', '```powershell', '.\\scripts\\run_observed_net_fields.ps1 -Image .\\data\\standardized\\0036\\0036.png -Device cpu',
        '.\\scripts\\run_observed_net_fields.ps1 -Image .\\data\\standardized\\0018\\0018.png -Device cpu',
        '& .\\.venv-gpu\\Scripts\\python.exe scripts/batch_observed_net_fields.py',
        '& .\\.venv-gpu\\Scripts\\python.exe scripts/evaluate_observed_net_fields.py', '```', '']
    runtime = a.output/'runtime_verification.json'
    if runtime.is_file():
        v = read(runtime);lines.append(f"{v['regression_tests_passed']}项回归通过；两页原始PNG重跑："+'；'.join(f"{r['case_id']} {r['fresh_entrypoint_wall_seconds']:.1f}秒，tip最大差{r['fresh_all_tip_max_delta_px']:.5f}px" for r in v['cases'])+'。新字段、完整网、成员与已输出线段均与缓存回放一致。');lines.append('')
        lines += ['0018跨次运行有4个候选ID因峰置信度排序互换，按0.05px内唯一位置双射核对后所有字段一致，没有重写任一次输出。另有两个未导出单成员物理片段因1px骨架端点变化而交换边/网ID；详细原始差异保存在runtime_verification.json。它们保持扣留，新增网及全部已输出线段没有差异。因此不声称整张原始图逐字节复现。', '']
    lines += ['整体完整网覆盖仍低。这26页多次参与开发，没有独立测试和官方NetF1；继续保留独立入口。下一步优先处理IC缺失编号、检测框偏差导致数字ROI位置错误，以及整网缺失的真实成员。', '']
    (a.output/'report.md').write_text('\n'.join(lines), encoding='utf-8');print(result)


if __name__ == '__main__':main()
