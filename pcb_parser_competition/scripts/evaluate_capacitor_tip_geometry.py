"""Offline fixed-reference and whole-net diagnostics for the S23 pilot."""
import argparse
from collections import Counter
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_capacitor_tip_geometry import ROOT, read, save, sha
from evaluate_pin_ownership import joint_match
from evaluate_pin_text import component_matches
from evaluate_wire_graph import match, metrics
from evaluate_engineering_networks import review_card
from pcb_parser.dataset_converter import validate_output


def point_card(case, change, reference, output):
    box = [max(0, int(min(change['old_point'][0], change['new_point'][0], reference['point'][0]))-24),
        max(0, int(min(change['old_point'][1], change['new_point'][1], reference['point'][1]))-24),
        int(max(change['old_point'][0], change['new_point'][0], reference['point'][0]))+25,
        int(max(change['old_point'][1], change['new_point'][1], reference['point'][1]))+25]
    with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
        image = source.convert('RGB').crop(box);image = image.resize((image.width*6, image.height*6), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (max(550, image.width), image.height+140), 'white');canvas.paste(image, (0, 38))
    d = ImageDraw.Draw(canvas);font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 14)
    d.text((8, 8), f"{case} {change['component_id']} {change['candidate_id']}", fill='#222', font=font)
    for point, label, color in [(change['old_point'], 'Native', '#bb2424'), (change['new_point'], '2x model', '#00834a'), (reference['point'], 'Draft ref', '#225ec4')]:
        x, y = (point[0]-box[0])*6, 38+(point[1]-box[1])*6
        d.ellipse((x-5, y-5, x+5, y+5), outline=color, width=2)
    d.text((8, image.height+49), f"Distance: {math.dist(change['old_point'], reference['point']):.3f} -> {math.dist(change['new_point'], reference['point']):.3f}px | 5px unchanged", fill='#333', font=font)
    for x, label, color in [(8, 'Red: native', '#bb2424'), (165, 'Green: 2x model', '#00834a'), (345, 'Blue: draft', '#225ec4')]:
        d.text((x, image.height+74), label, fill=color, font=font)
    d.text((8, image.height+99), 'Reference used only in offline review; no point snapping.', fill='#8c4320', font=font)
    canvas.save(output)


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s23_capacitor_geometry');a = p.parse_args()
    batch = read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S23 batch incomplete')
    ref_path = ROOT/'runs/s6_pin_text_v1/draft_field_references.json';digest = sha(ref_path);fields = read(ref_path)
    totals = {'before': Counter(), 'after': Counter()};changes = [];net_reviews = [];checked = 0
    for item in batch['cases']:
        case = item['case_id'];folder = a.output/f'predictions/{case}'
        for path, h in item['original_S21_input_sha256'].items():
            if sha(Path(path)) != h:raise ValueError('Frozen S21 input changed')
            checked += 1
        before_folder = ROOT/f'runs/s22_engineering_networks/predictions/{case}'
        before = read(before_folder/'engineering_pin_text_result.json');after = read(folder/'export/engineering_pin_text_result.json')
        bg = read(before_folder/'engineering_graph_proposals.json');ag = read(folder/'export/engineering_graph_proposals.json')
        old_payload = read(before_folder/'engineering_submission.json');payload = read(folder/'export/engineering_submission.json')
        records = read(folder/'export/completed_component_records.json')['records']
        if fields['image_sha256'][case] != after['image_sha256']:raise ValueError('Reference image differs')
        owners = component_matches([{'class_name': c['class_name'], 'bbox': c['bbox_top_left']} for c in after['components']], fields['detector_references'][case]['objects'])
        truth = [t for t in fields['references'] if t['case_id'] == case]
        if [(p['candidate_id'], p['owner_status'], p['owner_detection_index'], p['number'], p['pinname']) for p in before['pin_candidates']] != [(p['candidate_id'], p['owner_status'], p['owner_detection_index'], p['number'], p['pinname']) for p in after['pin_candidates']]:raise ValueError('Geometry changed semantic fields or ownership')
        if [(n['network_id'], n['pin_candidate_ids']) for n in bg['networks']] != [(n['network_id'], n['pin_candidate_ids']) for n in ag['networks']]:raise ValueError('Graph member sets changed')
        if {n: net['hyperGraph'] for n, net in old_payload['nets'].items()} != {n: net['hyperGraph'] for n, net in payload['nets'].items()}:raise ValueError('Complete hypergraphs changed')
        if any(validate_output(payload, *after['size'])):raise ValueError('Invalid output')
        for name, pins in [('before', before), ('after', after)]:
            count = joint_match([p for p in pins['pin_candidates'] if p['owner_status'] == 'assigned'], truth, owners)
            totals[name].update({k: count[k] for k in ['tp', 'fp', 'fn']})
        for change in read(folder/'geometry_audit.json')['changes']:
            pin = next(p for p in after['pin_candidates'] if p['candidate_id'] == change['candidate_id'])
            same = [t for t in truth if t['component_source_id'] == owners.get(pin['owner_detection_index'])]
            reference = min(same, key=lambda t: math.dist(pin['point'], t['point']), default=None)
            row = {'case_id': case, **change, 'nearest_same_owner_draft_reference': reference,
                'before_distance_px': math.dist(change['old_point'], reference['point']) if reference else None,
                'after_distance_px': math.dist(change['new_point'], reference['point']) if reference else None}
            if reference:
                row['review_card'] = f"tip_{case}_{change['candidate_id']}.png";point_card(case, change, reference, a.output/row['review_card'])
            changes.append(row)
        pm = {p['candidate_id']: p for p in after['pin_candidates']}
        assigned = [p for p in after['pin_candidates'] if p['owner_status'] == 'assigned'];pairs = match(assigned, truth, owners)
        matched = {assigned[i]['candidate_id']: truth[j] for i, j in pairs}
        old_assigned = [p for p in before['pin_candidates'] if p['owner_status'] == 'assigned'];old_pairs = match(old_assigned, truth, owners)
        old_matched = {old_assigned[i]['candidate_id'] for i, j in old_pairs}
        for proof in read(folder/'export/engineering_export_coverage.json')['atomic_complete_network_proofs']:
            contacts = set(proof['member_to_contact'].values())
            row = {'case_id': case, **proof, 'all_members_match_draft_owner_and_location_before': contacts.issubset(old_matched),
                'all_members_match_draft_owner_and_location_after': contacts.issubset(matched),
                'number_name_or_official_network_certified': False}
            row['review_card'] = f"whole_net_{case}_{proof['network_id']}.png"
            review_card(case, proof, after, ag, records, a.output/row['review_card']);net_reviews.append(row)
            if case == '0036':
                refs = payload['nets'][proof['network_id']]['hyperGraph'][1:-1].split(',');owner_set = {s.rsplit('.', 1)[0] for s in refs}
                single = {'components': {k: payload['components'][k] for k in owner_set},
                    'pins': {o: {k: v for k, v in payload['pins'][o].items() if o+'.'+k[4:] in refs} for o in owner_set},
                    'nets': {proof['network_id']: payload['nets'][proof['network_id']]}}
                if any(validate_output(single, *after['size'])):raise ValueError('Invalid standalone complete net')
                save(a.output/f"engineering_complete_{case}_{proof['network_id']}.json", single)
    if sha(ref_path) != digest:raise ValueError('Frozen draft changed')
    scores = {k: metrics(v['tp'], v['fp'], v['fn']) for k, v in totals.items()}
    result = {'aggregate': batch['aggregate'], 'joint_owner_and_location_5px_draft_proxy': scores,
        'refined_tips_matching_same_owner_draft_within_5px_before': sum(c['before_distance_px'] is not None and c['before_distance_px'] <= 5 for c in changes),
        'refined_tips_matching_same_owner_draft_within_5px_after': sum(c['after_distance_px'] is not None and c['after_distance_px'] <= 5 for c in changes),
        'complete_nets_with_all_members_matching_draft_owner_and_location_before': sum(n['all_members_match_draft_owner_and_location_before'] for n in net_reviews),
        'complete_nets_with_all_members_matching_draft_owner_and_location_after': sum(n['all_members_match_draft_owner_and_location_after'] for n in net_reviews),
        'frozen_S21_input_files_checked': checked, 'draft_reference_sha256': digest, 'reference_changes': 0,
        'original_tip_count_semantic_fields_and_net_membership_preserved': True, 'local_tip_and_edge_geometry_changed': True,
        'independent_test': False, 'official_PinF1': None, 'official_NetF1': None}
    save(a.output/'evaluation.json', result);save(a.output/'geometry_review.json', {'items': changes, 'used_in_inference': False, 'used_in_training': False})
    save(a.output/'whole_network_review.json', {'items': net_reviews, 'used_in_inference': False})
    lines = ['# S23：电容端点定位修正', '',
        '目标是修正已导出整网的电容端点偏差。使用冻结U-Net的原尺寸/1.5倍/2倍输出，固定规则只接受两个相对端点在三尺度中沿同一引线逐步向内移动的电容。新坐标取2倍模型峰值，原坐标保存在审计中；没有按参考点或检测框吸附。', '',
        f"26页回放：{batch['aggregate']['geometry']['refined_capacitors']}个电容、{batch['aggregate']['geometry']['refined_tips']}个tip修正，均在0036。工程出口仍为3个net/266个pin/214个元件，原3条hyperGraph、1477个tip候选及全部原网成员保留。", '',
        '两条0036三端网络的器件成员不变。C1.2与草稿最近同器件点的距离9.149→2.584px，C3.1为9.164→1.805px。8个修正tip在5px内的同器件草稿匹配0→8；两条整网全部成员的归属/5px坐标匹配0→2。这些检查不包含官方编号、名称和完整参考网集认证。', '',
        f"26页同一草稿归属＋位置代理F1：{scores['before']['f1']:.4%}→{scores['after']['f1']:.4%}，TP {scores['before']['tp']}→{scores['after']['tp']}；5px规则和参考均未修改。这26页反复用于开发，不能作为独立测试或官方PinF1。", '',
        '## 导线与来源', '',
        '原学习导线在电容极板附近也偏早终止。8条边只增加由native/2倍导线概率支持的局部端点路径，每个新增路径像素概率至少0.5，坐标取可连通且距新tip不超过3px的模型导线像素。遇到其他导线、器件、重复tip、缺失/竞争峰或未决附件时整个器件扣留。线段由模型像素路径压缩，偏差最多0.25px。原边主体不删除，原网成员和逻辑合并不改写。', '',
        '仅0036通过全部条件，其余页保持原几何。原图红色native点、绿色2倍模型点、蓝色参考点对照见geometry_review.json。原图核对另存data/reviews/capacitor_tip_geometry_v1.json，不进入推理/训练。', '',
        '新tip的confidence是三尺度相关分数的算术均值，明确标记not_calibrated；不是准确率或独立概率。原峰置信度、1.5倍与2倍峰、完整新增路径及最小像素概率均保留。工程编号和名称继续采用S22未验证约定；保守JSON原样保留。', '',
        '```powershell', '.\\scripts\\run_capacitor_tip_geometry.ps1 -Image .\\data\\standardized\\0036\\0036.png -Device cpu',
        '& .\\.venv-gpu\\Scripts\\python.exe scripts/batch_capacitor_tip_geometry.py',
        '& .\\.venv-gpu\\Scripts\\python.exe scripts/evaluate_capacitor_tip_geometry.py', '```', '']
    runtime = a.output/'runtime_verification.json'
    if runtime.is_file():
        v = read(runtime);lines += [f"验证：{v['regression_tests_passed']}项回归通过；0036原始PNG全流程{v['fresh_entrypoint_wall_seconds']:.1f}秒，新点最大数值差{v['fresh_refined_tip_max_delta_px']:.5f}px，字段、成员与导出整网线段一致。", '']
    lines += ['本轮提升定位质量，完整网数量没有增加，整体覆盖仍低。保持独立入口；后续按已有网的未决真实成员和缺失位号推进，并检验该规则能否推广到其他电容样式。', '']
    (a.output/'report.md').write_text('\n'.join(lines), encoding='utf-8');print(result)


if __name__ == '__main__':main()
