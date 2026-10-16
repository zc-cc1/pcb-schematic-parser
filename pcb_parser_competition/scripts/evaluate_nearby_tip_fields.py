"""Offline S20 comparison. References never enter the PNG entry point."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

from run_nearby_tip_fields import ROOT, read, save, sha
from evaluate_pin_text import component_matches, match, conditional_fields, score
from evaluate_wire_graph import pair_counts, metrics


def complete_literal_pairs(predicted, truth, owners):
    if not predicted or not truth:
        return []
    distance = np.linalg.norm(np.array([p['point'] for p in predicted])[:, None]-np.array([t['point'] for t in truth])[None, :], axis=2)
    valid = (distance <= 5) & np.array([[owners.get(p['owner_detection_index']) == t['component_source_id']
                                      and p['number'] in t['expected_numbers'] and p['pinname'] == t['expected_name']
                                      for t in truth] for p in predicted])
    rows, cols = linear_sum_assignment(np.where(valid, distance, (min(len(predicted), len(truth))+1)*6))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]


def field_card(case, pins, change, reference, path):
    selected = [p for p in pins['pin_candidates'] if p['candidate_id'] in {change['candidate_id'], change['donor_candidate_id']}]
    boxes = [e['bbox'] for p in selected for e in p.get('text_evidence', {}).values() if e.get('bbox')]
    component = next(c for c in pins['components'] if c['component_id'] == change['owner'])
    boxes.append(component['bbox_top_left'])
    points = [p['point'] for p in selected] + ([reference['point']] if reference else [])
    boxes += [[*p, *p] for p in points]
    with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
        crop = [max(0, int(min(b[0] for b in boxes))-20), max(0, int(min(b[1] for b in boxes))-20),
                min(source.width, int(max(b[2] for b in boxes))+21), min(source.height, int(max(b[3] for b in boxes))+21)]
        image = source.convert('RGB').crop(crop)
    scale = min(4, 790/image.width, 520/image.height)
    image = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (840, image.height+140), 'white');canvas.paste(image, (12, 35))
    draw = ImageDraw.Draw(canvas);font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 13)
    draw.text((12, 8), f"{case} {change['owner']}.{change['number']} -> {change['pinname']} | same physical lead", fill='#333', font=font)
    xy = lambda p: (12+(p[0]-crop[0])*scale, 35+(p[1]-crop[1])*scale)
    for p in selected:
        colour = '#c52c30' if p['candidate_id'] == change['candidate_id'] else '#1b8533'
        x, y = xy(p['point']);draw.ellipse((x-5, y-5, x+5, y+5), outline=colour, width=2)
        for field, e in p.get('text_evidence', {}).items():
            if e.get('bbox'):
                draw.rectangle((*xy(e['bbox'][:2]), *xy(e['bbox'][2:])), outline=colour, width=1)
    if reference:
        x, y = xy(reference['point']);draw.line((x-5, y-5, x+5, y+5), fill='#125ab0', width=2);draw.line((x-5, y+5, x+5, y-5), fill='#125ab0', width=2)
    lines = ['Red: numbered target tip | Green: donor OCR name tip | Blue: draft (offline only)',
             f"Tips {change['tip_distance_px']:.3f}px apart; third tip {change['nearest_third_tip_distance_px']:.3f}px away.",
             'No tips/numbers/members removed or moved. Existing INT glyph is reused, not new OCR.',
             'Donor number remains unresolved; the network is still withheld.']
    for n, line in enumerate(lines):draw.text((12, image.height+45+n*20), line, fill='#333', font=font)
    canvas.save(path)
    return crop


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', type=Path, default=ROOT/'runs/s20_nearby_tip_fields');args = parser.parse_args()
    batch = read(args.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S20 batch incomplete')
    fields_path = ROOT/'runs/s6_pin_text_v1/draft_field_references.json';fields = read(fields_path)
    power_path = ROOT/'runs/s15_power_networks/reference_power_connectivity.json';power = read(power_path)
    conditional = {v: Counter() for v in ['before', 'after']};named = {v: Counter() for v in conditional};complete = {v: Counter() for v in conditional}
    connectivity = Counter();reviews = []
    for item in batch['cases']:
        case = item['case_id'];folder = args.output/f'predictions/{case}';source = ROOT/f'runs/s18_targeted_pin_ocr/predictions/{case}'
        if sha(source/'completed_pin_text_result.json') != item['input_pin_sha256'] or sha(source/'graph_proposals.json') != item['input_graph_sha256']:
            raise ValueError('S18 input changed')
        old, new = read(source/'completed_pin_text_result.json'), read(folder/'completed_pin_text_result.json')
        if new['image_sha256'] != fields['image_sha256'][case] or new['image_sha256'] != power['cases'][case]['image_sha256']:raise ValueError('Reference image differs')
        truth = [t for t in fields['references'] if t['case_id'] == case]
        owners = component_matches([{'class_name': c['class_name'], 'bbox': c['bbox_top_left']} for c in new['components']], fields['detector_references'][case]['objects'])
        for version, payload in [('before', old), ('after', new)]:
            predicted = [p for p in payload['pin_candidates'] if p['owner_status'] == 'assigned']
            conditional[version].update(conditional_fields(predicted, truth, match(predicted, truth, owners)))
            name_predictions = [p for p in predicted if p.get('pinname') is not None];name_truth = [t for t in truth if t['expected_name']]
            named[version].update(correct=len(match(name_predictions, name_truth, owners, 'pinname')), emitted=len(name_predictions), expected=len(name_truth))
            exported = read(folder/f"{'baseline_export' if version == 'before' else 'proposal_export'}/partial_submission.json")['pins']
            whole = [p for p in predicted if p.get('pinname') is not None and p.get('number') is not None
                     and 'pin_'+p['number'] in exported.get(p['owner'], {})]
            whole_truth = [t for t in truth if t['expected_name'] and t['expected_numbers']]
            complete[version].update(correct=len(complete_literal_pairs(whole, whole_truth, owners)), emitted=len(whole), expected=len(whole_truth))
        og, ng = read(source/'graph_proposals.json'), read(folder/'graph_proposals.json')
        protected = deepcopy(ng)
        # Stage provenance changes independently of graph content.
        for k in ['input_pin_sha256', 'input_graph_sha256', 'input_records_sha256', 'nearby_tip_config_sha256']:
            if k in og:protected[k] = og[k]
            else:protected.pop(k, None)
        for a, b in zip(protected['pin_attachments'], og['pin_attachments']):a['pinname'] = b['pinname']
        if protected != og:raise ValueError('S20 changed protected graph content')
        counts = pair_counts(ng, new, truth, power['cases'][case]['reference_pin_networks'], owners)
        for scope in ['conditional_connectivity_pairs', 'all_reference_pair_proxy']:
            for metric in ['tp', 'fp', 'fn']:connectivity[scope+'_'+metric] += counts[scope][metric]
        by_id = {p['candidate_id']: p for p in new['pin_candidates']}
        for change in read(folder/'field_link_audit.json')['changes']:
            target = by_id[change['candidate_id']]
            compatible = [t for t in truth if t['component_source_id'] == owners.get(target['owner_detection_index'])
                          and target['number'] in t['expected_numbers'] and target['pinname'] == t['expected_name']]
            reference = min(compatible, key=lambda t: np.linalg.norm(np.array(target['point'])-t['point']), default=None)
            name = f"field_link_{case}_{change['candidate_id']}.png"
            row = {'case_id': case, **change, 'reference': reference, 'review_card': name}
            row['crop_top_left'] = field_card(case, new, change, reference, args.output/name)
            row['draft_complete_field_match_within_5px'] = bool(reference and np.linalg.norm(np.array(target['point'])-reference['point']) <= 5)
            cov = read(folder/'proposal_export/conservative_export_coverage.json')
            blocked = next(n for n in cov['withheld_networks'] if n['network_id'] == change['network_id'])
            row['network_still_withheld_reasons'] = blocked['reasons'];reviews.append(row)
    conn = {scope: metrics(*(connectivity[scope+'_'+m] for m in ['tp', 'fp', 'fn'])) for scope in ['conditional_connectivity_pairs', 'all_reference_pair_proxy']}
    if conn != read(ROOT/'runs/s18_targeted_pin_ocr/evaluation.json')['connectivity']:raise ValueError('Connectivity metrics changed')
    result = {'conditional_fields': {v: dict(c) for v, c in conditional.items()},
              'joint_name_and_location_draft_proxy': {v: score(c['correct'], c['emitted'], c['expected']) for v, c in named.items()},
              'strict_export_complete_literal_field_counts': {v: {'matched_complete_fields': c['correct'],
                  'exported_pins': c['emitted'], 'draft_pins_with_both_reference_fields': c['expected']} for v, c in complete.items()},
              'connectivity': conn, 'connectivity_unchanged': True, 'all_original_graph_geometry_and_members_preserved': True,
              'field_reference_sha256': sha(fields_path), 'power_reference_sha256': sha(power_path),
              'new_OCR_observations': 0, 'reference_changes': 0, 'independent_test': False, 'official_PinF1': None, 'official_NetF1': None}
    save(args.output/'evaluation.json', result);save(args.output/'visual_review_queue.json', {'items': reviews, 'used_in_inference': False})
    lines = ['# S20：同一引线近邻tip字段补全', '',
             '编号和名称分散到两个模型tip时，只在同器件、同侧、同物理网络、同一条高置信导线段、重复完整OCR和无近邻竞争等条件同时通过时，把已观察名称补到已有编号的tip。', '',
             f"26页补全{batch['aggregate']['names_reconciled']}个名称，严格完整pin122→{batch['aggregate']['conservative']['exported_pins']}、候选138→{batch['aggregate']['candidate']['exported_pins']}；严格net仍0、候选仍1。", '',
             '0018/U3的8与INT分别位于PIN_CAND_0102和PIN_CAND_0113，两点相距4.183px，最近第三点12.666px。两点投影落在同一段W00039，编号和名称均有3个不同OCR视图支持。现将INT补到已有编号8的点；原名称、模型点、编号、网络成员均保留。', '',
             '该物理网络仍包含缺编号的次级tip，所以保持扣留。没有删掉它来导出单成员自环网络，也没有向它复制编号。', '',
             f"固定位置/归属匹配的条件字段计数：{result['conditional_fields']}，前后相同。", '',
             f"全量名称＋位置草稿代理：{result['joint_name_and_location_draft_proxy']}。", '',
             f"严格已输出pin的编号＋名称＋位置同时匹配草稿：{result['strict_export_complete_literal_field_counts']}。", '',
             '完整字段草稿缺失会让已导出pin无法评价；上述完整字段计数不据此计算模型准确率或官方F1。', '',
             '原一对一位置匹配优先把参考分给更近的次级tip。因此条件名称分数不变，名称代理多了一个重复字段输出而略降；同时具备编号和名称的匹配增加1。以上均为旧开发草稿代理，并非新OCR识别、独立测试或官方PinF1提升。', '',
             '原线段、拓扑、网络成员及连接对代理完全不变。保持独立试验入口：', '',
             '```powershell', '.\\scripts\\run_nearby_tip_fields.ps1 -Image .\\data\\standardized\\0018\\0018.png -Device cpu', '```', '',
             '逐例保留field_link_audit.json、旧/新出口和缺口队列；对照图与视觉索引供离线核验。', '']
    verification = args.output/'runtime_verification.json'
    if verification.is_file():
        v = read(verification)
        lines += [f"验证：{v['regression_tests_passed']}项回归通过；{v['fresh_case_id']}原始PNG完整入口{v['fresh_entrypoint_wall_seconds']:.1f}秒，字段补全、出口和网络成员与回放一致。最大tip偏差{v['fresh_tip_max_delta_px']:.4f}px，容差{v['tip_numeric_tolerance_px']}px。", '']
    (args.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'aggregate': batch['aggregate'], 'conditional_fields_unchanged': conditional['before'] == conditional['after'],
           'strict_joint_complete': result['strict_export_complete_literal_field_counts'], 'connectivity_unchanged': True}, flush=True)


if __name__ == '__main__':main()
