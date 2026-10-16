"""Offline verification of S21 complete networks; no reference use in inference."""
import argparse
from collections import Counter
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_whole_networks import ROOT, read, save, sha
from evaluate_pin_text import component_matches
from evaluate_wire_graph import pair_counts, metrics


def card(case, row, pins, graph, records, path):
    selected = [p for p in pins['pin_candidates'] if p['candidate_id'] in row['all_original_member_ids']]
    rec = [r for r in records if r['detection_index'] in {p['owner_detection_index'] for p in selected}]
    boxes = [r['bbox_top_left'] for r in rec]
    edge_map = {e['edge_id']: e for e in graph['edges']}
    points = [pt for eid in row['model_edge_ids'] for pt in edge_map[eid]['polyline']]
    boxes += [[*p, *p] for p in points]
    with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
        crop = [max(0, int(min(b[0] for b in boxes))-25), max(0, int(min(b[1] for b in boxes))-25),
                min(source.width, int(max(b[2] for b in boxes))+26), min(source.height, int(max(b[3] for b in boxes))+26)]
        image = source.convert('RGB').crop(crop)
    factor = min(3, 1000/image.width, 620/image.height)
    image = image.resize((round(image.width*factor), round(image.height*factor)), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (max(1050, image.width+24), image.height+105), 'white');canvas.paste(image, (12, 35))
    draw = ImageDraw.Draw(canvas);font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 14)
    draw.text((12, 8), f"{case} {row['network_id']} | "+','.join(row['all_pin_references']), fill='#333', font=font)
    xy = lambda p: (12+(p[0]-crop[0])*factor, 35+(p[1]-crop[1])*factor)
    for eid in row['model_edge_ids']:draw.line([xy(p) for p in edge_map[eid]['polyline']], fill='#be247b', width=2)
    for p in selected:
        x, y = xy(p['point']);draw.ellipse((x-5, y-5, x+5, y+5), outline='#047e36', width=2)
        draw.text((x+7, y+6), f"{p['owner']}.{p['number']}", fill='#047e36', font=font, stroke_width=1, stroke_fill='white')
    draw.text((12, image.height+45), 'All model members resolve; original wire geometry and tip points retained.', fill='#333', font=font)
    draw.text((12, image.height+68), 'Observed-title identity is a local policy; no official Golden/evaluator certification.', fill='#8c4320', font=font)
    canvas.save(path);return crop


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', type=Path, default=ROOT/'runs/s21_whole_networks');args = parser.parse_args()
    batch = read(args.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S21 batch incomplete')
    fields_path = ROOT/'runs/s6_pin_text_v1/draft_field_references.json';fields = read(fields_path)
    power_path = ROOT/'runs/s15_power_networks/reference_power_connectivity.json';power = read(power_path)
    totals = Counter();guards = Counter();reviews = [];field_only = 0
    for item in batch['cases']:
        case = item['case_id'];folder = args.output/f'predictions/{case}';source = ROOT/f'runs/s20_nearby_tip_fields/predictions/{case}'
        if sha(source/'completed_pin_text_result.json') != item['input_pin_sha256'] or sha(source/'graph_proposals.json') != item['input_graph_sha256']:
            raise ValueError('S20 inputs changed')
        pins, graph, records = read(folder/'completed_pin_text_result.json'), read(folder/'graph_proposals.json'), read(folder/'completed_component_records.json')['records']
        if pins['image_sha256'] != fields['image_sha256'][case] or pins['image_sha256'] != power['cases'][case]['image_sha256']:raise ValueError('Reference image differs')
        truth = [t for t in fields['references'] if t['case_id'] == case]
        owners = component_matches([{'class_name': c['class_name'], 'bbox': c['bbox_top_left']} for c in pins['components']], fields['detector_references'][case]['objects'])
        counts = pair_counts(graph, pins, truth, power['cases'][case]['reference_pin_networks'], owners)
        for scope in ['conditional_connectivity_pairs', 'all_reference_pair_proxy']:
            for metric in ['tp', 'fp', 'fn']:totals[scope+'_'+metric] += counts[scope][metric]
        audit = read(folder/'whole_network_audit.json');guards.update(audit['summary']['blocker_counts'])
        field_only += read(folder/'conservative_completion_queue.json')['field_only_incomplete_networks']
        coverage = read(folder/'proposal_export/conservative_export_coverage.json')
        pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
        for proof in coverage['atomic_complete_network_proofs']:
            row = {'case_id': case, **proof, 'member_reference_diagnostics': []}
            for cid in proof['all_original_member_ids']:
                p = pmap[cid]
                same = [t for t in truth if t['component_source_id'] == owners.get(p['owner_detection_index'])]
                near = min(same, key=lambda t: math.dist(p['point'], t['point']), default=None)
                row['member_reference_diagnostics'].append({'candidate_id': cid, 'predicted_owner': p['owner'],
                    'predicted_number': p['number'], 'predicted_name': p['pinname'],
                    'nearest_same_owner_draft_reference': near,
                    'distance_px': math.dist(p['point'], near['point']) if near else None,
                    'within_5px': bool(near and math.dist(p['point'], near['point']) <= 5)})
            row['review_card'] = f"whole_net_{case}_{proof['network_id']}.png"
            row['crop_top_left'] = card(case, proof, pins, graph, records, args.output/row['review_card'])
            row['official_complete_network_certified'] = False;reviews.append(row)
    conn = {scope: metrics(*(totals[scope+'_'+m] for m in ['tp', 'fp', 'fn'])) for scope in ['conditional_connectivity_pairs', 'all_reference_pair_proxy']}
    if conn != read(ROOT/'runs/s20_nearby_tip_fields/evaluation.json')['connectivity']:raise ValueError('Whole-net export changed predicted connectivity')
    result = {'aggregate_export': batch['aggregate'], 'connectivity': conn, 'connectivity_unchanged': True,
              'whole_network_blocker_counts': dict(guards), 'field_only_remaining_networks': field_only,
              'local_complete_networks': len(reviews), 'official_complete_networks_certified': 0,
              'field_reference_sha256': sha(fields_path), 'power_reference_sha256': sha(power_path),
              'reference_changes': 0, 'independent_test': False, 'official_PinF1': None, 'official_NetF1': None}
    save(args.output/'evaluation.json', result);save(args.output/'whole_network_review.json', {'items': reviews, 'used_in_inference': False})
    lines = ['# S21：完整网络输出', '',
             '目标从零散引脚补全转向完整网络：恢复图中唯一接口标题、验证所有成员引用、拦截缺失端点成员及单接点重复tip片段。', '',
             f"26页本地保守出口：{batch['aggregate']['conservative']}；候选：{batch['aggregate']['candidate']}。", '',
             '首个本地完整网络为0042的`(JTAG.15,RESET#.1)`：原图可见RESET#到JTAG第15脚nSRST的整段线，两个模型成员都能解析，原线段和tip不移动。JTAG来自3视图支持的可见标题，JTAG-20PIN底部文字由2视图支持，且已有多处观察编号与20PIN吻合。没有生成J1或排序后缀；标题作为身份属于本地预测策略，官方规范身份仍未确认。', '',
             '87个网络发现导线端点邻近已检测元件但缺相应成员，1个网络确认两个tip属于同一物理接点；这些门槛也记录在字段已经扣留的网络中。计数包括单引脚片段与原有其他门槛，不能相加当作新增多引脚错误数。', '',
             f"经过所有门槛后，仅字段未决的多引脚网络剩{field_only}个，按完整网任务排序；保留原始模型所有成员，不靠删除未决pin缩小网集。", '',
             '已验证的同一引线tip别名可共同解析到已有完整接点，但必须逐项保留原成员→接点映射，至少两个不同接点才允许形成网络；未证明的成员、交点、断线及电源门槛保持扣留。', '',
             '## 参考限制', '',
             '草稿中JTAG接口身份为UNRESOLVED_C_1，15脚编号缺失，最近同器件参考点距预测5.021px；RESET#标签没有匹配器件参考。保留5px阈值、原坐标和参考，不据此宣称官方NetF1或完整参考匹配成功。连接对代理与S20一致；本轮改进可解析的整网出口，不是新连接识别成绩。', '',
             '```powershell', '.\\scripts\\run_whole_networks.ps1 -Image .\\data\\standardized\\0042\\0042.png -Device cpu', '```', '',
             '每个出口网络附atomic_complete_network_proofs：所有原模型成员、逐成员接点映射、完整pin引用及原模型边ID。whole_network_audit.json保留端点漏成员、重复tip和扣留原因。对照图见whole_network_review.json。', '']
    verification = args.output/'runtime_verification.json'
    if verification.is_file():
        v = read(verification);lines += [f"验证：{v['regression_tests_passed']}项回归通过；0042原始PNG全流程{v['fresh_entrypoint_wall_seconds']:.1f}秒，整网引用、几何和字段匹配回放，最大tip数值差{v['fresh_tip_max_delta_px']:.4f}px。", '']
    (args.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'export': batch['aggregate'], 'whole_network_guards': dict(guards), 'field_only_remaining': field_only, 'connectivity_unchanged': True}, flush=True)


if __name__ == '__main__':main()
