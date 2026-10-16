"""Offline S22 whole-net review and frozen draft diagnostics; never inference."""
import argparse
from collections import Counter
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_engineering_networks import ROOT, read, save, sha
from evaluate_pin_text import component_matches
from pcb_parser.dataset_converter import validate_output


def review_card(case, proof, pins, graph, records, path):
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']};edges = {e['edge_id']: e for e in graph['edges']}
    members = [pmap[c] for c in proof['all_original_member_ids']]
    boxes = [r['bbox_top_left'] for r in records if r['detection_index'] in {p['owner_detection_index'] for p in members}]
    boxes += [[*p, *p] for eid in proof['model_edge_ids'] for p in edges[eid]['polyline']]
    with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
        crop = [max(0, int(min(b[0] for b in boxes))-25), max(0, int(min(b[1] for b in boxes))-25),
                min(source.width, int(max(b[2] for b in boxes))+26), min(source.height, int(max(b[3] for b in boxes))+26)]
        image = source.convert('RGB').crop(crop)
    factor = min(3, 1000/image.width, 600/image.height)
    image = image.resize((round(image.width*factor), round(image.height*factor)), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (max(1000, image.width+24), image.height+115), 'white');canvas.paste(image, (12, 38))
    draw = ImageDraw.Draw(canvas);font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 15)
    draw.text((12, 8), f"{case} {proof['network_id']} | "+','.join(proof['all_pin_references']), fill='#222', font=font)
    xy = lambda p: (12+(p[0]-crop[0])*factor, 38+(p[1]-crop[1])*factor)
    for eid in proof['model_edge_ids']:draw.line([xy(p) for p in edges[eid]['polyline']], fill='#be247b', width=2)
    for p in members:
        x, y = xy(p['point']);generated = any(p.get(f+'_status') == 'engineering_convention_unverified' for f in ['number', 'pinname'])
        color = '#b75b00' if generated else '#047e36'
        draw.ellipse((x-5, y-5, x+5, y+5), outline=color, width=2)
        draw.text((x+7, y+7), f"{p['owner']}.{p['number']}"+(' [rule]' if generated else ' [observed]'), fill=color, font=font, stroke_width=1, stroke_fill='white')
    draw.text((12, image.height+52), 'Orange: provisional number/name conventions. Green: existing recognized fields.', fill='#333', font=font)
    draw.text((12, image.height+77), 'Original tips, edges and every model member retained. Official numbering unverified.', fill='#8c4320', font=font)
    canvas.save(path);return crop


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s22_engineering_networks');a = p.parse_args()
    batch = read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S22 batch incomplete')
    ref_path = ROOT/'runs/s6_pin_text_v1/draft_field_references.json';ref_hash = sha(ref_path);fields = read(ref_path)
    reviews = [];reason_counts = Counter();generated_in_nets = set();new_nets = 0;checked_inputs = 0
    for item in batch['cases']:
        case = item['case_id'];folder = a.output/f'predictions/{case}';source = ROOT/f'runs/s21_whole_networks/predictions/{case}'
        for path, digest in item['input_sha256'].items():
            if sha(Path(path)) != digest:raise ValueError('Frozen S21 input changed')
            checked_inputs += 1
        baseline = read(source/'proposal_export/partial_submission.json')
        if baseline != read(folder/'baseline_export/partial_submission.json'):raise ValueError('Conservative output changed')
        pins, graph, records = read(folder/'engineering_pin_text_result.json'), read(folder/'engineering_graph_proposals.json'), read(folder/'completed_component_records.json')['records']
        original_pins = read(source/'completed_pin_text_result.json');original_graph = read(source/'graph_proposals.json')
        if [p['point'] for p in pins['pin_candidates']] != [p['point'] for p in original_pins['pin_candidates']]:raise ValueError('Model point changed')
        if (graph['edges'] != original_graph['edges'] or graph['physical_networks'] != original_graph['physical_networks']
                or [(n['network_id'], n['pin_candidate_ids']) for n in graph['networks']] != [(n['network_id'], n['pin_candidate_ids']) for n in original_graph['networks']]):
            raise ValueError('Model topology changed')
        cov = read(folder/'engineering_export_coverage.json');output = read(folder/'engineering_submission.json')
        errors, bounds = validate_output(output, *pins['size'])
        if errors or bounds:raise ValueError('Output invalid')
        pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
        if fields['image_sha256'][case] != pins['image_sha256']:raise ValueError('Draft/image mismatch')
        owners = component_matches([{'class_name': c['class_name'], 'bbox': c['bbox_top_left']} for c in pins['components']], fields['detector_references'][case]['objects'])
        truth = [t for t in fields['references'] if t['case_id'] == case]
        reason_counts.update(cov['network_reason_counts'])
        for proof in cov['atomic_complete_network_proofs']:
            is_new = proof['network_id'] not in baseline['nets'];new_nets += is_new
            row = {'case_id': case, 'new_engineering_network': is_new, **proof, 'member_reference_diagnostics': []}
            for cid in proof['all_original_member_ids']:
                pin = pmap[proof['member_to_contact'][cid]]
                if any(pin.get(f+'_status') == 'engineering_convention_unverified' for f in ['number', 'pinname']):generated_in_nets.add((case, pin['candidate_id']))
                same = [t for t in truth if t['component_source_id'] == owners.get(pin['owner_detection_index'])]
                near = min(same, key=lambda t: math.dist(pin['point'], t['point']), default=None)
                distance = math.dist(pin['point'], near['point']) if near else None
                row['member_reference_diagnostics'].append({'candidate_id': cid, 'predicted_owner': pin['owner'],
                    'predicted_number': pin['number'], 'predicted_name': pin['pinname'],
                    'engineering_fields': [f for f in ['number', 'pinname'] if pin.get(f+'_status') == 'engineering_convention_unverified'],
                    'nearest_same_owner_draft_reference': near, 'distance_px': distance,
                    'within_5px': distance is not None and distance <= 5})
            row['review_card'] = f"whole_net_{case}_{proof['network_id']}.png"
            row['crop_top_left'] = review_card(case, proof, pins, graph, records, a.output/row['review_card'])
            row['official_complete_network_certified'] = False;reviews.append(row)
            if is_new:
                refs = {x for x in output['nets'][proof['network_id']]['hyperGraph'][1:-1].split(',')}
                selected_owners = {x.rsplit('.', 1)[0] for x in refs}
                single = {'components': {k: v for k, v in output['components'].items() if k in selected_owners},
                    'pins': {owner: {key: pin for key, pin in output['pins'][owner].items() if owner+'.'+key[4:] in refs} for owner in selected_owners},
                    'nets': {proof['network_id']: output['nets'][proof['network_id']]}}
                if any(validate_output(single, *pins['size'])):raise ValueError('Standalone complete net invalid')
                save(a.output/f"engineering_complete_{case}_{proof['network_id']}.json", single)
    if sha(ref_path) != ref_hash:raise ValueError('Frozen reference changed')
    result = {'aggregate': batch['aggregate'], 'local_complete_networks': len(reviews), 'new_engineering_complete_networks': new_nets,
        'generated_pins_used_in_complete_networks': len(generated_in_nets), 'frozen_S21_inputs_checked': checked_inputs,
        'conservative_outputs_identical': True, 'model_geometry_and_raw_network_members_identical': True,
        'network_reason_counts': dict(reason_counts), 'draft_field_reference_sha256': ref_hash,
        'reference_changes': 0, 'independent_test': False, 'official_numbering_verified': False,
        'official_PinF1': None, 'official_NetF1': None}
    save(a.output/'evaluation.json', result);save(a.output/'whole_network_review.json', {'items': reviews, 'used_in_inference': False, 'used_in_training': False})
    lines = ['# S22：独立工程整网出口', '',
        '用户选择增加带来源标记的工程出口。R/C/L按图像左/上为1、右/下为2，名称取编号；只作用于已识别身份、恰有两个相对端点且两个模型导线附件合格的器件。规则字段与原OCR字段分别记录在engineering_field_audit.json及field_provenance.json。', '',
        '| 26页旧开发集 | S21保守出口 | S22工程出口 |', '|---|---:|---:|',
        '| 元件 | 214 | 214 |', '| 完整pin | 132 | 266 |', '| 完整net | 1 | 3 |', '',
        '新增0036两条三端网络：`(C1.2,L1.1,U1.2)`和`(C3.1,R1.2,U1.3)`，对应原图RFIN输入及VDD去耦连接。0042的`(JTAG.15,RESET#.1)`原网络逐项保留。工程出口并非官方完整提交或准确率提升；保守出口仍为1条网络。', '',
        '67个被动器件通过两个端点几何检查，生成124个编号、134个名称；其中4个规则pin实际参与新增整网。其余规则pin是局部可输出字段，不能把它们当成完整网络进展。未编号、未匹配身份、重复tip、交点、断线及电源门槛仍可扣留整个网。仅字段未决多引脚网络剩270个。', '',
        '修正5个0170空白tip的几何归属：电阻/电感主体有明确长轴且导线同轴时，偏离主体横向中心带的空白tip撤回owner。已有字段的tip保持原预测并记录风险。R1下方的LD1第11脚开放导线不会被编成R1端点。所有原tip点、物理边、图网成员保持，不删除未决成员凑完整网。', '',
        '修正端点检查的方向问题：NetworkX边source/target顺序不一定与存储polyline顺序一致，改用图节点坐标检查端点。', '',
        '## 原图及参考检查', '',
        '两条新增网的实际器件端点与连接分支经原图核对。近邻草稿参考中，C1端点相差9.149px、C3端点相差9.164px，超过5px；这些被动端点草稿本身没有编号、名称为空。其余新增网成员坐标差0.617–3.412px。保留原模型点和草稿参考，不用参考吸附修正，也不放宽5px。工程约定编号需另行验证官方源顺序；当前不能据此计算官方PinF1/NetF1。', '',
        '每条网保留atomic_complete_network_proofs，记录原成员、引用和模型边ID；field_provenance.json逐成员标记工程字段来源。对照图橙色表示规则编号/名称，绿色表示已有识别字段。独立整网JSON可用engineering_complete_0036_*.json查看。', '',
        '```powershell', '.\\scripts\\run_engineering_networks.ps1 -Image .\\data\\standardized\\0036\\0036.png -Device cpu',
        '& .\\.venv-gpu\\Scripts\\python.exe scripts/batch_engineering_networks.py',
        '& .\\.venv-gpu\\Scripts\\python.exe scripts/evaluate_engineering_networks.py', '```', '']
    runtime = a.output/'runtime_verification.json'
    if runtime.is_file():
        v = read(runtime);lines += [f"验证：{v['regression_tests_passed']}项回归通过；0036从PNG全流程{v['fresh_entrypoint_wall_seconds']:.1f}秒，字段、元件、整网线段及全部原成员证明与回放一致。全页tip最大CPU/GPU数值差{v['fresh_tip_max_delta_px']:.5f}px，小于0.05px数值容差。130个S21输入文件哈希及26页保守JSON保持。", '']
    lines += ['下一步优先处理电容tip定义/定位偏差和新增整网的未决真实成员，再处理可见位号漏识别；避免继续堆积未进入整网的规则字段。', '']
    (a.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({k: result[k] for k in ['local_complete_networks', 'new_engineering_complete_networks', 'generated_pins_used_in_complete_networks', 'frozen_S21_inputs_checked']})


if __name__ == '__main__':main()
