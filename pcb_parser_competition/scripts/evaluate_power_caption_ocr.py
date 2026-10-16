"""Offline S15/S16 comparison against one frozen power reference snapshot."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_power_caption_ocr import ROOT, read, save, sha
from evaluate_wire_graph import pair_counts, metrics
from evaluate_pin_text import component_matches


def review_cards(output, case, before, after):
    previous = {q['detection_index']: q for q in before['component_review_queue']}
    rows = []
    with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as image:
        image = image.convert('RGB')
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 13)
        for row in after['component_review_queue']:
            old = previous[row['detection_index']]
            if old['bound_site_id'] is not None or row['bound_site_id'] is None:
                continue
            sites = [s for s in row['caption_candidates'] if not s['withheld_reasons']]
            boxes = [row['bbox_top_left']]+[s['bbox'] for s in sites]
            crop = [max(0, int(min(b[0] for b in boxes))-24), max(0, int(min(b[1] for b in boxes))-24),
                    min(image.width, int(max(b[2] for b in boxes))+25), min(image.height, int(max(b[3] for b in boxes))+25)]
            raw = image.crop(crop)
            factor = min(4, 600/raw.width, 360/raw.height)
            raw = raw.resize((max(1, round(raw.width*factor)), max(1, round(raw.height*factor))), Image.Resampling.NEAREST)
            canvas = Image.new('RGB', (max(620, raw.width+24), raw.height+100), 'white')
            canvas.paste(raw, (12, 35))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), f"{case} power detector {row['detection_index']} | original PNG", fill='#333', font=font)
            draw.text((12, raw.height+44), 'Observed captions: '+', '.join(sorted({s['label'] for s in sites})), fill='#333', font=font)
            draw.text((12, raw.height+66), 'OCR evidence only; full remote network not visually certified.', fill='#8b4320', font=font)
            name = f"caption_{case}_{row['detection_index']:03d}.png"
            canvas.save(output/name)
            rows.append({'case_id': case, 'detection_index': row['detection_index'], 'caption_sites': sites,
                         'crop_top_left': crop, 'review_card': name, 'visually_reviewed': False,
                         'complete_network_certified': False})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'runs/s16_power_caption_ocr')
    a = p.parse_args()
    batch = read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):
        raise ValueError('S16 batch incomplete')
    ref_path = ROOT/'runs/s15_power_networks/reference_power_connectivity.json'
    reference = read(ref_path)
    fields = read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json')
    totals = {v: Counter() for v in ['baseline_s15', 'localized_ocr']}
    rows, reviews = [], []
    for row in batch['cases']:
        case = row['case_id']
        source = ROOT/f'runs/s14_component_library/predictions/{case}'
        folder = a.output/f'predictions/{case}'
        pins = read(source/'pin_text/pin_text_result.json')
        before = read(folder/'baseline_power_graph.json')
        after = read(folder/'power_graph_proposals.json')
        if not (pins['image_sha256'] == reference['cases'][case]['image_sha256']
                == after['image_sha256'] == sha(ROOT/f'data/standardized/{case}/{case}.png')):
            raise ValueError('Prediction/reference image mismatch')
        if row['input_pin_sha256'] != sha(source/'pin_text/pin_text_result.json'):
            raise ValueError('Frozen pin prediction bytes changed')
        truth = [t for t in fields['references'] if t['case_id'] == case]
        owner_map = component_matches([{'class_name': c['class_name'], 'bbox': c['bbox_top_left']}
                                      for c in pins['components']], fields['detector_references'][case]['objects'])
        result = {'case_id': case}
        for version, graph in [('baseline_s15', before), ('localized_ocr', after)]:
            counts = pair_counts(graph, pins, truth, reference['cases'][case]['reference_pin_networks'], owner_map)
            result[version] = counts
            for scope in ['conditional_connectivity_pairs', 'all_reference_pair_proxy']:
                for metric in ['tp', 'fp', 'fn']:
                    totals[version][scope+'_'+metric] += counts[scope][metric]
        rows.append(result)
        reviews.extend(review_cards(a.output, case,
            read(ROOT/f'runs/s15_power_networks/predictions/{case}/power_binding_audit.json'),
            read(folder/'power_binding_audit.json')))
    aggregate = {v: {scope: metrics(*(counts[scope+'_'+m] for m in ['tp', 'fp', 'fn']))
                    for scope in ['conditional_connectivity_pairs', 'all_reference_pair_proxy']}
                 for v, counts in totals.items()}
    save(a.output/'evaluation.json', {'cases': rows, 'aggregate': aggregate, 'reference_sha256': sha(ref_path),
         'reference_scope': reference['scope'], 'official_NetF1': None, 'independent_test': False,
         'raw_annotation_reads': False, 'default_pipeline_adopted': False})
    save(a.output/'visual_review_queue.json', {'items': reviews, 'all_remote_paths_certified': False})
    totals = batch['aggregate']
    lines = ['# S16：电源文字局部 OCR 试验', '',
        '从 PNG 电源符号附近做带检测框的三尺度、双方向 OCR，保留原文字及冲突拼写。复用冻结 YOLO/U-Net/引脚库预测，仅处理单一归属且唯一接线的未决电源符号。原置信度≥0.98、至少3种视图、完整字面名及拓扑门槛保持。多个视图来自同一 OCR 模型，不是独立证据。', '',
        '| 同一开发集 | S15 | S16 |', '|---|---:|---:|']
    for key, label in [('observed_power_sites', '满足证据门槛的电源节点'), ('power_groups_applied', '电源逻辑候选组'),
                       ('input_network_groups_joined', '减少的网络分组'), ('power_components_without_qualified_caption', '文字/归属证据未决电源组件')]:
        lines.append(f"| {label} | {totals['before_power'][key]} | {totals['power'][key]} |")
    lines += ['', f"局部复读 {totals['recovery']['roi_count']} 个 ROI，新增 {len(reviews)} 处通过证据门槛的电源节点。", '',
              '| 同一冻结电源草稿参考 | S15 F1 | S16 F1 |', '|---|---:|---:|']
    for scope, label in [('conditional_connectivity_pairs', '已匹配归属/位置的引脚对'), ('all_reference_pair_proxy', '计入未匹配预测的全量配对代理')]:
        lines.append(f"| {label} | {aggregate['baseline_s15'][scope]['f1']:.2%} | {aggregate['localized_ocr'][scope]['f1']:.2%} |")
    conditional_before = aggregate['baseline_s15']['conditional_connectivity_pairs']
    conditional_after = aggregate['localized_ocr']['conditional_connectivity_pairs']
    all_before = aggregate['baseline_s15']['all_reference_pair_proxy']
    all_after = aggregate['localized_ocr']['all_reference_pair_proxy']
    lines += ['', f"条件连接对 TP {conditional_before['tp']}→{conditional_after['tp']}，FP {conditional_before['fp']}→{conditional_after['fp']}；条件精确率 {conditional_before['precision']:.2%}→{conditional_after['precision']:.2%}。全量代理 FP {all_before['fp']}→{all_after['fp']}。召回改善同时增加了草稿参考下的误合并，需核对0148/0149的网络关系，不能仅凭局部文字认证完整组网。",
              f"冻结预测回放含局部 OCR 与出口共 {sum(r['wall_seconds'] for r in batch['cases']):.1f} 秒；不包含 YOLO/U-Net/全图 OCR 冷启动总时长。"]
    lines += ['', '这些是已使用26页上的开发代理指标，参考未独立认证，不是官方 NetF1。评估只读取已有离线参考快照，预测入口不读取标注/CAD/Golden。', '',
        f"严格部分出口：{totals['conservative']['exported_components']} 元件、{totals['conservative']['exported_pins']} pin、{totals['conservative']['exported_nets']} net。候选：{totals['candidate']['exported_pins']} pin、{totals['candidate']['exported_nets']} net。",
        '所有原tip、字段、归属、物理线段和网络成员保持。没有猜测电源规范身份、编号或pinname；新增电源逻辑候选仍禁止作为完整赛事网络输出。原默认入口保留。', '',
        '```powershell', '.\\scripts\\run_power_caption_ocr.ps1 -Image .\\data\\standardized\\0018\\0018.png -Device cpu', '```', '',
        '产物：recovered_texts.json、power_caption_recovery.json、power_binding_audit.json、power_graph_proposals.json，以及旧/新严格和候选部分出口。', '',
        '新增文字裁剪与证据见 visual_review_queue.json；局部文字复核不等于完整远端网络认证。', '']
    verification = a.output/'runtime_verification.json'
    if verification.is_file():
        verified = read(verification)
        lines += [f"运行验证：{verified['regression_tests_passed']}项回归通过；{verified['fresh_case_id']}原始PNG完整入口耗时{verified['fresh_entrypoint_wall_seconds']:.1f}秒，字段与网络成员匹配回放，tip最大数值差{verified['fresh_tip_max_delta_px']:.4f}px（容差{verified['tip_numeric_tolerance_px']}px）。", '']
    (a.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'aggregate': aggregate, 'new_qualified_caption_cards': len(reviews)}, flush=True)


if __name__ == '__main__':
    main()
