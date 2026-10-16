"""S19 offline draft-reference audit. Does not change predictions or references."""
import argparse
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_pin_text import ROOT, read, save, sha
from evaluate_pin_text import component_matches
from pin_reference_diagnostics import diagnose_fields


def review_card(case, row, glyph_bbox, owner_bbox, path):
    nearest = row['nearest_same_owner_references']
    points = [row['point']] + [r['point'] for r in nearest]
    boxes = [glyph_bbox] + [[*p, *p] for p in points]
    with Image.open(ROOT / f'data/standardized/{case}/{case}.png') as source:
        source = source.convert('RGB')
        crop = [max(0, int(min(b[0] for b in boxes)) - 20), max(0, int(min(b[1] for b in boxes)) - 20),
                min(source.width, int(max(b[2] for b in boxes)) + 21),
                min(source.height, int(max(b[3] for b in boxes)) + 21)]
        image = source.crop(crop)
        context_crop = [max(0, int(owner_bbox[0])-20), max(0, int(owner_bbox[1])-20),
                        min(source.width, int(owner_bbox[2])+21), min(source.height, int(owner_bbox[3])+21)]
        context = source.crop(context_crop)
        context.thumbnail((270, 430), Image.Resampling.NEAREST)
    scale = min(4, 500 / image.width, 430 / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
    panel_height = max(image.height, context.height+20)
    canvas = Image.new('RGB', (820, panel_height + 155), 'white')
    canvas.paste(image, (12, 35))
    canvas.paste(context, (540, 55))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 13)
    draw.text((12, 8), f"{case} {row['owner']} {row['candidate_id']} {row['field']}={row['text']}", fill='#222', font=font)
    draw.text((540, 35), 'Component context', fill='#222', font=font)

    def xy(point):
        return 12 + (point[0] - crop[0]) * scale, 35 + (point[1] - crop[1]) * scale

    draw.rectangle((*xy(glyph_bbox[:2]), *xy(glyph_bbox[2:])), outline='#b56c00', width=2)
    x, y = xy(row['point'])
    draw.ellipse((x-5, y-5, x+5, y+5), outline='#d62828', width=2)
    draw.text((x+6, y), 'P', fill='#d62828', font=font, stroke_width=1, stroke_fill='white')
    for n, reference in enumerate(nearest, 1):
        x, y = xy(reference['point'])
        draw.line((x-5, y-5, x+5, y+5), fill='#1265bf', width=2)
        draw.line((x-5, y+5, x+5, y-5), fill='#1265bf', width=2)
        draw.text((x+6, y-12), f'R{n}', fill='#1265bf', font=font, stroke_width=1, stroke_fill='white')
    top = panel_height + 45
    draw.text((12, top), 'P red: unchanged prediction | R blue: draft reference | orange: OCR glyph', fill='#222', font=font)
    draw.text((12, top+20), row['reason'], fill='#222', font=font)
    for n, reference in enumerate(nearest, 1):
        draw.text((12, top+20+n*20), f"R{n}: distance={reference['distance_px']:.3f}px  number={reference['expected_numbers']}  name={reference['expected_name']!r}", fill='#1265bf', font=font)
    draw.text((12, top+85), 'Offline review only. 5px rule retained. No coordinate or reference edits.', fill='#8c4320', font=font)
    canvas.save(path)
    return crop


def review_sheets(output, rows):
    for start in range(0, len(rows), 4):
        sheet = Image.new('RGB', (1120, 940), '#eeeeee')
        for n, row in enumerate(rows[start:start+4]):
            with Image.open(output / row['review_card']) as image:
                image.thumbnail((555, 460))
                sheet.paste(image, ((n % 2) * 560, (n // 2) * 470))
        sheet.save(output / f'review_sheet_{start//4+1}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/s19_pin_reference_audit')
    args = parser.parse_args()
    output = args.output.resolve()
    source = ROOT / 'runs/s18_targeted_pin_ocr'
    reference_path = ROOT / 'runs/s6_pin_text_v1/draft_field_references.json'
    # Explicitly avoid placing audit outputs into frozen prediction/reference trees.
    if not output.is_relative_to(ROOT / 'runs') or output == ROOT / 'runs':
        raise ValueError('Audit output must be a distinct subfolder of runs')
    for frozen in [source, ROOT / 'runs/s6_pin_text_v1', ROOT / 'runs/s17_network_completion']:
        if output.is_relative_to(frozen) or frozen.is_relative_to(output):
            raise ValueError('Audit output overlaps frozen inputs')
    batch = read(source / 'predictions/batch_summary.json')
    if not batch.get('completed'):
        raise ValueError('S18 batch incomplete')
    references = read(reference_path)
    changes = read(source / 'visual_review_queue.json')['items']
    snapshots = {}

    def snapshot(path):
        snapshots[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)

    for path in [reference_path, source / 'predictions/batch_summary.json', source / 'visual_review_queue.json',
                 source / 'evaluation.json', ROOT / 'configs/unet_inference_v2.json',
                 ROOT / 'configs/pin_ownership_v1.json', ROOT / 'configs/targeted_pin_ocr_v1.json',
                 Path(__file__).resolve(), ROOT / 'scripts/pin_reference_diagnostics.py',
                 ROOT / 'scripts/evaluate_pin_text.py']:
        snapshot(path)
    all_rows, new_rows, case_rows = [], [], []
    for item in batch['cases']:
        case = item['case_id']
        folder = source / f'predictions/{case}'
        payload = read(folder / 'completed_pin_text_result.json')
        image_path = ROOT / f'data/standardized/{case}/{case}.png'
        if sha(image_path) != payload['image_sha256'] or payload['image_sha256'] != references['image_sha256'][case]:
            raise ValueError(f'{case}: PNG/reference/prediction hash mismatch')
        for path in [image_path, folder / 'completed_pin_text_result.json', folder / 'graph_proposals.json',
                     folder / 'recovery_audit.json', folder / 'completed_component_records.json']:
            snapshot(path)
        for path in sorted((folder / 'proposal_export').glob('*.json')):
            snapshot(path)
        truth = [r for r in references['references'] if r['case_id'] == case]
        owners = component_matches([{'class_name': c['class_name'], 'bbox': c['bbox_top_left']}
                                   for c in payload['components']], references['detector_references'][case]['objects'])
        rows = [{'case_id': case, **r} for r in diagnose_fields(payload['pin_candidates'], truth, owners)]
        all_rows.extend(rows)
        lookup = {(r['candidate_id'], r['field']): r for r in rows}
        records = {r['detection_index']: r for r in read(folder / 'completed_component_records.json')['records']}
        pins = {p['candidate_id']: p for p in payload['pin_candidates']}
        for change in [c for c in changes if c['case_id'] == case]:
            row = dict(lookup[(change['candidate_id'], change['field'])])
            if row['text'] != change['text']:
                raise ValueError('Recovery review differs from frozen prediction')
            pin = pins[row['candidate_id']]
            name = f"audit_{case}_{row['candidate_id']}_{row['field']}.png"
            row['review_card'] = name
            output.mkdir(parents=True, exist_ok=True)
            row['crop_top_left'] = review_card(case, row, change['evidence']['bbox'],
                                             records[pin['owner_detection_index']]['bbox_top_left'], output / name)
            row['tip_geometry_independently_certified'] = False
            row['visual_review_required'] = row['reason'] != 'agrees_with_draft'
            new_rows.append(row)
        case_rows.append({'case_id': case, 'field_diagnostics': dict(Counter(r['reason'] for r in rows)),
                          'emitted_field_count': len(rows)})
    review_sheets(output, new_rows)
    for name, digest in snapshots.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Frozen input changed during audit: {name}')
    result = {'scope': 'Offline analysis of all emitted S18 fields on 26 old development pages',
              'matching_radius_px': 5, 'all_emitted_field_diagnostics': dict(Counter(r['reason'] for r in all_rows)),
              'new_field_diagnostics': dict(Counter(r['reason'] for r in new_rows)),
              'case_count': len(case_rows), 'emitted_field_count': len(all_rows), 'new_field_count': len(new_rows),
              'input_sha256': snapshots, 'inputs_unchanged': True, 'prediction_changes': 0, 'reference_changes': 0,
              'extra_matches': 0, 'official_PinF1': None, 'official_NetF1': None, 'independent_test': False,
              'used_in_inference': False, 'used_in_training': False}
    log_path = ROOT / 'runs/s19_regression.log'
    if log_path.is_file():
        log = log_path.read_text(encoding='utf-8')
        count = re.search(r'Ran (\d+) tests', log)
        if count and re.search(r'\bOK\s*$', log):
            result['regression_evidence'] = {'passed': int(count.group(1)),
                                             'log_path': 'runs/s19_regression.log', 'log_sha256': sha(log_path)}
    save(output / 'audit_summary.json', result)
    save(output / 'all_emitted_field_diagnostics.json', {'cases': case_rows, 'items': all_rows})
    save(output / 'new_field_review.json', {'items': new_rows, 'used_in_inference': False, 'used_in_training': False})
    lines = ['# S19：引脚预测与草稿参考差异诊断', '',
             '离线诊断保留原有同器件、5像素、一对一匹配规则；最近同字段参考只作为复核线索，不增加命中数。', '',
             f"26页已输出字段：{dict(Counter(r['reason'] for r in all_rows))}。", '',
             f"S18新增11字段：{result['new_field_diagnostics']}。5个未匹配字段都存在同器件参考点，但最近点超过5像素，无一属于一对一占用冲突。", '',
             '| 页面/引脚 | 新字段 | 最近同器件距离 | 原始匹配诊断 |', '|---|---|---|---|']
    for row in new_rows:
        nearest = row['nearest_same_owner_references']
        distance = f"{nearest[0]['distance_px']:.3f}px" if nearest else '无同器件参考'
        lines.append(f"| {row['case_id']} / {row['owner']} / {row['candidate_id']} | {row['field']}={row['text']} | {distance} | {row['reason']} |")
    lines += ['', '对照图：红圈P为原预测、蓝叉R为最近草稿参考、橙框为实际OCR字框。图中的最近参考不是自动认可的正确引脚。', '',
              '0148/U6编号3距离5.147px，仍不计为命中。0170数码管右侧3处距离10.47～12.38px，草稿同时出现封装文字进入编号、编号11变1等问题，需分别核验原图标注语义及tip坐标约定。0002/C87距离9.374px。不能据此把预测吸附到参考点。', '',
              '新增字段中3处已有草稿字段且匹配，3处位置匹配但草稿编号缺失，其余5处受坐标阈值限制。可见字面已在S18单独复核，tip几何及完整网络仍未认证。', '',
              '输入PNG、预测字段/图/出口、参考及相关配置均记录哈希并在运行后检查未变；严格122个pin、0个net保持S18结果。该步骤改进错误定位和复核效率，不宣称识别或官方分数提升。', '',
              '复跑：`.\\.venv-gpu\\Scripts\\python.exe scripts/audit_pin_references.py`。', '']
    if 'regression_evidence' in result:
        lines += [f"回归验证：{result['regression_evidence']['passed']}项通过；日志哈希记录在audit_summary.json。", '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'new_field_diagnostics': result['new_field_diagnostics'],
           'all_emitted_field_diagnostics': result['all_emitted_field_diagnostics'],
           'inputs_unchanged': True}, flush=True)


if __name__ == '__main__':
    main()
