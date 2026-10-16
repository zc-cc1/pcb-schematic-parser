"""Evidence cards and Chinese report for the public pin library pilot."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from run_component_pin_library import ROOT, read, save


def main():
    output = ROOT/'runs/s14_component_library'
    batch = read(output/'predictions/batch_summary.json')
    evaluation = read(output/'evaluation.json')
    baseline = read(ROOT/'runs/s12_pin_output_rules/predictions/batch_summary.json')['aggregate']
    case = output/'predictions/0018'
    result = read(case/'pin_text/pin_text_result.json')
    audit = read(case/'library_binding_audit.json')
    pins = {p['candidate_id']:p for p in result['pin_candidates']}
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    cards = []
    with Image.open(ROOT/'data/standardized/0018/0018.png') as source:
        for cid in ['PIN_CAND_0008','PIN_CAND_0068','PIN_CAND_0106','PIN_CAND_0125']:
            pin = pins[cid]; x,y = pin['point']
            crop = [max(0,int(x)-225),max(0,int(y)-30),min(source.width,int(x)+225),min(source.height,int(y)+30)]
            original = source.convert('RGB').crop(crop).resize(((crop[2]-crop[0])*2,(crop[3]-crop[1])*2),Image.Resampling.NEAREST)
            overlay = original.copy(); draw = ImageDraw.Draw(overlay)
            sx,sy = (x-crop[0])*2,(y-crop[1])*2
            draw.ellipse([sx-7,sy-7,sx+7,sy+7],outline='#bd2727',width=2)
            draw.text((sx+9,sy+7),'number='+str(pin['number']),font=font,fill='#bd2727',stroke_width=1,stroke_fill='white')
            card = Image.new('RGB',(max(950,original.width*2+36),original.height+155),'white');d = ImageDraw.Draw(card)
            d.text((12,8),f'0018 {cid} | same learned tip, observed literal name, exact package',font=font,fill='#333')
            card.paste(original,(12,42));card.paste(overlay,(original.width+24,42));d=ImageDraw.Draw(card)
            rows = [c for c in audit['changes'] if c['candidate_id']==cid]
            for i,c in enumerate(rows):d.text((12,original.height+52+22*i),f"{c['field']}: {c['before']} -> {c['after']} ({c['reason']})",font=font,fill='#333')
            d.text((12,card.height-30),'Visual review only; no training labels or official scoring reference generated.',font=font,fill='#8d481e')
            name=f'library_review_{len(cards)+1:02d}.png';card.save(output/name)
            cards.append({'candidate_id':cid,'card':name,'visually_reviewed':False,'official_reference_verified':False})
    save(output/'review_manifest.json',{'case_id':'0018','cards':cards,'source_annotations_used_for_rendering':False})
    counts = batch['aggregate']; summary = counts['library']; metrics = evaluation['totals']; new_numbers=evaluation['new_library_numbers']
    lines = ['# S14：准确型号的公共引脚库试验','',
      '本轮以 ST 官方 DS5319 Rev 20 第26页 Figure 8 与第104页型号编码为依据，接入 STM32F103C8T6 的 LQFP48 编号映射。元件库来自厂家公开资料，不是赛方元件库，也不包含数据集源标注。',
      '', '[官方数据手册](https://www.st.com/resource/en/datasheet/stm32f103c8.pdf)','',
      '只有准确型号取得同位置两种 OCR 视图支持、引脚完整名称在元件框内取得两种强文字支持、几何与一对一编号关系通过校验时，才补编号。输出名称保留原图字面字符串。例如 PA5/AN5/SPI1_SCK 对应15；PC13/TEMPER_RTC 保留图中的 TEMPER，不改成手册的 TAMPER。新增编号明确标注为元件库推导，不能计入编号OCR识别数。','',
      '| 同一26页旧开发数据 | S12字段出口 | S14元件库试验 |','|---|---:|---:|']
    for title, key in [('严格出口字段齐全pin','conservative'),('候选出口字段齐全pin','candidate')]:
        lines.append(f"| {title} | {baseline[key]['exported_pins']} | {counts[key]['exported_pins']} |")
    lines += [f"| 严格出口net | {baseline['conservative']['exported_nets']} | {counts['conservative']['exported_nets']} |",
      f"| 候选出口net | {baseline['candidate']['exported_nets']} | {counts['candidate']['exported_nets']} |",'',
      f"本轮只匹配到0018的一颗U1。新增编号{summary['numbers_added_from_library']}个，补缺失名称{summary['missing_names_added']}个，补完整片段名称{summary['partial_names_completed']}个；把1个PB5文字误作物理编号的读数移入审计记录。所有1477个原tip、所有归属和物理线段原样保留，网络成员一致。",'',
      f"新增19个库推导编号中，{new_numbers['owner_location_draft_matched']}个能匹配草稿归属/5px位置，{new_numbers['conditional_draft_scorable']}个具有唯一非空草稿编号，其中{new_numbers['conditional_draft_correct']}个与草稿编号一致。其余缺口不能据此认定编号错误，也不能算正确。",'',
      '| 已匹配归属/位置且参考有字段的条件指标 | S11原文字 | S14试验 |','|---|---:|---:|']
    for field, label in [('number','编号'),('pinname','名称')]:
        old,new=metrics['baseline_s11']['conditional'][field],metrics['library']['conditional'][field]
        lines += [f"| {label}精确率 | {old['precision']:.2%} | {new['precision']:.2%} |",
                  f"| {label}覆盖率 | {old['recall']:.2%} | {new['recall']:.2%} |"]
    lines += ['', '这些是旧26页和未完善草稿字段上的开发代理，分母排除了缺失参考；没有官方PinF1/NetF1，没有独立测试。字段齐全pin增加表示可导出的记录增加，不能直接叫准确率提高。', '',
      '本轮没有训练，没有改YOLO/U-Net权重或原标注。S12仍为原默认入口；S14提供独立可执行的PNG入口。R/C无编号问题仍待赛方规则或人工核验，电源语义、未决交点和完整网络出口仍待推进。','',
      '## 运行','', '```powershell', r'.\scripts\run_component_pin_library.ps1 -Image .\data\standardized\0018\0018.png -Device cpu','```','',
      '输出在 runs/component_library_predictions/0018/output_rules：partial_submission.json 是严格部分出口，submission_candidate.json 仍包含S12的被动器件候选名称规则，两者都不是完整赛事答案。library_binding_audit.json 逐项记录旧值、新值、文字证据、元件型号/封装和公开来源。','',
      '## 局部复核图','']
    for card in cards:lines += [f"![{card['candidate_id']}]({card['card']})",'']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'report':str(output/'report.md'),'cards':len(cards),'strict_pins':counts['conservative']['exported_pins']},flush=True)


if __name__=='__main__':
    main()
