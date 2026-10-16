"""Coverage and visual review report; no changes to training or reference labels."""
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from run_pin_text_v2 import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s11_pin_text_v2');a=p.parse_args()
    evaluation=read(a.output/'evaluation.json');downstream=read(a.output/'downstream/batch_summary.json')
    if not downstream.get('completed'):raise ValueError('Downstream batch incomplete')
    baseline=read(ROOT/'runs/s10_submission_fields_v1/audit.json')['after_export'];after=Counter();reasons=Counter();reviews=[];cards=[];headers=[]
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',12)
    for row in downstream['cases']:
        case=row['case_id'];folder=a.output/'predictions'/case;pins=read(folder/'pin_text_result.json');pmap={p['candidate_id']:p for p in pins['pin_candidates']}
        records=read(folder/'component_records.json')['records'];cmap={r['detection_index']:r for r in records}
        for key in ['exported_components','exported_pins','exported_nets']:after[key]+=row['export'][key]
        reasons.update(row['export']['pin_reason_counts'])
        accepted=read(folder/'identity_recovery_debug.json')['accepted'];adopted=read(folder/'field_adoption_debug.json')['adopted_name_additions']
        rows=[('identity',r['designator'],r['evidence'],r['detection_index'],None) for r in accepted]
        rows += [('pinname',r['text'],r['evidence'],pmap[r['candidate_id']]['owner_detection_index'],r['candidate_id']) for r in adopted]
        headers.extend({'case_id':case,'detection_index':r['detection_index'],'designator':r['designator'],'Name':cmap[r['detection_index']]['Name']} for r in accepted)
        with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
            for kind,text,evidence,index,cid in rows:
                box=evidence['bbox'];point=pmap[cid]['point'] if cid else [(box[0]+box[2])/2,(box[1]+box[3])/2]
                crop=[max(0,int(min(box[0],point[0]))-30),max(0,int(min(box[1],point[1]))-30),
                      min(source.width,int(max(box[2],point[0]))+31),min(source.height,int(max(box[3],point[1]))+31)]
                patch=source.crop(crop).convert('RGB');draw=ImageDraw.Draw(patch)
                if cid:
                    x,y=point[0]-crop[0],point[1]-crop[1];draw.ellipse((x-3,y-3,x+3,y+3),outline='#bd2323',width=2)
                scale=min(290/patch.width,168/patch.height,3)
                patch=patch.resize((max(1,round(patch.width*scale)),max(1,round(patch.height*scale))),Image.Resampling.NEAREST)
                card=Image.new('RGB',(300,230),'white');card.paste(patch,(5,25));draw=ImageDraw.Draw(card)
                draw.text((5,5),f'{case} {kind} DET {index}',font=font,fill='#333')
                for i in range(0,min(len(text),72),36):draw.text((5,196+i//36*14),text[i:i+36],font=font,fill='#1f5688')
                name=f'{case}_{kind}_{cid or index}.png';dest=a.output/'review_crops'/name;dest.parent.mkdir(parents=True,exist_ok=True);card.save(dest);cards.append(card)
                reviews.append({'case_id':case,'kind':kind,'text':text,'detection_index':index,'candidate_id':cid,'bbox_top_left':box,
                                'point':point,'crop_top_left':crop,'image_sha256':pins['image_sha256'],'card':'review_crops/'+name,
                                'reviewed':False,'correct_text_and_role':None,'original_dataset_changed':False})
    for offset in range(0,len(cards),20):
        selected=cards[offset:offset+20];sheet=Image.new('RGB',(1500,230*((len(selected)+4)//5)),'white')
        for i,card in enumerate(selected):sheet.paste(card,((i%5)*300,(i//5)*230))
        sheet.save(a.output/f'field_review_{offset//20+1:02d}.png')
    save(a.output/'field_review_queue.json',{'items':reviews,'count':len(reviews),'reviewed':0,'automatic_label_changes':0})
    totals=evaluation['totals'];old,new=totals['v1'],totals['v2'];experiment=totals['full_v2_experiment']
    save(a.output/'export_audit.json',{'before':baseline,'after':dict(after),'missing_pin_fields':dict(reasons),
         'new_header_identities':headers,'physical_edges_and_tip_coordinates_preserved':True,'official_PinF1':None,'official_NetF1':None})
    def percent(value):return '—' if value is None else f'{100*value:.2f}%'
    lines=['# IC/连接器位号与引脚文字 v2','',
        '本轮处理大IC左上角位号、U1H等单元后缀、H/CN/LD等框式器件可见编号、密集引脚排及同一连接线上的竞争tip候选。YOLO、U-Net和训练集没有改变。',
        '采用策略：保留原编号和已有非空pinname，仅补入高置信度且至少两个已有OCR视图支持、没有被其他字段占用的缺失名称。几种视图来自同一个OCR模型，不是独立正确性验证。','',
        '## 同一26页草稿参考下的开发结果','',
        '这些图片均已用于开发；下表不是官方PinF1、独立测试成绩或全系统准确率。名称联合代理要求元件类别/框、归属、tip误差≤5px和名称严格匹配，尚不要求官方component key＋pin number匹配。','',
        '| 指标 | v1 | 本轮采用 |','|---|---:|---:|',
        f"| 编号恢复数量 | {old['counts']['number_observed']} | {new['counts']['number_observed']} |",
        f"| 名称恢复数量 | {old['counts']['name_observed']} | {new['counts']['name_observed']} |",
        f"| 名称＋归属＋位置代理F1 | {percent(old['joint_name_location']['f1'])} | {percent(new['joint_name_location']['f1'])} |",
        f"| 条件名称精确率 | {percent(old['conditional']['pinname']['precision'])} | {percent(new['conditional']['pinname']['precision'])} |",
        f"| 条件名称覆盖率 | {percent(old['conditional']['pinname']['recall'])} | {percent(new['conditional']['pinname']['recall'])} |",
        f"| 条件编号精确率 | {percent(old['conditional']['number']['precision'])} | {percent(new['conditional']['number']['precision'])} |",'',
        f"新增可见位号预测 {new['counts']['new_header_identities']} 个、型号文字 {new['counts']['new_model_names']} 个、缺失名称 {new['counts']['name_observed']-old['counts']['name_observed']} 个；数量不等于正确数量。",
        f"条件名称正确数 {old['conditional']['pinname']['correct']}→{new['conditional']['pinname']['correct']}，已输出 {old['conditional']['pinname']['emitted']}→{new['conditional']['pinname']['emitted']}，可评分参考仍为{new['conditional']['pinname']['expected']}。覆盖上升，但相同参考下精确率下降，需复核新增项。",'',
        f"完整v2编号实验的条件编号F1为 {percent(experiment['conditional']['number']['f1'])}，原v1为 {percent(old['conditional']['number']['f1'])}，因此没有采用该编号改法。`experimental_pin_text_result.json`单列试验输出；正式入口继续使用原编号证据。",'',
        'OCR与草稿参考存在字符串不一致，例如VDD_2与VDD 2、IN与N。这些差异需要逐项看图判断，不能一概归因于参考错误；本轮保留OCR原文，没有按参考改写字符，也未修订参考来提高分数。','',
        '## 部分JSON出口覆盖','',
        '| 项目 | 上一阶段 | 本轮 |','|---|---:|---:|']
    for key,label in [('exported_components','元件'),('exported_pins','字段齐全的pin'),('exported_nets','严格可导出net')]:lines.append(f'| {label} | {baseline[key]} | {after[key]} |')
    lines+=['','原1477个引脚候选和物理网络几何/接入关系均保留。对同侧、同行、沿法向分开的候选，只在文字分配时选代表；没有把坐标贴到bbox边缘，也没有删除假设或提高定位F1。',
            f"共{new['counts']['secondary_tip_hypotheses_text_withheld']}个候选在完整v2文字实验中视为竞争假设；采用版仍保留原已有字段，新增名称不能重复占用旧名称的OCR区域。",'',
            '## 新增位号预测','']
    for row in headers:lines.append(f"- {row['case_id']}：`{row['designator']}`；型号 `{row['Name']}`。" if row['Name'] else f"- {row['case_id']}：`{row['designator']}`；型号仍未知。")
    lines+=['','这些框式器件仍沿用冻结检测类别；例如连接器可能被检测为ic。没有凭位号改类别，也未确认新增身份均符合官方key匹配规则。','',
            '## 运行与复核','',
            '完整PNG入口：','',
            '```powershell',
            '.\\scripts\\run_pin_semantics.ps1 -Image data/standardized/0018/0018.png -Device cuda:0',
            '```','',
            '输出目录中`pin_text`保留采用结果、完整试验和证据；`physical`、`network`、`submission`依次接入既有导线、普通标签和字段出口。',
            f"`field_review_queue.json`及`field_review_*.png`包含{len(reviews)}项新增身份/名称预测，初始reviewed均为false，不会自动加入训练集。",
            '`submission/network_field_review_queue.json`继续指出每条网络缺哪些身份、编号和名称。strict nets仍为0，完整赛事提交未完成。','',
            '下一优先项：复核新增字段，明确无可见pinname的被动器件与电源符号出口规则，核对真实pin编号/空名标准，然后消除一个小电路网络的完整输出阻断。','']
    (a.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'review_items':len(reviews),'before_export':baseline,'after_export':dict(after),'name_f1':new['joint_name_location']['f1']},flush=True)


if __name__=='__main__':main()
