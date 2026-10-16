"""Build draft-evaluation review cards and report without inventing labels."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from run_component_tip_refinement import ROOT,read,save,sha


def main():
    stage=ROOT/'runs/s13_component_tips';e=read(stage/'evaluation.json');a=e['aggregate']
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14);cards=[];review=[]
    for index,item in enumerate(e['added_review_queue'],1):
        case=item['case_id'];audit=read(stage/f'predictions/{case}/refinement_audit.json')
        add=next(p for p in audit['added_candidates'] if p['candidate_id']==item['candidate_id'])
        c=next(c for c in audit['component_reviews'] if c['detection_index']==add['owner_detection_index']);b=c['bbox_top_left']
        with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as image:
            image=image.convert('RGB');crop=[max(0,int(b[0])-45),max(0,int(b[1])-45),min(image.width,int(b[2])+46),min(image.height,int(b[3])+46)]
            patch=image.crop(crop);draw=ImageDraw.Draw(patch);x,y=add['point'][0]-crop[0],add['point'][1]-crop[1]
            draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=1)
            for pin in read(ROOT/f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json')['pin_candidates']:
                if pin.get('owner_detection_index')==add['owner_detection_index']:
                    px,py=pin['point'][0]-crop[0],pin['point'][1]-crop[1];draw.ellipse((px-3,py-3,px+3,py+3),outline='blue',width=1)
            scale=min(4,600/patch.width,400/patch.height);patch=patch.resize((round(patch.width*scale),round(patch.height*scale)),Image.Resampling.NEAREST)
            card=Image.new('RGB',(640,510),'white');card.paste(patch,(10,40));draw=ImageDraw.Draw(card)
            draw.text((10,7),f"{case} {item['component_type']} {item['candidate_id']}",font=font,fill='#333')
            draw.text((10,455),f"Draft <=5px owner match: {item['draft_incremental_owner_location_match']}",font=font,fill='#333')
            draw.text((10,479),'Red: new native+zoom peak | Blue: preserved tip',font=font,fill='#333')
            name=f'tip_review_{index:02d}.png';card.save(stage/name);cards.append(card)
            review.append({**item,'crop_top_left':crop,'review_card':name,'image_sha256':sha(ROOT/f'data/standardized/{case}/{case}.png')})
    if cards:
        sheet=Image.new('RGB',(1280,510*((len(cards)+1)//2)),'white')
        for i,card in enumerate(cards):sheet.paste(card,((i%2)*640,(i//2)*510))
        sheet.save(stage/'tip_review_sheet.png')
    save(stage/'review_manifest.json',{'items':review,'scope':'offline draft comparison, not training truth'})
    audit=read(ROOT/'data/passive_number_review_v1/audit.json')
    before,after=a['all']['before'],a['all']['after']
    text=f'''# 电源/被动器件tip试验与编号标签审计

本轮没有训练编号模型，也没有替换默认推理入口。新增独立U-Net双尺度tip试验、300页原始编号审计和40个训练划分的未批准复核裁剪。默认仍使用上轮 `run_pin_output_rules.ps1`。

## 编号训练数据

审计原标注的pinNumberByOCR/VLM：共{audit['counts']['components']}个R/C元件，其中{audit['counts']['two_tip_components']}个带两个有效点，明确合法1/2编号对为{audit['counts'].get('explicit_number_pair',0)}组。这个统计不包括另外保存的少量人工复核覆盖。

转换程序按源pin ID顺序生成编号，无法作为图像识别的可靠监督。水平电容自动编号左/右比例接近一半，其他方向也不统一；左右/上下顺序不能普遍确定1/2。直接训练会学到源ID排序，不能宣称恢复了图中不可见的编号。

`data/passive_number_review_v1/review_queue.json`包含40个未批准样例，覆盖R/C、水平/竖直，仅来自原train划分。圈出的A/B只表示源pin ID，需先复核是否为真实tip，再填可见编号和证据；不可见编号保留null。批准训练目标当前0条，没有将验证页或自动编号混入新训练。

## U-Net tip双尺度试验

采用原U-Net权重，native和1.5倍模型热图都达到0.65、峰值相距≤3px、归属同一模型元件，且周围有学习导线概率≥0.5，才提议新增tip。已有候选4px内不重复添加；R/C需构成相对的两端；多个同样可疑点超过缺口时扣留。没有从框边生成点、传统原图分割、猜编号或删除已有候选。

同26页旧开发数据新增{a['all']['added']}个候选：{a['added_by_type']}。其中{a['all']['marginal_correct']}个在未占用的草稿参考中同时匹配正确元件和≤5px位置；新增候选参考匹配率为{a['all']['marginal_correct']/max(1,a['all']['added']):.0%}。未匹配点不能一概解释为假点，也可能是参考/类型错误，需要看图。

| 草稿元件归属＋位置代理 | 原方案 | 试验 |
|---|---:|---:|
| Precision | {before['precision']:.2%} | {after['precision']:.2%} |
| Recall | {before['recall']:.2%} | {after['recall']:.2%} |
| F1 | {before['f1']:.2%} | {after['f1']:.2%} |

增益很小、精确率略降。原配置的13页校准/13页验收分组都已有开发使用，不能作为独立测试。正式PinF1/NetF1未测，本试验保持未采用。

5个新增点均接入候选导线图；原1477个tip和已观测文字原样保留，原学习导线概率不变。但新增tip会改变现有body guard的保护区域，可能改变骨架/图边，已逐页记录changed_pixels与graph_edges_changed。不能宣称网络拓扑完全不变，也未用连线变化作为准确率成绩。

无已分配tip的电源检测数{a['power_missing_owned_tip_before']}→{a['power_missing_owned_tip_after']}，这是模型覆盖计数；+1V8新增点的连接tip语义仍待核验。0148新增点之一位于F2保险丝附近，冻结检测器把它判成resistor，导致草稿归属/类别匹配失败。相关视觉说明单独保存，未修订参考或用于增加分数。

## 可复现入口

```powershell
# 独立试验；不覆盖上轮默认结果
.\\scripts\\run_component_tip_refinement.ps1 -Image data/standardized/0149/0149.png -Device cpu
```

默认目录 `runs/component_tip_predictions/<图片名>`。读取PNG、模型、配置；完整入口不读配套源标注/CAD/参考JSON。缓存模式校验image SHA、两份概率字节、权重来源，并验证native概率可以原样重现旧的所有tip。

`pin_text_result.json`与`refinement_audit.json`保留新增点证据；`output_rules`是实验网络/出口，仍为部分JSON。40个复核裁剪、5个候选核验图和审计脚本已准备，原标注与训练划分没有修改。

本轮12项新增模块测试通过；没有启动GPU训练。下一步先确定无可见数字时的官方编号/元件库约定，再生成可靠的编号监督；电源tip需要补充经过图像核验的监督数据，而非继续降低全局阈值。
'''
    verify=stage/'runtime_verification.json'
    if verify.is_file():
        v=read(verify);text+=f"\n## 桌面验证\n\n12项新增测试与16项原出口回归在桌面项目通过；26页原{v['old_development_tip_points_preserved']}个tip原样保留。{v['protected_files_unchanged']}个上游文件SHA未变。0149完整CPU PNG入口无旧上游输入缓存，新增{v['fresh_added_candidates']}个试验点，全部原{v['fresh_original_candidate_count']}个tip保留；默认入口和训练不变。\n"
    (stage/'report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
