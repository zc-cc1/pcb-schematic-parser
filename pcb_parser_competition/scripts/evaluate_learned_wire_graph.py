"""Offline comparison. Inference never imports this reference-reading module."""
from collections import Counter,defaultdict
import argparse
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from run_wire_graph import ROOT,read,save,sha
from evaluate_wire_graph import metrics,pair_counts
from evaluate_pin_text import component_matches


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/s8_junction_classifier_v1');args=parser.parse_args()
    output=args.output;batch=read(output/'batch_summary.json')
    fields=read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json')
    physical=read(ROOT/'runs/s7_wire_graph_v1/reference_connectivity.json')
    reviews=read(ROOT/'data/junction_classifier_v1/reviewed_annotations.json')
    training=read(ROOT/'runs/training/junction_classifier_v1/evaluation.json')
    splits={case:s for s,cases in reviews['split_groups'].items() for case in cases}
    totals=defaultdict(Counter);structure=Counter();export=Counter();cases=[];junctions=[]
    manual={(r['case_id'],r['candidate']['junction_id']):r for r in reviews['records']}
    for row in batch['cases']:
        case=row['case_id'];scope=splits.get(case,'no_junction_review')
        pin=read(ROOT/f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json')
        truth=[r for r in fields['references'] if r['case_id']==case]
        reference=physical['cases'][case]
        if sha(ROOT/f'data/standardized/{case}/{case}.png')!=pin['image_sha256'] or reference['image_sha256']!=pin['image_sha256'] or fields['image_sha256'][case]!=pin['image_sha256']:
            raise ValueError('Offline references do not match input image')
        owner_map=component_matches([{'bbox':c['bbox_top_left'],'class_name':c['class_name']} for c in pin['components']],fields['detector_references'][case]['objects'])
        record={'case_id':case,'classifier_split':scope}
        for variant,path in [('v1',ROOT/f'runs/s7_wire_graph_v1/predictions/{case}/graph_proposals.json'),('learned',output/f'predictions/{case}/graph_proposals.json')]:
            graph=read(path)
            if graph['image_sha256']!=pin['image_sha256']:raise ValueError('Graph/image mismatch')
            score=pair_counts(graph,pin,truth,reference['reference_pin_networks'],owner_map);record[variant]=score
            for bucket in ['all_development',scope]:
                for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']:
                    for field in ['tp','fp','fn']:totals[(bucket,variant)][key+'_'+field]+=score[key][field]
                totals[(bucket,variant)]['localized_owned_pins']+=score['localized_owned_pins']
            if variant=='learned':
                structure.update({k:v for k,v in graph['summary'].items() if isinstance(v,int) and not isinstance(v,bool)})
                coverage=read(path.parent/'export_coverage.json')
                export.update({k:coverage[k] for k in ['exported_components','exported_pins','exported_nets']})
                for j in graph['junction_candidates']:
                    reviewed=manual.get((case,j['junction_id']))
                    junctions.append({'case_id':case,**j,'review_id':reviewed['review_id'] if reviewed else None,
                                      'expected':reviewed['label'] if reviewed else None,'classifier_split':scope})
        cases.append(record)
    aggregate=defaultdict(dict)
    for (bucket,variant),counts in totals.items():
        aggregate[bucket][variant]={'localized_owned_pins':counts['localized_owned_pins']}
        for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']:
            aggregate[bucket][variant][key]=metrics(*(counts[key+'_'+f] for f in ['tp','fp','fn']))
    status_counts=Counter(j['status'] for j in junctions)
    result={'cases':cases,'aggregate':dict(aggregate),'structure_totals':dict(structure),'partial_export_totals':dict(export),
            'junction_status_counts':dict(status_counts),'junctions':junctions,'classifier_training':training,
            'official_NetF1':None,'engineering_test_independent':False,'reference_is_official_Golden':False,
            'reviewer':'assistant_visual_inspection','pipeline_scope':'Same 26 upstream development validation pages. Scores are draft pin-pair proxies, not competition NetF1.',
            'classifier_test_scope':'13 labeled crops from 7 schematic groups, unseen by the junction classifier only. Upstream models have used these development pages.',
            'reference_connectivity_sha256':sha(ROOT/'runs/s7_wire_graph_v1/reference_connectivity.json'),
            'junction_model_sha256':sha(ROOT/'weights/junction_classifier_v1.pt')}
    save(output/'evaluation.json',result)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',12)
    for offset in range(0,len(junctions),24):
        rows=junctions[offset:offset+24];sheet=Image.new('RGB',(6*230,((len(rows)+5)//6)*250),'white')
        for i,j in enumerate(rows):
            row=manual.get((j['case_id'],j['junction_id']))
            if not row:continue
            card=Image.new('RGB',(230,250),'white')
            with Image.open(ROOT/'data/topology_review_v1/crops'/f"{row['review_id']}_preview.png") as crop:card.paste(crop.crop((0,0,220,218)),(5,0))
            d=j.get('classifier_decision',{});resolved=j['resolved']
            text=('JOIN' if j['status']=='learned_connected_junction' else 'CROSS' if resolved else 'REVIEW')
            draw=ImageDraw.Draw(card);draw.text((5,220),f"{text} minP={d.get('rotation_min_confidence',0):.3f} {j['classifier_split']}",font=font,fill='#087e33' if resolved else '#c06a00')
            expected={'connected_junction':'join','crossing_without_connection':'cross','segmentation_artifact':'artifact','uncertain':'uncertain'}[row['label']]
            draw.text((5,235),'Review reference: '+expected,font=font,fill='#333333')
            sheet.paste(card,((i%6)*230,(i//6)*250))
        sheet.save(output/f'junction_predictions_{offset//24+1:02d}.png')
    def pct(value):return '—' if value is None else f'{100*value:.2f}%'
    lines=['# 学习式交点判断与导线图 v2 试验','',
           'RGB 原图交点裁剪 → 双尺度小型 CNN → 四次旋转一致性检查 → 高置信度图连接决策。导线仍由冻结 U-Net 输出；分类器不使用颜色阈值提取导线。','',
           '## 数据与验证范围','',
           '60 个候选由助手看图做开发复核：23 个连接点、35 个无连接跨线、1 个文字分割伪影、1 个疑难点。原复核队列和 YOLO/U-Net 标注没有修改。',
           '原理图分组划分：32 个训练裁剪；校准组 14 个（其中 1 个伪影不用于两类损失）；留出 7 张原理图上的 13 个两类裁剪；另 1 个疑难点排除。1024 个程序绘制样本只进入训练。',
           '训练页、校准页、留出页的图像 SHA 互斥。校准交叉熵选权重，0.95 阈值在留出评估前固定；留出结果没有用来调参。',
           '**留出仅针对交点分类器。所有页来自上游模型已使用过的 26 张开发验证页；助手视觉复核也不是官方 Golden，不能称整个系统独立测试准确率。**','',
           '| 分类器分组 | 两类样本 | 直接预测正确 | 接受决策 | 接受后正确 | 待复核 |','|---|---:|---:|---:|---:|---:|']
    for split,label in [('calibration','校准'),('test','分类器留出')]:
        s=training['scores'][split];lines.append(f"| {label} | {s['count']} | {s['raw_correct']} | {s['accepted_count']} | {s['accepted_correct']} | {s['abstained']} |")
    lines += ['',f"CNN 参数 {training['parameters']:,}，训练 {len(training['history'])} 轮，采用第 {training['best_epoch']} 轮，耗时 {training['wall_seconds']:.1f} 秒。",'',
              '**试验模型尚不作为默认入口。留出样本 0149_junction_001 把可见连接点判成跨线，四次旋转最低分仍为 0.994。13 个留出裁剪直接预测正确 11 个；只接受 5 个，其中 4 个正确，另 8 个待复核。Softmax 分数不是经大量样本验证的正确概率。**',
              '冻结权重后没有利用这 13 个留出裁剪继续调参。v1 保守入口保持默认；v2 供开发比较和错误定位使用。','',
              '## 网络图草稿指标','',
              '仅从图像模型输出做推理；下面的参考连接关系只由离线评估脚本读取。原 edgeId 草稿存在缺失或逻辑/物理分组混杂，不是正式 NetF1。','',
              '| 范围 | 方案 | 引脚对指标 | P | R | F1 | TP / FP / FN |','|---|---|---|---:|---:|---:|---|']
    for bucket,label in [('all_development','26 页开发观察'),('test','分类器留出 7 页')]:
        for variant,title in [('v1','v1 保守分离'),('learned','v2 学习判断')]:
            for key,metric_label in [('conditional_connectivity_pairs','已匹配位置/归属'),('all_reference_pair_proxy','全量参考引脚对代理')]:
                s=aggregate[bucket][variant][key];lines.append(f"| {label} | {title} | {metric_label} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} | {s['tp']} / {s['fp']} / {s['fn']} |")
    lines += ['',f"交点状态：{dict(status_counts)}。剩余疑难点 {structure['uncertain_junctions']}；断线候选 {structure['gap_candidates_not_filled']}，没有自动填补。",'',
              f"部分格式输出：{export['exported_components']} 个元件、{export['exported_pins']} 个字段齐全引脚、{export['exported_nets']} 条严格网络。完整比赛 JSON 尚未完成。",'',
              '## 使用与后续','',
              '在项目目录运行 `powershell -File scripts/run_learned_wire_graph.ps1 -Image data/standardized/0002/0002.png -Device cuda:0`。这是新配置入口；原 `run_wire_graph.ps1` 继续使用 v1。',
              '绿色框是模型判断为连接的点，紫色框是模型判断为跨线的点，橙色框仍待复核。所有引脚原坐标保持原样；分类器不删除分割伪影、不猜测缺失引脚字段。',
              '下一阶段需补充真正新来源的带复核交点数据、断线与漏分割案例，并推进网络标签关联和引脚字段覆盖。当前二分类不能保证识别所有伪影或无圆点但实际连接的绘图规范。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'aggregate':aggregate['all_development'],'junction_status':dict(status_counts),'export':dict(export)})


if __name__=='__main__':main()
