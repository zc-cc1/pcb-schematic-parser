"""Coverage/provenance audit, not an official competition accuracy evaluator."""
import argparse
from collections import Counter
from pathlib import Path
from run_submission_fields import ROOT,read,save,sha
from pcb_parser.dataset_converter import validate_output


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s10_submission_fields_v1');a=p.parse_args()
    batch=read(a.output/'batch_summary.json');before=Counter();after=Counter();totals=Counter();reasons_before=Counter();reasons_after=Counter()
    identities=[];suggestions=[];cases=[]
    for row in batch['cases']:
        case=row['case_id'];folder=a.output/'predictions'/case;summary=read(folder/'summary.json')
        source=read(ROOT/f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json');pins=read(folder/'completed_pin_text_result.json')
        graph=read(ROOT/f'runs/s9_network_labels_v1/predictions/{case}/logical_graph_proposals.json')
        if summary['image_sha256']!=sha(ROOT/f'data/standardized/{case}/{case}.png') or pins['original_pin_text_sha256']!=sha(ROOT/f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json'):
            raise ValueError('Image/pin provenance changed')
        if summary['input_network_sha256'] != sha(ROOT/f'runs/s9_network_labels_v1/predictions/{case}/logical_graph_proposals.json'):
            raise ValueError('Input graph changed')
        if [(p['candidate_id'],p['point']) for p in source['pin_candidates']] != [(p['candidate_id'],p['point']) for p in pins['pin_candidates']]:
            raise ValueError('Pin tips changed')
        if [p['pinname'] for p in source['pin_candidates']] != [p['pinname'] for p in pins['pin_candidates']]:
            raise ValueError('Pin names guessed')
        submission=read(folder/'partial_submission.json');errors,bounds=validate_output(submission,*pins['size'])
        if errors or bounds:raise ValueError('Invalid partial JSON')
        for key in ['exported_components','exported_pins','exported_nets']:
            before[key]+=summary['before_export'][key];after[key]+=summary['after_export'][key]
        reasons_before.update(summary['before_export']['pin_reason_counts']);reasons_after.update(summary['after_export']['pin_reason_counts'])
        totals.update({k:summary['summary'][k] for k in ['new_label_identities','label_identity_proposals','single_tip_label_auto_numbers']})
        for key in ['multi_pin_networks','topology_clear_but_field_incomplete','local_OCR_rois','local_spelling_suggestions']:
            totals[key]+=summary[key]
        proposals=read(folder/'identity_proposals.json')['components']
        identities.extend({'case_id':case,**p} for p in proposals if p['new_identity_accepted'])
        readings=read(folder/'label_roi_readings.json')['readings']
        suggestions.extend({'case_id':case,'site_id':r['site_id'],'original_label':r['original_label'],'suggested_spelling':r['suggested_spelling'],'applied':False}
                           for r in readings if r['suggested_spelling'])
        cases.append({'case_id':case,'before':summary['before_export'],'after':summary['after_export'],'fields':summary['summary']})
    result={'cases':cases,'before_export':dict(before),'after_export':dict(after),'counts':dict(totals),
            'before_pin_missing_fields':dict(reasons_before),'after_pin_missing_fields':dict(reasons_after),
            'new_identity_predictions':identities,'unapplied_OCR_suggestions':suggestions,
            'original_pin_coordinates_preserved':True,'network_membership_changes':0,'pinname_guesses':0,
            'all_local_JSON_contracts_valid':True,'official_NetF1':None,'accuracy_improvement_measured':False,
            'scope':'26 reused development pages; output coverage and provenance counts, not independent recognition accuracy'}
    save(a.output/'audit.json',result)
    text=['# 输出字段补全与缺口定位 v1','',
          '本轮把普通标签身份、有限格式自动编号和网络字段缺口接入输出。检测器、U-Net、交点分类器、普通标签合并逻辑及训练数据均保持原样。',
          '运行只使用PNG及图像模型结果。字段身份仍是预测；没有官方Golden/评测器，不能把新增输出数量当作准确率提升。','',
          '## 赛题字段规则','',
          '已视觉核对0803Rev1第14–18页：重复designator统一加_1、_2等后缀，顺序依赖原始component ID的自然排序；pin按component key和number匹配，pinname严格比较；单引脚RUN标签可自动输出pin_1。',
          '原PDF SHA256：45eedbc195ad71bbb03044c4606ea85ccf3eccebbb3d9b3304d6e325e1acc5e8。PNG无法恢复原始component ID顺序，重复标签的视觉顺序后缀只保留在identity_proposals.json，不进入严格部分JSON。',
          'pinname必须是字符串。当前PDF没有给出未知名称应填空字符串的充分规则，因此未填空串、未复制网名到pinname。R/C/IC等多引脚器件的自动顺序未猜测。','',
          '## 26页输出覆盖','',
          '| 项目 | 上一阶段 | 本阶段 |','|---|---:|---:|']
    for key,label in [('exported_components','部分JSON元件数'),('exported_pins','字段齐全的pin数'),('exported_nets','通过当前严格门槛的net数')]:
        text.append(f'| {label} | {before[key]} | {after[key]} |')
    text+=['',f"新增标签身份预测 {totals['new_label_identities']} 个；单引脚标签格式编号 {totals['single_tip_label_auto_numbers']} 个。后者不表示识别出了编号字符，也不增加OCR识别准确率。",
           f"多引脚网络候选 {totals['multi_pin_networks']} 个，其中 {totals['topology_clear_but_field_incomplete']} 个没有记录到交叉/断线阻断，但仍缺字段。这不证明实际拓扑正确。",'',
           '| 未决引脚字段 | 上一阶段 | 本阶段 |','|---|---:|---:|']
    for key,label in [('component_identity_unresolved','元件身份'),('pin_number_unresolved','编号'),('pin_name_unresolved','名称')]:
        text.append(f'| {label} | {reasons_before[key]} | {reasons_after[key]} |')
    text+=['','新增标签身份预测：']
    for r in identities:text.append(f"- {r['case_id']}：`{r['accepted_component_key']}`，检测索引{r['detection_index']}，已有{r['existing_OCR_view_count']}个OCR视图支持。")
    text+=['','同图空格/下划线或大小写近似标签会阻止身份被误当成唯一编号；此门槛不改字串，也不合并网络。两端Net_short标签默认保留身份候选，等待更完整的符号及编号语义。','',
           '## 局部文字复读','',
           f"对 {totals['local_OCR_rois']} 个已有普通符号标签裁剪进行2/3/4倍RGB复读，得到 {totals['local_spelling_suggestions']} 个待复核拼写建议；应用替换0次。多个尺度来自同一个模型，不是独立验证。"]
    for r in suggestions:text.append(f"- {r['case_id']}：`{r['original_label']}` → `{r['suggested_spelling']}`，只记录建议。")
    text+=['','小裁剪仍会遗漏下划线，并可把此前正确的ADC_IN12读成ADC IN12；尚不采用全量覆盖策略。','',
           '## 调试文件与运行','',
           '`completed_pin_text_result.json`保留原tip与candidate ID，新增字段保存证据；`identity_proposals.json`区分原名、候选后缀、接受身份和阻断原因。',
           '`network_field_review_queue.json`按拓扑阻断及缺字段数量排序，列出每个pin的身份/编号/pinname缺口；`network_fields_*.png`和`field_review_sheet.png`定位原图区域。不要把复核队列自动作为训练真值。',
           '`partial_submission.json`是经过本地结构/引用/坐标检查的部分输出，仍不能完整提交。','',
           '完整图片入口：','',
           '```powershell',
           '.\\scripts\\run_submission_fields.ps1 -Image data/standardized/0170/0170.png -Device cuda:0',
           '```','',
           '冻结上游调试：运行`scripts/batch_submission_fields.py`和`scripts/audit_submission_fields.py`；它们重放旧26页模型预测，保留上游图不变。',
           '下一项优先推进：补齐IC/连接器的身份、编号与名称，并建立可核验的引脚字段参考；重复标签编号规则需要官方Golden确认。严格nets仍为0，不能宣称已完成赛事输出。','']
    (a.output/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print({'before':dict(before),'after':dict(after),'counts':dict(totals)},flush=True)


if __name__=='__main__':main()
