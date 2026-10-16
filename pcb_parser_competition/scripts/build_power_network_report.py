"""Visual evidence cards and paired-metric report for the power pilot."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from run_power_networks import ROOT,read,save


def main():
    output=ROOT/'runs/s15_power_networks';batch=read(output/'predictions/batch_summary.json');evaluation=read(output/'evaluation.json')
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',13);header=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    reviews=[]
    prior_path=ROOT/'data/reviews/power_networks_v1.json'
    prior=read(prior_path) if prior_path.is_file() else {'groups':[]}
    for item in evaluation['applied_power_groups']:
        case=item['case_id'];graph=read(output/f'predictions/{case}/power_graph_proposals.json')
        sites=[s for s in graph['power_sites'] if s['site_id'] in item['site_ids']]
        net=next(n for n in graph['networks'] if set(item['input_network_ids'])<=set(n.get('input_network_ids',[])))
        records=read(ROOT/f'runs/s14_component_library/predictions/{case}/pin_text/component_records.json')['records']
        cmap={r['detection_index']:r for r in records};edges={e['edge_id']:e for e in graph['edges']}
        card=Image.new('RGB',(1050,82+((len(sites)+2)//3)*258+80),'white');draw=ImageDraw.Draw(card)
        draw.text((12,8),f"{case}: exact {item['label']} | {len(sites)} observed power sites | logical proposal",font=header,fill='#333')
        draw.text((12,34),f"Retains {len(net['pin_candidate_ids'])} pin members / {len(net['edge_ids'])} physical wire edges. Canonical power IDs unresolved.",font=font,fill='#555')
        with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
            for i,site in enumerate(sites):
                b=site['bbox'];cb=cmap[site['detection_index']]['bbox_top_left'];x,y=site['point']
                crop=[max(0,int(min(b[0],cb[0],x))-24),max(0,int(min(b[1],cb[1],y))-32),
                      min(source.width,int(max(b[2],cb[2],x))+25),min(source.height,int(max(b[3],cb[3],y))+28)]
                patch=source.convert('RGB').crop(crop);d=ImageDraw.Draw(patch)
                for eid in net['edge_ids']:
                    d.line([(px-crop[0],py-crop[1]) for px,py in edges[eid]['polyline']],fill='#198a78',width=1)
                d.rectangle([b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]],outline='#96309d',width=1)
                xx,yy=x-crop[0],y-crop[1];d.ellipse([xx-4,yy-4,xx+4,yy+4],outline='#d52b27',width=1)
                scale=min(3,325/patch.width,182/patch.height);patch=patch.resize((round(patch.width*scale),round(patch.height*scale)),Image.Resampling.NEAREST)
                left,top=(i%3)*350+12,82+(i//3)*258
                card.paste(patch,(left,top));draw=ImageDraw.Draw(card)
                draw.text((left,top+188),site['candidate_id']+' | '+site['component_id'],font=font,fill='#333')
                draw.text((left,top+208),f"Views={len(site['strong_OCR_views'])}  {site['input_network_id']}",font=font,fill='#555')
        draw=ImageDraw.Draw(card)
        draw.text((12,card.height-52),'Green: existing learned wire. Purple: actual OCR glyph. Red: unchanged learned tip.',font=font,fill='#444')
        draw.text((12,card.height-30),'No remote line drawn. No member removed. Not an official complete network.',font=font,fill='#8a471d')
        name=f'power_group_{case}_{len(reviews)+1:02d}.png';card.save(output/name)
        reviewed=next((r for r in prior['groups'] if r['case_id']==case and r['site_ids']==item['site_ids']
                       and r.get('image_sha256')==graph['image_sha256']),None)
        reviews.append({'case_id':case,'label':item['label'],'network_id':net['network_id'],'site_ids':item['site_ids'],
                        'input_network_ids':item['input_network_ids'],'pin_candidate_ids':net['pin_candidate_ids'],
                        'card':name,'visually_reviewed':bool(reviewed),'caption_symbol_roles_correct':reviewed.get('caption_symbol_roles_correct') if reviewed else None,
                        'all_physical_member_paths_verified':False,'official_reference_verified':False})
    save(output/'visual_review_queue.json',{'groups':reviews,'training_labels_created':0,'source_annotations_modified':False})
    total=batch['aggregate'];metrics=evaluation['aggregate'];fmt=lambda n:'—' if n is None else f'{n:.2%}'
    lines=['# S15：电源逻辑连接试验','',
      '在 S14 的冻结检测/分割/文字与元件库结果之上，加入有实际文字、符号及唯一U-Net接线点证据的同名电源逻辑候选。仅在同一图中比较完整字面名称；GND、DGND、AGND、GND_ISO，以及VCC_3.3和3V3分别保留。',
      '每处名称须取得同位置至少3种置信度≥0.98的OCR视图支持。视图来自同一模型，不是独立统计证据。两种强视图支持另一拼写、缺/多引脚、接线不唯一、未决交点/断线、漏电源成员风险或普通飞线混入，均保留待决。',
      '赛题第23页说明VCC/GND不按普通飞线处理。本试验只合并网络成员关系，保留所有真实模型线段，不生成跨图连接、远距离导线或普通飞线零长度edges。原tip、所有字段与归属不变。','',
      '| 同一26页旧开发数据 | 数量 |','|---|---:|',
      f"| 有充分电源文字＋符号＋接线证据的节点 | {total['observed_power_sites']} |",
      f"| 应用的电源逻辑候选组 | {total['power_groups_applied']} |",
      f"| 减少的原网络分组 | {total['input_network_groups_joined']} |",
      f"| 检测到的电源组件复核条目 | {total['power_components_in_review']} |",
      f"| 未达到完整文字/接线证据门槛 | {total['power_components_without_qualified_caption']} |",'',
      '应用3组：0018的5处GND，0029的2处PVDD，0105的5处GND。0105的VCC5V0_SYS因邻近导线可能漏掉电源成员而未合并；0170的GND因交点待决而未合并。', '',
      '## 成对草稿指标','',
      '**两种方案均使用同一份新参考：原普通标签草稿连接＋源注释中具有唯一明确关联文字的电源组。没有文字的Gnd类别不默认归入GND。参考未独立核验，不是官方Golden；这张表不能直接与以前只评价物理连接或普通标签的分数比较。**','',
      '| 范围 | 方案 | P | R | F1 | TP / FP / FN |','|---|---|---:|---:|---:|---|']
    for key,title in [('conditional_connectivity_pairs','已匹配归属/5px位置的引脚对'),('all_reference_pair_proxy','计入未匹配预测的全量配对代理')]:
        for version,label in [('baseline_s14','S14原网络'),('power','S15电源试验')]:
            m=metrics[version][key];lines.append(f"| {title} | {label} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {m['tp']} / {m['fp']} / {m['fn']} |")
    lines += ['', '条件范围新增65个与草稿参考一致的连接对，FP仍3；计入无法匹配的预测引脚后，新增候选也带来43个代理FP。不能只宣传条件精确率，也不能把本轮分数叫NetF1或独立测试准确率。旧普通标签参考下的结果保存在 evaluation.json，用来展示新增电源语义如何改变评测含义。','',
      '## 出口与验证','',
      f"严格部分出口仍为{total['conservative']['exported_components']}个元件、{total['conservative']['exported_pins']}个字段齐全pin、{total['conservative']['exported_nets']}个net；候选仍{total['candidate']['exported_pins']}个pin、{total['candidate']['exported_nets']}个net。本轮不推断重复电源组件的源ID顺序，不补电源pin编号/pinname，也不删未决成员来凑完整hyperGraph。新增电源组显式标为provisional并禁止当完整赛事网络导出。",
      '原默认入口、训练数据和权重保持原样。本阶段提供独立PNG入口；没有新增训练。12项电源规则测试与既有出口回归检查、桌面原始PNG验证见runtime_verification.json。','',
      '```powershell',r'.\scripts\run_power_networks.ps1 -Image .\data\standardized\0018\0018.png -Device cpu','```','',
      '输出：power_graph_proposals.json（内部网络候选）、power_binding_audit.json（每组依据与未决理由）、proposal_export/network_field_queue.json（成员字段缺口）。baseline_export与proposal_export分别保存旧/新保守部分JSON；新增电源网络不会自动成为完整答案。','',
      '离线评估脚本可读取原注释/参考，PNG推理入口不导入或读取它们。数据集原注释、YOLO/U-Net训练清单、模型配置和S14结果保持原样。', '',
      '下一步缺口：208处电源组件尚未达到证据门槛，64处缺少明确归属tip；需要进一步改善电源接线点及文字数据，并解决规范组件身份和pin字段。交点与断线也仍限制完整网络。','',
      '## 局部候选复核图','']
    for row in reviews:lines += [f"![{row['case_id']} {row['label']}]({row['card']})",'']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8');print({'groups':len(reviews),'report':str(output/'report.md')},flush=True)


if __name__=='__main__':main()
