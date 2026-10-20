import copy
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
before=read(WORK/'pin_relocation_implementation_20261020/deployed/4_Pred.json');after=read(ROOT/'4_Pred.json')
audit=read(ROOT/'generation_audit.json');result=read(ROOT/'results.json')
assert len(audit['changes'])==2 and audit['groups'][0]['status']=='complete_observed_rekey_and_restore'
assert after['components']==before['components']
for owner,pins in before['pins'].items():
    for key,pin in pins.items():
        if owner=='U2' and key=='pin_3':assert after['pins'][owner]['pin_2']==pin
        else:assert after['pins'][owner][key]==pin
for nid,net in before['nets'].items():assert after['nets'][nid]['edges']==net['edges']
assert after['nets']['LOG_NET_CAND_00005']['hyperGraph']=='(C5.1,C5.3,U2.2)'
vin_nets=[nid for nid,net in after['nets'].items() if 'U2.3' in net['hyperGraph'].strip('()').split(',')]
assert vin_nets==['RESTORED_NET_CAND_00015']
new=after['nets'][vin_nets[0]]
item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='4')
graph=read(Path(item['model_cache'])/'p/export/engineering_graph_proposals.json')
physical=next(n for n in graph['physical_networks'] if n['network_id']=='NET_CAND_00015')
emap={e['edge_id']:e for e in graph['edges']};h=graph['size'][1]
observed_segments=[]
for eid in physical['edge_ids']:
    poly=emap[eid]['polyline']
    observed_segments.extend([[{'x':float(a[0]),'y':float(h-a[1])},{'x':float(b[0]),'y':float(h-b[1])}] for a,b in zip(poly,poly[1:])])
assert list(new['edges'].values())==observed_segments
plan=read(WORK/'pin_relocation_implementation_20261020/release_plan.json')
assert all(hashlib.sha256((Path(plan['target'])/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in plan['files'])
verification={'all_other_existing_pins_unchanged':True,'all_existing_wire_edges_unchanged':True,
              'new_branch_exactly_matches_model_segments':True,'stale_VIN_reference_removed_from_VOUT_net':True,
              'VIN_networks':vin_nets,'production_unchanged':True,'new_local_ocr_calls':4,
              'reused_local_ocr_calls':4,'external_api_calls':0,'full_model_runs':0}
(ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');rgb=Image.open('D:/pcb_open_gt_in/4.png');crop=[1285,170,1595,310]
fig,axes=plt.subplots(1,2,figsize=(14,4.5),dpi=150)
for ax,title in zip(axes,['修复前：3 号占在 VOUT 行，VIN 未导出','联合校正后：2/VOUT、3/VIN 各归其位']):
    ax.imshow(rgb.crop(crop),extent=[crop[0]-.5,crop[2]-.5,crop[3]-.5,crop[1]-.5],interpolation='nearest')
    ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.axis('off');ax.set_title(title,fontproperties=font,fontsize=14,pad=12)
def point(doc,key):
    p=doc['pins']['U2'][key]['point'];return p['x'],h-p['y']
old=point(before,'pin_3');vout=point(after,'pin_2');vin=point(after,'pin_3')
axes[0].plot(*old,'x',color='#d35400',markersize=12,markeredgewidth=2.5)
axes[0].annotate('误标为 3 / VOUT',xy=old,xytext=(1292,297),fontproperties=font,fontsize=11,color='#b74700',arrowprops={'arrowstyle':'->','color':'#d35400'})
for p,label,textxy in [(vout,'2 / VOUT',(1292,205)),(vin,'3 / VIN',(1292,297))]:
    axes[1].plot(*p,'o',markersize=10,markerfacecolor='none',markeredgewidth=2,color='#0077cc')
    axes[1].annotate(label,xy=p,xytext=textxy,fontproperties=font,fontsize=12,color='#0066aa',arrowprops={'arrowstyle':'->','color':'#0077cc','lw':1.5})
fig.text(.5,.025,'第四页实验：12 个参考 IC 引脚全部得分；VOUT 网络引用同步改号，VIN 恢复到真实 +5V 分支。',ha='center',fontproperties=font,fontsize=11)
fig.subplots_adjust(left=.01,right=.99,top=.85,bottom=.10,wspace=.04)
fig.savefig(ROOT/'page4_pairing_result.png',facecolor='white');plt.close(fig)
report=f'''# 第四页编号与名称联合配对实验

现象：同一个数字 3 的 OCR token PT_00038 被错误地用于上方 VOUT 的编号，
下方 VIN 因 text_used_by_other_pin 没有编号。已有字段恢复虽然重新读到 3，但因已有 pin_3 而弃权。
几何恢复的数字偏移又受到错绑记录影响。因此需要完整、一次性的联合改号。

本实验通过“共享数字 token + 同元件同侧候选”定位冲突，未硬编码目标编号、名称或参考坐标。
数字列和上方偏移从同侧可见数字估计，真实红色短引线确认候选行。
两个局部数字分别经 3x/5x OCR 验证：2 的置信度 0.99964/0.99964，3 为 0.99889/0.99878；
VOUT、VIN 各有 4 个元件 OCR 视图支持，均按既有 0.98 门槛和不同视图投票规则采信。

原 VOUT 的坐标保持不变，仅将 pin_3 改为 pin_2，同时将 LOG_NET_CAND_00005 的
U2.3 引用改为 U2.2。下方 VIN 使用真实短引线末端补为 pin_3。
随后将确认过的编号写入模型候选副本，复用现有 power_wire_export 逻辑恢复
RESTORED_NET_CAND_00015 电源分支。新分支的线段逐项等于模型已有 W00017，未凭空画线。
所有原有其他引脚、元件和导线线段保持不变，原 VOUT 网络不再残留 U2.3。

| 本地指标 | 基线 | 实验 |
| --- | ---: | ---: |
| 正确 IC 引脚 | 10 | 12 |
| 漏检 | 2 | 0 |
| 误检 | 1 | 0 |
| 引脚 F1 | 0.8696 | 1.0000 |
| 网络连接得分 | {result['before']['connectivity']:.6f} | {result['after']['connectivity']:.6f} |
| 导线 TP | 36 | 37 |
| 导线 F1 | 0.6792 | 0.6916 |
| 当前本地加权总分 | {result['before']['score_100']:.4f} | {result['after']['score_100']:.4f} |

预测和所有采信决定在读取 GT 之前冻结，指标是本地复核，未提交平台。
新增 4 次本地 OCR，复用 4 次观测，没有外部 API 或整套模型重跑。
正式代码文件哈希保持一致。本次仍是第四页定向实验，尚未完成通用运行模块、跨页回归和正式入口集成。
完整候选和网络变动见 generation_audit.json，一致性验证见 verification.json。
'''
(ROOT/'实验报告.md').write_text(report,encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False))
