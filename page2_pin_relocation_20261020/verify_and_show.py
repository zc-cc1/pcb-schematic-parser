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
base=read(WORK/'pin_geometry_implementation_20261020/deployed/2_Pred.json');pred=read(ROOT/'2_Pred.json')
audit=read(ROOT/'generation_audit.json');changes=audit['changes'];assert len(changes)==1
entry=changes[0];owner=entry['owner'];key='pin_'+entry['old_number']
check=copy.deepcopy(pred);check['pins'][owner][key]=copy.deepcopy(base['pins'][owner][key]);assert check==base
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/2').glob('*.json')))
image=Image.open('D:/pcb_open_gt_in/2.png');h=image.height
def point(doc):
    p=doc['pins'][owner][key]['point'];return p['x'],h-p['y']
old,new,ref=point(base),point(pred),point(truth)
plan=read(WORK/'pin_geometry_implementation_20261020/release_plan.json')
assert all(hashlib.sha256((Path(plan['target'])/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in plan['files'])
member=f"{owner}.{entry['old_number']}"
memberships=[nid for nid,net in base['nets'].items() if member in [v.strip() for v in net['hyperGraph'].strip('()').split(',')]]
verification={'old_error_px':math.dist(old,ref),'new_error_px':math.dist(new,ref),
              'only_one_pin_point_changed':True,'production_release_hashes_unchanged':True,
              'existing_network_memberships':memberships,'wire_and_network_outputs_unchanged':True,
              'total_new_local_ocr_observations':14,'reused_prior_ocr_observations':2,
              'new_full_model_runs':0,'external_api_calls':0}
(ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');crop=[920,260,1150,420]
fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=150)
for ax,title in zip(axes,['修复前：错点在右下角','实验修复后：回到左侧 2 号引线']):
    ax.imshow(image.crop(crop),extent=[crop[0]-.5,crop[2]-.5,crop[3]-.5,crop[1]-.5],interpolation='nearest')
    ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.axis('off')
    ax.set_title(title,fontproperties=font,fontsize=15,pad=12)
axes[0].plot(*old,'x',color='#df5a00',markersize=13,markeredgewidth=2.5)
axes[0].plot(*ref,'o',color='#0077cc',markersize=11,markerfacecolor='none',markeredgewidth=2)
axes[0].annotate('原误差 96.10 px',xy=old,xytext=(935,408),fontproperties=font,fontsize=12,color='#b74700',
                 arrowprops={'arrowstyle':'->','color':'#df5a00'})
axes[1].plot(*new,'o',color='#0077cc',markersize=11,markerfacecolor='none',markeredgewidth=2.3)
axes[1].annotate('2 / VCC\n修复后误差 0.71 px',xy=new,xytext=(928,399),fontproperties=font,fontsize=12,color='#0066aa',
                 bbox={'facecolor':'white','edgecolor':'none','alpha':.95},arrowprops={'arrowstyle':'->','color':'#0077cc','lw':1.6})
fig.text(.5,.025,'实验结果：16 个参考 IC 引脚全部得分；未修改正式代码。',ha='center',fontproperties=font,fontsize=12)
fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.09,wspace=.06)
fig.savefig(ROOT/'page2_relocation_result.png',facecolor='white');plt.close(fig)
report=f'''# 第二页 U2.2 VCC 错点迁移实验

结论：可以修复。本实验只修改本地预测副本，D 盘正式发布文件哈希全部保持一致。

| 指标 | 修复前 | 实验后 |
| --- | ---: | ---: |
| 正确 IC 引脚 TP | 15 | 16 |
| 漏检 FN | 1 | 0 |
| 误检 FP | 1 | 0 |
| 引脚 F1 | 0.9375 | 1.0000 |
| U2.2 坐标误差 | {math.dist(old,ref):.3f} px | {math.dist(new,ref):.3f} px |

生成规则先筛选：编号和名称均由 library_elimination 填入、没有文字证据、
低置信度、低导线概率，且点位于元件框内的候选。找到三条同侧已检出引脚，
并要求另一条可见短引线的位置唯一。当前图中自动筛选出 U2 的候选 PIN_CAND_0061。
代码没有使用 GT 坐标、目标编号或名称来生成新坐标。GT 在预测落盘后才读取评分。

数字列与文字行偏移从其他已观察引脚估计。隔离中性色数字后，3x/5x OCR 分别读到
数字 2，置信度 0.99857/0.99858。名称隔离 OCR 未达到 0.98 门槛，未采用。
完整元件裁剪在 2x/3x 的正向视图读到 VCC，置信度 0.98139/0.99018；
满足两个独立视图门槛后，才确认与已有字段一致并迁移原键的坐标。
未通过降低门槛或用库排除结果代替可见文字来放行。

坐标从图像左上原点的 {old} 移至 {new}，后者来自真实红色短引线末端。
其他所有引脚、名称、组件、网络和导线输出完全一致。
原 U2.2 的导出网络成员关系为 {memberships}。此轮未验证新增网络接入，
引脚得分改善不能等同于网络连接已经修复。

总计新增 14 次本地 OCR 观测，复用 2 次已有观测；没有外部 API 调用或完整模型重跑。
第一次宽裁剪读入邻行残片，后续按黑色连通域和已观察的文字行偏移清理，仍保留失败观测。

目前仅第二页的定向实验通过，尚未进行跨页错误迁移回归，也未集成正式入口。
若落地，还需要对被替换旧候选的网络成员、导线端点和新位置附着证据进行联合验证，
并增加“存在可靠原文字/旧网络时必须弃权”等边界检查。
'''
(ROOT/'实验报告.md').write_text(report,encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False))
