"""Diagnostic plot of frozen prediction and reference; no prediction changes."""
from pathlib import Path
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
pred=read(ROOT/'deployed/2_Pred.json')
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/2').glob('*.json')))
rgb=Image.open('D:/pcb_open_gt_in/2.png');h=rgb.height
crop=[885,255,1150,430]
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
def point(doc):
    p=doc['pins']['U2']['pin_2']['point'];return p['x'],h-p['y']
bad=point(pred);good=point(truth)
fig,axes=plt.subplots(1,2,figsize=(14,5.5),dpi=150)
for ax in axes:
    ax.imshow(rgb.crop(crop),extent=[crop[0]-.5,crop[2]-.5,crop[3]-.5,crop[1]-.5],interpolation='nearest')
    ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.axis('off')
axes[0].set_title('第二页 U2：局部原图',fontproperties=font,fontsize=17,pad=15)
ax=axes[1];ax.set_title('当前预测点与正确位置对照',fontproperties=font,fontsize=17,pad=15)
ax.plot(*good,marker='o',markersize=12,markerfacecolor='none',markeredgewidth=2.3,color='#0077cc')
ax.plot(*bad,marker='x',markersize=12,markeredgewidth=2.3,color='#df5a00')
ax.plot([good[0],bad[0]],[good[1],bad[1]],'--',color='#777777',linewidth=1.4)
ax.annotate('正确的 2 号脚 VCC\n位于左侧短引线末端',xy=good,xytext=(897,390),
            fontproperties=font,fontsize=12,color='#0066aa',
            bbox={'facecolor':'white','alpha':.96,'edgecolor':'none','pad':4},
            arrowprops={'arrowstyle':'->','color':'#0077cc','lw':1.6})
ax.annotate('当前错点\n误差约 96 px',xy=bad,xytext=(1104,350),
            fontproperties=font,fontsize=12,color='#b74700',
            bbox={'facecolor':'white','alpha':.96,'edgecolor':'none','pad':4},
            arrowprops={'arrowstyle':'->','color':'#df5a00','lw':1.6})
fig.text(.5,.025,'蓝圈为参考标注位置，仅用于事后核对；橙叉为当前实际导出位置。',
         ha='center',fontproperties=font,fontsize=12,color='#444444')
fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.10,wspace=.06)
dest=ROOT/'page2_U2_VCC_diagnostic.png';fig.savefig(dest,facecolor='white');plt.close(fig)
print(str(dest));print('distance',math.dist(good,bad))
