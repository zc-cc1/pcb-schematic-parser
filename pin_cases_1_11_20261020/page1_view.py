"""Create page-one-only diagnostic views from existing results; no model calls."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
row=next(r for r in json.loads((ROOT/'trace.json').read_text(encoding='utf-8')) if r['case']=='1')
im=Image.open('D:/pcb_open_gt_in/1.png').convert('RGB')
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
draw=ImageDraw.Draw(im)
counts={c:sum(d['failure']['component']==c for d in row['details']) for c in row['components']}
for c,item in row['components'].items():
    x1,y1,x2,y2=item['record']['bbox_top_left']
    draw.rectangle((x1-12,y1-12,x2+12,y2+12),outline='#ad1481',width=3)
    label=f'{c}：{counts[c]} 个未得分'
    y=max(2,int(y1)-46)
    box=draw.textbbox((x1,y),label,font=font)
    draw.rectangle((box[0]-4,box[1]-4,box[2]+4,box[3]+4),fill='white')
    draw.text((x1,y),label,font=font,fill='#ad1481')
im.save(ROOT/'第1页_问题分布.png')

source=Image.open('D:/pcb_open_gt_in/1.png').convert('RGB')
for c in ['USB1','USB2','U1','U2','U3']:
    x1,y1,x2,y2=row['components'][c]['record']['bbox_top_left']
    cropbox=(max(0,int(x1)-55),max(0,int(y1)-60),min(source.width,int(x2)+90),min(source.height,int(y2)+55))
    scale=3
    crop=source.crop(cropbox).resize(((cropbox[2]-cropbox[0])*scale,(cropbox[3]-cropbox[1])*scale))
    canvas=Image.new('RGB',(crop.width,crop.height+55),'white')
    canvas.paste(crop,(0,55));d=ImageDraw.Draw(canvas)
    d.text((12,10),f'{c}：红圈 = 未得分引脚的参考端点；蓝叉 = 位置错误的输出',font=small,fill='black')
    for detail in row['details']:
        f=detail['failure']
        if f['component']!=c:continue
        p=f['gt_point'];px=(p['x']-cropbox[0])*scale;py=(source.height-p['y']-cropbox[1])*scale+55
        d.ellipse((px-11,py-11,px+11,py+11),outline='#df0076',width=4)
        if f['export_stage']=='wrong_position':
            q=f['exported']['point'];qx=(q['x']-cropbox[0])*scale;qy=(source.height-q['y']-cropbox[1])*scale+55
            d.line((px,py,qx,qy),fill='#0067d5',width=3)
            d.line((qx-8,qy-8,qx+8,qy+8),fill='#0067d5',width=4)
            d.line((qx-8,qy+8,qx+8,qy-8),fill='#0067d5',width=4)
    canvas.save(ROOT/f'第1页_{c}_问题点.png')
print('Saved page 1 overview and five detail images.')
