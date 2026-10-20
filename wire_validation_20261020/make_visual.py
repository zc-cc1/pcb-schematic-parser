"""Prediction-only visual check of a changed real wire; no GT is read."""
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent


def segments(payload):
    return [tuple((float(p['x']),float(p['y'])) for p in points)
            for net in payload['nets'].values() for points in net['edges'].values() if len(points)==2]


def key(segment):
    return tuple(sorted(tuple(round(v,2) for v in p) for p in segment))


def main():
    case='11'
    baseline=json.loads((HERE/f'baseline/final_results/{case}_Pred.json').read_text())
    candidate=json.loads((HERE/f'candidate_path/final_results/{case}_Pred.json').read_text())
    a,b=segments(baseline),segments(candidate)
    new={key(s) for s in b}
    changed=[s for s in a if key(s) not in new]
    angled=[s for s in changed if min(abs(s[0][0]-s[1][0]),abs(s[0][1]-s[1][1]))>4]
    chosen=max(angled or changed,key=lambda s:math.dist(*s))
    image=Image.open('D:/pcb_open_gt_in/'+case+'.png').convert('RGB')
    height=image.height
    xs=[p[0] for p in chosen];ys=[height-p[1] for p in chosen]
    box=(max(0,int(min(xs))-30),max(0,int(min(ys))-30),
         min(image.width,int(max(xs))+31),min(image.height,int(max(ys))+31))
    crop=image.crop(box)
    scale=min(3.0,550/max(crop.size))
    size=tuple(max(1,round(v*scale)) for v in crop.size)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    panels=[]
    for title,lines,color in [('Original PNG',[],None),('Baseline',a,'#E02424'),('Preserve path',b,'#1565C0')]:
        panel=Image.new('RGB',(size[0],size[1]+30),'white')
        panel.paste(crop.resize(size),(0,30))
        draw=ImageDraw.Draw(panel);draw.text((8,6),title,fill='#111111',font=font)
        for segment in lines:
            pts=[((x-box[0])*scale,(height-y-box[1])*scale+30) for x,y in segment]
            if all((p[0]<0 or p[0]>size[0] or p[1]<30 or p[1]>size[1]+30) for p in pts):
                continue
            draw.line(pts,fill=color,width=2)
            for x,y in pts:draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        draw.rectangle((0,0,size[0],29),fill='white')
        draw.text((8,6),title,fill='#111111',font=font)
        panels.append(panel)
    result=Image.new('RGB',(sum(p.width for p in panels)+20,max(p.height for p in panels)),'#DDDDDD')
    x=0
    for panel in panels:result.paste(panel,(x,0));x+=panel.width+10
    result.save(HERE/'case11_wire_comparison.png')
    (HERE/'visual_crop.json').write_text(json.dumps({'case':case,'box':box,'baseline_changed_segment':chosen},indent=2))
    print(HERE/'case11_wire_comparison.png')


if __name__=='__main__':main()
