"""Generic rendered digit comparison, ignoring the observed red occlusion band."""
from pathlib import Path
import json,cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw
ROOT=Path(__file__).resolve().parent
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/1.png').convert('RGB'))
doc=json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
results=[]
for index in [7,8,9]:
    l=doc['rows'][index];x=l['point'][0];edge=l['edge'];box=[int(x-20),int(edge-12),int(x-1),int(edge+12)]
    p=rgb[box[1]:box[3],box[0]:box[2]].astype(int)
    ink=(p.max(2)-p.min(2)<35)&(p.mean(2)<230)
    n,lab,stats,_=cv2.connectedComponentsWithStats(ink.astype('uint8'),8)
    ink=np.isin(lab,[i for i in range(1,n) if stats[i,4]>=4])
    yy,xx=np.where(ink);a,b,c,d=min(xx),min(yy),max(xx)+1,max(yy)+1
    ink=ink[b:d,a:c];invalid=((p[:,:,0]>p[:,:,1]+35)&(p[:,:,0]>p[:,:,2]+35))[b:d,a:c]
    ink=np.rot90(ink,3);valid=~np.rot90(invalid,3)
    rankings={}
    for digit in '0123456789':
        best=0;bestparam=None
        for filename in ['arial.ttf','tahoma.ttf','segoeui.ttf']:
            for size in range(10,23):
                font=ImageFont.truetype('C:/Windows/Fonts/'+filename,size)
                im=Image.new('L',(60,60),255);ImageDraw.Draw(im).text((10,10),digit,font=font,fill=0)
                ar=np.array(im);sy,sx=np.where(ar<205);mask=ar[min(sy):max(sy)+1,min(sx):max(sx)+1]<205
                templ=cv2.resize(mask.astype('uint8'),(ink.shape[1],ink.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                score=((templ&ink)&valid).sum()/max(1,((templ|ink)&valid).sum())
                if score>best:best=score;bestparam=(filename,size)
        rankings[digit]={'score':float(best),'font':bestparam}
    results.append({'index':index,'shape':ink.shape,'ranking':sorted(rankings.items(),key=lambda p:-p[1]['score'])})
(ROOT/'experiment_font.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results))
