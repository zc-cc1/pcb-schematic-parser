"""Corroborate complete grayscale number groups with a fixed compact border."""
from pathlib import Path
exec(Path(__file__).with_name('experiment_binary.py').read_text(encoding='utf-8').split('result=[]')[0])
result=[]
for entry in read(ROOT/'experiment_grouped.json'):
    box=entry['crop'];patch=rgb[box[1]:box[3],box[0]:box[2]];v=patch.astype(np.int16)
    mask=((v.max(2)-v.min(2)<35)&(v.mean(2)<205)).astype(np.uint8)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    chosen=[]
    for i in range(1,n):
        x,y,cw,ch,area=map(int,stats[i]);cx,cy=centroids[i]
        if 1<=cw<=24 and 2<=ch<=17 and area>=2 and -14<=box[1]+cy-entry['lead']['row']<=0:chosen.append(i)
    selected=np.isin(labels,chosen);ys,xs=np.where(selected);a,b,c,d=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
    patch2=np.full_like(patch,255);patch2[selected]=patch[selected]
    for pad in [4]:
        isolated=cv2.copyMakeBorder(patch2[b:d,a:c],pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255));tokens=[]
        for scale in [3.,5.]:tokens.extend(ocr(isolated,scale,f"group_gray_{entry['lead']['row']}_p{pad}_s{scale}"))
        value,support=vote([t for t in tokens if t['text'].isdigit()],DEFAULT)
        result.append({'row':entry['lead']['row'],'number':value,'support':support,'readings':tokens})
(ROOT/'experiment_gray.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
