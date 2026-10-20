"""Renderer-labeled training examples plus source-group-separated image reviews."""
import random

from PIL import Image,ImageDraw,ImageFilter
import torch
from torch.utils.data import Dataset

from .junction_classifier import centered_views,CLASSES


def render_synthetic(seed):
    rng=random.Random(seed);scale=3;size=97
    background=(rng.randint(244,255),)*3
    image=Image.new('RGB',(size*scale,size*scale),background);draw=ImageDraw.Draw(image)
    palette=[(0,0,0),(70,70,70),(145,145,145),(0,0,150),(85,160,90),(0,145,40)]
    wire=rng.choice(palette);dot=rng.choice([wire,wire,(180,40,45)])
    width=rng.choice([.7,1.,1.2,1.8,2.5,3.])
    # The center label comes from the renderer, not from image thresholding.
    connected=rng.randrange(2)
    cx,cy=48+rng.uniform(-1.2,1.2),48+rng.uniform(-1.2,1.2)
    def line(points,color=wire,w=width):draw.line([(int(x*scale),int(y*scale)) for x,y in points],fill=color,width=max(1,round(w*scale)))
    for _ in range(rng.randrange(5)):
        coordinate=rng.choice([-1,1])*rng.uniform(9,38)
        if rng.random()<.5:line([(0,cy+coordinate),(96,cy+coordinate)])
        else:line([(cx+coordinate,0),(cx+coordinate,96)])
        if rng.random()<.6:
            dx,dy=(cx,cy+coordinate) if rng.random()<.5 else (cx+coordinate,cy)
            r=rng.uniform(1.6,3.5)*scale
            draw.ellipse((dx*scale-r,dy*scale-r,dx*scale+r,dy*scale+r),fill=dot)
    line([(0,cy),(96,cy)])
    # T nodes with visible dots can be promoted to four arms by segmentation.
    line([(cx,0),(cx,cy if connected and rng.random()<.25 else 96)])
    if connected:
        radius=max(width*.8,rng.uniform(1.3,3.5))*scale
        draw.ellipse((cx*scale-radius,cy*scale-radius,cx*scale+radius,cy*scale+radius),fill=dot)
    if rng.random()<.65:
        draw.text((rng.randrange(10,80)*scale,rng.choice([10,72])*scale),rng.choice(['R12','GND','100K','3V3','PA15']),fill=rng.choice(palette))
    image=image.resize((size,size),Image.Resampling.LANCZOS)
    if rng.random()<.2:image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.15,.55)))
    return centered_views(image,[48,48]),connected


class ReviewedDataset(Dataset):
    def __init__(self,root,records,augment=False,repeats=1):
        self.augment=augment;self.repeats=repeats
        self.items=[]
        for row in records:
            if not row['training_eligible']:continue
            with Image.open(root/row['crop_file']) as image:
                box=row['crop_top_left'];point=[row['center_top_left'][0]-box[0],row['center_top_left'][1]-box[1]]
                self.items.append((centered_views(image,point),CLASSES.index(row['label'])))

    def __len__(self):return len(self.items)*self.repeats

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        x,y=self.items[index%len(self.items)]
        if self.augment:
            x=torch.rot90(x,random.randrange(4),(-2,-1))
            if random.random()<.5:x=x.flip(-1)
            # Mild photometric variation cannot change junction semantics.
            x=(x*random.uniform(.9,1.08)+random.uniform(-.02,.02)).clamp(0,1)
        return x,torch.tensor(y,dtype=torch.long)
