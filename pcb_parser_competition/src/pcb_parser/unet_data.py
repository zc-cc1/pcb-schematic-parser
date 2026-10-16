"""Offline annotation rasterization and image-only U-Net tile inputs."""
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def point(value):
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y")]
    if not isinstance(value, (list, tuple)) or len(value)!=2:
        return None
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in value):
        return None
    return tuple(float(v) for v in value)


def polylines(value):
    if not isinstance(value,(list,tuple)) or not value:
        return
    if all(point(p) is not None for p in value):
        if len(value)>=2:
            yield [point(p) for p in value]
        return
    for child in value:
        yield from polylines(child)


def annotation_targets(data, width, height, line_width=3, pin_sigma=2.):
    """No pixel segmentation: draw source geometry only, excluding flywires."""
    if data["canvas"]["width"]!=width or data["canvas"]["height"]!=height:
        raise ValueError("Annotation canvas and PNG dimensions differ")
    wire = Image.new("L",(width,height),0)
    draw = ImageDraw.Draw(wire)
    stats = {"physical_edges":0,"flywires_excluded":0,"missing_physical_paths":0,
             "out_of_bounds_paths":0,"invalid_pin_tips":0,"unowned_root_pins":0}
    for edge in data.get("edges",{}).values():
        if edge.get("state")=="deleted":
            continue
        if edge.get("isFlywire") is True or str(edge.get("isFlywire")).lower()=="true":
            stats["flywires_excluded"]+=1
            continue
        if edge.get("category","wire")!="wire":
            continue
        stats["physical_edges"]+=1
        paths=edge.get("keyPointPaths",{})
        paths=list(polylines(paths.get("wire",[]) if isinstance(paths,dict) else paths))
        if not paths:
            stats["missing_physical_paths"]+=1
        for path in paths:
            if any(not (0<=x<width and 0<=y<height) for x,y in path):
                stats["out_of_bounds_paths"]+=1
                continue
            draw.line([(round(x),round(y)) for x,y in path],fill=255,width=line_width)
    tips=[]
    owned=set()
    for cid, component in data.get("components",{}).items():
        if component.get("state")=="deleted":
            continue
        for pid, local in component.get("pins",{}).items():
            if local.get("state")=="deleted":
                continue
            owned.add(pid)
            raw={**data.get("pins",{}).get(pid,{}),**local}
            p=point(raw.get("tip"))
            if p is None or not (0<=p[0]<width and 0<=p[1]<height):
                stats["invalid_pin_tips"]+=1
                continue
            tips.append({"component_source_id":cid,"pin_source_id":pid,"point":list(p)})
    stats["unowned_root_pins"]=sum(pid not in owned for pid,pin in data.get("pins",{}).items() if pin.get("state")!="deleted")
    heat=np.zeros((height,width),dtype=np.float32)
    radius=math.ceil(pin_sigma*3)
    for tip in tips:
        x,y=tip["point"]
        x1,x2=max(0,math.floor(x)-radius),min(width,math.floor(x)+radius+1)
        y1,y2=max(0,math.floor(y)-radius),min(height,math.floor(y)+radius+1)
        yy,xx=np.mgrid[y1:y2,x1:x2]
        gaussian=np.exp(-((xx-x)**2+(yy-y)**2)/(2*pin_sigma**2))
        heat[y1:y2,x1:x2]=np.maximum(heat[y1:y2,x1:x2],gaussian)
    stats["pin_tips"]=len(tips)
    stats["wire_positive_pixels"]=int(np.count_nonzero(np.asarray(wire)))
    return wire,Image.fromarray(np.round(heat*255).astype(np.uint8)),tips,stats


def tile_starts(length,size,stride):
    if length<=size:
        return [0]
    return sorted(set(list(range(0,length-size+1,stride))+[length-size]))


class SegmentationTiles(Dataset):
    def __init__(self, rows, tile_size=512, stride=384):
        self.rows=rows
        self.size=tile_size
        self.tiles=[(i,x,y) for i,row in enumerate(rows)
                    for y in tile_starts(row["height"],tile_size,stride)
                    for x in tile_starts(row["width"],tile_size,stride)]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self,index):
        i,x,y=self.tiles[index]
        row=self.rows[i]
        s=self.size
        image=np.full((s,s,3),255,np.uint8)
        targets=np.zeros((2,s,s),np.float32)
        valid=np.zeros((s,s),np.float32)
        w,h=min(s,row["width"]-x),min(s,row["height"]-y)
        with Image.open(row["image"]) as im:
            image[:h,:w]=np.asarray(im.convert("RGB").crop((x,y,x+w,y+h)))
        for channel,key in enumerate(["wire_mask","pin_heatmap"]):
            with Image.open(row[key]) as im:
                targets[channel,:h,:w]=np.asarray(im.crop((x,y,x+w,y+h)),dtype=np.float32)/255
        valid[:h,:w]=1
        return torch.from_numpy(image.transpose(2,0,1).copy()).float()/255,torch.from_numpy(targets),torch.from_numpy(valid)


@torch.inference_mode()
def predict_tiled(model,rgb,device="cpu",tile_size=512,stride=384):
    """Original-scale tiles; never resize away thin wires or read labels."""
    height,width=rgb.shape[:2]
    output=np.zeros((2,height,width),np.float32)
    count=np.zeros((height,width),np.float32)
    model.eval()
    for y in tile_starts(height,tile_size,stride):
        for x in tile_starts(width,tile_size,stride):
            h,w=min(tile_size,height-y),min(tile_size,width-x)
            tile=np.full((tile_size,tile_size,3),255,np.uint8)
            tile[:h,:w]=rgb[y:y+h,x:x+w]
            batch=torch.from_numpy(tile.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/255
            probability=model(batch).sigmoid()[0,:,:h,:w].cpu().numpy()
            output[:,y:y+h,x:x+w]+=probability
            count[y:y+h,x:x+w]+=1
    return output/count[None]
