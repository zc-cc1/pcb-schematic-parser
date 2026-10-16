"""Fixed-quota sampling of manually checked training-only nonwire regions."""
import numpy as np

from .unet_retraining import CachedTrainingTiles


def paired_schedule(base_tiles, focus_tiles, fraction, seed, epoch):
    if not base_tiles or not 0<=fraction<=1 or epoch<1:
        raise ValueError("Invalid schedule")
    if fraction>0 and not focus_tiles:
        raise ValueError("Focus sampling needs checked regions")
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,7301]))
    order=rng.permutation(len(base_tiles));control=[base_tiles[i] for i in order]
    treatment=list(control);n=round(fraction*len(control))
    replaced=sorted(rng.choice(len(control),size=n,replace=False).tolist())
    for i in replaced:treatment[i]=focus_tiles[int(rng.integers(len(focus_tiles)))]
    return control,treatment,replaced


class NegativeFocusTrainingTiles(CachedTrainingTiles):
    def __init__(self, rows, regions, focus_fraction, seed=20261014, tile_size=512, stride=384):
        if any(r.get("split")!="train" or r.get("source_split","train")!="train" for r in rows):
            raise ValueError("Training sampler rejects validation/calibration rows")
        super().__init__(rows,tile_size,stride,augmentation=None,seed=seed)
        self.base_tiles=list(self.tiles);self.focus_fraction=focus_fraction;self.seed=seed
        self.focus_tiles=[];self.focus_regions=[]
        indices={r["case_id"]:i for i,r in enumerate(rows)}
        if len(indices)!=len(rows):raise ValueError("Duplicate case IDs")
        for region in regions:
            if not region["hard_negative"]:continue
            c=region["case_id"]
            if region.get("split")!="train" or c not in indices:
                raise ValueError("Checked region outside training partition")
            i=indices[c];row=rows[i];x0,y0,x1,y1=region["box"]
            if not (0<=x0<x1<=row["width"] and 0<=y0<y1<=row["height"]):raise ValueError("Invalid region box")
            _,wire,_=self.source(i)
            if wire[y0:y1,x0:x1].any():raise ValueError("Checked nonwire region contains GT wire")
            x=max(0,min(max(0,row["width"]-tile_size),(x0+x1)//2-tile_size//2))
            y=max(0,min(max(0,row["height"]-tile_size),(y0+y1)//2-tile_size//2))
            self.focus_tiles.append((i,x,y));self.focus_regions.append(region)
        self.set_epoch(1)

    def set_epoch(self, epoch):
        control,treatment,replaced=paired_schedule(self.base_tiles,self.focus_tiles,self.focus_fraction,self.seed,epoch)
        self.tiles=treatment if self.focus_fraction else control
        self.replaced_slots=replaced
        return dict(epoch=epoch,tiles=len(self.tiles),focus_slots=len(replaced),focus_fraction=len(replaced)/len(self.tiles))

