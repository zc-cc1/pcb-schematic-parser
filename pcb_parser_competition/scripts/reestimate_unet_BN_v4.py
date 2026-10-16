"""Train-image-only BN buffer diagnostic, with every learned weight frozen."""
from functools import lru_cache
import hashlib
import json
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from unet_audit_common import ROOT, OUT, read, write, sha, assert_protected
from pcb_parser.unet_data import tile_starts
from pcb_parser.unet_model import WirePinUNet


class RGBOnlyCrops(Dataset):
    def __init__(self,rows):
        assert all(r["split"]=="train" and r["source_split"]=="train" for r in rows)
        self.rows=rows;self.tiles=[(i,x,y) for i,r in enumerate(rows) for y in tile_starts(r["height"],512,384) for x in tile_starts(r["width"],512,384)]
    def __len__(self):return len(self.tiles)
    @lru_cache(maxsize=None)
    def rgb(self,index):
        r=self.rows[index];a=np.asarray(Image.open(r["image"]).convert("RGB"))
        assert a.shape[:2]==(r["height"],r["width"]);return a
    def __getitem__(self,index):
        i,x,y=self.tiles[index];rgb=self.rgb(i);crop=rgb[y:y+512,x:x+512];out=np.full((512,512,3),255,np.uint8)
        out[:crop.shape[0],:crop.shape[1]]=crop
        return torch.from_numpy(out.transpose(2,0,1).copy()).float()/255


def fixed_weights_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        if k.endswith(("running_mean","running_var","num_batches_tracked")):continue
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    run=ROOT/"runs/unet_negative_sampling_pilot_v4";protocol=read(run/"BN_diagnostic_protocol.json")
    complete=read(run/"summary.json");assert complete["status"]=="paired_fixed16_training_complete_external_evaluation_not_started"
    rows=[json.loads(s) for s in (ROOT/"runs/unet_training_label_review_v4/training_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows=[r for r in rows if r["split"]=="train"];assert len(rows)==93
    frozen=read(run/"protected_inputs.json")|read(run/"additional_source_attestation.json")
    assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    torch.set_num_threads(4);dataset=RGBOnlyCrops(rows);assert len(dataset)==925
    summaries=[];start=time.perf_counter()
    for arm in protocol["arms"]:
        path=run/arm/"last.pt";ckpt=torch.load(path,map_location="cpu",weights_only=True);before=ckpt["state_dict"]
        model=WirePinUNet(ckpt["base_channels"]).to("cuda:0");model.load_state_dict(before,strict=True);model.eval()
        bns=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)];assert len(bns)==18
        for p in model.parameters():p.requires_grad_(False)
        for bn in bns:bn.reset_running_stats();bn.momentum=None;bn.train()
        loader=DataLoader(dataset,batch_size=protocol["batch_size"],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(protocol["order_seed"]))
        with torch.inference_mode():
            for batch,images in enumerate(loader):
                model(images.to("cuda:0"))
                if (batch+1)%20==0:print(json.dumps(dict(arm=arm,batches=batch+1,total=len(loader))),flush=True)
        model.eval();after={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        assert fixed_weights_hash(before)==fixed_weights_hash(after)
        changed=[k for k in before if not torch.equal(before[k],after[k])]
        assert changed and all(k.endswith(("running_mean","running_var","num_batches_tracked")) for k in changed)
        folder=run/(arm+"_BN_reestimated");folder.mkdir(exist_ok=True)
        new=dict(ckpt,state_dict=after,kind="train_only_BN_buffer_diagnostic",parent_checkpoint_sha256=sha(path),
            BN_diagnostic_protocol_sha256=sha(run/"BN_diagnostic_protocol.json"))
        torch.save(new,folder/"last.pt")
        record=dict(arm=arm,parent_sha256=sha(path),diagnostic_sha256=sha(folder/"last.pt"),changed_keys=changed,
            non_BN_weights_sha256=fixed_weights_hash(after),learned_weights_unchanged=True,gradient_updates=0,
            calibration_or_external26_images_read=False,training_images=93,tiles=925,batches=len(loader),production_adopted=False)
        write(folder/"summary.json",record);summaries.append(record);del model;torch.cuda.empty_cache()
    assert all(sha(p)==v for p,v in frozen.items());assert_protected()
    write(run/"BN_diagnostic_summary.json",dict(records=summaries,elapsed_seconds=time.perf_counter()-start,protocol=protocol))


if __name__=="__main__":main()
