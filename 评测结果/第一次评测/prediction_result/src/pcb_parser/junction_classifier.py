"""Small learned RGB classifier. No wire extraction or raw color/dot rules."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

CLASSES=['crossing_without_connection','connected_junction']
RADII=[8,24]


class JunctionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(6,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*4*4,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,2))

    def forward(self,x):return self.head(self.features(x))


def centered_views(image,point,radii=RADII):
    """Two original-pixel scales, centered by the model candidate coordinate."""
    if not isinstance(image,Image.Image):image=Image.fromarray(image)
    image=image.convert('RGB')
    x,y=point
    if not np.isfinite([x,y]).all() or not 0<=x<image.width or not 0<=y<image.height:
        raise ValueError('Junction center outside original image')
    views=[]
    for radius in radii:
        # Floating extent retains subpixel centers. White padding at page edges.
        patch=image.transform((64,64),Image.Transform.EXTENT,(x-radius,y-radius,x+radius,y+radius),
                              Image.Resampling.BILINEAR,fillcolor='white')
        views.append(np.asarray(patch,dtype=np.float32).transpose(2,0,1)/255.)
    return torch.from_numpy(np.concatenate(views))


class JunctionPredictor:
    def __init__(self,weights,device='cpu',expected_sha256=None):
        weights=Path(weights)
        self.sha256=hashlib.sha256(weights.read_bytes()).hexdigest()
        if expected_sha256 and self.sha256!=expected_sha256:raise ValueError('Junction checkpoint hash mismatch')
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        if checkpoint.get('kind')!='junction_cnn_development_trained':raise ValueError('Unsupported junction checkpoint')
        if checkpoint['classes']!=CLASSES or checkpoint['radii']!=RADII:raise ValueError('Junction input contract mismatch')
        self.device=torch.device(device)
        self.model=JunctionCNN().to(self.device).eval()
        self.model.load_state_dict(checkpoint['state_dict'],strict=True)

    @torch.inference_mode()
    def predict(self,rgb,candidates,threshold=.95):
        if not 0.5<threshold<=1:raise ValueError('Use a conservative junction confidence threshold')
        if not candidates:return {}
        image=Image.fromarray(rgb).convert('RGB')
        inputs=torch.stack([centered_views(image,c['point']) for c in candidates]).to(self.device)
        probabilities=torch.stack([self.model(torch.rot90(inputs,k,(-2,-1))).softmax(1) for k in range(4)])
        mean=probabilities.mean(0).cpu().numpy()
        voted=probabilities.argmax(2).cpu().numpy()
        decisions={}
        for i,candidate in enumerate(candidates):
            label=int(mean[i].argmax())
            minimum=float(probabilities[:,i,label].min().cpu())
            stable=bool((voted[:,i]==label).all())
            accepted=stable and minimum>=threshold and candidate['degree']==4
            decisions[candidate['junction_id']]={
                'label':CLASSES[label],'confidence':float(mean[i,label]),'rotation_min_confidence':minimum,
                'probabilities':dict(zip(CLASSES,map(float,mean[i]))),'rotation_agreement':stable,
                'accepted':accepted,'decision_source':'learned_RGB_junction_CNN',
                'model_sha256':self.sha256,'threshold':threshold,
                'abstention_reason':None if accepted else 'unsupported_degree' if candidate['degree']!=4 else 'low_confidence_or_rotation_disagreement'}
        return decisions
