import torch
from torch import nn
import timm

def build_model(name):
    if name.endswith('_cap'):
        return ClassAttentionPool(name[:-4])
    return timm.create_model(name, pretrained=False, num_classes=10, in_chans=3)

class ClassAttentionPool(nn.Module):
    """Independent disease-specific spatial pooling, initialized to global averaging.

    Inspired by class attention literature; this is not a reproduction of C2A.
    """
    def __init__(self, name):
        super().__init__()
        self.backbone=timm.create_model(name,pretrained=False,num_classes=0,global_pool='')
        channels=self.backbone.num_features
        self.attention=nn.Conv2d(channels,10,1)
        nn.init.zeros_(self.attention.weight);nn.init.zeros_(self.attention.bias)
        self.weight=nn.Parameter(torch.empty(10,channels))
        self.bias=nn.Parameter(torch.zeros(10))
        nn.init.normal_(self.weight,std=0.01)
    def forward(self,x):
        f=self.backbone(x)
        attention=self.attention(f).flatten(2).float().softmax(-1).to(f.dtype)
        descriptors=torch.einsum('bcn,bkn->bkc',f.flatten(2),attention)
        return (descriptors*self.weight.unsqueeze(0)).sum(-1)+self.bias

class AsymmetricLoss(nn.Module):
    def forward(self, logits, target):
        p = logits.float().sigmoid()
        neg = (1-p+0.05).clamp(max=1)
        log = target * p.clamp(min=1e-8).log() + (1-target) * neg.clamp(min=1e-8).log()
        with torch.no_grad():
            w = (1 - p*target - neg*(1-target)).pow(4*(1-target))
        return -(w*log).sum(1).mean()
