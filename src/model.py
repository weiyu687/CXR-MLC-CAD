import torch
from torch import nn
import timm

def build_model(name):
    return timm.create_model(name, pretrained=False, num_classes=10, in_chans=3)

class AsymmetricLoss(nn.Module):
    def forward(self, logits, target):
        p = logits.float().sigmoid()
        neg = (1-p+0.05).clamp(max=1)
        log = target * p.clamp(min=1e-8).log() + (1-target) * neg.clamp(min=1e-8).log()
        with torch.no_grad():
            w = (1 - p*target - neg*(1-target)).pow(4*(1-target))
        return -(w*log).sum(1).mean()
