"""CheXFusion's class-weighted ASL formula, with portable registered buffers.

Reference: dongkyunk/CheXFusion model/loss.py, ASLwithClassWeight.
The focusing factor participates in autograd, unlike our older ASL variant.
"""
import torch
from torch import nn


class CheXFusionWeightedASL(nn.Module):
    def __init__(self, positive_counts, total_count, gamma_negative=4., gamma_positive=1., clip=.05):
        super().__init__()
        counts=torch.as_tensor(positive_counts,dtype=torch.float32)
        if counts.ndim!=1 or total_count<=0 or not torch.isfinite(counts).all() or (counts<0).any() or (counts>total_count).any():
            raise ValueError('Invalid training-only class frequencies')
        frequency=counts/total_count
        self.register_buffer('positive_weights',torch.exp(1-frequency))
        self.register_buffer('negative_weights',torch.exp(frequency))
        self.gamma_negative=gamma_negative
        self.gamma_positive=gamma_positive
        self.clip=clip

    def forward(self, logits, target):
        positive=logits.sigmoid()
        negative=(1-positive+self.clip).clamp(max=1.)
        class_weight=target*self.positive_weights+(1-target)*self.negative_weights
        likelihood=target*positive.clamp(min=1e-8).log()+(1-target)*negative.clamp(min=1e-8).log()
        probability=positive*target+negative*(1-target)
        focusing=(1-probability).pow(self.gamma_positive*target+self.gamma_negative*(1-target))
        return -(likelihood*class_weight*focusing).mean()
