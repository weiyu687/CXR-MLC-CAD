"""Serializable LibAUC v2 objective and multi-label positive/negative coverage.

Formula: LibAUC MultiLabelAUCMLoss(version='v2'), arxiv:2012.03173.
Unlike the original unregistered vectors, a/b/alpha are saved parameters.
Batch coverage adapts binary DualSampler to ten simultaneous labels; it is
documented as an adaptation, not the original binary sampling algorithm.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler


class MultiLabelAUCMV2(nn.Module):
    def __init__(self, labels=10, margin=1.):
        super().__init__()
        self.margin=margin
        self.a=nn.Parameter(torch.zeros(labels))
        self.b=nn.Parameter(torch.zeros(labels))
        self.alpha=nn.Parameter(torch.zeros(labels))

    def forward(self, probabilities, targets, counts=None, constant_fraction=1.):
        positive=targets
        negative=1-targets
        if counts is None:
            pos_count=positive.sum(0)
            neg_count=negative.sum(0)
        else:
            pos_count,neg_count=counts
        if (pos_count<=0).any() or (neg_count<=0).any():
            raise ValueError('Every AUC batch must contain both classes for each label')
        pos_square=((probabilities-self.a).square()*positive).sum(0)/pos_count
        neg_square=((probabilities-self.b).square()*negative).sum(0)/neg_count
        pos_mean=(probabilities*positive).sum(0)/pos_count
        neg_mean=(probabilities*negative).sum(0)/neg_count
        return (pos_square+neg_square+2*self.alpha*(neg_mean-pos_mean)+
                constant_fraction*(2*self.alpha*self.margin-self.alpha.square())).mean()


class CoverageBatchSampler(Sampler):
    def __init__(self, targets, batch_size=32, seed=2049):
        self.targets=np.asarray(targets,dtype=np.int64)
        if self.targets.ndim!=2 or not np.isin(self.targets,[0,1]).all():
            raise ValueError('Binary multi-label targets required')
        if batch_size<2*self.targets.shape[1]:
            raise ValueError('At least two anchors per label are required')
        self.batch_size,self.seed,self.epoch=batch_size,seed,0
        self.pools=[(np.flatnonzero(self.targets[:,i]==1),np.flatnonzero(self.targets[:,i]==0))
                    for i in range(self.targets.shape[1])]
        if any(not len(pos) or not len(neg) for pos,neg in self.pools):
            raise ValueError('Training split lacks a positive or negative class')

    def set_epoch(self, epoch):
        self.epoch=epoch

    def __len__(self):
        return math.ceil(len(self.targets)/self.batch_size)

    def __iter__(self):
        random=np.random.default_rng(self.seed+self.epoch)
        for _ in range(len(self)):
            indices=[]
            for positive,negative in self.pools:
                indices.extend([int(random.choice(positive)),int(random.choice(negative))])
            indices.extend(random.integers(0,len(self.targets),size=self.batch_size-len(indices)).tolist())
            random.shuffle(indices)
            yield indices
