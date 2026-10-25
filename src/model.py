import torch
from torch import nn
import timm
import hashlib
from pathlib import Path

def build_model(name,pretrained=False):
    if name == 'tv_densenet121_query':
        base = build_model('tv_densenet121', pretrained=pretrained)
        model = DenseNetLabelQuery(base)
        if pretrained:
            model.pretraining_provenance = base.pretraining_provenance
        return model
    if name.startswith('tv_'):
        from torchvision import models
        catalog={
            'tv_resnet50':(models.resnet50,models.ResNet50_Weights.IMAGENET1K_V2,'fc'),
            'tv_densenet121':(models.densenet121,models.DenseNet121_Weights.IMAGENET1K_V1,'classifier'),
            'tv_efficientnet_v2_s':(models.efficientnet_v2_s,models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,'classifier')}
        constructor,weights,head=catalog[name]
        model=constructor(weights=weights if pretrained else None)
        if isinstance(getattr(model,head),nn.Sequential):
            linear=getattr(model,head)[-1]
            getattr(model,head)[-1]=nn.Linear(linear.in_features,10)
        else:
            setattr(model,head,nn.Linear(getattr(model,head).in_features,10))
        if pretrained:
            weight_path=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
            model.pretraining_provenance={'name':name,'weights':str(weights),'url':weights.url,
                'sha256':hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                'cached_file':str(weight_path),'training_data':'ImageNet-1K (non-medical natural images)',
                'license_url':'https://raw.githubusercontent.com/pytorch/vision/v0.23.0/LICENSE',
                'documentation':'https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.'+name[3:]+'.html'}
        return model
    if pretrained:raise ValueError('External pretraining is restricted to explicitly audited torchvision ImageNet weights')
    if name.endswith('_cap'):
        return ClassAttentionPool(name[:-4])
    return timm.create_model(name, pretrained=False, num_classes=10, in_chans=3)


class LabelQueryHead(nn.Module):
    """Small disease-query cross-attention head on a single image's feature map.

    Learnable class queries and class-specific readout follow the general
    Query2Label idea. This intentionally smaller module is not a reproduction
    of Query2Label or ML-Decoder. No attention operates across image batches.
    """
    def __init__(self, channels, classes=10, dimension=256, heads=4):
        super().__init__()
        self.projection = nn.Conv2d(channels, dimension, 1)
        self.queries = nn.Parameter(torch.empty(classes, dimension))
        nn.init.normal_(self.queries, std=.02)
        self.key_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=.1, batch_first=True)
        self.query_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(nn.Linear(dimension, 2*dimension), nn.GELU(),
                                          nn.Dropout(.1), nn.Linear(2*dimension, dimension))
        self.output_norm = nn.LayerNorm(dimension)
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        self.bias = nn.Parameter(torch.zeros(classes))
        nn.init.normal_(self.weight, std=.01)

    def forward(self, features):
        tokens = self.projection(features).flatten(2).transpose(1, 2)
        tokens = self.key_norm(tokens)
        queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        attended, _ = self.attention(queries, tokens, tokens, need_weights=False)
        descriptors = self.query_norm(queries + attended)
        descriptors = self.output_norm(descriptors + self.feed_forward(descriptors))
        return (descriptors * self.weight.unsqueeze(0)).sum(-1) + self.bias


class DenseNetLabelQuery(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.backbone = base.features
        self.classifier = LabelQueryHead(base.classifier.in_features)

    def forward(self, x):
        features = torch.nn.functional.relu(self.backbone(x), inplace=False)
        return self.classifier(features)


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
