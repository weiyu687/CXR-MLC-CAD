"""CheXFusion image/study modules, adapted to ten labels and masked batches.

References: arxiv:2308.03968 and github.com/dongkyunk/CheXFusion.
Only generic non-medical initializations are accepted. The upstream fusion
batch-axis and padding bugs are corrected explicitly, not reproduced.
"""
import hashlib
import math
from pathlib import Path

import timm
import torch
from torch import nn

from .ml_decoder import MLDecoder


class SpatialPosition(nn.Module):
    """NCHW sinusoidal positions matching positional_encodings' 2D convention.

    Formula adapted from Peter Tatkowski's MIT implementation; see
    docs/pretraining/positional-encoding-LICENSE.txt. Use spatial axes rather
    than interpreting an NCHW tensor as the upstream NHWC API does.
    """
    def __init__(self, channels):
        super().__init__()
        self.half_channels = math.ceil(channels/4)*2
        self.register_buffer('inverse_frequency',
            1/(10000**(torch.arange(0, self.half_channels, 2).float()/self.half_channels)))

    def forward(self, x):
        _, channels, height, width = x.shape
        def encoding(length):
            phase = torch.outer(torch.arange(length, device=x.device, dtype=self.inverse_frequency.dtype),
                                self.inverse_frequency)
            return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        position = x.new_zeros(height, width, 2*self.half_channels)
        position[:, :, :self.half_channels] = encoding(height).unsqueeze(1)
        position[:, :, self.half_channels:] = encoding(width).unsqueeze(0)
        return x+position[:, :, :channels].permute(2, 0, 1).unsqueeze(0)


def generic_backbone(name, pretrained):
    if name == 'convnext_tiny_1k':
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights if pretrained else None)
        metadata = None
        if pretrained:
            path = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/', 1)[-1]
            metadata = {'name': name, 'weights': str(weights), 'url': weights.url,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cached_file': str(path),
                'training_data': 'ImageNet-1K (non-medical natural images)',
                'license_url': 'https://github.com/pytorch/vision/blob/v0.23.0/LICENSE',
                'documentation': 'https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.convnext_tiny.html'}
        return model.features, 768, metadata
    if name == 'convnext_base_22k':
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth'
        digest = 'b007c30d8173f247df47699bd991f534d0ba5284c49929ac71444f7d82d8cfcf'
        kwargs = {}
        metadata = None
        if pretrained:
            path = Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/', 1)[-1]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Run prepare_reference_backbones first; audited ImageNet-22K weight required')
            kwargs['pretrained_cfg_overlay'] = {'file': str(path), 'hf_hub_id': None}
            metadata = {'name': name, 'weights': 'convnext_base.fb_in22k', 'url': url,
                'sha256': digest, 'cached_file': str(path), 'training_data': 'ImageNet-22K (non-medical natural images)',
                'license_url': 'https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE',
                'documentation': 'https://huggingface.co/timm/convnext_base.fb_in22k'}
        model = timm.create_model('convnext_base.fb_in22k', pretrained=pretrained, num_classes=0, **kwargs)
        # Keep norm_pre and expose the original spatial feature map.
        model.head = nn.Identity()
        return model, 1024, metadata
    raise ValueError('Unreviewed backbone: '+name)


class CheXFusionImage(nn.Module):
    def __init__(self, backbone='convnext_tiny_1k', head='ml_decoder', pretrained=False):
        super().__init__()
        self.backbone, self.channels, self.pretraining_provenance = generic_backbone(backbone, pretrained)
        self.head_type = head
        self.position = SpatialPosition(self.channels) if head == 'ml_decoder' else nn.Identity()
        if head == 'ml_decoder':
            self.classifier = MLDecoder(10, initial_num_features=self.channels)
        elif head == 'gap':
            self.classifier = nn.Sequential(nn.LayerNorm(self.channels), nn.Linear(self.channels, 10))
        else:
            raise ValueError('Unsupported classification head')

    def forward(self, x):
        features = self.backbone(x)
        if self.head_type == 'gap':
            return self.classifier(features.mean((-2, -1)))
        return self.classifier(self.position(features))


class CheXFusionStudy(nn.Module):
    """Paper stage two: frozen CNN, spatial reduction, encoder, ML-Decoder.

    Explicit view validity propagates into BOTH attention blocks. Batches
    contain independent studies. Missing images never enter the CNN.
    """
    def __init__(self, backbone='convnext_tiny_1k', max_views=4):
        super().__init__()
        self.backbone, channels, self.pretraining_provenance = generic_backbone(backbone, False)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.max_views = max_views
        self.projection = nn.Conv2d(channels, 768, kernel_size=3, stride=2, padding=1)
        self.position = SpatialPosition(768)
        self.segment_embedding = nn.Parameter(torch.randn(max_views, 768))
        layer = nn.TransformerEncoderLayer(768, 8, dim_feedforward=2048, dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = MLDecoder(10, initial_num_features=768)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def fuse_features(self, features, valid_views):
        batch, views, channels, height, width = features.shape
        if views > self.max_views or valid_views.shape != (batch, views) or valid_views.dtype != torch.bool:
            raise ValueError('Invalid study/view mask')
        if not valid_views.any(dim=1).all():
            raise ValueError('Every study must have at least one real image')
        selected = self.projection(features[valid_views])
        selected = self.position(selected)
        spatial_shape = selected.shape[-2:]
        positioned = selected.new_zeros(batch, views, 768, *spatial_shape)
        segment = self.segment_embedding[:views].view(1, views, 768, 1, 1)
        positioned[valid_views] = selected
        positioned = positioned+segment.to(positioned.dtype)
        tokens = positioned.permute(0, 1, 3, 4, 2).reshape(batch, -1, 768)
        mask = (~valid_views).unsqueeze(-1).expand(batch, views, spatial_shape[0]*spatial_shape[1]).reshape(batch, -1)
        tokens = self.encoder(tokens, src_key_padding_mask=mask)
        return self.classifier(tokens, memory_key_padding_mask=mask)

    def forward(self, images, valid_views):
        batch, views = images.shape[:2]
        if valid_views.shape != (batch, views) or not valid_views.any(dim=1).all():
            raise ValueError('Invalid study/view mask')
        with torch.no_grad():
            selected = self.backbone(images[valid_views])
        features = selected.new_zeros(batch, views, *selected.shape[1:])
        features[valid_views] = selected
        return self.fuse_features(features, valid_views)


class DinoV2MeanClassifier(nn.Module):
    """Generic DINOv2 + linear readout, following vit-med's token mean.

    The non-register variant matches the paper's DINOv2-base. The optional
    register variant excludes its four registers from the token mean.
    """
    def __init__(self, pretrained=False, registers=True):
        super().__init__()
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth'
        digest = '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
        model_name='vit_base_patch14_reg4_dinov2.lvd142m'
        if not registers:
            url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
            digest='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
            model_name='vit_base_patch14_dinov2.lvd142m'
        kwargs = {}
        self.pretraining_provenance = None
        if pretrained:
            path = Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/',1)[-1]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Audited generic DINOv2 weight missing or changed')
            kwargs['pretrained_cfg_overlay'] = {'file': str(path), 'hf_hub_id': None}
            self.pretraining_provenance = {'name': 'dinov2_base_reg_mean' if registers else 'dinov2_base_mean',
                'weights': model_name, 'url': url, 'sha256': digest,
                'cached_file': str(path), 'training_data': 'LVD-142M (generic non-medical visual pretraining)',
                'license_url': 'https://github.com/facebookresearch/dinov2/blob/main/LICENSE',
                'documentation': 'https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md'}
        self.backbone = timm.create_model(model_name,
            pretrained=pretrained, num_classes=0, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        self.classifier = nn.Linear(768, 10)

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        selected = torch.cat((tokens[:, :1], tokens[:, self.backbone.num_prefix_tokens:]), dim=1)
        return self.classifier(selected.mean(dim=1))
