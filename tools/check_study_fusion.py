"""Regression checks for masked, independent study-level attention."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from src.reference_models import SpatialPosition, CheXFusionImage, CheXFusionStudy

    torch.set_num_threads(2)
    torch.manual_seed(2045)
    path = project/'outputs/research_review/repos/tatp22/multidim-positional-encoding/positional_encodings/torch_encodings.py'
    spec = importlib.util.spec_from_file_location('audited_positions', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    for channels in [32, 768, 1024]:
        features = torch.randn(2, channels, 3, 5)
        expected = features+original.PositionalEncodingPermute2D(channels)(features)
        torch.testing.assert_close(SpatialPosition(channels)(features), expected, rtol=0, atol=0)

    image = CheXFusionImage().eval()
    inputs = torch.randn(2, 3, 64, 64)
    logits = image(inputs)
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert image.backbone[0][0].weight.grad is not None
    assert image.classifier.decoder.query_embed.weight.grad is None
    del image

    study = CheXFusionStudy().eval()
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    features = torch.randn(2, 4, 768, 4, 4)
    with torch.no_grad():
        together = study.fuse_features(features, mask)
        alone = study.fuse_features(features[:1, :1], mask[:1, :1])
        torch.testing.assert_close(together[:1], alone, atol=2e-6, rtol=2e-5)
        changed = features.clone()
        changed[1] *= 100
        torch.testing.assert_close(study.fuse_features(changed, mask)[:1], alone, atol=2e-6, rtol=2e-5)
        changed = features.clone()
        changed[~mask] = torch.randn_like(changed[~mask])*1e6
        torch.testing.assert_close(study.fuse_features(changed, mask), together, atol=2e-6, rtol=2e-5)
        pictures = torch.randn(2, 4, 3, 64, 64)
        via_images = study(pictures, mask)
        encoded = study.backbone(pictures[mask])
        padded = encoded.new_zeros(2, 4, *encoded.shape[1:])
        padded[mask] = encoded
        torch.testing.assert_close(study.fuse_features(padded, mask), via_images, atol=1e-6, rtol=1e-5)
    study.train()
    assert not study.backbone.training and study.encoder.training
    study.fuse_features(features, mask).square().mean().backward()
    assert all(parameter.grad is None for parameter in study.backbone.parameters())
    assert study.projection.weight.grad is not None
    assert torch.isfinite(study.projection.weight.grad).all()
    assert study.classifier.decoder.duplicate_pooling.grad is not None
    for invalid_mask in [torch.zeros_like(mask), torch.ones(2, 3, dtype=torch.bool)]:
        try:
            study.fuse_features(features, invalid_mask)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid masks must fail explicitly')
    result = {'python': sys.version.split()[0], 'torch': str(torch.__version__),
              'spatial_position_matches_NCHW_reference': True,
              'image_backbone_receives_gradient': True,
              'fixed_decoder_queries_receive_no_gradient': True,
              'study_patient_batch_independence': True,
              'study_padding_invariance': True,
              'image_and_cached_feature_fusion_equivalence': True,
              'stage2_backbone_frozen_and_eval': True,
              'stage2_projection_and_decoder_receive_gradient': True,
              'invalid_masks_rejected': True,
              'scope': 'Synthetic CPU correctness only; no validation performance claims'}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
