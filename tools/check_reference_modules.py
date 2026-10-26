"""Numerical and gradient checks against locally audited author implementations."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.ml_decoder import MLDecoder
    from src.paper_losses import CheXFusionWeightedASL

    torch.set_num_threads(2)
    torch.manual_seed(2045)
    upstream = root/'outputs/research_review/repos'
    reference = import_file('audited_ml_decoder', upstream/'Alibaba-MIIL/ML_Decoder/src_files/ml_decoder/ml_decoder.py')
    original_loss = import_file('audited_chexfusion_loss', upstream/'dongkyunk/CheXFusion/model/loss.py')
    candidate = MLDecoder(10, initial_num_features=32, decoder_embedding=32).double().eval()
    original = reference.MLDecoder(10, initial_num_features=32, decoder_embedding=32).double().eval()
    original.load_state_dict(candidate.state_dict(), strict=True)

    def original_forward(x):
        # The upstream block itself runs unchanged. Bypass only the old
        # nn.TransformerDecoder container's incompatible modern API.
        memory = torch.relu(original.decoder.embed_standart(x)).transpose(0, 1)
        query = original.decoder.query_embed.weight.unsqueeze(1).expand(-1, x.shape[0], -1)
        decoded = original.decoder.layers[0](query, memory).transpose(0, 1)
        output = torch.zeros(*decoded.shape[:2], original.decoder.duplicate_factor, dtype=decoded.dtype)
        original.decoder.group_fc(decoded, original.decoder.duplicate_pooling, output)
        return output.flatten(1)[:, :10]+original.decoder.duplicate_pooling_bias

    features = torch.randn(2, 9, 32, dtype=torch.float64, requires_grad=True)
    upstream_features = features.detach().clone().requires_grad_(True)
    actual = candidate(features)
    expected = original_forward(upstream_features)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    probe = torch.randn_like(actual)
    (actual*probe).sum().backward()
    (expected*probe).sum().backward()
    torch.testing.assert_close(features.grad, upstream_features.grad, atol=1e-10, rtol=1e-9)
    maximum_parameter_gradient_difference = 0.
    for (name, parameter), (original_name, original_parameter) in zip(candidate.named_parameters(), original.named_parameters()):
        assert name == original_name
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, original_parameter.grad, atol=1e-10, rtol=1e-9)
            maximum_parameter_gradient_difference = max(maximum_parameter_gradient_difference,
                (parameter.grad-original_parameter.grad).abs().max().item())
    assert not candidate.decoder.query_embed.weight.requires_grad
    with torch.no_grad():
        single = candidate(features[:1])
        torch.testing.assert_close(single, actual[:1], atol=1e-10, rtol=1e-9)
        changed = features.detach().clone()
        changed[1] = torch.randn_like(changed[1])*100
        torch.testing.assert_close(candidate(changed)[:1], single, atol=1e-10, rtol=1e-9)
        padded = torch.cat([features, torch.randn(2, 5, 32, dtype=torch.float64)*100], dim=1)
        mask = torch.zeros(2, 14, dtype=torch.bool)
        mask[:, 9:] = True
        torch.testing.assert_close(candidate(padded, memory_key_padding_mask=mask), actual, atol=1e-10, rtol=1e-9)
        spatial = features.transpose(1, 2).reshape(2, 32, 3, 3)
        torch.testing.assert_close(candidate(spatial), actual, atol=1e-10, rtol=1e-9)

    counts = [1, 3, 8, 15, 25, 40, 60, 75, 93, 99]
    objective = CheXFusionWeightedASL(counts, 100).double()
    published_objective = original_loss.ASLwithClassWeight(counts, 100)
    logits = (torch.randn(7, 10, dtype=torch.float64)*3).requires_grad_(True)
    original_logits = logits.detach().clone().requires_grad_(True)
    target = torch.randint(0, 2, logits.shape).double()
    loss = objective(logits, target)
    # The author loss unconditionally calls .cuda() on constant tensors.
    # Adapt this device transfer only; the published forward is untouched.
    with patch.object(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self):
        expected_loss = published_objective(original_logits, target)
    torch.testing.assert_close(loss, expected_loss, atol=1e-12, rtol=1e-11)
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, original_logits.grad, atol=1e-12, rtol=1e-10)
    assert torch.autograd.gradcheck(objective, (logits.detach().requires_grad_(True), target))
    reconstructed = CheXFusionWeightedASL([0]*10, 100).double()
    reconstructed.load_state_dict(objective.state_dict(), strict=True)
    torch.testing.assert_close(reconstructed(logits, target), loss)
    result = {'torch': str(torch.__version__), 'python': sys.version.split()[0],
              'ml_decoder_output_max_difference': (actual-expected).abs().max().item(),
              'ml_decoder_input_gradient_max_difference': (features.grad-upstream_features.grad).abs().max().item(),
              'ml_decoder_parameter_gradient_max_difference': maximum_parameter_gradient_difference,
              'weighted_asl_loss_difference': (loss-expected_loss).abs().item(),
              'weighted_asl_gradient_max_difference': (logits.grad-original_logits.grad).abs().max().item(),
              'fixed_queries': True, 'patient_batch_independence': True, 'padding_invariance': True,
              'weighted_asl_finite_difference_gradcheck': True, 'weighted_asl_state_roundtrip': True,
              'author_source_sha256': {
                  str(path.relative_to(upstream)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [upstream/'Alibaba-MIIL/ML_Decoder/src_files/ml_decoder/ml_decoder.py',
                               upstream/'dongkyunk/CheXFusion/model/loss.py']},
              'scope': 'Synthetic CPU inputs; interface and mathematical parity, not classification performance'}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
