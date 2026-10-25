"""Check image independence and trainability before any query-head experiment."""
import copy
import torch
from src.model import LabelQueryHead, build_model
from src.training import configure_train_stage


def main():
    torch.manual_seed(2038)
    torch.set_num_threads(2)
    head = LabelQueryHead(12, dimension=16, heads=4).eval()
    features = torch.randn(3, 12, 4, 5)
    with torch.no_grad():
        together = head(features)
        serial = torch.cat([head(x.unsqueeze(0)) for x in features])
        torch.testing.assert_close(together, serial, atol=1e-6, rtol=1e-5)
        changed = features.clone()
        changed[1:] = torch.randn_like(changed[1:]) * 100
        torch.testing.assert_close(head(changed)[0], together[0], atol=1e-6, rtol=1e-5)
        # The head has no positional encoding: reordering spatial tokens alone
        # must not introduce cross-image interactions or change pooled scores.
        permuted = features.flatten(2)[:, :, torch.randperm(20)].reshape_as(features)
        torch.testing.assert_close(head(permuted), together, atol=1e-6, rtol=1e-5)
        assert (head(changed)[1:] - together[1:]).abs().max() > 1e-4
    head.train()
    head(features).square().mean().backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert head.queries.grad.abs().sum() > 0

    model = build_model('tv_densenet121_query').eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = model(images)
    restored = build_model('tv_densenet121_query').eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
    allowed = {name for name, p in model.named_parameters() if p.requires_grad}
    initial = copy.deepcopy(model.backbone.state_dict())
    configure_train_stage(model, True, allowed)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    optimizer.zero_grad()
    model(images).square().mean().backward()
    optimizer.step()
    assert all(torch.equal(value, initial[name]) for name, value in model.backbone.state_dict().items())
    assert not model.backbone.training and model.classifier.training
    configure_train_stage(model, False, allowed)
    assert model.backbone.training and all(p.requires_grad for p in model.parameters())
    print('PASS: query-head batch/serial and other-image independence, spatial permutation, gradients, strict state restore, DenseNet backbone/BN freeze')


if __name__ == '__main__':
    main()
