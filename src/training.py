"""Training objectives and classifier warmup shared by controlled experiments."""
from typing import Optional, Set

import torch


HEAD_PREFIXES = ('fc.', 'classifier.', 'head.')


def mixed_target_loss(criterion, logits, targets, paired_targets=None,
                      mixing_coefficient=None, objective='hard_loss'):
    """Mix losses for hard labels, preserving nonlinear focal/ASL weighting.

    BCE is affine in the target, so its two forms are equivalent. ASL is not:
    its focal exponent and weight also depend on the target. ``soft_target``
    remains available to reproduce checkpoints trained with the old recipe.
    """
    if paired_targets is None:
        return criterion(logits, targets)
    if mixing_coefficient is None or not 0 <= mixing_coefficient <= 1:
        raise ValueError('Mixup coefficient must be in [0, 1]')
    lam = mixing_coefficient
    if objective == 'hard_loss':
        return lam * criterion(logits, targets) + (1-lam) * criterion(logits, paired_targets)
    if objective == 'soft_target':
        return criterion(logits, lam * targets + (1-lam) * paired_targets)
    raise ValueError('Unknown Mixup objective: ' + objective)


def configure_train_stage(model, head_only: bool,
                          trainable_names: Optional[Set[str]] = None):
    """Freeze backbone parameters AND training-time buffers during head warmup.

    A model-wide train() with requires_grad=False still updates BatchNorm.
    Only classifier modules enter training mode in the head-only stage.
    Preserve originally frozen parameters when full fine-tuning begins.
    """
    allowed = trainable_names if trainable_names is not None else {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    heads = {name for name in allowed if name.startswith(HEAD_PREFIXES)}
    if head_only and not heads:
        raise ValueError('Classifier parameters not found for head warmup')
    model.train()
    if head_only:
        model.eval()
        for name, child in model.named_children():
            if name + '.' in HEAD_PREFIXES:
                child.train()
    for name, param in model.named_parameters():
        param.requires_grad_(name in (heads if head_only else allowed))
