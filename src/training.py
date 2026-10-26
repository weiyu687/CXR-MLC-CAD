"""Training objectives and classifier warmup shared by controlled experiments."""
from typing import Optional, Set
import json
import math
from pathlib import Path

import torch


HEAD_PREFIXES = ('fc.', 'classifier.', 'head.')


def learning_rate(progress, epochs, base_lr, warmup_epochs=2., min_lr_ratio=0.):
    """Keep the legacy schedule exactly when the new options use defaults."""
    if epochs <= 0 or base_lr <= 0 or warmup_epochs < 0 or not 0 <= min_lr_ratio <= 1:
        raise ValueError('Invalid learning-rate schedule')
    progress = min(max(progress, 0.), epochs)
    warmup = min(1., (progress + .1) / warmup_epochs) if warmup_epochs else 1.
    cosine = .5 * (1 + math.cos(math.pi * progress / epochs))
    return base_lr * warmup * (min_lr_ratio + (1-min_lr_ratio)*cosine)


def optimizer_groups(model, head_multiplier=1., exclude_bias_norm=False):
    """Partition parameters once by classifier LR and optional decay exclusion."""
    groups = {}
    found_head = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = name.startswith(HEAD_PREFIXES)
        found_head |= is_head
        scale = head_multiplier if is_head else 1.
        no_decay = exclude_bias_norm and (parameter.ndim <= 1 or name.endswith('.bias'))
        key = (scale, no_decay)
        if key not in groups:
            groups[key] = {'params': [], 'lr_scale': scale}
            if no_decay:
                groups[key]['weight_decay'] = 0.
        groups[key]['params'].append(parameter)
    if head_multiplier != 1. and not found_head:
        raise ValueError('Classifier parameters not found')
    return list(groups.values())


def save_validation_candidate(run, checkpoint, record, resume_state=False):
    """Preserve a separate official-validation-selected EMA model.

    The original patient-holdout best.pt remains untouched. Training resumes
    store zero-based epoch indices and separate raw/EMA states; deployed
    snapshots store one-based indices and must always use the EMA state.
    """
    epoch = checkpoint['epoch'] + int(resume_state)
    score = record['macro_auc']
    if epoch != record['epoch'] or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Validation record and checkpoint disagree')
    folder = Path(run) / 'official_valid_selected'
    metadata = folder / 'metrics.json'
    if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8'))['macro_auc'] >= score:
        return False
    state = checkpoint['ema'] if resume_state else checkpoint['state_dict']
    snapshot = {'state_dict': {key: value.detach().cpu() for key, value in state.items()},
                'epoch': epoch, 'config': checkpoint['config'], 'provenance': checkpoint.get('provenance'),
                'selection_aggregation': record['aggregation'],
                'metrics': {key: record[key] for key in ['macro_auc', 'auc', 'macro_f1_05', 'macro_f1_tuned', 'gates']},
                'holdout_metrics': record.get('holdout'), 'selection_basis': 'official_valid_macro_auc',
                'original_selection_score': record['selection_score']}
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / 'best.pending.pt'
    torch.save(snapshot, temporary)
    temporary.replace(folder / 'best.pt')
    pending_metadata = folder / 'metrics.pending.json'
    pending_metadata.write_text(json.dumps(record, indent=2), encoding='utf-8')
    pending_metadata.replace(metadata)
    return True


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
