"""CheXFusion stage two on official training studies, with a frozen CNN.

Retains the exact patient holdout of stage one. Test data are never opened.
This research checkpoint needs a study-aware inference path before delivery.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics import evaluate
from src.paper_losses import CheXFusionWeightedASL
from src.paper_studies import PaperStudies
from src.reference_models import CheXFusionStudy
from src.training import learning_rate


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    result = np.zeros((len(loader.dataset), 10), dtype=np.float32)
    for pictures, mask, _, indices in loader:
        with torch.autocast(device.type, enabled=device.type=='cuda'):
            logits = model(pictures.to(device, non_blocking=True), mask.to(device, non_blocking=True))
        result[indices.numpy()] = logits.float().sigmoid().cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root', default='raw_data/cxr_first')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2046)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--memory-fraction', type=float, default=.8)
    args = parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.workers<0:
        raise ValueError('Invalid training dimensions')
    source = Path(args.checkpoint).resolve()
    original = torch.load(source, map_location='cpu', weights_only=False)
    cfg = original['config']
    if not cfg['model'].endswith('_mldecoder') or not cfg.get('image_training'):
        raise ValueError('Stage one must be an all-view ML-Decoder run')
    held_path = source.parent/'holdout_patients.json'
    if not held_path.is_file():
        raise ValueError('A training-only patient holdout is required')
    held = json.loads(held_path.read_text(encoding='utf-8'))
    backbone = {'chexfusion_tiny_mldecoder': 'convnext_tiny_1k',
                'chexfusion_base22k_mldecoder': 'convnext_base_22k'}[cfg['model']]
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('Preserve previous stage-two work')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    training = PaperStudies(args.root, args.cache, 'train', cfg['size'], train=True, held_patients=held)
    holdout = PaperStudies(args.root, args.cache, 'train', cfg['size'], held_patients=held)
    validation = PaperStudies(args.root, args.cache, 'valid', cfg['size'])
    assert not set(training.patients)&set(holdout.patients)
    assert not (set(training.patients)|set(holdout.patients))&set(validation.patients)
    parameters = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type=='cuda',
                      persistent_workers=args.workers>0)
    train_loader = DataLoader(training, shuffle=True, generator=torch.Generator().manual_seed(args.seed+731), **parameters)
    hold_loader = DataLoader(holdout, shuffle=False, **parameters)
    valid_loader = DataLoader(validation, shuffle=False, **parameters)
    model = CheXFusionStudy(backbone).to(device)
    model.backbone.load_state_dict({key[len('backbone.'):]: value for key, value in original['state_dict'].items()
                                    if key.startswith('backbone.')}, strict=True)
    training_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(training_parameters, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    counts = np.asarray(training.targets).sum(0)
    criterion = CheXFusionWeightedASL(counts, len(training)).to(device)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    configuration = {**vars(args), 'model': 'chexfusion_study_'+backbone, 'backbone': backbone,
                     'size': cfg['size'], 'normalization': 'imagenet', 'max_views': 4, 'stage': 2,
                     'inference_unit': 'study', 'selection_basis': 'training_patient_holdout_macro_auc',
                     'delivery_inference_integrated': False}
    provenance = {'source_commit': commit, 'python': platform.python_version(), 'torch': str(torch.__version__),
                  'pretraining': original['provenance'].get('pretraining'),
                  'stage_one_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'stage_one_epoch': original['epoch'], 'loss_counts': counts.tolist(),
                  'loss_total_studies': len(training), 'paper': 'https://arxiv.org/abs/2308.03968',
                  'backbone_frozen': True, 'attention_batch_first': True, 'explicit_view_mask': True,
                  'test_data_read': False}
    (out/'config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (out/'holdout_patients.json').write_text(json.dumps(held), encoding='utf-8')
    targets_valid = np.asarray(validation.targets)
    targets_holdout = np.asarray(holdout.targets)
    del original
    print(json.dumps({'stage': 2, 'device': str(device), 'train_studies': len(training), 'holdout_studies': len(holdout),
                      'valid_studies': len(validation), 'trainable_parameters': sum(p.numel() for p in training_parameters)}), flush=True)
    best = -1.
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        total = 0.
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats()
        for step, (pictures, mask, target, _) in enumerate(train_loader):
            rate = learning_rate(epoch+step/len(train_loader), args.epochs, args.lr, 0.)
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            pictures = pictures.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type=='cuda'):
                logits = model(pictures, mask)
                loss = criterion(logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite study loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(training_parameters, 5.)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            if step%100==0:
                print(f'epoch={epoch+1} step={step}/{len(train_loader)} loss={loss.item():.5f}', flush=True)
        training_seconds = time.time()-started
        predictions_valid = predict(model, valid_loader, device)
        predictions_holdout = predict(model, hold_loader, device)
        metrics = evaluate(targets_valid, predictions_valid)
        holdout_metrics = evaluate(targets_holdout, predictions_holdout)
        record = {'epoch': epoch+1, 'loss': total/len(train_loader), 'training_seconds': training_seconds,
                  'seconds': time.time()-started, 'holdout': holdout_metrics,
                  'selection_score': holdout_metrics['macro_auc'], **metrics,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None}
        with (out/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        improved = holdout_metrics['macro_auc']>best
        best = max(best, holdout_metrics['macro_auc'])
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                      'epoch': epoch+1, 'config': configuration, 'provenance': provenance,
                      'metrics': metrics, 'holdout_metrics': holdout_metrics, 'best': best,
                      'criterion': criterion.state_dict(), 'random': random.getstate(), 'numpy': np.random.get_state(),
                      'torch': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        torch.save(checkpoint, out/'last.pt')
        if improved:
            torch.save({key: checkpoint[key] for key in ['state_dict','epoch','config','provenance','metrics','holdout_metrics']}, out/'best.pt')
            np.savez_compressed(out/'valid_predictions.npz', study_ids=validation.study_ids,
                                study_probabilities=predictions_valid, targets=targets_valid)
            (out/'metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    (out/'complete.json').write_text(json.dumps({'epochs_completed': args.epochs, 'best_holdout_auc': best,
        'official_website_submission_performed': False, 'delivery_inference_integrated': False}, indent=2), encoding='utf-8')
    print('STUDY_TRAINING_COMPLETE', flush=True)


if __name__=='__main__':
    main()
