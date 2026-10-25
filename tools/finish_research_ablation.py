"""Finish the finite research experiment queue and prepare a reviewable bundle.

No scheduling or website submission. Existing experiments and bundles are never
overwritten. The updated user objective prioritizes official-valid metrics and
a reproducible submission CSV; final results still need a completion audit.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_extra_candidates(path, root):
    """Load an explicitly configured registry of validated fixed checkpoints."""
    raw=Path(path).read_bytes()
    payload=json.loads(raw.decode('utf-8'))
    if payload.get('schema_version')!=1 or not isinstance(payload.get('candidates'),list):
        raise ValueError('Invalid supplemental candidate registry')
    import numpy as np
    expected_csv=hashlib.sha256((Path(root)/'valid.csv').read_bytes()).hexdigest()
    runs=[]
    for entry in payload['candidates']:
        run=Path(entry['run']).resolve();checkpoint=run/'best.pt'
        if not checkpoint.is_file() or checkpoint.stat().st_size>500_000_000:
            raise ValueError('Invalid supplemental checkpoint: '+str(checkpoint))
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report_path=Path(entry['validation_report'])
        report=read(report_path)
        if digest!=entry['checkpoint_sha256'] or report['checkpoint_sha256']!=[digest]:
            raise ValueError('Supplemental checkpoint/report hash mismatch')
        settings=report['arguments'];runtime=report['runtime']
        if not report['validation_recomputed'] or not settings.get('validation_only') or not settings.get('export_images'):
            raise ValueError('Supplemental candidate lacks a validation-only export')
        if settings['batch_size']!=1 or runtime.get('device')!='cuda' or runtime['torch'].split('+')[0]!='2.10.0' or not runtime['python'].startswith('3.12.'):
            raise ValueError('Supplemental candidate uses a different target runtime/batch size')
        with np.load(report_path.with_name('predictions.npz')) as saved:
            metadata=json.loads(str(saved['image_prediction_metadata']))
        if (metadata['checkpoint_sha256']!=digest or metadata['validation_csv_sha256']!=expected_csv
                or metadata.get('runtime')!=runtime or metadata.get('batch_size')!=1
                or metadata.get('tta')!=settings['tta'] or not metadata.get('deterministic_inference')):
            raise ValueError('Supplemental prediction/model/dataset binding differs')
        score=report['validation']['macro_auc'];declared=entry['validation_macro_auc']
        if not np.isfinite([score,declared]).all() or not 0<=score<=1 or abs(score-declared)>1e-12:
            raise ValueError('Supplemental validation score differs from its report')
        runs.append(str(run))
    if len(set(runs))!=len(runs):raise ValueError('Duplicate supplemental candidate')
    evidence={'path':str(Path(path).resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'candidates':payload['candidates']}
    return runs,evidence,raw


def completed_training(run):
    """Reject a stopped or partially written training run, even if best.pt exists."""
    run = Path(run)
    config = read(run / 'config.json')
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
    if not history or [r['epoch'] for r in history] != list(range(1, len(history)+1)):
        raise ValueError('Incomplete or inconsistent training history: ' + str(run))
    import numpy as np
    scores = np.asarray([r['selection_score'] for r in history])
    if not np.isfinite(scores).all() or len(history) > config['epochs']:
        raise ValueError('Invalid training scores or epoch count: ' + str(run))
    best_index = int(np.argmax(scores))
    stale = len(history)-1-best_index
    finished = len(history) == config['epochs'] or (config.get('patience', 0) > 0 and stale >= config['patience'])
    if not finished:
        raise ValueError('Training stopped before its epoch limit or patience rule: ' + str(run))
    import torch
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    last = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    if best['epoch'] != best_index+1 or last['epoch'] != len(history)-1:
        raise ValueError('Checkpoint and history epochs disagree: ' + str(run))
    if best['config'] != config or last['config'] != config or abs(last['best']-scores[best_index]) > 1e-12:
        raise ValueError('Checkpoint configuration/score differs from training records: ' + str(run))
    return {'run': str(run.resolve()), 'epochs_completed': len(history), 'selected_epoch': best_index+1,
            'selection_score': float(scores[best_index]), 'official_valid_macro_auc': history[best_index]['macro_auc'],
            'config': config}


def wait_windows_process(pid):
    """Wait for the explicitly identified queue process; never signal or stop it."""
    if os.name != 'nt':
        raise RuntimeError('--wait-process currently supports the Windows training host only')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Cannot wait for the requested queue PID; verify completed runs and omit --wait-process')
    try:
        print('WAITING_FOR_TRAINING_QUEUE pid=' + str(pid), flush=True)
        while True:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result == 0:
                break
            if result != 258:
                raise OSError(ctypes.get_last_error(), 'Failed waiting for training queue')
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(), 'Cannot read training queue exit code')
        if code.value != 0:
            raise RuntimeError('Training queue failed with exit code ' + str(code.value))
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=[
        'outputs/research_ablation_s2036/bce_baseline',
        'outputs/research_ablation_s2036/bce_head_warmup',
        'outputs/research_ablation_s2036/asl_head_warmup'])
    parser.add_argument('--reference', default='outputs/resnet50_imagenet_hold_s2030')
    parser.add_argument('--reference-full', default='outputs/resnet50_imagenet_full_s2031')
    parser.add_argument('--extra-runs',nargs='*',default=[],help='Explicit additional validation-screened fixed candidates')
    parser.add_argument('--extra-runs-manifest',help='Explicit validated-candidate registry, reloaded and frozen before final selection')
    parser.add_argument('--target-python', required=True, help='Verified Python 3.12 / PyTorch 2.10 interpreter')
    parser.add_argument('--root', default='raw_data/cxr_first')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--ssd-root', required=True)
    parser.add_argument('--timing-root', required=True, help='Original valid images; computation screened before final test')
    parser.add_argument('--work', default='outputs/research_delivery_valid_s2038')
    parser.add_argument('--refinement', default='outputs/research_refinement_s2038')
    parser.add_argument('--efficientnet', default='outputs/effv2s_research_full_s2039')
    parser.add_argument('--efficientnet-448', default='outputs/effv2s_research_448_s2040')
    parser.add_argument('--out', default='submission/round2_bundle')
    parser.add_argument('--wait-process', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(vars(args), indent=2))
        print('PRINT_ONLY: finite holdout comparison -> selected DenseNet refinement -> validation selection -> raw replay -> cost check; add --execute')
        return
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    work = Path(args.work).resolve()
    refinement = Path(args.refinement).resolve()
    efficientnet = Path(args.efficientnet).resolve()
    efficientnet_448 = Path(args.efficientnet_448).resolve()
    bundle = Path(args.out).resolve()
    for path in [work, refinement, efficientnet, efficientnet_448, bundle]:
        if path.exists():
            raise FileExistsError('Existing output preserved: ' + str(path))
    if len(set(map(str, map(Path, args.runs)))) != len(args.runs):
        raise ValueError('Duplicate comparison run')
    for run in args.extra_runs:
        if not (Path(run)/'best.pt').is_file():raise FileNotFoundError(Path(run)/'best.pt')
    if args.extra_runs_manifest:load_extra_candidates(args.extra_runs_manifest,args.root)
    work.mkdir(parents=True)
    controller_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()

    def status(stage, **extra):
        state = {'stage': stage, 'updated_unix': time.time(), 'controller_source_commit': controller_commit, **extra}
        temporary = work / 'status.pending.json'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(work / 'status.json')
        print(json.dumps(state), flush=True)

    def execute(arguments, log, python=None):
        status('RUNNING', command=list(map(str, arguments)), log=str(log))
        with Path(log).open('w', encoding='utf-8') as stream:
            subprocess.run([python or sys.executable, '-u', *map(str, arguments)], cwd=project,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        if args.wait_process:
            status('WAITING_FOR_TRAINING_QUEUE', pid=args.wait_process)
            wait_windows_process(args.wait_process)
        records = [completed_training(run) for run in args.runs]
        status('TRAINING_QUEUE_VERIFIED', runs=records)
        comparison = work / 'patient_comparison'
        execute(['-m', 'tools.compare_patient_holdout', '--runs', args.reference, *args.runs,
                 '--root', args.root, '--cache', args.cache, '--out', comparison, '--bootstrap', '1000'],
                work / 'patient_comparison.log', python=args.target_python)
        compared = read(comparison / 'comparison.json')
        # The user updated the goal to prioritize official validation metrics.
        # Keep original holdout checkpoints for the controlled comparison, and
        # explicitly select separate EMA snapshots by official validation here.
        for record in records:
            candidate=Path(record['run'])/'official_valid_selected'
            candidate_metrics=read(candidate/'metrics.json')
            history=[json.loads(line) for line in (Path(record['run'])/'history.jsonl').read_text(encoding='utf-8').splitlines()]
            if abs(candidate_metrics['macro_auc']-max(r['macro_auc'] for r in history))>1e-12:
                raise ValueError('Archived official-validation peak is missing: '+str(candidate))
            record['official_validation_candidate']=str(candidate)
            record['official_validation_peak']=candidate_metrics['macro_auc']
        chosen = max(records, key=lambda record: record['official_validation_peak'])
        choice = {'selected': chosen, 'patient_comparison': compared,
                  'selection_scope': 'Highest official-valid Macro-AUC among separate EMA snapshots of the predefined DenseNet recipes; repeated-validation model selection estimate',
                  'full_refinement_plan': {'size': 384, 'epochs': 6, 'patience': 3, 'seed': 2038, 'lr': 3e-5},
                  'full_refinement_invalidates_old_patient_holdout_independence': True}
        (work / 'recipe_choice.json').write_text(json.dumps(choice, indent=2), encoding='utf-8')
        config = chosen['config']
        command = ['train.py', '--model', config['model'], '--init', Path(chosen['official_validation_candidate']) / 'best.pt',
                   '--size', '384', '--batch-size', '24', '--epochs', '6', '--patience', '3',
                   '--lr', '0.00003', '--head-lr-mult', '1', '--weight-decay', '0.0001',
                   '--loss', config['loss'], '--frontal-training', '--regularize', '--horizontal-flip',
                   '--mixup', '0', '--selection-aggregation', 'frontal', '--seed', '2038',
                   '--workers', '2', '--root', args.root, '--cache', args.cache, '--out', refinement]
        if config['positive_weight']:
            command.append('--positive-weight')
        execute(command, work / 'refinement.log')
        refined = completed_training(refinement)
        # Add an audited, structurally different CNN family after selecting a
        # loss recipe. Both runs use official train only; valid selects epochs.
        common=['--model','tv_efficientnet_v2_s','--loss',config['loss'],
                '--frontal-training','--regularize','--horizontal-flip','--mixup','0',
                '--weight-decay','0.0001','--selection-aggregation','frontal',
                '--workers','2','--root',args.root,'--cache',args.cache]
        if config['positive_weight']:
            common.append('--positive-weight')
        execute(['train.py',*common,'--pretrained','--freeze-backbone-epochs','2',
                 '--size','320','--batch-size','24','--epochs','12','--patience','4',
                 '--lr','0.00015','--head-lr-mult','5','--seed','2039','--out',efficientnet],
                work/'efficientnet.log')
        efficient_record=completed_training(efficientnet)
        execute(['train.py',*common,'--init',efficientnet/'best.pt',
                 '--size','448','--batch-size','16','--epochs','6','--patience','3',
                 '--lr','0.00003','--head-lr-mult','1','--seed','2040','--out',efficientnet_448],
                work/'efficientnet_448.log')
        efficient_448_record=completed_training(efficientnet_448)
        extra_runs=list(args.extra_runs);extra_registry=None
        if args.extra_runs_manifest:
            registered,extra_registry,raw=load_extra_candidates(args.extra_runs_manifest,args.root)
            extra_runs.extend(registered)
            (work/'frozen_extra_candidates.json').write_bytes(raw)
        extra_runs=list(dict.fromkeys(str(Path(path).resolve()) for path in extra_runs))
        status('FINAL_CANDIDATES_FROZEN',extra_runs=extra_runs,extra_registry=extra_registry)
        execute(['tools/finalize.py', '--runs', args.reference_full, refinement,
                 efficientnet, efficientnet_448,
                 *[r['official_validation_candidate'] for r in records], *extra_runs,
                 '--max-members','3','--root', args.root,
                 '--cache', args.cache, '--ssd-root', args.ssd_root, '--timing-root', args.timing_root,
                 '--timing-split', 'valid', '--study-timing', '--batch-size', '1',
                 '--latency-statistic', 'mean', '--latency-ms', '100', '--samples', '100',
                 '--work', work / 'final_selection', '--out', bundle], work / 'finalization.log', python=args.target_python)
        # A fixed random sample is used solely for the frozen final script's
        # reproduction/cost check. Its results never trigger reselection.
        import numpy as np
        import pandas as pd
        root = Path(args.ssd_root)
        meta = pd.read_csv(root / 'test.csv')
        sampled = np.random.default_rng(123).choice(len(meta), 105, replace=False)
        measured = work / 'pictures_100'
        warmup = work / 'pictures_5'
        for i, index in enumerate(sampled):
            row = meta.iloc[index]
            relative = Path(row.Subject_id) / row.Study_id / (row.Dicom_id + '.jpg')
            source = root / 'test' / relative
            destination = (warmup if i < 5 else measured) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                raise ValueError('Timing sample copy differs from official image')
        execute([bundle / 'cost.py', bundle / 'submission.csv', measured, warmup,
                 '--bundle', bundle, '--root', root, '--out', work / 'cost_results'],
                work / 'cost.log', python=args.target_python)
        cost = read(work / 'cost_results' / 'cost_report.json')
        if not cost['official_sample_protocol'] or not cost['submission_predictions_match'] or not cost['local_mean_limit_passed']:
            raise RuntimeError('Frozen final cost/reproduction check failed; preserve evidence without test-based model reselection')
        summary = {'bundle': str(bundle), 'recipe_choice': str(work / 'recipe_choice.json'),
                   'refinement': refined, 'efficientnet':efficient_record,'efficientnet_448':efficient_448_record,
                   'extra_validation_candidates':extra_runs,'extra_registry':extra_registry,
                   'finalization': read(work / 'final_selection' / 'complete.json'),
                   'cost': cost, 'official_website_submission_performed': False,
                   'overall_deliverable_complete': False,
                   'remaining': ['Review final validation gain and patient comparisons',
                                 'Provide verified submission CSV and publish aggregate results; T4 timing remains unverified locally']}
        (work / 'prepared.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        status('PREPARED_FOR_MATERIAL_REVIEW', bundle=str(bundle), overall_deliverable_complete=False)
    except Exception as error:
        status('FAILED', error=type(error).__name__ + ': ' + str(error))
        raise


if __name__ == '__main__':
    main()
