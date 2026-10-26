"""Finite, source-frozen CheXFusion replication and paired head ablation.

Does not read test data, submit to a website, or modify Git. Run without
--execute to review the exact commands. Results remain local until reviewed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='raw_data/cxr_first')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--work', default='outputs/matched_sota_20261026')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    work = Path(args.work).resolve()
    recipes = []
    for model in ['chexfusion_tiny_mldecoder', 'chexfusion_tiny_gap', 'chexfusion_base22k_mldecoder']:
        out = work/model
        command = [sys.executable, '-u', 'train.py', '--root', str(Path(args.root).resolve()), '--cache', args.cache,
            '--out', str(out), '--model', model, '--pretrained', '--size', '384', '--batch-size', '16', '--epochs', '7',
            '--workers', str(args.workers), '--lr', '3e-5', '--seed', '2045', '--loss', 'chexfusion_asl',
            '--image-training', '--augmentation', 'chexfusion', '--selection-aggregation', 'mean',
            '--holdout-aggregation', 'mean', '--patient-holdout', '.10', '--split-seed', '2030',
            '--head-lr-mult', '1', '--weight-decay', '.01', '--warmup-epochs', '0', '--ema-decay', '0',
            '--cuda-memory-fraction', '.80']
        recipes.append({'model': model, 'out': str(out), 'command': command})
    if not args.execute:
        print(json.dumps({'stage_one_recipes': recipes, 'stage_two': {
            'selection': 'Higher train-patient-holdout AUC among the two ML-Decoder runs',
            'epochs': 6, 'learning_rate': 3e-5, 'weight_decay': .01, 'batch_size': 16}}, indent=2))
        return
    if work.exists():
        raise FileExistsError('Preserve earlier research queue: '+str(work))
    work.mkdir(parents=True)
    code = [project/'train.py', *sorted((project/'src').glob('*.py')),
            project/'tools/train_chexfusion_study.py', Path(__file__).resolve()]
    hashes = {str(path.relative_to(project)): digest(path) for path in code}
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=project, capture_output=True, text=True, check=True).stdout.strip()
    status_output = subprocess.run(['git','status','--porcelain'], cwd=project, capture_output=True, text=True, check=True).stdout
    if status_output.strip():
        raise RuntimeError('Commit the reviewed source and documentation before starting the fixed queue')
    plan = {'recipes': recipes, 'source_commit': commit, 'source_sha256': hashes,
        'training_csv_sha256': digest(Path(args.root)/'train.csv'),
        'validation_csv_sha256': digest(Path(args.root)/'valid.csv'),
        'paper': 'https://arxiv.org/abs/2308.03968', 'stage_one_epochs': 7, 'stage_two_epochs': 6,
        'changes_from_paper': ['10 labels; only official competition train',
            'Tiny ImageNet-1K 384 ablation and Base ImageNet-22K 384 hardware adaptation',
            'Patient holdout instead of image-level random validation',
            'Canonical padded 448 cache before author training augmentations; existing deterministic validation resizing',
            'Fixed all-view mean for stage-one validation and holdout; gradient norm clipped at 5',
            'Modern decoder interface; corrected spatial positional axes and fusion batch/padding masks'],
        'target_valid_auc': .90, 'target_achieved': False,
        'test_data_used_for_training_or_selection': False, 'official_website_submission_performed': False}
    (work/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    status = {'stage': 'PLAN_FROZEN', 'started_unix': time.time(), 'completed_runs': [], 'source_commit': commit}
    def save():
        temporary = work/'status.pending.json'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(work/'status.json')
        print(json.dumps(status), flush=True)
    def check_source():
        for relative, expected in hashes.items():
            if digest(project/relative)!=expected:
                raise ValueError('Frozen training implementation changed: '+relative)
    def execute(command, log):
        check_source()
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT, check=True)
    save()
    try:
        for recipe in recipes:
            status.update(stage='STAGE_ONE_TRAINING', current_model=recipe['model'])
            save()
            execute(recipe['command'], work/(recipe['model']+'.log'))
            folder = Path(recipe['out'])
            history = [json.loads(line) for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines()]
            if len(history)!=7 or [row['epoch'] for row in history]!=list(range(1,8)):
                raise ValueError('Incomplete seven-epoch stage one')
            metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            status['completed_runs'].append({'model': recipe['model'], 'checkpoint': str(folder/'best.pt'),
                'checkpoint_sha256': digest(folder/'best.pt'), 'epochs_completed': 7,
                'selected_epoch': metrics['epoch'], 'holdout_auc': metrics['holdout']['macro_auc'],
                'official_valid_auc_at_holdout_selection': metrics['macro_auc'],
                'history_aggregate': [{key: row[key] for key in ['epoch','loss','macro_auc','selection_score',
                                         'training_seconds','training_peak_allocated_mib']} for row in history]})
            save()
        eligible = [item for item in status['completed_runs'] if item['model'].endswith('_mldecoder')]
        selected = max(eligible, key=lambda item: item['holdout_auc'])
        if digest(selected['checkpoint'])!=selected['checkpoint_sha256']:
            raise ValueError('Stage-one checkpoint changed')
        status.update(stage='STAGE_TWO_TRAINING', current_model=selected['model'],
                      stage_two_initialization=selected['checkpoint_sha256'])
        save()
        execute([sys.executable, '-u', '-m', 'tools.train_chexfusion_study', '--checkpoint', selected['checkpoint'],
            '--root', str(Path(args.root).resolve()), '--cache', args.cache, '--out', str(work/'study_fusion'),
            '--epochs', '6', '--batch-size', '16', '--workers', str(args.workers), '--lr', '3e-5', '--seed', '2046'],
            work/'study_fusion.log')
        metrics = json.loads((work/'study_fusion/metrics.json').read_text(encoding='utf-8'))
        status.update(stage='TRAINING_COMPLETE_REVIEW_PENDING', current_model=None, completed_unix=time.time(),
            study_fusion={'epochs_completed': 6, 'selected_epoch': metrics['epoch'],
                'holdout_auc': metrics['holdout']['macro_auc'], 'official_valid_auc_at_holdout_selection': metrics['macro_auc'],
                'checkpoint_sha256': digest(work/'study_fusion/best.pt'), 'delivery_inference_integrated': False},
            official_website_submission_performed=False)
        save()
    except Exception as error:
        status.update(stage='FAILED', error=repr(error))
        save()
        raise


if __name__=='__main__':
    main()
