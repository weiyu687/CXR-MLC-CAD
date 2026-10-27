"""Audit the completed paper replication and publish aggregate results only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='outputs/matched_sota_20261026')
    parser.add_argument('--root', default='raw_data/cxr_first')
    parser.add_argument('--out', default='docs/research/matched_sota_verified_20261027.json')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    work = Path(args.work)
    status, plan = read(work/'status.json'), read(work/'plan.json')
    if status['stage']!='TRAINING_COMPLETE_REVIEW_PENDING':
        raise ValueError('The fixed queue is not complete')
    for relative, expected in plan['source_sha256'].items():
        if sha(project/relative)!=expected:
            raise ValueError('Frozen training source changed before review: '+relative)
    from src.data import frame
    from src.metrics import evaluate
    data = frame(args.root, 'valid').drop_duplicates('Study_id').set_index('Study_id')
    results = []
    patient_sets = []
    names = [entry['model'] for entry in status['completed_runs']]+['study_fusion']
    for name in names:
        folder = work/name
        history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        expected_epochs = 6 if name=='study_fusion' else 7
        if [row['epoch'] for row in history]!=list(range(1,expected_epochs+1)):
            raise ValueError('Incomplete planned training')
        metrics = read(folder/'metrics.json')
        selected = max(history, key=lambda row: row['selection_score'])
        if selected['epoch']!=metrics['epoch']:
            raise ValueError('Checkpoint was not selected by the patient holdout')
        expected = status['study_fusion'] if name=='study_fusion' else next(
            entry for entry in status['completed_runs'] if entry['model']==name)
        if sha(folder/'best.pt')!=expected['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        # These are our hash-bound local training exports; older versions
        # stored string IDs as NumPy object arrays.
        with np.load(folder/'valid_predictions.npz', allow_pickle=True) as saved:
            ids, probabilities, targets = saved['study_ids'], saved['study_probabilities'], saved['targets']
        np.testing.assert_array_equal(targets, np.stack(data.loc[ids].target))
        if len(ids)!=859 or len(set(ids))!=859:
            raise ValueError('Validation study coverage differs')
        rebuilt = evaluate(targets, probabilities)
        if abs(rebuilt['macro_auc']-metrics['macro_auc'])>1e-12:
            raise ValueError('Saved predictions and selected metrics disagree')
        patient_sets.append(set(read(folder/'holdout_patients.json')))
        results.append({'name': name, 'epochs_completed': expected_epochs, 'selected_epoch': metrics['epoch'],
            'selected_holdout_auc': metrics['selection_score'], 'selected_training_runtime_valid_auc': rebuilt['macro_auc'],
            'max_official_valid_auc_over_epochs': max(row['macro_auc'] for row in history),
            'max_official_valid_is_a_separate_selection': True,
            'checkpoint_sha256': expected['checkpoint_sha256'], 'prediction_sha256': sha(folder/'valid_predictions.npz'),
            'validation': rebuilt,
            'history': [{key: row[key] for key in ['epoch','loss','macro_auc','selection_score','training_seconds']}
                        for row in history]})
    if any(held!=patient_sets[0] for held in patient_sets):
        raise ValueError('Patient holdout differs between comparisons')
    output = Path(args.out)
    if output.exists():
        raise FileExistsError('Preserve earlier review')
    report = {'source_commit': plan['source_commit'], 'training_runtime': 'Python 3.9 / PyTorch 2.8 CUDA 12.6',
        'source_files_verified': len(plan['source_sha256']), 'same_training_patient_holdout': True,
        'holdout_patients': len(patient_sets[0]), 'valid_studies': 859, 'valid_patients': 228, 'experiments': results,
        'incumbent_target_runtime_valid_auc': .8111219096556889,
        'incumbent_user_reported_test_auc': .7678, 'target_valid_auc': .9, 'target_achieved': False,
        'new_hidden_test_result_known': False, 'official_website_submission_performed': False,
        'interpretation': 'No demonstrated gain over the incumbent. Source-bound training metrics; do not confuse the official-valid epoch peak with patient-holdout selection.'}
    target_review = project/'outputs/matched_sota_review_20261027/base22k_selected/report.json'
    if target_review.exists():
        target = read(target_review)
        if target['checkpoint_sha256']!=[results[2]['checkpoint_sha256']] or not target['arguments']['validation_only']:
            raise ValueError('Target runtime review identity differs')
        report['target_runtime_recheck'] = {key: target[key] for key in ['validation','runtime','checkpoint_sha256']}
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    for experiment, axis in zip(results, axes.flat):
        rows = experiment['history']
        x = [row['epoch'] for row in rows]
        axis.plot(x, [row['macro_auc'] for row in rows], label='Official valid', color='#155e75')
        axis.plot(x, [row['selection_score'] for row in rows], label='Training patient holdout', color='#7c3aed')
        axis.axvline(experiment['selected_epoch'], color='#e11d48', linestyle=':', linewidth=1)
        axis.set_title(experiment['name'].replace('chexfusion_', '').replace('_', ' '))
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Macro-AUC')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle('Paper replication: selected by the same training-patient holdout')
    figure.tight_layout()
    figure.savefig(project/'docs/figures/matched_sota_20261027.png', dpi=160)
    plt.close(figure)
    print(json.dumps({'aggregate': str(output), 'experiments_reviewed': len(results),
                      'target_achieved': False, 'source_hashes_verified': True}))


if __name__=='__main__':
    main()
