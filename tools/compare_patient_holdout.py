"""Export and compare train-patient holdout predictions, never reading test data.

Checkpoints must have been selected on exactly the same patient holdout, trained
from scratch or audited generic initialization. Full-train refinements and
checkpoints initialized from competition checkpoints are deliberately rejected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data import Images, aggregate
from src.inference import configure_deterministic_inference, load_deployed_model, runtime_information
from src.metrics import submission_probabilities
from tools.patient_statistics import paired_cluster_bootstrap
from train import predict


def checked_holdout(root, run, config):
    if not 0 < config.get('patient_holdout', 0) < .5 or config.get('init') or config.get('resume'):
        raise ValueError('Checkpoint is not an independently trained patient-holdout run: ' + str(run))
    from src.data import frame
    patients = np.sort(frame(root, 'train').Subject_id.unique())
    seed = config.get('split_seed', config['seed'])
    expected = sorted(np.random.default_rng(seed).choice(
        patients, int(len(patients) * config['patient_holdout']), replace=False).tolist())
    recorded = json.loads((run / 'holdout_patients.json').read_text(encoding='utf-8'))
    if recorded != expected:
        raise ValueError('Saved patient holdout does not match the recorded split parameters')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Reference first; at least two runs')
    parser.add_argument('--root', default='raw_data/cxr_first')
    parser.add_argument('--cache', default='data_cache/448')
    parser.add_argument('--out', required=True, help='New local-only output directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2037)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    runs = [Path(run).resolve() for run in args.runs]
    if len(runs) < 2 or len(set(runs)) != len(runs) or len({run.name for run in runs}) != len(runs):
        raise ValueError('Provide at least two distinct runs with distinct directory names')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Preserve existing holdout exports: ' + str(out))
    # Validate every candidate before writing outputs or running inference.
    configs = []
    reference_patients = None
    for run in runs:
        checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        config = checkpoint['config']
        held = checked_holdout(args.root, run, config)
        if reference_patients is not None and held != reference_patients:
            raise ValueError('Patient partitions differ between comparison runs')
        reference_patients = held
        configs.append(config)
        del checkpoint
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    configure_deterministic_inference()
    device = torch.device(args.device)
    predictions = {}
    provenance = {}
    reference_ids = reference_y = reference_subjects = None
    for run, config in zip(runs, configs):
        model, checkpoint = load_deployed_model(run / 'best.pt', device)
        ds = Images(args.root, args.cache, 'train', config['size'], normalization=config.get('normalization', 'legacy'))
        ds.df = ds.df[ds.df.Subject_id.isin(reference_patients)].reset_index(drop=True)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                        pin_memory=device.type == 'cuda')
        image_p = predict(model, dl, device)
        ids, p, y = aggregate(ds.df, image_p, 'frontal')
        p = submission_probabilities(p)
        subjects = ds.df.drop_duplicates('Study_id').set_index('Study_id').loc[ids].Subject_id.to_numpy()
        if reference_ids is not None and not all(np.array_equal(a, b) for a, b in
                [(ids, reference_ids), (y, reference_y), (subjects, reference_subjects)]):
            raise ValueError('Study labels or patient ordering differ across candidates')
        reference_ids, reference_y, reference_subjects = ids, y, subjects
        predictions[run.name] = p
        np.savez_compressed(out / (run.name + '.npz'), probabilities=p, targets=y,
                            study_ids=ids, patient_ids=subjects)
        provenance[run.name] = {
            'checkpoint_sha256': hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest(),
            'selected_epoch': checkpoint['epoch'], 'training_config': config,
            'training_provenance': checkpoint.get('provenance'),
        }
        print('EXPORTED ' + run.name, flush=True)
        del model, checkpoint
    result = paired_cluster_bootstrap(reference_y, predictions, reference_subjects, args.bootstrap, args.seed)
    result.update(runtime=runtime_information(device), batch_size=args.batch_size, aggregation='frontal', tta=False,
                  train_csv_sha256=hashlib.sha256((Path(args.root) / 'train.csv').read_bytes()).hexdigest(),
                  holdout_patients_sha256=hashlib.sha256(json.dumps(reference_patients).encode()).hexdigest(),
                  checkpoint_provenance=provenance)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
