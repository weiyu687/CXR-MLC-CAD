"""Run the real training entrypoint on synthetic images, without a GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError('Preserve previous synthetic checks')
    destination.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from src.model import build_model
    rng = np.random.default_rng(2045)
    for split, patients in [('train', 8), ('valid', 2)]:
        folder = destination/'cache'/split
        folder.mkdir(parents=True)
        records = []
        for patient in range(patients):
            for study in range(2):
                for view in range(2):
                    identifier = f'{split}_{patient}_{study}_{view}'
                    Image.fromarray(rng.integers(0, 256, (448, 448, 3), dtype=np.uint8)).save(folder/(identifier+'.jpg'))
                    records.append({'Dicom_id': identifier, 'Subject_id': f'{split}_p{patient}',
                        'Study_id': f'{split}_s{patient}_{study}', 'ViewPosition': 'PA' if view==0 else 'LATERAL',
                        'Predict_class': ';'.join(map(str, range(1, 11))) if study else ''})
        pd.DataFrame(records).to_csv(destination/(split+'.csv'), index=False)
    results = []
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}
    for name in ['chexfusion_tiny_gap', 'chexfusion_tiny_mldecoder']:
        output = destination/name
        command = [sys.executable, '-u', 'train.py', '--root', str(destination), '--cache', str(destination/'cache'),
            '--out', str(output), '--model', name, '--size', '64', '--batch-size', '4', '--epochs', '1',
            '--workers', '0', '--lr', '3e-5', '--loss', 'chexfusion_asl', '--image-training',
            '--augmentation', 'chexfusion', '--normalization', 'imagenet', '--patient-holdout', '.25',
            '--split-seed', '2030', '--selection-aggregation', 'mean', '--head-lr-mult', '1',
            '--warmup-epochs', '0', '--ema-decay', '0', '--weight-decay', '.01']
        with (destination/(name+'.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        saved = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
        history = json.loads((output/'history.jsonl').read_text(encoding='utf-8').strip())
        assert history['training_peak_allocated_mib'] is None, 'CPU check unexpectedly used a GPU'
        provenance = saved['provenance']['loss_reference']
        assert provenance['total_count'] == 24
        assert provenance['class_counts'] == [12.]*10
        assert provenance['heldout_patients_excluded']
        assert saved['config']['image_training'] and saved['epoch'] == 1
        torch.set_num_threads(2)
        restored = build_model(name).eval()
        restored.load_state_dict(saved['state_dict'], strict=True)
        with torch.inference_mode():
            prediction = restored(torch.randn(1, 3, 64, 64))
        assert prediction.shape == (1, 10) and torch.isfinite(prediction).all()
        results.append({'model': name, 'entrypoint_completed': True, 'image_training_holdout_exclusion': True,
                        'training_only_loss_counts': True, 'checkpoint_restored': True, 'finite_logits': True})
        del restored, saved
    report = {'scope': 'Synthetic CPU entrypoint checks only, not competition scores', 'results': results}
    (destination/'complete.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
