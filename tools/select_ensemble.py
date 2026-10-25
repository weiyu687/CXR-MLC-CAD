"""Compare equal-weight subsets and four pooling rules, using validation only."""
import argparse
import itertools
import json
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import frame,aggregate
from src.metrics import evaluate
from src.ensemble import combine_probabilities

def select(runs, root, max_bytes=500_000_000, prediction_files=None, max_members=0, weight_grid='equal'):
    if max_members<0:raise ValueError('max_members must be nonnegative; 0 means unlimited')
    if weight_grid not in ['equal','quarter']:raise ValueError('Unknown fixed weight grid')
    df=frame(root,'valid')
    modes=['mean','max','frontal','weighted']
    pooled={};checkpoint_paths=[];sizes=[]
    reference_ids=None;reference_y=None
    export_settings=None
    if prediction_files is not None and len(prediction_files)!=len(runs):
        raise ValueError('Provide one deterministic prediction file per run, in the same order')
    for index,run in enumerate(runs):
        run=Path(run);checkpoint=run/'best.pt'
        checkpoint_paths.append(str(checkpoint));sizes.append(checkpoint.stat().st_size)
        if prediction_files is None:
            with np.load(run/'valid_predictions.npz') as saved:probabilities=saved['probabilities']
        else:
            with np.load(prediction_files[index]) as saved:
                metadata=json.loads(str(saved['image_prediction_metadata']))
                if not metadata['deterministic_inference']:raise ValueError('Predictions are not deterministic deployment exports')
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=metadata['checkpoint_sha256']:
                    raise ValueError('Prediction/checkpoint hash mismatch')
                if hashlib.sha256((Path(root)/'valid.csv').read_bytes()).hexdigest()!=metadata['validation_csv_sha256']:
                    raise ValueError('Validation metadata changed since prediction export')
                if not np.array_equal(saved['valid_image_ids'],df.Dicom_id.to_numpy(dtype=str)):
                    raise ValueError('Exported validation image order differs')
                settings=(metadata['batch_size'],metadata['tta'],json.dumps(metadata.get('runtime',{}),sort_keys=True))
                if export_settings is None:export_settings=settings
                if settings!=export_settings:raise ValueError('Exported batch/TTA/runtime settings differ between members')
                probabilities=saved['valid_images']
        if probabilities.shape!=(len(df),10):raise ValueError('Validation image counts differ')
        for mode in modes:
            ids,p,y=aggregate(df,probabilities,mode)
            if reference_ids is None:reference_ids,reference_y=ids,y
            if not np.array_equal(ids,reference_ids) or not np.array_equal(y,reference_y):
                raise ValueError('Validation study alignment differs')
            pooled[index,mode]=p
    candidates=[]
    for count in range(1,min(len(runs),max_members or len(runs))+1):
        for subset in itertools.combinations(range(len(runs)),count):
            model_bytes=sum(sizes[i] for i in subset)
            if model_bytes>max_bytes:continue
            weights=[None]
            if weight_grid=='quarter' and count==2:
                weights.extend([[.25,.75],[.75,.25]])
            if weight_grid=='quarter' and count==3:
                weights.extend([[.5 if j==i else .25 for j in range(3)] for i in range(3)])
            for mode in modes:
                for weight in weights:
                    # Match predict.py: pool each member first, then combine.
                    p=combine_probabilities([pooled[i,mode] for i in subset],weight)
                    metrics=evaluate(reference_y,p)
                    candidates.append({'checkpoints':[checkpoint_paths[i] for i in subset],
                                       'aggregation':mode,'model_bytes':model_bytes,'metrics':metrics,
                                       'ensemble_weights':weight,
                                       'tta':export_settings[1] if export_settings else False})
    if not candidates:raise ValueError('No candidate satisfies model size limit')
    candidates.sort(key=lambda x:(-x['metrics']['macro_auc'],-x['metrics']['macro_f1_tuned'],len(x['checkpoints']),x['model_bytes']))
    return {'selected':candidates[0],'candidates':candidates,
            'scope':'Validation only; fixed global weight grid; scores are model-selection estimates, not hidden-test results',
            'weight_grid':weight_grid,
            'latency_verified':False,'tta':export_settings[1] if export_settings else False,
            'batch_size':export_settings[0] if export_settings else None,
            'runtime':json.loads(export_settings[2]) if export_settings else None,
            'prediction_source':'deterministic deployment exports' if prediction_files is not None else 'training validation snapshots'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--prediction-files',nargs='+',help='Deterministic predict.py --export-images NPZ files, one per run')
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--max-members',type=int,default=0,help='Limit ensemble size; 0 compares all subsets')
    ap.add_argument('--weight-grid',choices=['equal','quarter'],default='equal')
    ap.add_argument('--out',default='outputs/ensemble_selection.json')
    args=ap.parse_args()
    result=select(args.runs,args.root,prediction_files=args.prediction_files,max_members=args.max_members,weight_grid=args.weight_grid)
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['selected'],indent=2))
if __name__=='__main__':main()
