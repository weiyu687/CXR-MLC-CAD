"""Compare equal-weight subsets and four pooling rules, using validation only."""
import argparse
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import frame,aggregate
from src.metrics import evaluate

def select(runs, root, max_bytes=500_000_000):
    df=frame(root,'valid')
    modes=['mean','max','frontal','weighted']
    pooled={};checkpoint_paths=[];sizes=[]
    reference_ids=None;reference_y=None
    for index,run in enumerate(runs):
        run=Path(run);checkpoint=run/'best.pt'
        checkpoint_paths.append(str(checkpoint));sizes.append(checkpoint.stat().st_size)
        with np.load(run/'valid_predictions.npz') as saved:probabilities=saved['probabilities']
        if probabilities.shape!=(len(df),10):raise ValueError('Validation image counts differ')
        for mode in modes:
            ids,p,y=aggregate(df,probabilities,mode)
            if reference_ids is None:reference_ids,reference_y=ids,y
            if not np.array_equal(ids,reference_ids) or not np.array_equal(y,reference_y):
                raise ValueError('Validation study alignment differs')
            pooled[index,mode]=p
    candidates=[]
    for count in range(1,len(runs)+1):
        for subset in itertools.combinations(range(len(runs)),count):
            model_bytes=sum(sizes[i] for i in subset)
            if model_bytes>max_bytes:continue
            for mode in modes:
                # Match predict.py: pool each member first, then average members.
                p=np.mean([pooled[i,mode] for i in subset],axis=0)
                metrics=evaluate(reference_y,p)
                candidates.append({'checkpoints':[checkpoint_paths[i] for i in subset],
                                   'aggregation':mode,'model_bytes':model_bytes,'metrics':metrics})
    if not candidates:raise ValueError('No candidate satisfies model size limit')
    candidates.sort(key=lambda x:(-x['metrics']['macro_auc'],len(x['checkpoints']),x['model_bytes']))
    return {'selected':candidates[0],'candidates':candidates,
            'scope':'Validation only; scores are model-selection estimates, not hidden-test results',
            'latency_verified':False,'tta':False}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--out',default='outputs/ensemble_selection.json')
    args=ap.parse_args()
    result=select(args.runs,args.root)
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['selected'],indent=2))
if __name__=='__main__':main()
