"""Fit a small, patient-cross-fitted fusion grid using validation exports only.

Checkpoint selection already used this validation set. Even the fusion OOF
scores are not independent estimates of the entire image-model pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import roc_auc_score
from src.data import frame,aggregate
from src.metrics import evaluate,submission_probabilities


GLOBAL_GRID=np.array([[1/3]*3,[.5,.25,.25],[.25,.5,.25],[.25,.25,.5]])
CLASS_GRID=np.array([[a/4,b/4,(4-a-b)/4] for a in range(5) for b in range(5-a)]+[[1/3]*3])
STRENGTHS=(0.,.25,.5,.75,1.)
SEEDS=tuple(range(2045,2050))


def auc_columns(y,p):
    p=submission_probabilities(p)
    return np.array([roc_auc_score(y[:,c],p[:,c]) if len(np.unique(y[:,c]))==2 else np.nan
                     for c in range(10)])


def fit_weights(y,members):
    """Choose global AUC first; break per-class AUC ties toward that global fit."""
    if members.shape!=(3,len(y),10) or not np.isfinite(members).all():
        raise ValueError('Expected three finite, aligned validation members')
    scores=[np.nanmean(auc_columns(y,np.tensordot(w,members,axes=(0,0)))) for w in GLOBAL_GRID]
    if not np.isfinite(scores).all():raise ValueError('Fitting subset lacks binary label variation')
    global_weights=GLOBAL_GRID[int(np.argmax(scores))]
    grid_predictions=np.tensordot(CLASS_GRID,members,axes=(1,0))
    grid_scores=np.stack([auc_columns(y,p) for p in grid_predictions])
    class_weights=np.empty((3,10))
    for c in range(10):
        if not np.isfinite(grid_scores[:,c]).any():
            class_weights[:,c]=global_weights
            continue
        # Keep strict score ordering, including machine-level differences, to
        # reproduce the original finite-grid analysis exactly. Only exact
        # score ties use distance to the global weights, then grid order.
        index=max(range(len(CLASS_GRID)),key=lambda i:(grid_scores[i,c],
                  -np.sum((CLASS_GRID[i]-global_weights)**2)))
        class_weights[:,c]=CLASS_GRID[index]
    return global_weights,class_weights


def cross_fit(y,members,patients):
    patients=np.asarray(patients,dtype=str)
    if len(patients)!=len(y) or len(np.unique(patients))<5:
        raise ValueError('Five-fold fitting requires at least five validation patients')
    records=[];predictions={}
    for seed in SEEDS:
        shuffled=np.unique(patients);np.random.default_rng(seed).shuffle(shuffled)
        oof={strength:np.full_like(y,np.nan,dtype=float) for strength in STRENGTHS}
        folds=[]
        for fold,held_patients in enumerate(np.array_split(shuffled,5)):
            held=np.isin(patients,held_patients);training=~held
            assert not set(patients[held]) & set(patients[training])
            global_weights,class_weights=fit_weights(y[training],members[:,training])
            for strength in STRENGTHS:
                weights=(1-strength)*global_weights[:,None]+strength*class_weights
                oof[strength][held]=np.sum(weights[:,None,:]*members[:,held],axis=0)
            folds.append({'fold':fold,'train_studies':int(training.sum()),'held_studies':int(held.sum()),
                          'held_patients':len(held_patients),'global_weights':global_weights.tolist(),
                          'class_weights':class_weights.tolist()})
        scores={str(s):float(np.nanmean(auc_columns(y,oof[s]))) for s in STRENGTHS}
        assert all(np.isfinite(p).all() for p in oof.values())
        records.append({'seed':seed,'oof_macro_auc':scores,'folds':folds})
        predictions.update({f'oof_seed{seed}_strength{s}':p for s,p in oof.items()})
    mean_scores={str(s):float(np.mean([r['oof_macro_auc'][str(s)] for r in records])) for s in STRENGTHS}
    selected=max(STRENGTHS,key=lambda s:mean_scores[str(s)])
    global_weights,class_weights=fit_weights(y,members)
    weights=(1-selected)*global_weights[:,None]+selected*class_weights
    metrics=evaluate(y,np.sum(weights[:,None,:]*members,axis=0))
    return {'seeds':list(SEEDS),'folds':5,'patients':len(np.unique(patients)),'studies':len(y),
            'records':records,'mean_oof_macro_auc':mean_scores,'selected_shrinkage':selected,
            'global_grid':GLOBAL_GRID.tolist(),'class_grid':CLASS_GRID.tolist(),
            'weights':weights.tolist(),'refit_validation':metrics},predictions


def load_exports(checkpoints,exports,root,aggregation):
    if len(checkpoints)!=3 or len(exports)!=3:raise ValueError('Provide exactly three members and exports')
    df=frame(root,'valid');valid_hash=hashlib.sha256((Path(root)/'valid.csv').read_bytes()).hexdigest()
    hashes=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in checkpoints]
    if len(set(hashes))!=3:raise ValueError('Duplicate source checkpoints')
    reference_ids=reference_y=settings=None;members=[]
    for checkpoint_hash,export in zip(hashes,exports):
        with np.load(export) as saved:
            metadata=json.loads(str(saved['image_prediction_metadata']))
            if metadata['checkpoint_sha256']!=checkpoint_hash or metadata['validation_csv_sha256']!=valid_hash:
                raise ValueError('Export model or validation metadata differs')
            if metadata['batch_size']!=1 or metadata['tta'] or not metadata['deterministic_inference']:
                raise ValueError('Fusion fitting expects deterministic batch=1 exports without TTA')
            current=json.dumps(metadata['runtime'],sort_keys=True)
            if settings is not None and settings!=current:raise ValueError('Source runtimes differ')
            settings=current
            np.testing.assert_array_equal(saved['valid_image_ids'],df.Dicom_id.to_numpy(dtype=str))
            if saved['valid_images'].shape!=(len(df),10):raise ValueError('Export image dimensions differ')
            ids,p,y=aggregate(df,saved['valid_images'],aggregation)
        if reference_ids is not None:
            np.testing.assert_array_equal(reference_ids,ids);np.testing.assert_array_equal(reference_y,y)
        reference_ids,reference_y=ids,y;members.append(p)
    patients=df.drop_duplicates('Study_id').set_index('Study_id').loc[reference_ids].Subject_id.to_numpy(dtype=str)
    return reference_y,np.asarray(members),patients,hashes,valid_hash,json.loads(settings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',nargs=3,required=True)
    parser.add_argument('--prediction-files',nargs=3,required=True)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--aggregation',choices=['frontal','mean','max','weighted'],default='frontal')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError('Existing analysis preserved: '+str(out))
    y,members,patients,hashes,valid_hash,runtime=load_exports(args.checkpoint,args.prediction_files,args.root,args.aggregation)
    result,oof=cross_fit(y,members,patients)
    protocol={'method':'quarter-simplex class grid, shrunk toward global fusion; five repeated five-fold patient-group CV',
              'seeds':list(SEEDS),'shrinkage':result['selected_shrinkage'],
              'mean_oof_macro_auc':result['mean_oof_macro_auc'],
              'refit_macro_auc':result['refit_validation']['macro_auc'],
              'refit_macro_f1_tuned':result['refit_validation']['macro_f1_tuned'],
              'scope':'Validation patients only; checkpoints already selected on valid; refit metrics are optimistic model-selection estimates, not hidden-test scores'}
    spec={'schema_version':1,'checkpoint_sha256':hashes,'class_ids':list(range(1,11)),
          'weights':result['weights'],'selection_protocol':protocol}
    result.update(checkpoint_sha256=hashes,validation_csv_sha256=valid_hash,runtime=runtime,
                  aggregation=args.aggregation,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope=protocol['scope'])
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'weights.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    np.savez_compressed(out/'private_oof.npz',targets=y,patients=patients,**oof)
    print(json.dumps({'selected_shrinkage':result['selected_shrinkage'],
                      'mean_oof_macro_auc':result['mean_oof_macro_auc'],
                      'refit_validation':result['refit_validation'],'scope':result['scope']},indent=2))


if __name__=='__main__':main()
