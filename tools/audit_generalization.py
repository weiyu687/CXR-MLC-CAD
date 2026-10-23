"""Patient-cluster uncertainty and view shift; no test labels or external data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.data import frame, LABELS

def main():
    root=Path('raw_data/cxr_first')
    report={'leaderboard_feedback':{'score':72,'rank':432,'participants':488,
                                  'file':'submission/final_bundle/submission.csv','source':'user'}}
    for split in ['train','valid','test']:
        df=frame(root,split)
        studies=df.drop_duplicates('Study_id')
        counts=studies.groupby('Subject_id').size()
        report[split]={'patients':len(counts),'studies':len(studies),
                       'studies_per_patient_mean':float(counts.mean()),
                       'studies_per_patient_max':int(counts.max()),
                       'image_views':df.ViewPosition.value_counts().to_dict(),
                       'studies_with_frontal':int(df[df.ViewPosition.isin(['AP','PA','PA LLD'])].Study_id.nunique())}
        if split!='test':
            report[split]['class_prevalence']=dict(zip(LABELS,np.stack(studies.target).mean(0).tolist()))
    data=np.load('outputs/final_selection_v1/selected_predictions/predictions.npz')
    print('Prediction fields:',data.files)
    valid=frame(root,'valid').drop_duplicates('Study_id').set_index('Study_id')
    ids=valid.index.to_numpy()
    p=data['valid']; y=np.stack(valid.loc[ids].target)
    np.testing.assert_array_equal(y,data['targets'])
    subjects=valid.loc[ids].Subject_id.to_numpy()
    unique=np.unique(subjects); positions={s:np.flatnonzero(subjects==s) for s in unique}
    rng=np.random.default_rng(2026); scores=[]
    for _ in range(400):
        idx=np.concatenate([positions[s] for s in rng.choice(unique,len(unique),replace=True)])
        auc=[roc_auc_score(y[idx,k],p[idx,k]) for k in range(10) if len(np.unique(y[idx,k]))==2]
        if len(auc)==10:scores.append(np.mean(auc))
    report['old_validation_patient_bootstrap']={'replicates':len(scores),
        'macro_auc_95_interval':np.quantile(scores,[0.025,0.975]).tolist(),
        'warning':'Patient-cluster interval describes validation uncertainty, not a test-score estimate.'}
    Path('outputs/generalization_audit_v2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
