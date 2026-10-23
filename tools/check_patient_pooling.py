"""Exploratory within-patient inference, without target labels in pooling."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from src.data import frame, LABELS
from src.metrics import evaluate

def patient_pool(studies, probabilities, alpha):
    # IDs align to the existing first-appearance study order, never sorted by patient.
    patients=studies.Subject_id.to_numpy()
    result=probabilities.copy()
    for subject in np.unique(patients):
        idx=np.flatnonzero(patients==subject)
        if len(idx)>1:
            result[idx]=(1-alpha)*probabilities[idx]+alpha*probabilities[idx].mean(0)
    return result

def main():
    data=np.load('outputs/final_selection_v1/selected_predictions/predictions.npz')
    studies=frame('raw_data/cxr_first','valid').drop_duplicates('Study_id')
    y=np.stack(studies.target); p=data['valid']
    np.testing.assert_array_equal(y,data['targets'])
    report={'scope':'Exploration on official validation only; no hidden-test score claim','candidates':[]}
    for alpha in [0.,0.25,0.5,0.75]:
        q=patient_pool(studies,p,alpha)
        score=evaluate(y,q)
        report['candidates'].append({'alpha':alpha,'metrics':score})
    patients=studies.Subject_id.to_numpy(); unique=np.unique(patients)
    positions={s:np.flatnonzero(patients==s) for s in unique}
    rng=np.random.default_rng(2027)
    differences={str(alpha):[] for alpha in [0.25,0.5,0.75]}
    variants={alpha:patient_pool(studies,p,alpha) for alpha in [0.25,0.5,0.75]}
    for _ in range(300):
        idx=np.concatenate([positions[s] for s in rng.choice(unique,len(unique),replace=True)])
        if any(len(np.unique(y[idx,k]))<2 for k in range(10)):continue
        base=roc_auc_score(y[idx],p[idx],average='macro')
        for alpha,q in variants.items():
            differences[str(alpha)].append(roc_auc_score(y[idx],q[idx],average='macro')-base)
    report['paired_patient_bootstrap_delta_95']={a:np.quantile(v,[0.025,0.975]).tolist() for a,v in differences.items()}
    Path('outputs/patient_pooling_exploration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for item in report['candidates']:
        print(item['alpha'],item['metrics']['macro_auc'],item['metrics']['auc'])
    print(report['paired_patient_bootstrap_delta_95'])

if __name__=='__main__':main()
