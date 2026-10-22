import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

PROBABILITY_DECIMALS=12
CSV_DECIMAL_PLACES=15

def submission_probabilities(p):
    return np.clip(np.round(np.asarray(p,dtype=np.float64),PROBABILITY_DECIMALS),1e-12,1-1e-12)

def evaluate(y, p):
    p=submission_probabilities(p)
    auc = [float(roc_auc_score(y[:,i],p[:,i])) if len(np.unique(y[:,i]))==2 else None for i in range(10)]
    gates = []
    for i in range(10):
        precision, recall, thresholds = precision_recall_curve(y[:,i],p[:,i])
        f = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-9)
        index=int(np.argmax(f))
        lower=thresholds[index-1] if index>0 else 0.
        scale=10**PROBABILITY_DECIMALS
        lo=int(round(float(lower)*scale));hi=int(round(float(thresholds[index])*scale))
        # Half-grid gates cannot equal any emitted 12-decimal probability.
        middle=(lo+hi)//2
        gates.append((2*middle+1)/(2*scale))
    return {'macro_auc': float(np.mean([a for a in auc if a is not None])), 'auc': auc,
            'macro_f1_05':float(f1_score(y,p>=0.5,average='macro',zero_division=0)),
            'macro_f1_tuned':float(f1_score(y,p>=np.array(gates),average='macro',zero_division=0)),
            'gates':gates}
