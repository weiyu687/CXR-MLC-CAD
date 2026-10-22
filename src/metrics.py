import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

def evaluate(y, p):
    auc = [float(roc_auc_score(y[:,i],p[:,i])) if len(np.unique(y[:,i]))==2 else None for i in range(10)]
    gates = []
    for i in range(10):
        precision, recall, thresholds = precision_recall_curve(y[:,i],p[:,i])
        f = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-9)
        gates.append(float(thresholds[np.argmax(f)]))
    return {'macro_auc': float(np.mean([a for a in auc if a is not None])), 'auc': auc,
            'macro_f1_05':float(f1_score(y,p>=0.5,average='macro',zero_division=0)),
            'macro_f1_tuned':float(f1_score(y,p>=np.array(gates),average='macro',zero_division=0)),
            'gates':gates}
