"""Small fixed comparison of probability, logit and empirical-quantile averaging."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata
from src.data import frame,aggregate
from src.metrics import evaluate

def main():
    folder=Path('outputs/final_selection_v1/exports/plain')
    runs=['02_resnet18_cap_asl_s44','03_efficientnetv2_b0_s123']
    df=frame('raw_data/cxr_first','valid')
    predictions=[]
    for run in runs:
        data=np.load(folder/run/'predictions.npz',allow_pickle=True)
        assert np.array_equal(data['valid_image_ids'],df.Dicom_id.to_numpy())
        _,p,y=aggregate(df,data['valid_images'],'frontal')
        predictions.append(p)
    stack=np.stack(predictions)
    clipped=np.clip(stack,1e-7,1-1e-7)
    logits=np.log(clipped/(1-clipped)).mean(0)
    rank=np.stack([np.stack([rankdata(p[:,k],method='average')/(len(p)+1) for k in range(10)],axis=1) for p in predictions]).mean(0)
    candidates={'probability':stack.mean(0),'logit':1/(1+np.exp(-logits)),'quantile':rank}
    scores={k:evaluate(y,p) for k,p in candidates.items()}
    Path('outputs/ensemble_scales_exploration.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    print({k:v['macro_auc'] for k,v in scores.items()})

if __name__=='__main__':main()
