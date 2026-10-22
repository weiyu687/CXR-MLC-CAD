"""Reproduce a study-level competition CSV; never submits to the website."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data import Images,aggregate
from src.model import build_model
from src.metrics import evaluate
from train import predict

def validate_submission(path, root):
    s=pd.read_csv(path)
    expected=set(pd.read_csv(Path(root)/'test.csv').Study_id)
    assert list(s.columns)==['Subject_id','Study_id','Predict_class','Probability','Gate']
    meta=pd.read_csv(Path(root)/'test.csv').drop_duplicates('Study_id').set_index('Study_id')
    assert (s.Subject_id.values==meta.loc[s.Study_id].Subject_id.values).all()
    assert set(s.Study_id)==expected and len(s)==10*len(expected)
    assert not s.duplicated(['Study_id','Predict_class']).any()
    assert s.groupby('Study_id').Predict_class.apply(lambda x:set(x)==set(range(1,11))).all()
    assert np.isfinite(s[['Probability','Gate']].values).all()
    assert s.Probability.between(0,1).all() and s.Gate.between(0,1).all()
    return {'rows':len(s),'studies':len(expected),'classes':10,'valid':True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',nargs='+',required=True)
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--cache',default='data_cache/448')
    ap.add_argument('--out',default='submission')
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--tta',action='store_true')
    ap.add_argument('--aggregation',choices=['mean','max','frontal','weighted'],default='mean')
    args=ap.parse_args(); torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=True
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    all_probs={'valid':[],'test':[]}
    for checkpoint in args.checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=build_model(ck['config']['model']); model.load_state_dict(ck['state_dict'])
        model=model.to(device,memory_format=torch.channels_last)
        for split in all_probs:
            ds=Images(args.root,args.cache,split,ck['config']['size'])
            dl=DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda')
            p=predict(model,dl,device,args.tta)
            ids,probs,y=aggregate(ds.df,p,args.aggregation)
            all_probs[split].append(probs)
            if split=='valid': valid_y=y
            else: test_ids=ids
        del model
    metrics=evaluate(valid_y,np.mean(all_probs['valid'],axis=0))
    probs=np.mean(all_probs['test'],axis=0)
    s=pd.DataFrame({'Study_id':np.repeat(test_ids,10),'Predict_class':np.tile(np.arange(1,11),len(test_ids)),
                    'Probability':probs.reshape(-1),'Gate':np.tile(metrics['gates'],len(test_ids))})
    meta=pd.read_csv(Path(args.root)/'test.csv').drop_duplicates('Study_id').set_index('Study_id')
    s.insert(0,'Subject_id',meta.loc[s.Study_id].Subject_id.values)
    s.to_csv(out/'submission.csv',index=False,encoding='utf-8',float_format='%.8f')
    report={'validation':metrics,'submission':validate_submission(out/'submission.csv',args.root),'arguments':vars(args)}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(out/'predictions.npz',valid=np.mean(all_probs['valid'],axis=0),targets=valid_y,test=probs,study_ids=test_ids)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
