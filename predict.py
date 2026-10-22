"""Reproduce a study-level competition CSV; never submits to the website."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data import Images,aggregate
from src.model import build_model
from src.metrics import evaluate
from train import predict
from src.inference import configure_deterministic_inference

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
    ap.add_argument('--calibration',help='Frozen report.json; enables inference with only test.csv/test images')
    ap.add_argument('--export-images',action='store_true',help='Export single-model deterministic per-image probabilities for model selection')
    args=ap.parse_args(); torch.set_num_threads(4)
    configure_deterministic_inference()
    frozen=json.loads(Path(args.calibration).read_text(encoding='utf-8')) if args.calibration else None
    if args.export_images and (frozen or len(args.checkpoint)!=1):
        raise ValueError('--export-images requires one checkpoint and an unfrozen validation run')
    checkpoint_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.checkpoint]
    if frozen:
        if frozen['checkpoint_sha256']!=checkpoint_hashes:
            raise ValueError('Calibration does not match checkpoint contents/order')
        args.tta=frozen['arguments']['tta']
        args.aggregation=frozen['arguments']['aggregation']
        args.batch_size=frozen['arguments']['batch_size']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    all_probs={'test':[]} if frozen else {'valid':[],'test':[]}
    image_saved={}
    for checkpoint in args.checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=build_model(ck['config']['model']); model.load_state_dict(ck['state_dict'])
        model=model.to(device,memory_format=torch.channels_last)
        for split in all_probs:
            ds=Images(args.root,args.cache,split,ck['config']['size'])
            dl=DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda')
            p=predict(model,dl,device,args.tta)
            if args.export_images:
                image_saved[split+'_images']=p
                image_saved[split+'_image_ids']=ds.df.Dicom_id.to_numpy(dtype=str)
            ids,probs,y=aggregate(ds.df,p,args.aggregation)
            all_probs[split].append(probs)
            if split=='valid': valid_y=y
            else: test_ids=ids
        del model
    metrics=frozen['validation'] if frozen else evaluate(valid_y,np.mean(all_probs['valid'],axis=0))
    probs=np.mean(all_probs['test'],axis=0)
    s=pd.DataFrame({'Study_id':np.repeat(test_ids,10),'Predict_class':np.tile(np.arange(1,11),len(test_ids)),
                    'Probability':probs.reshape(-1),'Gate':np.tile(metrics['gates'],len(test_ids))})
    meta=pd.read_csv(Path(args.root)/'test.csv').drop_duplicates('Study_id').set_index('Study_id')
    s.insert(0,'Subject_id',meta.loc[s.Study_id].Subject_id.values)
    s.to_csv(out/'submission.csv',index=False,encoding='utf-8',float_format='%.8f')
    report={'validation':metrics,'submission':validate_submission(out/'submission.csv',args.root),
            'checkpoint_sha256':checkpoint_hashes,'arguments':vars(args),
            'deterministic_inference':True,
            'validation_recomputed':not bool(frozen)}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    saved={'test':probs,'study_ids':test_ids}
    if not frozen:saved.update(valid=np.mean(all_probs['valid'],axis=0),targets=valid_y)
    if args.export_images:
        metadata={'checkpoint_sha256':checkpoint_hashes[0],'batch_size':args.batch_size,'tta':args.tta,
                  'deterministic_inference':True,
                  'validation_csv_sha256':hashlib.sha256((Path(args.root)/'valid.csv').read_bytes()).hexdigest(),
                  'test_csv_sha256':hashlib.sha256((Path(args.root)/'test.csv').read_bytes()).hexdigest()}
        saved.update(image_saved,image_prediction_metadata=np.array(json.dumps(metadata)))
    np.savez_compressed(out/'predictions.npz',**saved)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
