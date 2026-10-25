"""Local batch-one end-to-end latency; does not certify performance on a T4."""
import argparse
import json
import hashlib
import sys
import time
import csv
from contextlib import nullcontext
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.data import Images
from src.model import build_model
from src.inference import configure_deterministic_inference,runtime_information
from src.inference import load_deployed_model,forward_probabilities
from src.metrics import submission_probabilities
from src.deployment import StudyPredictor

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',nargs='+',required=True)
    ap.add_argument('--tta',action='store_true')
    ap.add_argument('--calibration',help='Frozen prediction report; verifies model hashes and TTA')
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--samples',type=int,default=60)
    ap.add_argument('--out',default='outputs/latency.json')
    ap.add_argument('--split',choices=['train','valid','test'],default='test')
    ap.add_argument('--study-level',action='store_true',help='Conservative full-study inference, 5 warmup, 3 rounds, flushed result writes')
    ap.add_argument('--aggregation',choices=['mean','max','frontal','weighted'],default='frontal')
    args=ap.parse_args();torch.set_num_threads(4)
    configure_deterministic_inference()
    checkpoint_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.checkpoint]
    gates=np.full(10,.5)
    if args.calibration:
        frozen=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
        actual=checkpoint_hashes
        if actual!=frozen['checkpoint_sha256']:raise ValueError('Calibration/model mismatch')
        args.tta=frozen['arguments']['tta']
        args.aggregation=frozen['arguments']['aggregation']
        gates=np.array(frozen['validation']['gates'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor=None;members=[]
    if args.study_level:
        predictor=StudyPredictor(args.checkpoint,args.root,device,args.aggregation,args.tta,args.split)
        members=predictor.members
    else:
        for checkpoint in args.checkpoint:
            model,ck=load_deployed_model(checkpoint,device)
            ds=Images(args.root,'__no_cache__',args.split,ck['config']['size'],normalization=ck['config'].get('normalization','legacy'))
            members.append((model,ds))
    ds=members[0][1]
    warmup=5 if args.study_level else 10
    rounds=3 if args.study_level else 1
    indices=np.random.default_rng(123).choice(len(ds),min(len(ds),args.samples+warmup),replace=False)
    elapsed=np.zeros((rounds,len(indices)-warmup))
    with torch.inference_mode():
        if predictor:
            for idx in indices[:warmup]:predictor(predictor.original_path(int(idx)))
        for round_index in range(rounds):
            output=Path(args.out).with_name(Path(args.out).stem+f'_round_{round_index+1}.csv')
            with (output.open('w',encoding='utf-8',newline='') if predictor else nullcontext(None)) as stream:
                writer=csv.writer(stream) if stream else None
                if writer:writer.writerow(['Study_id','Predict_class','Probability','Gate'])
                for n,idx in enumerate(indices[warmup:] if predictor else indices):
                    if device.type=='cuda':torch.cuda.synchronize()
                    t=time.perf_counter()
                    if predictor:
                        study,p=predictor(predictor.original_path(int(idx)))
                    else:
                        member_predictions=[]
                        for model,member_ds in members:
                            x,_,_=member_ds[int(idx)]
                            member_predictions.append(forward_probabilities(model,x.unsqueeze(0),device,args.tta).cpu().numpy())
                        p=submission_probabilities(np.mean(member_predictions,axis=0))
                    decision=p>=gates
                    if writer:
                        for k in range(10):writer.writerow([study,k+1,f'{p[k]:.15f}',f'{gates[k]:.15f}'])
                        stream.flush()
                    if device.type=='cuda':torch.cuda.synchronize()
                    if predictor:elapsed[round_index,n]=1000*(time.perf_counter()-t)
                    elif n>=warmup:elapsed[round_index,n-warmup]=1000*(time.perf_counter()-t)
    averaged=elapsed.mean(0)
    report={'device':torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
            'torch':torch.__version__,'samples':len(averaged),'warmup':warmup,'rounds':rounds,'batch_size':1,'cache':False,'models':len(members),'tta':args.tta,
            'deterministic_inference':True,
            'checkpoint_sha256':checkpoint_hashes,'runtime':runtime_information(device),
            'data_root':str(Path(args.root).resolve()),
            'split':args.split,'dataset_csv_sha256':hashlib.sha256((Path(args.root)/(args.split+'.csv')).read_bytes()).hexdigest(),
            'test_csv_sha256':hashlib.sha256((Path(args.root)/'test.csv').read_bytes()).hexdigest() if args.split=='test' else None,
            'includes':'disk read, original JPEG decode, preprocessing, transfer, forward, sigmoid, CPU output, ensemble mean, probability quantization and threshold comparison',
            'mean_ms':float(np.mean(averaged)),'p50_ms':float(np.median(averaged)),
            'p95_ms':float(np.percentile(averaged,95)),'max_ms':float(max(averaged)),
            'checkpoint_bytes':sum(Path(p).stat().st_size for p in args.checkpoint),'T4_verified':False,
            'preprocessing':'Each member reads and preprocesses the original independently, matching sequential ensemble inference',
            'excludes':'Model startup, study-level aggregation and dataset CSV export; first 10 image predictions are warmup'}
    if predictor:report.update(aggregation=args.aggregation,conservative_full_study=True,
        preprocessing='Original reads and canonical decode shared only within the current study call; matching size/normalization tensors reused between fixed members; no cross-call cache',
        includes='Specified file read, complete fixed study image reads, preprocessing, transfer, forward, sigmoid, CPU output, study/model aggregation, quantization, thresholds and flushed CSV writes',
        excludes='Model loading and 5 separate warmup images; 100 inputs measured serially over 3 rounds when --samples 100')
    Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
