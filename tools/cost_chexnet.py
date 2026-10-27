"""Measure frozen CheXNet full-study read/infer/postprocess/write latency.

Use the official pictures directories for the 5/100/3 protocol. Model loading
is excluded. A supplied lateral input still reproduces its study's frozen
frontal-first prediction, including any required companion-image I/O.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.chexnet import CheXNetTransform,load_checkpoint,crop_probabilities,CACHE_SIGNATURE
from src.data import frame,aggregate
from src.inference import configure_deterministic_inference,runtime_information
from src.metrics import submission_probabilities


class StudyPredictor:
    def __init__(self,checkpoint,root,calibration,device):
        self.root=Path(root)
        self.device=device
        self.report=json.loads(Path(calibration).read_text(encoding='utf-8'))
        digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if self.report['checkpoint_sha256']!=digest or self.report['preprocessing']!=CACHE_SIGNATURE:
            raise ValueError('Frozen calibration identity differs')
        if self.report['arguments']['batch_size']!=1:raise ValueError('Freeze batch=1 for per-input reproduction')
        self.model,_=load_checkpoint(checkpoint,device)
        self.crop=self.report['arguments']['crop']
        self.transform=CheXNetTransform(self.crop)
        self.df=frame(root,'test')
        self.positions={identifier:index for index,identifier in enumerate(self.df.Dicom_id)}
        self.groups={identifier:group.index.tolist() for identifier,group in self.df.groupby('Study_id',sort=False)}

    def path(self,index):
        row=self.df.iloc[index]
        return self.root/'test'/row.Subject_id/row.Study_id/(row.Dicom_id+'.jpg')

    @torch.inference_mode()
    def __call__(self,path):
        path=Path(path)
        index=self.positions[path.stem]
        supplied=path.read_bytes()
        original=self.path(index)
        if path.resolve()!=original.resolve() and hashlib.sha256(supplied).digest()!=hashlib.sha256(original.read_bytes()).digest():
            raise ValueError('Timing input differs from the original image')
        row=self.df.iloc[index]
        subset=self.df.iloc[self.groups[row.Study_id]].copy()
        frontal=subset.ViewPosition.isin(['AP','PA','PA LLD'])
        if frontal.any():subset=subset[frontal]
        probabilities=[]
        for current in subset.index:
            data=supplied if current==index else self.path(current).read_bytes()
            with Image.open(io.BytesIO(data)) as image:pictures=self.transform(image).unsqueeze(0)
            if self.crop=='ten':probability,_=crop_probabilities(self.model,pictures,self.device)
            else:
                pictures=pictures.to(self.device)
                if self.device.type=='cuda':pictures=pictures.to(memory_format=torch.channels_last)
                with torch.autocast(self.device.type,enabled=self.device.type=='cuda'):
                    probability=self.model(pictures).float().sigmoid()
            probabilities.append(probability.cpu().numpy()[0])
        _,pooled,_=aggregate(subset.reset_index(drop=True),np.stack(probabilities),'frontal')
        return row.Subject_id,row.Study_id,submission_probabilities(pooled)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--submission',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--pictures')
    parser.add_argument('--warmup-pictures')
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--out',required=True)
    parser.add_argument('--cpu',action='store_true',help='Synthetic script checks; does not certify GPU latency')
    args=parser.parse_args()
    if min(args.samples,args.warmup,args.rounds)<1:raise ValueError('Invalid timing sample counts')
    out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve prior timing evidence')
    configure_deterministic_inference()
    torch.set_num_threads(4)
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if device.type=='cuda':
        from tools.gpu_idle import wait_for_idle_gpu
        out.parent.mkdir(parents=True,exist_ok=True)
        wait_for_idle_gpu(out.with_suffix('.gpu_load.jsonl'),max_wait_seconds=240)
    engine=StudyPredictor(args.checkpoint,args.root,args.calibration,device)
    runtime=runtime_information(device)
    if runtime!=engine.report['runtime']:raise ValueError('Timing and frozen inference runtime differ')
    original=[engine.path(index) for index in range(len(engine.df))]
    pictures=sorted(Path(args.pictures).glob('*.jpg')) if args.pictures else original[:args.samples]
    warmup=sorted(Path(args.warmup_pictures).glob('*.jpg')) if args.warmup_pictures else original[args.samples:args.samples+args.warmup]
    if len(pictures)!=args.samples or len(warmup)!=args.warmup:raise ValueError('Timing input counts differ')
    expected=pd.read_csv(args.submission,float_precision='round_trip')
    lookup={identifier:group.sort_values('Predict_class') for identifier,group in expected.groupby('Study_id')}
    out.mkdir(parents=True)
    for path in warmup:engine(path)
    if device.type=='cuda':torch.cuda.synchronize()
    times=[]
    maximum_difference=0.
    with (out/'timed_predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['Subject_id','Study_id','Predict_class','Probability','Gate'])
        for repeat in range(args.rounds):
            round_times=[]
            for path in pictures:
                if device.type=='cuda':torch.cuda.synchronize()
                started=time.perf_counter()
                subject,study,probabilities=engine(path)
                for label,probability in enumerate(probabilities,1):
                    writer.writerow([subject,study,label,format(probability,'.15f'),format(engine.report['validation']['gates'][label-1],'.15f')])
                stream.flush()
                os.fsync(stream.fileno())
                if device.type=='cuda':torch.cuda.synchronize()
                round_times.append((time.perf_counter()-started)*1000)
                reference=lookup[study]
                if reference.Predict_class.tolist()!=list(range(1,11)):raise ValueError('Submitted classes differ')
                maximum_difference=max(maximum_difference,float(np.max(np.abs(probabilities-reference.Probability.to_numpy()))))
            times.append(round_times)
    mean=float(np.asarray(times).mean(axis=0).mean())
    protocol=bool(args.pictures and args.warmup_pictures and args.samples==100 and args.warmup==5 and args.rounds==3)
    report={'samples':args.samples,'warmup':args.warmup,'rounds':args.rounds,'official_sample_protocol':protocol,
        'mean_ms':mean,'all_rounds_ms':times,'max_probability_difference':maximum_difference,
        'submission_predictions_match':maximum_difference<=1e-9,'local_mean_limit_passed':mean<=100,
        'runtime':runtime,'T4_verified':False,'checkpoint_sha256':engine.report['checkpoint_sha256'],
        'scope':'Full-study original-image I/O and flushed/fsynced result writing; model initialization excluded'}
    (out/'cost_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if maximum_difference>1e-9:raise ValueError('Frozen predictions were not reproduced')


if __name__=='__main__':main()
