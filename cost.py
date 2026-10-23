"""Official 5-warmup / 100-input / 3-round protocol, conservative study inference.

For a multi-image study the complete fixed study inference is repeated for each
timed input. This includes MORE work than a single-image forward, and reproduces
the study-level CSV rather than comparing an image score to a study score.
Provide --root (or CXR_DATA_ROOT) when companion study images are outside the
timing folders. No labels, model updates or cross-study statistics are used.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from src.data import Images,aggregate
from src.inference import configure_deterministic_inference,load_deployed_model,forward_probabilities,runtime_information
from src.metrics import submission_probabilities
from src.deployment import StudyPredictor

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('submission')
    ap.add_argument('pictures_100')
    ap.add_argument('pictures_5')
    ap.add_argument('--root',default=os.environ.get('CXR_DATA_ROOT'))
    ap.add_argument('--bundle',default=str(Path(__file__).resolve().parent))
    ap.add_argument('--out',default='cost_results')
    ap.add_argument('--samples',type=int,default=100,help='Change only for script checks; official protocol uses 100')
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--tolerance',type=float,default=1e-6)
    args=ap.parse_args();torch.set_num_threads(4);configure_deterministic_inference()
    bundle=Path(args.bundle)
    report=json.loads((bundle/'report.json').read_text(encoding='utf-8'))
    info=json.loads((bundle/'checkpoint_info.json').read_text(encoding='utf-8'))
    assert len(info)==len(report['checkpoint_sha256'])>0
    root=Path(args.root or report['arguments']['root'])
    if not (root/'test.csv').exists():raise FileNotFoundError('Provide --root or CXR_DATA_ROOT with test.csv and complete original test images')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_paths=[]
    for item,digest in zip(info,report['checkpoint_sha256']):
        path=bundle/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        checkpoint_paths.append(path)
    assert len(checkpoint_paths)==len(report['checkpoint_sha256'])>0
    predictor=StudyPredictor(checkpoint_paths,root,device,report['arguments']['aggregation'],report['arguments']['tta'])
    from predict import validate_submission
    validate_submission(args.submission,root)
    meta=predictor.df
    positions={name:i for i,name in enumerate(meta.Dicom_id)}
    study_rows={s:g.index.to_numpy() for s,g in meta.groupby('Study_id',sort=False)}
    expected=pd.read_csv(args.submission).set_index(['Study_id','Predict_class'])
    def image_list(directory,count):
        paths=sorted(p for p in Path(directory).rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png'])
        if len(paths)!=count:raise ValueError(f'{directory}: expected {count} images, got {len(paths)}')
        if len({p.stem for p in paths})!=count:raise ValueError('Duplicate image ID in timing folder')
        for path in paths:
            if path.stem not in positions:raise ValueError('Timing image not in official test metadata')
        return paths
    measured=image_list(args.pictures_100,args.samples)
    warmup=image_list(args.pictures_5,args.warmup)
    if {p.stem for p in measured}&{p.stem for p in warmup}:raise ValueError('Warmup and measured images overlap')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    gates=np.array(report['validation']['gates'])
    actual_gates=expected.groupby(level='Predict_class').Gate.first().sort_index().to_numpy()
    assert np.allclose(actual_gates,gates,rtol=0,atol=1e-15),'Submission gates differ from frozen report'
    elapsed=np.zeros((args.rounds,args.samples));max_difference=0.
    infer=predictor
    with torch.inference_mode():
        for path in warmup:infer(path)
        for round_index in range(args.rounds):
            with (out/f'round_{round_index+1}.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['Dicom_id','Study_id','Predict_class','Probability','Gate'])
                for i,path in enumerate(measured):
                    if device.type=='cuda':torch.cuda.synchronize()
                    started=time.perf_counter()
                    study,p=infer(path)
                    for k in range(10):writer.writerow([path.stem,study,k+1,f'{p[k]:.15f}',f'{gates[k]:.15f}'])
                    stream.flush()
                    if device.type=='cuda':torch.cuda.synchronize()
                    elapsed[round_index,i]=1000*(time.perf_counter()-started)
                    actual=expected.loc[[(study,k+1) for k in range(10)],'Probability'].to_numpy()
                    difference=float(np.max(np.abs(actual-p)))
                    max_difference=max(max_difference,difference)
                    if difference>args.tolerance:raise AssertionError(f'Submission reproduction mismatch for {study}: {difference}')
    mean_by_image=elapsed.mean(0)
    result={'samples':args.samples,'warmup':args.warmup,'rounds':args.rounds,
        'official_sample_protocol':args.samples==100 and args.warmup==5 and args.rounds==3,
        'mean_ms':float(mean_by_image.mean()),'max_image_mean_ms':float(mean_by_image.max()),
        'all_rounds_ms':elapsed.tolist(),'max_probability_difference':max_difference,
        'probability_tolerance':args.tolerance,'submission_predictions_match':True,
        'local_mean_limit_passed':bool(mean_by_image.mean()<=100),'runtime':runtime_information(device),
        'checkpoint_sha256':report['checkpoint_sha256'],
        'submission_sha256':hashlib.sha256(Path(args.submission).read_bytes()).hexdigest(),
        'metadata_sha256':hashlib.sha256((root/'test.csv').read_bytes()).hexdigest(),
        'scope':'Conservative full-study inference repeated per input, including companion images, disk I/O and flushed CSV writes; initialization excluded',
        'T4_verified':bool(device.type=='cuda' and 'T4' in torch.cuda.get_device_name())}
    (out/'cost_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='all_rounds_ms'},indent=2))
    print(f'[CONCLUSION] {args.samples} inputs, batch=1, {args.rounds} rounds; mean {result["mean_ms"]:.2f} ms; max probability difference {max_difference:.8g}; precision {result["runtime"]["inference_dtype"]}')

if __name__=='__main__':main()
