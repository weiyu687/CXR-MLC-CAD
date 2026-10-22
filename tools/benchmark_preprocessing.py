"""Measure canonical raw-image preprocessing, including private HDD/SSD reads."""
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.preprocessing import decode_original_like_cache

def summarize(values):
    ordered=sorted(values)
    return {'samples':len(values),'mean_ms':statistics.mean(values),
            'p50_ms':statistics.median(values),'p95_ms':ordered[min(len(ordered)-1,int(.95*len(ordered)))],
            'max_ms':max(values)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--samples',type=int,default=40)
    ap.add_argument('--ssd-copy-dir',default='C:/Users/Administrator/.cache/cxr-mlc-cad/latency_originals')
    ap.add_argument('--out',default='outputs/preprocessing_latency.json')
    args=ap.parse_args()
    with (Path(args.root)/'test.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    selected=random.Random(123).sample(rows,min(args.samples,len(rows)))
    destination=Path(args.ssd_copy_dir);destination.mkdir(parents=True,exist_ok=True)
    original_times=[];ssd_times=[]
    def measure(path):
        start=time.perf_counter()
        with Image.open(path) as image:result=decode_original_like_cache(image)
        result.close()
        return (time.perf_counter()-start)*1000
    for row in selected:
        path=Path(args.root)/'test'/row['Subject_id']/row['Study_id']/(row['Dicom_id']+'.jpg')
        original_times.append(measure(path))
        copy=destination/path.name
        shutil.copyfile(path,copy)
        ssd_times.append(measure(copy))
    result={'original_storage':summarize(original_times),'ssd_copy':summarize(ssd_times),
            'includes':'file open/read, JPEG decode, grayscale, bicubic padding to 448, quality-95 encode/decode',
            'excludes':'tensor normalization, GPU transfer and model inference',
            'conditions':'Local machine during ongoing training; SSD file reads immediately follow copying and may be OS-cached',
            'T4_verified':False}
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
