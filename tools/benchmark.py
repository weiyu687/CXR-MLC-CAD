"""Local batch-one end-to-end latency; does not certify performance on a T4."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.data import Images
from src.model import build_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--samples',type=int,default=60)
    ap.add_argument('--out',default='outputs/latency.json')
    args=ap.parse_args();torch.set_num_threads(4)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(ck['config']['model']);model.load_state_dict(ck['state_dict'])
    model=model.to(device,memory_format=torch.channels_last).eval()
    torch.backends.cudnn.benchmark=True
    ds=Images(args.root,'__no_cache__','test',ck['config']['size'])
    indices=np.random.default_rng(123).choice(len(ds),min(len(ds),args.samples+10),replace=False)
    elapsed=[]
    with torch.inference_mode():
        for n,idx in enumerate(indices):
            if device.type=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            x,_,_=ds[int(idx)]
            x=x.unsqueeze(0).to(device,memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                p=model(x).float().sigmoid().cpu().numpy()
            if device.type=='cuda':torch.cuda.synchronize()
            if n>=10:elapsed.append(1000*(time.perf_counter()-t))
    report={'device':torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
            'torch':torch.__version__,'samples':len(elapsed),'batch_size':1,'cache':False,
            'includes':'disk read, original JPEG decode, preprocessing, transfer, forward, sigmoid, CPU output',
            'mean_ms':float(np.mean(elapsed)),'p50_ms':float(np.median(elapsed)),
            'p95_ms':float(np.percentile(elapsed,95)),'max_ms':float(max(elapsed)),
            'checkpoint_bytes':Path(args.checkpoint).stat().st_size,'T4_verified':False}
    Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
