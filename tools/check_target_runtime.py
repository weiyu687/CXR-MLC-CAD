"""Check official Python/PyTorch API versions on CPU or local CUDA, without claiming T4 validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torchvision
import timm
import PIL
from src.data import Images
from src.model import build_model
from src.inference import configure_deterministic_inference,runtime_information

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',nargs='+',required=True)
    ap.add_argument('--architecture',nargs='*',default=[])
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--cache',default='data_cache/448')
    ap.add_argument('--samples',type=int,default=2)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    assert sys.version_info[:2]==(3,12),'Run with the isolated Python 3.12 interpreter'
    assert str(torch.__version__).split('+')[0]=='2.10.0'
    assert str(torchvision.__version__).split('+')[0]=='0.25.0'
    torch.set_num_threads(2);configure_deterministic_inference()
    device=torch.device(args.device)
    if device.type=='cuda':assert torch.cuda.is_available(),'CUDA runtime is unavailable'
    cases=[]
    def forward(model,size):
        ds=Images(args.root,args.cache,'test',size)
        x=torch.stack([ds[i][0] for i in range(min(args.samples,len(ds)))]).to(device,memory_format=torch.channels_last)
        model=model.to(device,memory_format=torch.channels_last).eval()
        with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            p=model(x).float().sigmoid()
        assert p.shape==(len(x),10) and torch.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        return len(x)
    for path in args.checkpoint:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model=build_model(ck['config']['model']);model.load_state_dict(ck['state_dict'],strict=True)
        count=forward(model,ck['config']['size'])
        cases.append({'checkpoint':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      'architecture':ck['config']['model'],'weight_loading_verified':True,'images':count,'finite_probabilities':True})
        del ck,model
    for name in args.architecture:
        model=build_model(name)
        count=forward(model,384)
        cases.append({'checkpoint':None,'architecture':name,'weight_loading_verified':False,
                      'random_initialization_only':True,'images':count,'finite_probabilities':True})
        del model
    report={**runtime_information(device),'cases':cases,'passed':True,
            'scope':'Weight loading and finite forward on the selected device only; not accuracy, latency or cross-runtime numerical equivalence',
            'T4_verified':False}
    Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
