"""Download only the explicitly audited official, non-medical ImageNet weights."""
import json
from pathlib import Path
import urllib.request
import torch
from src.model import build_model

def main():
    torch.set_num_threads(4)
    materials=Path('docs/pretraining');materials.mkdir(parents=True,exist_ok=True)
    license_url='https://raw.githubusercontent.com/pytorch/vision/v0.23.0/LICENSE'
    with urllib.request.urlopen(license_url,timeout=40) as response:
        license_text=response.read().decode('utf-8')
    assert 'Redistribution and use' in license_text
    (materials/'torchvision-LICENSE.txt').write_text(license_text,encoding='utf-8')
    records=[]
    for name in ['tv_resnet50','tv_efficientnet_v2_s']:
        model=build_model(name,pretrained=True).cuda().eval()
        with torch.inference_mode(),torch.autocast('cuda'):
            output=model(torch.randn(2,3,320,320,device='cuda').to(memory_format=torch.channels_last))
        assert output.shape==(2,10) and output.isfinite().all()
        metadata=model.pretraining_provenance
        records.append(metadata)
        print(json.dumps(metadata),flush=True)
        del model
    Path('outputs/imagenet_pretraining_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    public=[{k:v for k,v in record.items() if k!='cached_file'} for record in records]
    (materials/'imagenet_sources.json').write_text(json.dumps(public,indent=2),encoding='utf-8')

if __name__=='__main__':main()
