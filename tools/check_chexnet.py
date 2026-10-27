"""Numerical checks against the user-specified CheXNet inference repository."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T

from src.chexnet import CheXNetTransform,CheXNetClassifier,CheXNetImages,crop_probabilities
from tools.prepare_chexnet_cache import prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve previous CheXNet check')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled=False
    torch.manual_seed(2050)
    rng=np.random.default_rng(2050)
    image=Image.fromarray(rng.integers(0,256,(317,481,3),dtype=np.uint8))
    normalize=T.Normalize([.485,.456,.406],[.229,.224,.225])
    expected=torch.stack([normalize(T.ToTensor()(crop)) for crop in T.TenCrop(224)(T.Resize(256)(image))])
    actual=CheXNetTransform('ten')(image)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(CheXNetTransform('center')(image),actual[4],atol=0,rtol=0)
    reference_folder=Path('outputs/chexnet_reference_20261027').resolve()
    sys.path.insert(0,str(reference_folder))
    spec=importlib.util.spec_from_file_location('audited_chexnet_model',reference_folder/'model.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constructor=torchvision.models.densenet121
    with patch.object(torchvision.models,'densenet121',lambda pretrained=True:constructor(weights=None)):
        reference=module.DenseNet121(10).eval()
    adapted=CheXNetClassifier().eval()
    mapping={key.replace('.classifier.0.','.classifier.'):value for key,value in reference.state_dict().items()}
    adapted.load_state_dict(mapping,strict=True)
    inputs=torch.randn(2,3,64,64)
    with torch.no_grad():
        expected_probabilities=reference(inputs)
        actual_probabilities=adapted(inputs).sigmoid()
        torch.testing.assert_close(actual_probabilities,expected_probabilities,atol=1e-7,rtol=1e-6)
    logits=torch.randn(5,10,requires_grad=True)
    other=logits.detach().clone().requires_grad_(True)
    targets=torch.randint(0,2,logits.shape).float()
    stable=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none').sum(1).mean()
    explicit=torch.nn.functional.binary_cross_entropy(other.sigmoid(),targets,reduction='none').sum(1).mean()
    stable.backward()
    explicit.backward()
    torch.testing.assert_close(stable,explicit,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(logits.grad,other.grad,atol=1e-7,rtol=1e-6)
    class Probe(torch.nn.Module):
        def forward(self,x):return x[:,0,0,0,None]*4+torch.arange(10)[None]*.1
    pictures=actual.unsqueeze(0)
    ten,center=crop_probabilities(Probe(),pictures,torch.device('cpu'))
    per_crop=Probe()(pictures.flatten(0,1)).sigmoid().reshape(1,10,10)
    torch.testing.assert_close(ten,per_crop.mean(1))
    torch.testing.assert_close(center,per_crop[:,4])
    assert not torch.allclose(ten,Probe()(pictures.flatten(0,1)).reshape(1,10,10).mean(1).sigmoid())
    import pandas as pd
    raw=out/'raw'
    raw.mkdir()
    for split,patients in [('train',4),('valid',2)]:
        records=[]
        for patient in range(patients):
            for study in range(2):
                for view in range(2):
                    subject=f'{split}_p{patient}'
                    study_id=f'{split}_s{patient}_{study}'
                    identifier=f'{study_id}_{view}'
                    folder=raw/split/subject/study_id
                    folder.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(rng.integers(0,256,(283,347,3),dtype=np.uint8)).save(folder/(identifier+'.jpg'))
                    records.append({'Dicom_id':identifier,'Subject_id':subject,'Study_id':study_id,
                        'ViewPosition':'PA' if view==0 else 'LATERAL',
                        'Predict_class':';'.join(map(str,range(1,11))) if study else ''})
        pd.DataFrame(records).to_csv(raw/(split+'.csv'),index=False)
    cache=out/'cache'
    cache.mkdir()
    for split in ['train','valid']:prepare(raw,cache,split,2)
    packed=CheXNetImages(raw,cache,'valid')
    original=CheXNetImages(raw,None,'valid')
    for index in range(len(packed)):
        torch.testing.assert_close(packed[index][0],original[index][0],atol=0,rtol=0)
    packed.packed._mmap.close()
    assert len(CheXNetImages(raw,cache,'train',mode='train',frontal_only=True))==8
    report={'torch':str(torch.__version__),'repo_tencrop_tensors_exact':True,
        'ten_label_architecture_probability_parity':True,'stable_bce_loss_and_gradient_parity':True,
        'probabilities_averaged_after_sigmoid':True,'center_crop_member_identified':True,
        'raw_and_lossless_cache_tensors_exact':True,'frontal_training_selection':True,
        'scope':'Synthetic CPU checks; no competition score measured or medical checkpoint loaded'}
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
