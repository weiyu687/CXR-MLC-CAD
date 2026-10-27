"""Check CXR transfer preprocessing, generic DINO and the actual train entry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from src.model import build_model
from src.preprocessing import equalize_grayscale


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve earlier checks')
    out.mkdir(parents=True)
    random=np.random.default_rng(2047)
    for index in range(50):
        values=random.integers(0,256,(37+index,41+index),dtype=np.uint8)
        if index==0:values[:]=127
        if index==1:values=np.where(values>127,255,0).astype(np.uint8)
        actual=np.asarray(equalize_grayscale(Image.fromarray(values)))[:,:,0]
        np.testing.assert_array_equal(actual,cv2.equalizeHist(values))
    torch.set_num_threads(2)
    model=build_model('dinov2_base_reg_mean',pretrained=True).eval()
    assert model.pretraining_provenance['sha256']=='73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
    with torch.inference_mode():
        x=torch.randn(2,3,56,56)
        features=model.backbone.forward_features(x)
        explicit=model.classifier(torch.cat((features[:,:1],features[:,5:]),dim=1).mean(1))
        torch.testing.assert_close(model(x),explicit)
        torch.testing.assert_close(model(x[:1]),model(x)[:1],atol=1e-5,rtol=1e-5)
    del model
    source=Path('outputs/reference_training_smoke_20261026_v2').resolve()
    if not (source/'complete.json').is_file():raise ValueError('Prior synthetic fixture missing')
    run=out/'entrypoint'
    command=[sys.executable,'-u','train.py','--root',str(source),'--cache',str(source/'cache'),
        '--out',str(run.resolve()),'--model','chexfusion_tiny_gap','--size','64','--batch-size','4',
        '--epochs','2','--workers','0','--lr','1e-5','--loss','inverse_bce','--optimizer','adam',
        '--schedule','constant','--image-training','--frontal-images','--augmentation','transfer',
        '--normalization','unit_equalized','--patient-holdout','.25','--split-seed','2030',
        '--selection-aggregation','frontal','--holdout-aggregation','frontal','--head-lr-mult','1',
        '--warmup-epochs','0','--ema-decay','0','--weight-decay','0']
    with (out/'entrypoint.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},stdout=log,stderr=subprocess.STDOUT,check=True)
    saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    loss=saved['provenance']['loss_reference']
    assert loss['class_counts']==[6.]*10 and loss['total_count']==12 and loss['weights']==[1.]*10
    assert json.loads((run/'holdout_patients.json').read_text())==json.loads((source/'chexfusion_tiny_gap/holdout_patients.json').read_text())
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    assert len(history)==2 and all(row['backbone_lr_last']==1e-5 and row['training_peak_allocated_mib'] is None for row in history)
    report={'histogram_equalization_matches_opencv':True,'generic_dinov2_weight_verified':True,
        'token_readout_matches_paper_mean_excluding_registers':True,'patient_batch_independence':True,
        'full_cpu_training_entrypoint':True,'frontal_image_counts_and_patient_exclusion':True,
        'inverse_bce_training_only_counts':True,'constant_adam_schedule':True,
        'scope':'Synthetic correctness only; no competition AUC measured'}
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
