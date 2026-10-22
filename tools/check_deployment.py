"""Synthetic CPU smoke test: package and reproduce without any validation files."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.model import build_model
from src.preprocessing import resize_original

def main():
    project=Path(__file__).resolve().parents[1]
    torch.set_num_threads(2);torch.manual_seed(123)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
    with tempfile.TemporaryDirectory() as folder:
        tmp=Path(folder);root=tmp/'data';cache=tmp/'cache'
        root.mkdir();(cache/'test').mkdir(parents=True)
        pd.DataFrame({'Dicom_id':['x1','x2','x3'],'Subject_id':['p1','p1','p2'],
                      'Study_id':['s1','s1','s2'],'ViewPosition':['PA','LL','AP']}).to_csv(root/'test.csv',index=False)
        for i in range(1,4):
            raw=root/'test'/('p1' if i<3 else 'p2')/('s1' if i<3 else 's2')/f'x{i}.jpg'
            raw.parent.mkdir(parents=True,exist_ok=True)
            image=np.tile(np.arange(896,dtype=np.uint16),(640,1))
            Image.fromarray(((image+50*i)%256).astype(np.uint8)).save(raw,quality=90)
            with Image.open(raw) as original:
                resize_original(original).save(cache/'test'/f'x{i}.jpg',quality=95)
        checkpoint=tmp/'test.pt'
        torch.save({'state_dict':build_model('resnet18').state_dict(),'config':{'model':'resnet18','size':64}},checkpoint)
        frozen={'validation':{'gates':[.5]*10,'macro_auc':None,'note':'synthetic test; no measured validation score'},
                'checkpoint_sha256':[hashlib.sha256(checkpoint.read_bytes()).hexdigest()],
                'arguments':{'tta':False,'aggregation':'weighted'}}
        calibration=tmp/'frozen.json';calibration.write_text(json.dumps(frozen))
        def run(arguments,cwd=project):
            result=subprocess.run([sys.executable,*map(str,arguments)],cwd=cwd,env=env,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(f'Process exit code {result.returncode}: '+result.stdout+'\n'+result.stderr)
        prediction=tmp/'predictions';bundle=tmp/'bundle';reproduced=tmp/'reproduced'
        run(['predict.py','--checkpoint',checkpoint,'--calibration',calibration,'--root',root,
             '--cache',cache,'--out',prediction,'--workers','0','--batch-size','2'])
        run(['tools/package.py','--checkpoint',checkpoint,'--predictions',prediction,'--root',root,'--out',bundle])
        run(['predict.py','--checkpoint','models/model_1.pt','--calibration','report.json','--root',root,
             '--cache',tmp/'missing_cache','--out',reproduced,'--workers','0','--batch-size','2'],cwd=bundle)
        a=pd.read_csv(prediction/'submission.csv');b=pd.read_csv(reproduced/'submission.csv')
        pd.testing.assert_frame_equal(a,b)
        assert len(a)==20 and not (root/'valid.csv').exists()
        manifest=json.loads((bundle/'manifest.json').read_text())
        for name,expected in manifest['files'].items():
            assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==expected
        frozen['checkpoint_sha256']=['0'*64];calibration.write_text(json.dumps(frozen))
        result=subprocess.run([sys.executable,'predict.py','--checkpoint',str(checkpoint),'--calibration',str(calibration),
                               '--root',str(root),'--out',str(tmp/'must_fail')],cwd=project,env=env,capture_output=True,text=True)
        assert result.returncode!=0 and 'Calibration does not match' in result.stderr
    print('PASS: test-only frozen calibration, cached/raw packaged reproduction, identical CSV, manifest hashes, mismatched model rejection')
if __name__=='__main__':main()
