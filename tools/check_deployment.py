"""Synthetic CPU smoke test: package and reproduce without any validation files."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import shutil
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
                'arguments':{'tta':False,'aggregation':'weighted','batch_size':2}}
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
        valid=pd.read_csv(root/'test.csv')
        valid['Dicom_id']=['v1','v2','v3']
        valid['Predict_class']=['1;2;3;4;5;6;7;8;9','1;2;3;4;5;6;7;8;9','10']
        valid.to_csv(root/'valid.csv',index=False)
        (cache/'valid').mkdir()
        for i in range(1,4):shutil.copy2(cache/'test'/f'x{i}.jpg',cache/'valid'/f'v{i}.jpg')
        # Candidate selection must also work when test metadata/images do not
        # exist. Its NPZ still feeds the same ensemble-selection implementation.
        valid_only_root=tmp/'only_valid';valid_only_root.mkdir()
        shutil.copy2(root/'valid.csv',valid_only_root/'valid.csv')
        valid_only=tmp/'validation_export'
        run(['predict.py','--checkpoint',checkpoint,'--root',valid_only_root,'--cache',cache,
             '--out',valid_only,'--workers','0','--batch-size','2','--export-images','--validation-only'])
        assert not (valid_only/'submission.csv').exists()
        validation_report=json.loads((valid_only/'report.json').read_text())
        assert validation_report['submission'] is None and validation_report['validation_recomputed']
        with np.load(valid_only/'predictions.npz') as saved:
            assert 'valid_images' in saved and 'test' not in saved and 'test_images' not in saved
            metadata=json.loads(str(saved['image_prediction_metadata']))
            assert 'test_csv_sha256' not in metadata
        assert not (valid_only_root/'test.csv').exists()
        calibrated=tmp/'calibrated';new_bundle=tmp/'calibrated_bundle';new_reproduced=tmp/'new_reproduced'
        run(['predict.py','--checkpoint',checkpoint,'--root',root,'--cache',cache,'--out',calibrated,
             '--workers','0','--batch-size','2','--export-images'])
        generated=json.loads((calibrated/'report.json').read_text())
        assert generated['validation']==validation_report['validation']
        with np.load(calibrated/'predictions.npz') as full, np.load(valid_only/'predictions.npz') as validation:
            np.testing.assert_array_equal(full['valid_images'],validation['valid_images'])
        assert generated['probability_decimals']==12 and generated['csv_decimal_places']==15
        assert generated['gate_policy']=='half_step_between_distinct_validation_probabilities'
        run(['tools/package.py','--checkpoint',checkpoint,'--predictions',calibrated,'--root',root,'--out',new_bundle])
        only_test=tmp/'only_test';only_test.mkdir()
        shutil.copy2(root/'test.csv',only_test/'test.csv');shutil.copytree(root/'test',only_test/'test')
        run(['predict.py','--checkpoint','models/model_1.pt','--calibration','report.json','--root',only_test,
             '--cache',tmp/'missing_cache','--out',new_reproduced,'--workers','0'],cwd=new_bundle)
        run(['tools/verify_bundle.py','--bundle',new_bundle,'--root',only_test,
             '--reproduced',new_reproduced/'submission.csv','--out',tmp/'verification.json'])
        actual=json.loads((tmp/'verification.json').read_text())
        assert actual['max_probability_difference']==0 and actual['runtime_recorded']
        assert not (only_test/'valid.csv').exists()
        # A two-model weighted result must retain its exact blend after
        # packaging and raw-image reproduction without validation labels.
        second_checkpoint=tmp/'second.pt'
        second=torch.load(checkpoint,map_location='cpu',weights_only=False)
        second['state_dict']['fc.bias']+=torch.linspace(-.5,.5,10)
        torch.save(second,second_checkpoint)
        weighted=tmp/'weighted';weighted_bundle=tmp/'weighted_bundle';weighted_raw=tmp/'weighted_raw'
        run(['predict.py','--checkpoint',checkpoint,second_checkpoint,'--ensemble-weights','.25','.75',
             '--root',root,'--cache',cache,'--out',weighted,'--workers','0','--batch-size','1'])
        weighted_report=json.loads((weighted/'report.json').read_text())
        assert weighted_report['arguments']['ensemble_weights']==[.25,.75]
        run(['tools/package.py','--checkpoint',checkpoint,second_checkpoint,'--predictions',weighted,
             '--root',root,'--out',weighted_bundle])
        run(['predict.py','--checkpoint','models/model_1.pt','models/model_2.pt','--calibration','report.json',
             '--root',only_test,'--cache',tmp/'missing_cache','--out',weighted_raw,'--workers','0'],cwd=weighted_bundle)
        run(['tools/verify_bundle.py','--bundle',weighted_bundle,'--root',only_test,
             '--reproduced',weighted_raw/'submission.csv','--out',tmp/'weighted_verification.json'])
        assert json.loads((tmp/'weighted_verification.json').read_text())['max_probability_difference']==0
        frozen['checkpoint_sha256']=['0'*64];calibration.write_text(json.dumps(frozen))
        result=subprocess.run([sys.executable,'predict.py','--checkpoint',str(checkpoint),'--calibration',str(calibration),
                               '--root',str(root),'--out',str(tmp/'must_fail')],cwd=project,env=env,capture_output=True,text=True)
        assert result.returncode!=0 and 'Calibration does not match' in result.stderr
    print('PASS: validation-only export without test access; legacy and half-step calibration; equal/weighted cached/raw package reproduction; identical CSV; runtime/model binding; test-only inference')
if __name__=='__main__':main()
