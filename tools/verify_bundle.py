"""Inspect actual bundle contents, model hashes, CSV semantics and optional reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from predict import validate_submission
from src.data import LABELS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',required=True)
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--reproduced',help='Optional independently generated submission.csv')
    ap.add_argument('--out',required=True)
    args=ap.parse_args();bundle=Path(args.bundle).resolve()
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    required={'predict.py','train.py','requirements.txt','requirements-competition.txt','README.md',
              'labels.json','checkpoint_info.json','report.json','submission.csv','src/data.py','src/preprocessing.py','src/inference.py'}
    if not required.issubset(manifest['files']):raise ValueError('Required package files missing from manifest')
    for relative,expected in manifest['files'].items():
        path=(bundle/relative).resolve()
        if bundle not in path.parents:raise ValueError('Manifest path escapes package')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing or changed package file: '+relative)
    labels=json.loads((bundle/'labels.json').read_text())
    assert labels=={str(i):name for i,name in enumerate(LABELS,1)}
    info=json.loads((bundle/'checkpoint_info.json').read_text())
    report=json.loads((bundle/'report.json').read_text())
    assert report.get('deterministic_inference') is True,'Package must declare deterministic deployment settings'
    hashes=[hashlib.sha256((bundle/item['path']).read_bytes()).hexdigest() for item in info]
    assert hashes==report['checkpoint_sha256']
    model_bytes=sum((bundle/item['path']).stat().st_size for item in info)
    assert model_bytes==manifest['model_bytes'] and model_bytes<=500_000_000
    checks=validate_submission(bundle/'submission.csv',args.root)
    csv=pd.read_csv(bundle/'submission.csv')
    assert csv.groupby('Predict_class').Gate.nunique().eq(1).all()
    gates=csv.groupby('Predict_class').Gate.first().sort_index().values
    assert np.allclose(gates,report['validation']['gates'],rtol=0,atol=5.1e-9)
    result={'folder':str(bundle),'files_verified':len(manifest['files']),'model_bytes':model_bytes,
            'submission':checks,'frozen_gates_match':True,'model_hashes_match':True,
            'independent_inference_reproduced':False,'T4_verified':False}
    if args.reproduced:
        validate_submission(args.reproduced,args.root)
        reproduced_report=json.loads(Path(args.reproduced).with_name('report.json').read_text(encoding='utf-8'))
        assert reproduced_report['checkpoint_sha256']==report['checkpoint_sha256']
        assert reproduced_report['validation_recomputed'] is False,'Reproduction must use frozen calibration'
        for key in ['aggregation','tta','batch_size']:
            assert reproduced_report['arguments'][key]==report['arguments'][key]
        if report.get('runtime'):
            assert reproduced_report.get('runtime_matches_calibration') is True,'Reproduction runtime differs'
        a=csv.sort_values(['Study_id','Predict_class']).reset_index(drop=True)
        b=pd.read_csv(args.reproduced).sort_values(['Study_id','Predict_class']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[['Subject_id','Study_id','Predict_class','Gate']],b[['Subject_id','Study_id','Predict_class','Gate']])
        difference=float(np.max(np.abs(a.Probability.values-b.Probability.values)))
        assert difference<=1e-6,'Reproduced probabilities differ by more than 1e-6'
        result.update(independent_inference_reproduced=True,max_probability_difference=difference,
                      reproduction_configuration_matches=True,runtime_recorded=bool(report.get('runtime')))
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
