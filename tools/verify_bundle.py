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
    if 'cost.py' in manifest['files']:assert 'src/deployment.py' in manifest['files']
    for relative,expected in manifest['files'].items():
        path=(bundle/relative).resolve()
        if bundle not in path.parents:raise ValueError('Manifest path escapes package')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing or changed package file: '+relative)
    labels=json.loads((bundle/'labels.json').read_text())
    assert labels=={str(i):name for i,name in enumerate(LABELS,1)}
    info=json.loads((bundle/'checkpoint_info.json').read_text())
    report=json.loads((bundle/'report.json').read_text())
    if 'ensemble_weights' in report['arguments']:assert 'src/ensemble.py' in manifest['files']
    assert report.get('deterministic_inference') is True,'Package must declare deterministic deployment settings'
    hashes=[hashlib.sha256((bundle/item['path']).read_bytes()).hexdigest() for item in info]
    assert hashes==report['checkpoint_sha256']
    model_bytes=sum((bundle/item['path']).stat().st_size for item in info)
    assert model_bytes==manifest['model_bytes'] and model_bytes<=500_000_000
    deployment_bytes=model_bytes+sum((bundle/name).stat().st_size for name in ['report.json','labels.json','checkpoint_info.json'])
    if 'deployment_bytes' in manifest:assert deployment_bytes==manifest['deployment_bytes'] and deployment_bytes<=500_000_000
    for item in info:
        upstream=(item.get('provenance') or {}).get('pretraining')
        if upstream:
            upstream_path=bundle/'pretrain_model'/upstream['url'].rsplit('/',1)[-1]
            assert hashlib.sha256(upstream_path.read_bytes()).hexdigest()==upstream['sha256']
            assert (bundle/'licenses'/'torchvision-LICENSE.txt').is_file()
    checks=validate_submission(bundle/'submission.csv',args.root)
    csv=pd.read_csv(bundle/'submission.csv')
    assert csv.groupby('Predict_class').Gate.nunique().eq(1).all()
    gates=csv.groupby('Predict_class').Gate.first().sort_index().values
    gate_tolerance=max(1e-15,.51*10**(-report.get('csv_decimal_places',8)))
    assert np.allclose(gates,report['validation']['gates'],rtol=0,atol=gate_tolerance)
    result={'folder':str(bundle),'files_verified':len(manifest['files']),'model_bytes':model_bytes,'deployment_bytes':deployment_bytes,
            'submission':checks,'frozen_gates_match':True,'model_hashes_match':True,
            'independent_inference_reproduced':False,'T4_verified':False}
    if report.get('gate_policy')=='half_step_between_distinct_validation_probabilities':
        scale=10**report['probability_decimals']
        p=csv.Probability.values;g=csv.Gate.values
        assert np.all(np.abs(p*scale-np.round(p*scale))<.001)
        assert np.all(np.abs(g*scale-np.floor(g*scale)-.5)<.001)
        assert not np.any(p==g)
        result.update(emitted_threshold_equalities=0,disjoint_probability_and_gate_grids=True)
    if (bundle/'latency.json').exists():
        assert 'latency.json' in manifest['files']
        latency=json.loads((bundle/'latency.json').read_text(encoding='utf-8'))
        assert latency['checkpoint_sha256']==hashes
        assert latency['checkpoint_bytes']==model_bytes and latency['models']==len(info)
        assert latency['runtime']==report['runtime'] and latency['tta']==report['arguments']['tta']
        assert latency.get('ensemble_weights')==report['arguments'].get('ensemble_weights')
        assert latency['batch_size']==1 and latency['cache'] is False
        timing_split=latency.get('split','test')
        if timing_split=='test':assert latency['test_csv_sha256']==hashlib.sha256((Path(args.root)/'test.csv').read_bytes()).hexdigest()
        else:
            assert timing_split in ['train','valid'] and len(latency['dataset_csv_sha256'])==64
            result['latency_selection_split']=timing_split
        selection=json.loads((bundle/'selection.json').read_text(encoding='utf-8')) if (bundle/'selection.json').exists() else {}
        if selection:
            assert selection['selected'].get('ensemble_weights')==report['arguments'].get('ensemble_weights')
        statistic=selection.get('local_limit_statistic','max')
        limit=selection.get('local_sample_limit_ms',100)
        assert statistic in ['max','mean'] and 0<limit<=100
        assert latency['samples']>=10 and np.isfinite(latency[statistic+'_ms']) and latency[statistic+'_ms']<=limit
        result.update(local_latency_sample_limit_passed=True,local_latency_samples=latency['samples'],
                      local_latency_max_ms=latency['max_ms'],local_latency_limit_statistic=statistic,
                      local_latency_mean_ms=latency['mean_ms'])
    if args.reproduced:
        validate_submission(args.reproduced,args.root)
        reproduced_report=json.loads(Path(args.reproduced).with_name('report.json').read_text(encoding='utf-8'))
        assert reproduced_report['checkpoint_sha256']==report['checkpoint_sha256']
        assert reproduced_report['validation_recomputed'] is False,'Reproduction must use frozen calibration'
        for key in ['aggregation','tta','batch_size']:
            assert reproduced_report['arguments'][key]==report['arguments'][key]
        assert reproduced_report['arguments'].get('ensemble_weights')==report['arguments'].get('ensemble_weights')
        for key,default in [('probability_decimals',None),('csv_decimal_places',8),('gate_policy','legacy_threshold_value')]:
            assert reproduced_report.get(key,default)==report.get(key,default)
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
