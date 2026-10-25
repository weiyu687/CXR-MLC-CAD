"""Check a small global blend grid and compatibility with unweighted reports."""
from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
from src.ensemble import combine_probabilities,normalized_weights,read_weight_spec
from tools.select_ensemble import select


def main():
    values=np.arange(60,dtype=np.float32).reshape(3,2,10)/60
    np.testing.assert_array_equal(combine_probabilities(values),np.mean(values,axis=0))
    expected=.25*values[0].astype(np.float64)+.75*values[1].astype(np.float64)
    np.testing.assert_array_equal(combine_probabilities(values[:2],[1,3]),expected)
    canonical=normalized_weights([1,2,7],3)
    assert canonical==normalized_weights(canonical,3)
    first=np.linspace(.2,.8,10);matrix=np.stack([first,1-first])
    expected=np.empty_like(values[0],dtype=float)
    for c in range(10):expected[:,c]=matrix[0,c]*values[0,:,c]+matrix[1,c]*values[1,:,c]
    np.testing.assert_allclose(combine_probabilities(values[:2],matrix),expected,rtol=0,atol=1e-15)
    np.testing.assert_allclose(combine_probabilities(values[:2,0],matrix),expected[0],rtol=0,atol=1e-15)
    normalized=normalized_weights(matrix*3,2)
    np.testing.assert_allclose(np.sum(normalized,axis=0),np.ones(10),rtol=0,atol=1e-15)
    assert normalized==normalized_weights(normalized,2)
    sparse=np.zeros((3,10));sparse[0,::2]=1;sparse[1,1::2]=1
    expected=np.stack([values[c%2,:,c] for c in range(10)],axis=-1)
    np.testing.assert_array_equal(combine_probabilities(values,sparse),expected)
    empty_class=np.ones((3,10));empty_class[:,4]=0
    for invalid_matrix in [np.zeros((3,10)),empty_class,np.full((3,10),-1.)]:
        try:
            normalized_weights(invalid_matrix,3)
            raise AssertionError('Empty or negative classwise weights accepted')
        except ValueError:pass
    for weights in [[0,1],[-1,2],[np.nan,1],[np.inf,1],[1],[1,2,3]]:
        try:
            normalized_weights(weights,2)
            raise AssertionError('Invalid weights accepted')
        except ValueError:pass
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        spec=root/'weights.json';hashes=['a'*64,'b'*64]
        payload={'schema_version':1,'checkpoint_sha256':hashes,'class_ids':list(range(1,11)),'weights':matrix.tolist()}
        spec.write_text(json.dumps(payload))
        assert read_weight_spec(spec,hashes)['weights']==matrix.tolist()
        for broken_hashes in [hashes[::-1],['0'*64,hashes[1]]]:
            try:
                read_weight_spec(spec,broken_hashes)
                raise AssertionError('Weight/model order mismatch accepted')
            except ValueError:pass
        spec.write_text(json.dumps({**payload,'class_ids':list(range(10,0,-1))}))
        try:
            read_weight_spec(spec,hashes)
            raise AssertionError('Wrong class order accepted')
        except ValueError:pass
        pd.DataFrame({'Dicom_id':['a','b','c','d'],'Subject_id':['p1','p2','p3','p4'],
                      'Study_id':['s1','s2','s3','s4'],'ViewPosition':['PA']*4,
                      'Predict_class':['','','1;2;3;4;5;6;7;8;9;10','1;2;3;4;5;6;7;8;9;10']}).to_csv(root/'valid.csv',index=False)
        runs=[]
        for i,p in enumerate([[.9,.1,.5,.5],[.2,.6,.5,.9]]):
            run=root/str(i);run.mkdir();runs.append(run)
            (run/'best.pt').write_bytes(b'synthetic!')
            np.savez(run/'valid_predictions.npz',probabilities=np.repeat(np.asarray(p)[:,None],10,axis=1))
        equal=select(runs,root)
        weighted=select(runs,root,weight_grid='quarter')
        assert len(equal['candidates'])==12 and len(weighted['candidates'])==20
        assert equal['selected']['metrics']['macro_auc']==.75
        assert weighted['selected']['metrics']['macro_auc']==1
        assert weighted['selected']['ensemble_weights']==[.25,.75]
        assert len(select(runs,root,weight_grid='quarter',max_members=1)['candidates'])==8
        assert len(select(runs,root,weight_grid='quarter',max_bytes=10)['candidates'])==8
    print('PASS: legacy arithmetic unchanged, normalized global weights, invalid weights rejected, validation grid improves the constructed ranking example, size/member gates preserved')


if __name__=='__main__':main()
