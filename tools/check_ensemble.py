"""Guard ensemble pooling order and the aggregate model-file size gate."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from tools.select_ensemble import select

def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        pd.DataFrame({'Dicom_id':['a','b','c','d'],'Subject_id':['p1','p1','p2','p2'],
                      'Study_id':['s1','s1','s2','s2'],'ViewPosition':['PA','LL','PA','LL'],
                      'Predict_class':['1;2;3;4;5;6;7;8;9;10']*2+['']*2}).to_csv(root/'valid.csv',index=False)
        runs=[]
        for i,values in enumerate([[.9,.1,.2,.3],[.1,.9,.2,.4]]):
            run=root/f'run_{i}';run.mkdir();runs.append(run)
            (run/'best.pt').write_bytes(b'synthetic!')
            np.savez(run/'valid_predictions.npz',probabilities=np.repeat(np.array(values)[:,None],10,axis=1))
        result=select(runs,root)
        assert len(result['candidates'])==12
        combined=next(x for x in result['candidates'] if len(x['checkpoints'])==2 and x['aggregation']=='max')
        assert np.allclose(combined['metrics']['gates'],.9)
        # Averaging before max would produce .5; production does max before averaging.
        assert not np.allclose(combined['metrics']['gates'],.5)
        restricted=select(runs,root,max_bytes=10)
        assert len(restricted['candidates'])==8
        assert all(len(x['checkpoints'])==1 for x in restricted['candidates'])
        assert len(result['selected']['checkpoints'])==1
    print('PASS: ensemble pooling order, subset enumeration, size exclusion, smaller-model tie break')
if __name__=='__main__':main()
