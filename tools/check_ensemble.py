"""Guard ensemble pooling order and the aggregate model-file size gate."""
from pathlib import Path
import sys
import tempfile
import hashlib
import json
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
        files=[]
        for run in runs:
            with np.load(run/'valid_predictions.npz') as saved:probabilities=saved['probabilities']
            metadata={'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
                      'batch_size':64,'tta':False,'deterministic_inference':True,
                      'validation_csv_sha256':hashlib.sha256((root/'valid.csv').read_bytes()).hexdigest()}
            path=run/'deployment.npz';files.append(path)
            np.savez(path,valid_images=probabilities,valid_image_ids=np.array(['a','b','c','d']),
                     image_prediction_metadata=np.array(json.dumps(metadata)))
        exported=select(runs,root,prediction_files=files)
        assert exported['selected']==result['selected']
        assert exported['prediction_source']=='deterministic deployment exports'
        np.savez(files[0],valid_images=probabilities,valid_image_ids=np.array(['b','a','c','d']),
                 image_prediction_metadata=np.array(json.dumps({**metadata,
                    'checkpoint_sha256':hashlib.sha256((runs[0]/'best.pt').read_bytes()).hexdigest()})))
        try:
            select(runs,root,prediction_files=files)
            raise RuntimeError('Shuffled image IDs incorrectly accepted')
        except ValueError as error:assert 'image order' in str(error)
    print('PASS: pooling order, subset/size gates, deterministic export equivalence and image-order rejection')
if __name__=='__main__':main()
