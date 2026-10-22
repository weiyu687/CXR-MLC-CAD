"""Regression checks for competition-critical labels, aggregation and CSV completeness."""
from pathlib import Path
import sys
import tempfile
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.data import frame,aggregate,Images
from PIL import Image
from src.model import AsymmetricLoss
from predict import validate_submission
from src.metrics import evaluate,submission_probabilities

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        meta=pd.DataFrame({'Dicom_id':['a','b','c'],'Subject_id':['p1','p1','p2'],
                           'Study_id':['s1','s1','s2'],'ViewPosition':['PA','LATERAL','LL'],
                           'Predict_class':['1;10','1;10','2']})
        meta.to_csv(root/'train.csv',index=False)
        meta.drop(columns='Predict_class').to_csv(root/'test.csv',index=False)
        df=frame(root,'train')
        assert df.target.iloc[0]==[1.,0.,0.,0.,0.,0.,0.,0.,0.,1.]
        probabilities=np.array([[.2]*10,[.6]*10,[.8]*10])
        ids,p,y=aggregate(df,probabilities)
        assert ids.tolist()==['s1','s2']
        assert np.allclose(p[:,0],[.4,.8]) and y.shape==(2,10)
        assert np.allclose(aggregate(df,probabilities,'frontal')[1][:,0],[.2,.8])
        assert np.allclose(aggregate(df,probabilities,'weighted')[1][:,0],[(.2+.6*.35)/1.35,.8])
        cache=root/'cache';(cache/'train').mkdir(parents=True)
        offsets={}
        with (cache/'train.bin').open('wb') as packed:
            for image_id in ['a','b','c']:
                path=cache/'train'/(image_id+'.jpg')
                Image.new('L',(64,64),123).save(path)
                data=path.read_bytes();offsets[image_id]=[packed.tell(),len(data)];packed.write(data)
        unpacked=Images(root,cache,'train',32)[0][0]
        (cache/'train.index.json').write_text(json.dumps(offsets))
        packed_ds=Images(root,cache,'train',32)
        assert torch.equal(unpacked,packed_ds[0][0])
        packed_ds.packed_bytes._mmap.close()
        s=pd.DataFrame({'Subject_id':np.repeat(['p1','p2'],10),'Study_id':np.repeat(ids,10),
                        'Predict_class':np.tile(np.arange(1,11),2),'Probability':p.flatten(),'Gate':.5})
        s.to_csv(root/'submission.csv',index=False)
        assert validate_submission(root/'submission.csv',root)['valid']
        s.iloc[:-1].to_csv(root/'broken.csv',index=False)
        try:
            validate_submission(root/'broken.csv',root)
            raise RuntimeError('Incomplete CSV incorrectly accepted')
        except AssertionError:pass
        logits=torch.tensor([[100.,-100.]*5],requires_grad=True)
        loss=AsymmetricLoss()(logits,torch.tensor([[0.,1.]*5]))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    y=np.repeat(np.array([[0],[0],[1],[1]]),10,axis=1)
    p=np.repeat(np.array([[.1],[.3],[.6],[.9]]),10,axis=1)
    metrics=evaluate(y,p);gates=np.array(metrics['gates'])
    assert np.allclose(gates,.45) and metrics['macro_f1_tuned']==1.
    assert all(format(value,'.15f').endswith('500') for value in gates)
    assert np.array_equal(p>=gates,p>gates)
    close_p=np.repeat(np.array([[.1],[.100000000001],[.2],[.3]]),10,axis=1)
    close_y=np.repeat(np.array([[0],[1],[1],[1]]),10,axis=1)
    close_metrics=evaluate(close_y,close_p)
    written_p=np.array([[float(format(value,'.15f')) for value in row] for row in submission_probabilities(close_p)])
    written_gates=np.array([float(format(value,'.15f')) for value in close_metrics['gates']])
    assert np.array_equal(written_p>written_gates,close_y.astype(bool))
    assert np.array_equal(written_p>=written_gates,close_y.astype(bool))
    clipped=submission_probabilities(np.array([[0.,1.]]))
    assert clipped.min()>0 and clipped.max()<1
    print('PASS: labels, pooling, packed cache, CSV completeness, ASL gradients, midpoint gate boundaries and precision')
if __name__=='__main__':main()
