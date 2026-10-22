"""Regression checks for competition-critical labels, aggregation and CSV completeness."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.data import frame,aggregate
from src.model import AsymmetricLoss
from predict import validate_submission

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        meta=pd.DataFrame({'Dicom_id':['a','b','c'],'Subject_id':['p1','p1','p2'],
                           'Study_id':['s1','s1','s2'],'Predict_class':['1;10','1;10','2']})
        meta.to_csv(root/'train.csv',index=False)
        meta.drop(columns='Predict_class').to_csv(root/'test.csv',index=False)
        df=frame(root,'train')
        assert df.target.iloc[0]==[1.,0.,0.,0.,0.,0.,0.,0.,0.,1.]
        probabilities=np.array([[.2]*10,[.6]*10,[.8]*10])
        ids,p,y=aggregate(df,probabilities)
        assert ids.tolist()==['s1','s2']
        assert np.allclose(p[:,0],[.4,.8]) and y.shape==(2,10)
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
    print('PASS: label IDs 1/10, multi-view averaging, submission completeness, finite ASL gradients')
if __name__=='__main__':main()
