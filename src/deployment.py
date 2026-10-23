"""Conservative serial full-study prediction shared by timing entry points."""
from pathlib import Path
import hashlib
import numpy as np
from .data import Images,aggregate
from .inference import load_deployed_model,forward_probabilities
from .metrics import submission_probabilities

class StudyPredictor:
    def __init__(self,checkpoints,root,device,aggregation='frontal',tta=False,split='test'):
        self.root=Path(root);self.device=device;self.aggregation=aggregation;self.tta=tta;self.split=split
        self.members=[]
        for path in checkpoints:
            model,ck=load_deployed_model(path,device)
            ds=Images(root,'__no_cache__',split,ck['config']['size'],normalization=ck['config'].get('normalization','legacy'))
            self.members.append((model,ds))
        self.df=self.members[0][1].df
        self.positions={name:i for i,name in enumerate(self.df.Dicom_id)}
        self.study_rows={s:g.index.to_numpy() for s,g in self.df.groupby('Study_id',sort=False)}

    def original_path(self,idx):
        row=self.df.iloc[idx]
        return self.root/self.split/row.Subject_id/row.Study_id/(row.Dicom_id+'.jpg')

    def __call__(self,path):
        path=Path(path)
        input_bytes=path.read_bytes()
        index=self.positions[path.stem]
        original=self.original_path(index)
        if path.resolve()!=original.resolve():
            if hashlib.sha256(input_bytes).digest()!=hashlib.sha256(original.read_bytes()).digest():
                raise ValueError('Timing input differs from the original image with the same ID')
        study=self.df.iloc[index].Study_id
        rows=self.study_rows[study]
        small=self.df.iloc[rows].reset_index(drop=True)
        if self.aggregation=='frontal':
            fronts=small.ViewPosition.isin(['AP','PA','PA LLD']).to_numpy()
            if fronts.any():rows=rows[fronts];small=small.loc[fronts].reset_index(drop=True)
        probs=[]
        for model,ds in self.members:
            image=[]
            for idx in rows:
                x,_,_=ds[int(idx)]
                image.append(forward_probabilities(model,x.unsqueeze(0),self.device,self.tta).cpu().numpy()[0])
            _,p,_=aggregate(small,np.stack(image),self.aggregation)
            probs.append(p[0])
        return study,submission_probabilities(np.mean(probs,axis=0))
