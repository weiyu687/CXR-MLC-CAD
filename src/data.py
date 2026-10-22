"""Competition-only data handling; no network access or external weights."""
from pathlib import Path
import random
import io
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

LABELS = ['Enlarged Cardiomediastinum', 'Pneumothorax', 'Consolidation',
          'Pneumonia', 'Edema', 'Cardiomegaly', 'Atelectasis', 'Lung Opacity',
          'Pleural Effusion', 'No Finding']

def frame(root, split):
    df = pd.read_csv(Path(root) / (split + '.csv')).fillna('')
    if df.Dicom_id.duplicated().any():
        raise ValueError('Duplicate Dicom_id')
    if 'Predict_class' in df:
        df['target'] = df.Predict_class.map(lambda s: [float(str(i) in str(s).split(';')) for i in range(1, 11)])
        for _, g in df.groupby('Study_id'):
            if len(set(g.Predict_class.astype(str))) != 1:
                raise ValueError('Inconsistent labels within study')
    return df

class Images(Dataset):
    def __init__(self, root, cache, split, size, train=False, study_sampling=True):
        self.root, self.cache, self.split = Path(root), Path(cache), split
        index_path = self.cache / (split + '.index.json')
        self.packed_index = json.loads(index_path.read_text()) if index_path.exists() else None
        self.packed_bytes = None
        self.df = frame(root, split)
        self.train, self.study_sampling = train, train and study_sampling
        self.groups = [g.index.tolist() for _, g in self.df.groupby('Study_id', sort=False)]
        aug = [T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
               T.RandomAffine(6, translate=(0.025, 0.025)),
               T.ColorJitter(brightness=0.12, contrast=0.12)] if train else [T.Resize((size, size))]
        self.transform = T.Compose(aug + [T.ToTensor(), T.Normalize([0.5]*3, [0.25]*3)])
    def __len__(self):
        return len(self.groups) if self.study_sampling else len(self.df)
    def __getitem__(self, idx):
        if self.study_sampling:
            idx = random.choice(self.groups[idx])
        r = self.df.iloc[idx]
        if self.packed_index is not None:
            if self.packed_bytes is None:
                self.packed_bytes = np.memmap(self.cache / (self.split+'.bin'), dtype=np.uint8, mode='r')
            offset, length = self.packed_index[r.Dicom_id]
            source = io.BytesIO(self.packed_bytes[offset:offset+length].tobytes())
        else:
            source = None
        p = self.cache / self.split / (r.Dicom_id + '.jpg')
        if source is None and not p.exists():
            p = self.root / self.split / r.Subject_id / r.Study_id / (r.Dicom_id + '.jpg')
        with Image.open(source if source is not None else p) as im:
            im = im.convert('RGB')
            if im.size != (448, 448):
                im = ImageOps.pad(im, (448, 448), method=Image.Resampling.BICUBIC, color=0)
            x = self.transform(im)
        y = torch.tensor(r.target if 'target' in r else [0.]*10, dtype=torch.float32)
        return x, y, idx

def aggregate(df, probabilities, mode='mean'):
    p = pd.DataFrame(probabilities, columns=list(range(10)))
    p['Study_id'] = df.Study_id.values
    if mode == 'max':
        out = p.groupby('Study_id', sort=False).max()
    elif mode in ['frontal','weighted']:
        frontal=df.ViewPosition.isin(['AP','PA','PA LLD']).to_numpy()
        if mode=='weighted':
            weights=np.where(frontal,1.,0.35)
        else:
            has_frontal=pd.Series(frontal).groupby(df.Study_id,sort=False).transform('max').to_numpy()
            weights=np.where(has_frontal,frontal.astype(float),1.)
        values=p[list(range(10))].multiply(weights,axis=0)
        values['Study_id']=df.Study_id.values
        numerator=values.groupby('Study_id',sort=False).sum()
        denominator=pd.Series(weights).groupby(df.Study_id,sort=False).sum()
        out=numerator.div(denominator,axis=0)
    else:
        out = p.groupby('Study_id', sort=False).mean()
    y = None
    if 'target' in df:
        y = np.stack(df.drop_duplicates('Study_id').set_index('Study_id').loc[out.index].target)
    return out.index.to_numpy(), out.to_numpy(), y
