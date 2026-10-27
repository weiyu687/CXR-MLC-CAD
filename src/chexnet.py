"""CheXNet adaptation for official ten-label studies.

Reference: arnoweng/CheXNet (inference) and arxiv:1711.05225 (training).
This is an independent adaptation using torchvision, not a redistribution of
that repository's NIH checkpoint. Only generic ImageNet initialization is used.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

from .data import frame


CACHE_SIGNATURE='chexnet-rgb-bilinear-shortside256-lossless-png-v1'
MEAN=[.485,.456,.406]
STD=[.229,.224,.225]


def canonical_image(image):
    return T.Resize(256,interpolation=T.InterpolationMode.BILINEAR)(image.convert('RGB'))


class CheXNetTransform:
    def __init__(self,mode='ten'):
        if mode not in ['train','ten','center']:raise ValueError('Unknown CheXNet crop mode')
        self.mode=mode
        self.tensor=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
        self.flip=T.RandomHorizontalFlip()

    def __call__(self,image):
        image=canonical_image(image)
        if self.mode=='train':
            return self.tensor(self.flip(T.Resize((224,224))(image)))
        if self.mode=='center':return self.tensor(T.CenterCrop(224)(image))
        return torch.stack([self.tensor(crop) for crop in T.TenCrop(224)(image)])


class CheXNetClassifier(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        from .model import build_model
        self.densenet121=build_model('tv_densenet121',pretrained=pretrained)
        self.pretraining_provenance=getattr(self.densenet121,'pretraining_provenance',None)

    def forward(self,x):
        # Return logits for stable BCEWithLogitsLoss. Sigmoid is applied once
        # per crop before probability averaging, exactly matching the repo.
        return self.densenet121(x)


class CheXNetImages(Dataset):
    def __init__(self,root,cache,split,mode='ten',frontal_only=False):
        self.root=Path(root)
        self.cache=Path(cache) if cache else None
        self.split=split
        self.df=frame(root,split)
        if frontal_only:
            self.df=self.df[self.df.ViewPosition.isin(['AP','PA','PA LLD'])].reset_index(drop=True)
        self.transform=CheXNetTransform(mode)
        self.index=None
        self.packed=None
        if self.cache and (self.cache/(split+'.meta.json')).is_file():
            meta=json.loads((self.cache/(split+'.meta.json')).read_text(encoding='utf-8'))
            csv_hash=hashlib.sha256((self.root/(split+'.csv')).read_bytes()).hexdigest()
            if meta['signature']!=CACHE_SIGNATURE or meta['csv_sha256']!=csv_hash:
                raise ValueError('CheXNet cache preprocessing or CSV identity differs')
            self.index=json.loads((self.cache/(split+'.index.json')).read_text(encoding='utf-8'))
            if set(self.df.Dicom_id)-set(self.index):raise ValueError('Cache does not cover selected images')

    def __len__(self):return len(self.df)

    def __getitem__(self,index):
        row=self.df.iloc[index]
        if self.index is not None:
            if self.packed is None:
                self.packed=np.memmap(self.cache/(self.split+'.bin'),mode='r',dtype=np.uint8)
            entry=self.index[row.Dicom_id]
            data=self.packed[entry['offset']:entry['offset']+entry['length']].tobytes()
            with Image.open(io.BytesIO(data)) as image:picture=self.transform(image)
        else:
            path=self.root/self.split/row.Subject_id/row.Study_id/(row.Dicom_id+'.jpg')
            with Image.open(path) as image:picture=self.transform(image)
        target=torch.tensor(row.target if 'target' in row else [0.]*10,dtype=torch.float32)
        return picture,target,index


def crop_probabilities(model,pictures,device):
    """Return ten-crop mean and its center-crop member without a second pass."""
    if pictures.ndim!=5 or pictures.shape[1]!=10:
        raise ValueError('Expected B x 10 x 3 x 224 x 224 CheXNet crops')
    batch,views=pictures.shape[:2]
    flat=pictures.flatten(0,1).to(device,non_blocking=True)
    if device.type=='cuda':flat=flat.to(memory_format=torch.channels_last)
    with torch.autocast(device.type,enabled=device.type=='cuda'):
        probabilities=model(flat).float().sigmoid().reshape(batch,views,10)
    return probabilities.mean(1),probabilities[:,4]


@torch.inference_mode()
def collect_predictions(model,loader,device):
    model.eval()
    result={mode:np.zeros((len(loader.dataset),10),dtype=np.float32) for mode in ['ten','center']}
    for pictures,_,indices in loader:
        ten,center=crop_probabilities(model,pictures,device)
        result['ten'][indices.numpy()]=ten.cpu().numpy()
        result['center'][indices.numpy()]=center.cpu().numpy()
    return result


def load_checkpoint(path,device):
    from .model import build_model
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    config=checkpoint['config']
    if config.get('preprocessing')!=CACHE_SIGNATURE or config['model']!='chexnet_densenet121':
        raise ValueError('Checkpoint is not a CheXNet RGB/256 adaptation')
    model=build_model(config['model'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    if device.type=='cuda':model=model.to(device,memory_format=torch.channels_last)
    else:
        torch.backends.mkldnn.enabled=False
        model=model.to(device)
    return model.eval(),checkpoint
