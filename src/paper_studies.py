"""Study batches with an explicit valid-view mask and deterministic evaluation."""
import random

import torch
from torch.utils.data import Dataset

from .data import Images


class PaperStudies(Dataset):
    def __init__(self, root, cache, split, size, train=False, held_patients=None, max_views=4):
        self.images = Images(root, cache, split, size, normalization='imagenet')
        self.train = train
        self.max_views = max_views
        held_patients = None if held_patients is None else set(held_patients)
        if held_patients is not None:
            retained = self.images.df.Subject_id.isin(held_patients)
            self.images.df = self.images.df[~retained if train else retained].reset_index(drop=True)
        if train:
            from .paper_augmentation import CheXFusionAugment
            self.images.transform = CheXFusionAugment(size)
        self.groups = []
        self.study_ids = []
        self.patients = []
        self.targets = []
        for identifier, group in self.images.df.groupby('Study_id', sort=False):
            # Frontal first; stable tie breaks. Never inspect labels or scores
            # when selecting images. Extra training views are sampled randomly.
            indices = sorted(group.index.tolist(), key=lambda i: (
                self.images.df.iloc[i].ViewPosition not in ['AP', 'PA', 'PA LLD'],
                self.images.df.iloc[i].ViewPosition, self.images.df.iloc[i].Dicom_id))
            self.groups.append(indices)
            self.study_ids.append(identifier)
            self.patients.append(group.Subject_id.iloc[0])
            self.targets.append(group.target.iloc[0])
        if not self.groups:
            raise ValueError('Empty study dataset')

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        indices = self.groups[index]
        if self.train and len(indices)>self.max_views:
            indices = sorted(random.sample(indices, self.max_views))
        else:
            indices = indices[:self.max_views]
        images = [self.images[i][0] for i in indices]
        padding = [torch.zeros_like(images[0]) for _ in range(self.max_views-len(images))]
        valid = torch.arange(self.max_views)<len(images)
        return torch.stack(images+padding), valid, torch.tensor(self.targets[index], dtype=torch.float32), index
