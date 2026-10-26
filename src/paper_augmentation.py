"""The published CheXFusion augmentation sequence, using its 2023 API.

Source: github.com/dongkyunk/CheXFusion/blob/main/dataset/transforms.py.
Receives the project's canonical padded RGB image. The validation/deployment
preprocessing remains the existing deterministic ImageNet-normalized path.
"""
import numpy as np
import torch


class CheXFusionAugment:
    def __init__(self, size):
        import albumentations as A
        import cv2
        if A.__version__ != '1.3.1':
            raise RuntimeError('Use requirements-reference-train.txt for the paper augmentation API')
        cv2.setNumThreads(0)
        self.pipeline = A.Compose([
            A.RandomResizedCrop(size, size, scale=(.9, 1.), p=1., interpolation=cv2.INTER_LANCZOS4),
            A.HorizontalFlip(p=.5),
            A.ShiftScaleRotate(p=.5),
            A.RandomBrightnessContrast(p=.5),
            A.OneOf([A.OpticalDistortion(), A.GridDistortion(), A.ElasticTransform()], p=.2),
            A.OneOf([A.GaussNoise(), A.GaussianBlur(), A.MotionBlur(), A.MedianBlur()], p=.2),
            A.Resize(size, size, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(),
        ])

    def __call__(self, image):
        output = self.pipeline(image=np.asarray(image))['image']
        return torch.from_numpy(np.ascontiguousarray(output.transpose(2, 0, 1)))
