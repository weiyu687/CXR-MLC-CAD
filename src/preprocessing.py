"""Canonical grayscale resize/JPEG transform shared by cache and raw inference."""
import io
import numpy as np
from PIL import Image, ImageOps


def equalize_grayscale(image):
    """OpenCV-style histogram equalization without an inference dependency.

    Includes the constant-image case and round-to-nearest-even conversion.
    Kept as a top-level function so Windows DataLoader workers can pickle it.
    """
    values = np.asarray(image.convert('L'), dtype=np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256)
    occupied = np.flatnonzero(histogram)
    if len(occupied)<=1:
        return image.convert('RGB')
    low = occupied[0]
    count = values.size-histogram[low]
    cumulative = histogram.cumsum()-histogram[low]
    scale = np.float32(255.)/np.float32(count)
    lookup = np.rint(cumulative.astype(np.float32)*scale).clip(0,255).astype(np.uint8)
    return Image.fromarray(lookup[values], mode='L').convert('RGB')

def resize_original(image, size=448):
    return ImageOps.pad(image.convert('L'),(size,size),method=Image.Resampling.BICUBIC,color=0)

def decode_original_like_cache(image):
    """Match the saved quality-95 cache, including its JPEG quantization."""
    buffer=io.BytesIO()
    resize_original(image).save(buffer,format='JPEG',quality=95)
    buffer.seek(0)
    with Image.open(buffer) as cached:
        return cached.convert('RGB')
