"""Canonical grayscale resize/JPEG transform shared by cache and raw inference."""
import io
from PIL import Image, ImageOps

def resize_original(image, size=448):
    return ImageOps.pad(image.convert('L'),(size,size),method=Image.Resampling.BICUBIC,color=0)

def decode_original_like_cache(image):
    """Match the saved quality-95 cache, including its JPEG quantization."""
    buffer=io.BytesIO()
    resize_original(image).save(buffer,format='JPEG',quality=95)
    buffer.seek(0)
    with Image.open(buffer) as cached:
        return cached.convert('RGB')
