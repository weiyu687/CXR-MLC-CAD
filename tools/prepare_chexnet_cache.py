"""Lossless cache for the referenced CheXNet RGB/short-side-256 preprocessing.

Prepares only official train/valid. Does not download images or weights.
Cache identity is separate from the old padded grayscale/JPEG 448 cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image

from src.chexnet import canonical_image,CACHE_SIGNATURE
from src.data import frame


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def prepare(root,cache,split,workers):
    csv_hash=sha(root/(split+'.csv'))
    meta_path=cache/(split+'.meta.json')
    if meta_path.exists():
        prior=json.loads(meta_path.read_text(encoding='utf-8'))
        if prior['signature']!=CACHE_SIGNATURE or prior['csv_sha256']!=csv_hash:
            raise ValueError('Existing cache identity differs')
        if sha(cache/(split+'.bin'))!=prior['bin_sha256']:
            raise ValueError('Existing packed cache changed')
        print('ALREADY_VERIFIED '+split,flush=True)
        return prior
    data=frame(root,split)
    if split=='train':data=data[data.ViewPosition.isin(['AP','PA','PA LLD'])].reset_index(drop=True)
    binary=cache/(split+'.bin')
    if binary.exists():raise FileExistsError('Preserve incomplete cache: '+str(binary))
    started=time.time()
    def process(record):
        identifier,subject,study=record
        original=(root/split/subject/study/(identifier+'.jpg')).read_bytes()
        with Image.open(io.BytesIO(original)) as image:
            small=canonical_image(image)
            stream=io.BytesIO()
            small.save(stream,format='PNG',compress_level=1)
            shape=small.size
        return identifier,stream.getvalue(),hashlib.sha256(original).hexdigest(),shape
    records=list(data[['Dicom_id','Subject_id','Study_id']].itertuples(index=False,name=None))
    index={}
    # Keep the number of outstanding source images bounded on Python 3.9.
    with binary.open('xb') as stream,ThreadPoolExecutor(max_workers=workers) as executor:
        for begin in range(0,len(records),workers*4):
            for identifier,png,source_hash,shape in executor.map(process,records[begin:begin+workers*4]):
                index[identifier]={'offset':stream.tell(),'length':len(png),'source_sha256':source_hash,
                                   'width':shape[0],'height':shape[1]}
                stream.write(png)
            if begin%1024<workers*4:
                print(json.dumps({'split':split,'images':len(index),'total':len(records),'seconds':time.time()-started}),flush=True)
    (cache/(split+'.index.json')).write_text(json.dumps(index),encoding='utf-8')
    meta={'signature':CACHE_SIGNATURE,'csv_sha256':csv_hash,'images':len(index),'frontal_only':split=='train',
        'bin_sha256':sha(binary),'bytes':binary.stat().st_size,'seconds':time.time()-started,
        'competition_split':split,'external_images_used':False}
    meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    cache=Path(args.cache)
    cache.mkdir(parents=True,exist_ok=True)
    for split in ['train','valid']:prepare(Path(args.root),cache,split,args.workers)


if __name__=='__main__':main()
