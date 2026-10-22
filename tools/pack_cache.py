"""Pack private compressed cache for efficient Windows memory-mapped reads."""
import argparse
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',default='data_cache/448')
    ap.add_argument('--out',help='Optional SSD folder for packed cache')
    args=ap.parse_args();root=Path(args.cache)
    destination=Path(args.out) if args.out else root
    destination.mkdir(parents=True,exist_ok=True)
    for split in ['train','valid','test']:
        import pandas as pd
        ids=pd.read_csv(Path('raw_data/cxr_first')/(split+'.csv')).Dicom_id.tolist()
        if not all((root/split/(i+'.jpg')).exists() for i in ids):
            raise FileNotFoundError('Resized cache is incomplete; finish prepare.py first')
        index={}
        tmp=destination/(split+'.bin.tmp')
        with tmp.open('wb') as out:
            for image_id in ids:
                data=(root/split/(image_id+'.jpg')).read_bytes()
                index[image_id]=[out.tell(),len(data)];out.write(data)
        tmp.replace(destination/(split+'.bin'))
        (destination/(split+'.index.json')).write_text(json.dumps(index),encoding='utf-8')
        print(split,len(ids),(destination/(split+'.bin')).stat().st_size,flush=True)
if __name__=='__main__':main()
