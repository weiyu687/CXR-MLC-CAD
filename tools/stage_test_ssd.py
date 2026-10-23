"""Make a private, checksum-verified test-only copy for raw SSD inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--split',choices=['train','valid','test'],default='test')
    args=ap.parse_args();source=Path(args.root);out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for split in ['train','valid','test'] if split!=args.split for name in [split,split+'.csv']):
        raise ValueError('Destination must contain only the selected split')
    metadata=(source/(args.split+'.csv')).read_bytes()
    (out/(args.split+'.csv')).write_bytes(metadata)
    with (source/(args.split+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
    def copy(row):
        relative=Path(args.split)/row['Subject_id']/row['Study_id']/(row['Dicom_id']+'.jpg')
        data=(source/relative).read_bytes();digest=hashlib.sha256(data).hexdigest()
        destination=out/relative
        if out.resolve() not in destination.resolve().parents:
            raise ValueError('Image path escapes staging directory')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
                raise ValueError('Existing destination differs from official source: '+str(relative))
        else:
            temporary=destination.with_suffix('.jpg.tmp')
            temporary.write_bytes(data)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:
                raise ValueError('Staged image checksum failed')
            temporary.replace(destination)
        return relative.as_posix(),{'sha256':digest,'bytes':len(data)}
    with ThreadPoolExecutor(args.workers) as pool:
        files=dict(pool.map(copy,rows))
    report={'images':len(files),'studies':len({r['Study_id'] for r in rows}),
            'bytes':sum(f['bytes'] for f in files.values()),'metadata_sha256':hashlib.sha256(metadata).hexdigest(),
            'all_images_match_original':True,'contains_training_or_validation':args.split!='test','split':args.split,'files':files}
    (out/'staging_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
if __name__=='__main__':main()
