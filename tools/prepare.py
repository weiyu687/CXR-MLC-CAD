"""Audit official splits and create deterministic, private resized image cache."""
import argparse
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import frame, LABELS
from src.preprocessing import resize_original
from PIL import Image, ImageOps
from tqdm import tqdm

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--root', default='raw_data/cxr_first')
    a.add_argument('--cache', default='data_cache/448')
    a.add_argument('--size', type=int, default=448)
    a.add_argument('--workers', type=int, default=8)
    args = a.parse_args()
    dfs = {s: frame(args.root, s) for s in ['train', 'valid', 'test']}
    report = {'labels': dict(enumerate(LABELS, 1)), 'splits': {}, 'overlaps': {}}
    jobs = []
    for s, df in dfs.items():
        Path(args.cache, s).mkdir(parents=True, exist_ok=True)
        studies = df.drop_duplicates('Study_id')
        report['splits'][s] = {'images': len(df), 'studies': len(studies), 'patients': df.Subject_id.nunique(),
                               'views': df.ViewPosition.value_counts().to_dict()}
        if 'target' in df:
            import numpy as np
            report['splits'][s]['positives'] = dict(zip(LABELS, np.stack(studies.target).sum(0).astype(int).tolist()))
        for r in df.itertuples():
            jobs.append((Path(args.root, s, r.Subject_id, r.Study_id, r.Dicom_id+'.jpg'), Path(args.cache,s,r.Dicom_id+'.jpg')))
    for s, t in [('train','valid'), ('train','test'), ('valid','test')]:
        report['overlaps'][s+'_'+t] = {k: len(set(dfs[s][k]) & set(dfs[t][k])) for k in ['Subject_id','Study_id','Dicom_id']}
        if report['overlaps'][s+'_'+t]['Study_id']:
            raise ValueError('Study leakage')
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    def convert(job):
        src, dst = job
        if dst.exists():
            return
        with Image.open(src) as im:
            im = resize_original(im,args.size)
            im.save(dst, quality=95)
    with ThreadPoolExecutor(args.workers) as pool:
        for _ in tqdm(pool.map(convert, jobs), total=len(jobs)):
            pass
if __name__ == '__main__':
    main()
