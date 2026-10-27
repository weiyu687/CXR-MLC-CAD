"""Deterministic CheXNet evaluation or frozen test-only CSV reproduction."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.chexnet import CheXNetImages,CACHE_SIGNATURE,load_checkpoint,crop_probabilities
from src.data import aggregate
from src.inference import configure_deterministic_inference,runtime_information
from src.metrics import evaluate,submission_probabilities,PROBABILITY_DECIMALS,CSV_DECIMAL_PLACES


@torch.inference_mode()
def predict_images(model,loader,device,crop):
    output=np.zeros((len(loader.dataset),10),dtype=np.float32)
    model.eval()
    for pictures,_,indices in loader:
        if crop=='ten':probabilities,_=crop_probabilities(model,pictures,device)
        else:
            pictures=pictures.to(device,non_blocking=True)
            if device.type=='cuda':pictures=pictures.to(memory_format=torch.channels_last)
            with torch.autocast(device.type,enabled=device.type=='cuda'):
                probabilities=model(pictures).float().sigmoid()
        output[indices.numpy()]=probabilities.cpu().numpy()
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',default=None,help='Omit to replay original RGB JPEG files')
    parser.add_argument('--out',required=True)
    parser.add_argument('--crop',choices=['ten','center'],default='ten')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--workers',type=int,default=2)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validation-only',action='store_true')
    mode.add_argument('--calibration',help='Frozen validation report; opens only test metadata/images')
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<0:raise ValueError('Invalid inference loader')
    digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    frozen=json.loads(Path(args.calibration).read_text(encoding='utf-8')) if args.calibration else None
    if frozen:
        if frozen['checkpoint_sha256']!=digest or frozen['preprocessing']!=CACHE_SIGNATURE:
            raise ValueError('Calibration and model preprocessing identity differ')
        args.crop=frozen['arguments']['crop']
        args.batch_size=frozen['arguments']['batch_size']
    configure_deterministic_inference()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,checkpoint=load_checkpoint(args.checkpoint,device)
    split='test' if frozen else 'valid'
    dataset=CheXNetImages(args.root,args.cache,split,mode=args.crop)
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,
        pin_memory=device.type=='cuda')
    images=predict_images(model,loader,device,args.crop)
    ids,probabilities,targets=aggregate(dataset.df,images,'frontal')
    probabilities=submission_probabilities(probabilities)
    metrics=frozen['validation'] if frozen else evaluate(targets,probabilities)
    runtime=runtime_information(device)
    if frozen and runtime!=frozen['runtime']:
        raise ValueError('Frozen reproduction requires the same runtime')
    out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve prior prediction exports')
    out.mkdir(parents=True)
    submission=None
    if frozen:
        metadata=dataset.df.drop_duplicates('Study_id').set_index('Study_id')
        table=pd.DataFrame({'Study_id':np.repeat(ids,10),'Predict_class':np.tile(np.arange(1,11),len(ids)),
            'Probability':probabilities.ravel(),'Gate':np.tile(metrics['gates'],len(ids))})
        table.insert(0,'Subject_id',metadata.loc[table.Study_id].Subject_id.to_numpy())
        if len(table)!=10*dataset.df.Study_id.nunique() or table.duplicated(['Study_id','Predict_class']).any():
            raise ValueError('Submission study/class coverage is incomplete')
        if not np.isfinite(table[['Probability','Gate']].to_numpy()).all():raise ValueError('Invalid submitted probabilities')
        table.to_csv(out/'submission.csv',index=False,float_format='%.'+str(CSV_DECIMAL_PLACES)+'f')
        submission={'rows':len(table),'studies':len(ids),'classes':10,'valid':True}
    report={'validation':metrics,'submission':submission,'checkpoint_sha256':digest,'preprocessing':CACHE_SIGNATURE,
        'arguments':vars(args),'runtime':runtime,'probability_decimals':PROBABILITY_DECIMALS,
        'csv_decimal_places':CSV_DECIMAL_PLACES,'deterministic_inference':True,
        'dataset_csv_sha256':hashlib.sha256((Path(args.root)/(split+'.csv')).read_bytes()).hexdigest(),
        'validation_recomputed':not bool(frozen),'official_website_submission_performed':False}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    saved={'study_ids':ids.astype(str),'image_ids':dataset.df.Dicom_id.to_numpy(dtype=str),'images':images,
        'probabilities':probabilities,'metadata':np.array(json.dumps(report))}
    if not frozen:saved['targets']=targets
    np.savez_compressed(out/'predictions.npz',**saved)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
