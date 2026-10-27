"""CheXNet DenseNet121 training adapted to ten-label official study outputs.

The linked repository exposes inference only. Training follows the original
paper: ImageNet initialization, Adam, unweighted multi-label BCE, horizontal
flip, and loss-plateau LR reduction. Checkpoint selection by validation loss
and by competition AUC are saved separately. No medical checkpoint is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.chexnet import CheXNetImages,CACHE_SIGNATURE,collect_predictions
from src.data import aggregate,LABELS
from src.metrics import evaluate
from src.model import build_model


def save_checkpoint(checkpoint,path):
    temporary=path.with_suffix('.pending.pt')
    torch.save(checkpoint,temporary)
    temporary.replace(path)


def validation(model,loader,device):
    predictions=collect_predictions(model,loader,device)
    results={}
    for mode,images in predictions.items():
        ids,probabilities,targets=aggregate(loader.dataset.df,images,'frontal')
        metrics=evaluate(targets,probabilities)
        clipped=np.clip(probabilities.astype(np.float64),1e-7,1-1e-7)
        loss=float(-(targets*np.log(clipped)+(1-targets)*np.log1p(-clipped)).sum(1).mean())
        results[mode]={'metrics':metrics,'loss':loss,'ids':ids,'probabilities':probabilities,'targets':targets}
    return results,predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--min-epochs',type=int,default=20)
    parser.add_argument('--early-stopping',type=int,default=12)
    parser.add_argument('--plateau-patience',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--eval-batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2050)
    parser.add_argument('--random-initialization',action='store_true',help='Synthetic checks only; official runs use ImageNet')
    parser.add_argument('--resume')
    args=parser.parse_args()
    if not 1<=args.min_epochs<=args.epochs or args.batch_size<1 or args.eval_batch_size<1 or args.workers<0:
        raise ValueError('Invalid training schedule or loader dimensions')
    resume=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if resume:
        previous=resume['config']
        for name in ['lr','epochs','min_epochs','early_stopping','plateau_patience','batch_size','eval_batch_size','seed']:
            if getattr(args,name)!=previous[name]:raise ValueError('Resume recipe differs: '+name)
    out=Path(args.out)
    if out.exists() and not resume:raise FileExistsError('Preserve existing CheXNet experiment')
    out.mkdir(parents=True,exist_ok=bool(resume))
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.8)
        torch.backends.cudnn.benchmark=True
    else:torch.backends.mkldnn.enabled=False
    train=CheXNetImages(args.root,args.cache,'train',mode='train',frontal_only=True)
    valid=CheXNetImages(args.root,args.cache,'valid',mode='ten')
    if set(train.df.Subject_id)&set(valid.df.Subject_id):raise ValueError('Official train/valid patient overlap')
    generator=torch.Generator().manual_seed(args.seed+731)
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
        persistent_workers=args.workers>0,pin_memory=device.type=='cuda',generator=generator)
    valid_loader=DataLoader(valid,batch_size=args.eval_batch_size,shuffle=False,num_workers=args.workers,
        persistent_workers=args.workers>0,pin_memory=device.type=='cuda')
    model=build_model('chexnet_densenet121',pretrained=not args.random_initialization and not bool(resume))
    provenance={'reference_repository':'https://github.com/arnoweng/CheXNet',
        'paper':'https://arxiv.org/abs/1711.05225','pretraining':model.pretraining_provenance,
        'python':platform.python_version(),'torch':str(torch.__version__),
        'source_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        'training_csv_sha256':hashlib.sha256((Path(args.root)/'train.csv').read_bytes()).hexdigest(),
        'validation_csv_sha256':hashlib.sha256((Path(args.root)/'valid.csv').read_bytes()).hexdigest(),
        'labels':LABELS,'training_images':len(train),'training_patients':int(train.df.Subject_id.nunique()),
        'training_unit':'frontal image','medical_pretrained_weights_used':False,'test_data_read':False,
        'original_repo_medical_checkpoint_downloaded':False}
    config={**vars(args),'model':'chexnet_densenet121','size':224,'normalization':'imagenet',
        'preprocessing':CACHE_SIGNATURE,'crop':'ten','aggregation':'frontal','loss':'unweighted_bce_sum_classes',
        'optimizer':'Adam','weight_decay':0.,'class_count':10}
    if device.type=='cuda':model=model.to(device,memory_format=torch.channels_last)
    else:model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999),weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.1,
        patience=args.plateau_patience,threshold=1e-4,min_lr=1e-7)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start,best_auc,best_loss,stale=0,-1.,float('inf'),0
    if resume:
        for key in ['training_csv_sha256','validation_csv_sha256']:
            if provenance[key]!=resume['provenance'][key]:raise ValueError('Resume data identity differs')
        model.load_state_dict(resume['state_dict'],strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        start,best_auc,best_loss,stale=resume['epoch'],resume['best_auc'],resume['best_loss'],resume['stale']
        provenance=resume['provenance']
        random.setstate(resume['random'])
        np.random.set_state(resume['numpy'])
        torch.set_rng_state(resume['torch_rng'])
        generator.set_state(resume['loader_generator'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'device':str(device),'train_images':len(train),'train_studies':int(train.df.Study_id.nunique()),
        'valid_images':len(valid),'valid_studies':int(valid.df.Study_id.nunique()),'initial_lr':args.lr,
        'max_epochs':args.epochs,'minimum_epochs':args.min_epochs}),flush=True)
    for epoch in range(start,args.epochs):
        started=time.time()
        model.train()
        total,seen=0.,0
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for step,(pictures,targets,_) in enumerate(loader):
            pictures=pictures.to(device,non_blocking=True)
            if device.type=='cuda':pictures=pictures.to(memory_format=torch.channels_last)
            targets=targets.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=='cuda'):
                logits=model(pictures).float()
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none').sum(1).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite CheXNet loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total+=loss.item()*len(pictures)
            seen+=len(pictures)
            if step%100==0:print(f'epoch={epoch+1} step={step}/{len(loader)} loss={loss.item():.5f}',flush=True)
        if device.type=='cuda':torch.cuda.synchronize()
        training_seconds=time.time()-started
        peak=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
        result,images=validation(model,valid_loader,device)
        selected=result['ten']
        metrics=selected['metrics']
        scheduler.step(selected['loss'])
        record={'epoch':epoch+1,'loss':total/seen,'training_seconds':training_seconds,'seconds':time.time()-started,
            'train_images':seen,'lr_next':optimizer.param_groups[0]['lr'],'validation_bce':selected['loss'],
            'center':result['center']['metrics'],'peak_training_allocated_mib':peak,**metrics}
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        auc_improved=metrics['macro_auc']>best_auc
        loss_improved=selected['loss']<best_loss
        best_auc=max(best_auc,metrics['macro_auc'])
        best_loss=min(best_loss,selected['loss'])
        stale=0 if loss_improved else stale+1
        checkpoint={'state_dict':model.state_dict(),'epoch':epoch+1,'config':config,'provenance':provenance,
            'metrics':metrics,'center_metrics':result['center']['metrics'],'selection_aggregation':'frontal',
            'validation_bce':selected['loss']}
        if auc_improved:
            save_checkpoint({**checkpoint,'selection_basis':'official_valid_study_macro_auc'},out/'best.pt')
            (out/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            np.savez_compressed(out/'valid_predictions.npz',probabilities=images['ten'],center_images=images['center'],
                study_ids=selected['ids'].astype(str),study_probabilities=selected['probabilities'],targets=selected['targets'])
        if loss_improved:
            save_checkpoint({**checkpoint,'selection_basis':'official_valid_study_bce'},out/'loss_best.pt')
            (out/'loss_best_metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        save_checkpoint({**checkpoint,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'best_auc':best_auc,'best_loss':best_loss,'stale':stale,
            'random':random.getstate(),'numpy':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'loader_generator':generator.get_state()},out/'last.pt')
        if args.early_stopping and epoch+1>=args.min_epochs and stale>=args.early_stopping:
            print('EARLY_STOP validation loss no longer improves',flush=True)
            break
    (out/'complete.json').write_text(json.dumps({'epochs_completed':epoch+1,'best_training_runtime_valid_auc':best_auc,
        'best_validation_bce':best_loss,'official_website_submission_performed':False,
        'selection_bias':'Repeated official-validation selection; not hidden-test confirmation'},indent=2),encoding='utf-8')
    print('CHEXNET_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
