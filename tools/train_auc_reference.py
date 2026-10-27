"""Paired BCE/PESG-AUC fine-tuning from the same held-out-patient checkpoint.

Both arms reset the classifier, use identical positive/negative coverage,
augmentation and PESG primal updates. Only the objective/auxiliary state differ.
The coverage sampler is a documented multi-label adaptation of DualSampler.
No validation or test examples participate in training or class sampling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.auc_optimizer import PESG
from src.auc_training import MultiLabelAUCMV2, CoverageBatchSampler
from src.data import Images, aggregate
from src.metrics import evaluate
from src.model import build_model
from train import predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init',required=True)
    parser.add_argument('--objective',choices=['auc','bce'],required=True)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--micro-batch',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2049)
    parser.add_argument('--lr',type=float,default=.01)
    parser.add_argument('--epoch-decay',type=float,default=.002)
    parser.add_argument('--margin',type=float,default=1.)
    args=parser.parse_args()
    if args.epochs<2 or not 1<=args.micro_batch<=args.batch_size or args.batch_size<20:
        raise ValueError('Invalid AUC experiment dimensions')
    source=Path(args.init).resolve()
    original=torch.load(source,map_location='cpu',weights_only=False)
    cfg=original['config']
    held_path=source.parent/'holdout_patients.json'
    if not cfg.get('patient_holdout') or cfg.get('loss')!='bce' or not held_path.is_file():
        raise ValueError('Use BCE pretraining that excluded a saved patient holdout')
    held=set(json.loads(held_path.read_text(encoding='utf-8')))
    out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve previous AUC experiment')
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.8)
    else:torch.backends.mkldnn.enabled=False
    memory_format=torch.channels_last if device.type=='cuda' else torch.contiguous_format
    normalization=cfg.get('normalization','legacy')
    if normalization!='imagenet':raise ValueError('This paired recipe expects ImageNet-normalized BCE pretraining')
    training=Images(args.root,args.cache,'train',cfg['size'],train=True,frontal_training=True,normalization=normalization)
    training.groups=[group for group in training.groups if training.df.iloc[group[0]].Subject_id not in held]
    training.transform=T.Compose([T.Resize((cfg['size'],cfg['size'])),T.RandomHorizontalFlip(),
        T.RandomRotation(7),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    labels=np.stack(training.df.iloc[[group[0] for group in training.groups]].target)
    holdout=Images(args.root,args.cache,'train',cfg['size'],normalization=normalization)
    holdout.df=holdout.df[holdout.df.Subject_id.isin(held)].reset_index(drop=True)
    validation=Images(args.root,args.cache,'valid',cfg['size'],normalization=normalization)
    train_patients=set(training.df.iloc[[group[0] for group in training.groups]].Subject_id)
    if train_patients&held or (train_patients|held)&set(validation.df.Subject_id):
        raise ValueError('Patient split leakage')
    sampler=CoverageBatchSampler(labels,args.batch_size,args.seed)
    generator=torch.Generator().manual_seed(args.seed+731)
    loader=DataLoader(training,batch_sampler=sampler,num_workers=args.workers,
        persistent_workers=args.workers>0,pin_memory=device.type=='cuda',generator=generator)
    evaluation_kwargs=dict(batch_size=args.micro_batch,num_workers=args.workers,
        persistent_workers=args.workers>0,pin_memory=device.type=='cuda')
    hold_loader=DataLoader(holdout,shuffle=False,**evaluation_kwargs)
    valid_loader=DataLoader(validation,shuffle=False,**evaluation_kwargs)
    model=build_model(cfg['model'])
    model.load_state_dict(original['state_dict'],strict=True)
    if not isinstance(getattr(model,'fc',None),torch.nn.Linear):
        raise ValueError('This faithful head-reset comparison currently supports a linear ResNet fc')
    model.fc.reset_parameters()
    head_hash=hashlib.sha256(b''.join(value.detach().numpy().tobytes() for value in model.fc.state_dict().values())).hexdigest()
    model=model.to(device,memory_format=memory_format)
    criterion=MultiLabelAUCMV2(margin=args.margin).to(device) if args.objective=='auc' else torch.nn.BCEWithLogitsLoss()
    auxiliary=criterion if args.objective=='auc' else SimpleNamespace(a=None,b=None,alpha=None,margin=args.margin)
    parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer=PESG(parameters,auxiliary,lr=args.lr,mode='sgd',momentum=.9,weight_decay=0.,
        epoch_decay=args.epoch_decay,device=device,verbose=False)
    # Paper Algorithm 1 initializes v_ref to the pretrained v_0; the current
    # upstream default random references are explicitly corrected here.
    if args.epoch_decay:
        reference_parameters=parameters+([criterion.a,criterion.b] if args.objective=='auc' else [])
        for reference,parameter in zip(optimizer.model_ref,reference_parameters):
            reference.copy_(parameter.detach())
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    configuration={**vars(args),'model':cfg['model'],'size':cfg['size'],'normalization':normalization,
        'patient_holdout':cfg['patient_holdout'],'aggregation':'frontal','selection_aggregation':'frontal',
        'selection_basis':'training_patient_holdout_auc','classifier_reset':True}
    provenance={'source_commit':commit,'torch':str(torch.__version__),'initial_checkpoint_sha256':sha(source),
        'initial_classifier_sha256':head_hash,'pretraining':original.get('provenance',{}).get('pretraining'),
        'held_patient_file_sha256':sha(held_path),'paper':'https://arxiv.org/abs/2012.03173',
        'sampler':'Multi-label positive/negative coverage; adaptation of binary DualSampler',
        'criterion':'AUCMLoss v2 / PESG' if args.objective=='auc' else 'BCE / identical PESG primal updates',
        'paper_reference_initialization':True,'training_studies':len(training),'test_data_read':False}
    (out/'config.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (out/'holdout_patients.json').write_text(json.dumps(sorted(held)),encoding='utf-8')
    del original
    best=-1.
    print(json.dumps({'objective':args.objective,'device':str(device),'training_studies':len(training),
        'optimizer_steps_per_epoch':len(sampler),'effective_batch':args.batch_size,'micro_batch':args.micro_batch}),flush=True)
    for epoch in range(args.epochs):
        if epoch in {args.epochs//2,3*args.epochs//4}:
            if args.epoch_decay:optimizer.update_regularizer(decay_factor=3.)
            else:optimizer.update_lr(decay_factor=3.)
        sampler.set_epoch(epoch)
        model.train()
        started=time.time()
        total=0.
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for step,(pictures,targets,_) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            criterion.zero_grad(set_to_none=True)
            pictures=pictures.to(device,non_blocking=True)
            targets=targets.to(device,non_blocking=True)
            counts=(targets.sum(0),(1-targets).sum(0))
            for begin in range(0,len(pictures),args.micro_batch):
                x=pictures[begin:begin+args.micro_batch].to(memory_format=memory_format)
                y=targets[begin:begin+args.micro_batch]
                with torch.autocast(device.type,enabled=device.type=='cuda'):
                    logits=model(x).float()
                    if args.objective=='auc':
                        loss=criterion(logits.sigmoid(),y,counts,len(x)/len(pictures))
                    else:
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum')/targets.numel()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training objective')
                scaler.scale(loss).backward()
                total+=loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            if step%100==0:print(f'epoch={epoch+1} step={step}/{len(loader)} objective={total/(step+1):.5f}',flush=True)
        training_seconds=time.time()-started
        vp=predict(model,valid_loader,device)
        ids,probabilities,targets=aggregate(validation.df,vp,'frontal')
        metrics=evaluate(targets,probabilities)
        hp=predict(model,hold_loader,device)
        _,hold_p,hold_y=aggregate(holdout.df,hp,'frontal')
        hold_metrics=evaluate(hold_y,hold_p)
        record={'epoch':epoch+1,'loss':total/len(loader),'training_seconds':training_seconds,
            'seconds':time.time()-started,'holdout':hold_metrics,'selection_score':hold_metrics['macro_auc'],
            'lr':optimizer.param_groups[0]['lr'],**metrics,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None}
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        improved=record['selection_score']>best
        best=max(best,record['selection_score'])
        checkpoint={'state_dict':model.state_dict(),'config':configuration,'provenance':provenance,
            'epoch':epoch+1,'metrics':metrics,'holdout_metrics':hold_metrics,'selection_aggregation':'frontal'}
        torch.save({**checkpoint,'optimizer':optimizer.state_dict(),'criterion':criterion.state_dict(),
            'scaler':scaler.state_dict(),'best':best,'random':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'loader_generator':generator.get_state()},out/'last.pt')
        if improved:
            torch.save(checkpoint,out/'best.pt')
            np.savez_compressed(out/'valid_predictions.npz',study_ids=ids.astype(str),study_probabilities=probabilities,targets=targets)
            (out/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (out/'complete.json').write_text(json.dumps({'objective':args.objective,'epochs_completed':args.epochs,
        'best_holdout_auc':best,'official_website_submission_performed':False},indent=2),encoding='utf-8')
    print('AUC_REFERENCE_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
