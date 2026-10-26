"""Train on official data with random or audited non-medical ImageNet initialization."""
import argparse
import copy
import hashlib
import json
import math
import random
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data import Images, aggregate
from src.model import build_model, AsymmetricLoss
from src.metrics import evaluate
from src.inference import forward_probabilities
from src.training import (configure_train_stage, mixed_target_loss, save_validation_candidate,
                          learning_rate, optimizer_groups)

@torch.inference_mode()
def predict(model, loader, device, tta=False):
    model.eval()
    result = np.zeros((len(loader.dataset.df), 10), dtype=np.float32)
    for x, _, idx in loader:
        p=forward_probabilities(model,x,device,tta)
        result[idx.numpy()] = p.cpu().numpy()
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='raw_data/cxr_first')
    ap.add_argument('--cache', default='data_cache/448')
    ap.add_argument('--out', default='outputs/resnet18_s42')
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--size', type=int, default=320)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--loss', choices=['bce','asl','chexfusion_asl'], default='bce')
    ap.add_argument('--selection-aggregation',choices=['mean','max','frontal','weighted','auto'],default='mean')
    ap.add_argument('--holdout-aggregation', choices=['mean','max','frontal','weighted'], default='frontal')
    ap.add_argument('--resume')
    ap.add_argument('--init',help='Initialize from an existing competition checkpoint')
    ap.add_argument('--pretrained',action='store_true',help='Audited non-medical ImageNet weights permitted by official PDF article 1')
    ap.add_argument('--normalization',choices=['legacy','imagenet'],default='legacy')
    ap.add_argument('--frontal-training',action='store_true')
    ap.add_argument('--image-training', action='store_true', help='Visit every training image, including lateral views, each epoch')
    ap.add_argument('--augmentation', choices=['legacy','chexfusion'], default='legacy')
    ap.add_argument('--regularize',action='store_true')
    ap.add_argument('--mixup',type=float,default=0.)
    ap.add_argument('--mixup-objective',choices=['hard_loss','soft_target'],default='hard_loss',
                    help='Mix hard-target losses; soft_target reproduces the original ASL/Mixup recipe')
    ap.add_argument('--freeze-backbone-epochs',type=int,default=0,
                    help='Train only the classifier first, keeping backbone BatchNorm/dropout frozen')
    ap.add_argument('--weight-decay',type=float,default=0.02)
    ap.add_argument('--positive-weight',action='store_true',help='Square-root inverse positive frequency, capped at 5')
    ap.add_argument('--patient-holdout',type=float,default=0.,help='Official training patients reserved for an independent selection split')
    ap.add_argument('--split-seed',type=int,default=2030,help='Patient split seed, independent of model initialization')
    ap.add_argument('--head-lr-mult',type=float,help='Default 5 for a newly initialized ImageNet classifier, otherwise 1')
    ap.add_argument('--patience',type=int,default=0,help='Stop after this many epochs without an improvement in the selection split; 0 disables')
    ap.add_argument('--min-epochs',type=int,default=0,help='Do not apply early stopping before this many completed epochs')
    ap.add_argument('--warmup-epochs',type=float,default=2.)
    ap.add_argument('--min-lr-ratio',type=float,default=0.)
    ap.add_argument('--ema-decay',type=float,default=.99)
    ap.add_argument('--no-decay-bias-norm',action='store_true')
    ap.add_argument('--cuda-memory-fraction',type=float,default=1.,help='Per-process CUDA allocator ceiling')
    ap.add_argument('--horizontal-flip',action=argparse.BooleanOptionalAction,default=True,help='Training only; labels have no side-specific classes')
    args = ap.parse_args()
    if args.image_training and args.frontal_training:
        raise ValueError('Image training visits all views; do not combine with frontal-only study sampling')
    if args.augmentation == 'chexfusion' and (args.regularize or not args.horizontal_flip):
        raise ValueError('Use the published augmentation as a complete sequence')
    if args.pretrained:
        if args.init or args.resume:raise ValueError('Choose pretrained initialization or a competition checkpoint')
        args.normalization='imagenet'
    if args.init:
        initial_config=torch.load(args.init,map_location='cpu',weights_only=False)['config']
        args.normalization=initial_config.get('normalization','legacy')
    if args.resume:
        resume_metadata=torch.load(args.resume,map_location='cpu',weights_only=False)
        resume_config=resume_metadata.get('config')
        if resume_config is None:
            config_file=Path(args.resume).parent/'config.json'
            resume_config=json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
        if resume_config:
            if resume_config['model']!=args.model:raise ValueError('Resume model differs from the stored architecture')
            args.normalization=resume_config.get('normalization','legacy')
            if args.head_lr_mult is None:args.head_lr_mult=resume_config.get('head_lr_mult',1.)
            args.mixup_objective=resume_config.get('mixup_objective','soft_target')
            args.freeze_backbone_epochs=resume_config.get('freeze_backbone_epochs',0)
            args.split_seed=resume_config.get('split_seed',resume_config.get('seed',args.seed))
            for setting, default in [('warmup_epochs',2.),('min_lr_ratio',0.),('ema_decay',.99),
                                     ('no_decay_bias_norm',False),('min_epochs',0)]:
                setattr(args,setting,resume_config.get(setting,default))
        del resume_metadata
    if args.head_lr_mult is None:args.head_lr_mult=5. if args.pretrained else 1.
    if not 0<args.head_lr_mult<=100:raise ValueError('Invalid classifier learning-rate multiplier')
    if not 0<=args.freeze_backbone_epochs<args.epochs:raise ValueError('Invalid classifier warmup duration')
    if not 0<=args.min_epochs<=args.epochs or args.patience<0:raise ValueError('Invalid early stopping duration')
    if not 0<=args.ema_decay<1:raise ValueError('Invalid EMA decay')
    if not 0<args.cuda_memory_fraction<=1:raise ValueError('Invalid CUDA memory fraction')
    learning_rate(0,args.epochs,args.lr,args.warmup_epochs,args.min_lr_ratio)
    def git_output(arguments):
        try:
            result=subprocess.run(['git',*arguments],capture_output=True,text=True,timeout=5)
            return result.stdout.strip() if result.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):return None
    import timm
    import PIL
    status=git_output(['status','--porcelain'])
    provenance={'source_commit':git_output(['rev-parse','HEAD']),
                'source_worktree_dirty':bool(status) if status is not None else None,
                'python':platform.python_version(),'torch':str(torch.__version__),
                'timm':timm.__version__,'Pillow':PIL.__version__}
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    train = Images(args.root,args.cache,'train',args.size,train=True,study_sampling=not args.image_training,
                   frontal_training=args.frontal_training,regularize=args.regularize,normalization=args.normalization,
                   horizontal_flip=args.horizontal_flip)
    if args.augmentation == 'chexfusion':
        if args.normalization != 'imagenet':
            raise ValueError('The CheXFusion augmentation uses ImageNet normalization')
        from src.paper_augmentation import CheXFusionAugment
        train.transform = CheXFusionAugment(args.size)
    valid = Images(args.root,args.cache,'valid',args.size,normalization=args.normalization)
    holdout=None
    if args.patient_holdout:
        if args.init or args.resume:raise ValueError('Patient holdout must train from scratch to avoid prior checkpoint leakage')
        if not 0<args.patient_holdout<0.5:raise ValueError('Invalid holdout fraction')
        patients=np.sort(train.df.Subject_id.unique())
        rng=np.random.default_rng(args.split_seed)
        held=set(rng.choice(patients,int(len(patients)*args.patient_holdout),replace=False))
        holdout=Images(args.root,args.cache,'train',args.size,normalization=args.normalization)
        holdout.df=holdout.df[holdout.df.Subject_id.isin(held)].reset_index(drop=True)
        if args.image_training:
            train.df=train.df[~train.df.Subject_id.isin(held)].reset_index(drop=True)
            train.groups=[g.index.tolist() for _,g in train.df.groupby('Study_id',sort=False)]
        else:
            train.groups=[g for g in train.groups if train.df.iloc[g[0]].Subject_id not in held]
        assert not set(train.df.iloc[[g[0] for g in train.groups]].Subject_id)&set(holdout.df.Subject_id)
        (out/'holdout_patients.json').write_text(json.dumps(sorted(held)),encoding='utf-8')
    kwargs = dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    training_random = {'generator': torch.Generator().manual_seed(args.seed+731)} if args.augmentation=='chexfusion' else {}
    tl = DataLoader(train,shuffle=True,drop_last=True,**training_random,**kwargs)
    vl = DataLoader(valid,shuffle=False,**kwargs)
    hl=DataLoader(holdout,shuffle=False,**kwargs) if holdout is not None else None
    model = build_model(args.model,pretrained=args.pretrained).to(device, memory_format=torch.channels_last)
    if args.pretrained:
        provenance['pretraining']=model.pretraining_provenance
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    if args.init:
        provenance['initial_checkpoint_sha256']=hashlib.sha256(Path(args.init).read_bytes()).hexdigest()
        init=torch.load(args.init,map_location='cpu',weights_only=False)
        if init.get('provenance',{}).get('pretraining'):
            provenance['pretraining']=init['provenance']['pretraining']
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        state=init['state_dict']
        if args.model.endswith('_cap') and not init['config']['model'].endswith('_cap'):
            model.backbone.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
            with torch.no_grad():
                model.weight.copy_(state['fc.weight']);model.bias.copy_(state['fc.bias'])
        else:
            model.load_state_dict(state)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    original_trainable={name for name,param in model.named_parameters() if param.requires_grad}
    parameters=optimizer_groups(model,args.head_lr_mult,args.no_decay_bias_norm)
    optimizer = torch.optim.AdamW(parameters,lr=args.lr,weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    criterion = AsymmetricLoss() if args.loss=='asl' else torch.nn.BCEWithLogitsLoss()
    if args.loss == 'chexfusion_asl':
        from src.paper_losses import CheXFusionWeightedASL
        targets = np.stack(train.df.target if args.image_training else train.df.iloc[[g[0] for g in train.groups]].target)
        criterion = CheXFusionWeightedASL(targets.sum(0), len(targets)).to(device)
        provenance['loss_reference'] = {
            'paper': 'https://arxiv.org/abs/2308.03968',
            'author_file': 'https://github.com/dongkyunk/CheXFusion/blob/main/model/loss.py',
            'class_counts': targets.sum(0).tolist(), 'total_count': len(targets),
            'frequency_unit': 'train images' if args.image_training else 'train studies',
            'gamma_positive': 1, 'gamma_negative': 4, 'clip': .05,
            'focusing_in_autograd': True, 'reduction': 'mean',
            'heldout_patients_excluded': holdout is not None}
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    if args.positive_weight:
        if args.loss!='bce':raise ValueError('Positive weighting requires BCE')
        targets=np.stack(train.df.target if args.image_training else train.df.iloc[[g[0] for g in train.groups]].target)
        pos=targets.sum(0)
        weights=np.sqrt((len(targets)-pos)/np.maximum(pos,1)).clip(1,5)
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,dtype=torch.float32,device=device))
    best, start,stale_epochs = -1., 0,0
    if args.resume:
        ck = torch.load(args.resume,map_location='cpu',weights_only=False)
        previous_provenance=ck.get('provenance')
        if previous_provenance is None:
            provenance_file=Path(args.resume).parent/'provenance.json'
            previous_provenance=json.loads(provenance_file.read_text(encoding='utf-8')) if provenance_file.exists() else {}
        if previous_provenance.get('pretraining'):
            provenance['pretraining']=previous_provenance['pretraining']
            (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        model.load_state_dict(ck['state_dict']); ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start,best = ck['epoch']+1,ck['best']
        stale_epochs=ck.get('stale_epochs',0)
        random.setstate(ck['random']); np.random.set_state(ck['numpy']); torch.set_rng_state(ck['torch'])
        if 'cuda_rng' in ck and device.type=='cuda': torch.cuda.set_rng_state_all(ck['cuda_rng'])
    print(json.dumps({'device':str(device),'model':args.model,'train_examples':len(train),
        'training_unit':'image' if args.image_training else 'study','train_studies':len(train.groups),
        'valid_images':len(valid),'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(start,args.epochs):
        t0=time.time(); total=0
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        head_only=epoch<args.freeze_backbone_epochs
        configure_train_stage(model,head_only,original_trainable)
        print('TRAINING_STAGE '+('classifier_warmup' if head_only else 'full_finetune'),flush=True)
        for step,(x,y,_) in enumerate(tl):
            progress=(epoch+step/len(tl))
            lr=learning_rate(progress,args.epochs,args.lr,args.warmup_epochs,args.min_lr_ratio)
            for group in optimizer.param_groups: group['lr']=lr*group.get('lr_scale',1.)
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            paired_targets=None;lam=None
            if args.mixup>0 and random.random()<0.5:
                lam=float(np.random.beta(args.mixup,args.mixup))
                order=torch.randperm(len(x),device=device)
                x=lam*x+(1-lam)*x[order]
                paired_targets=y[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(x)
                loss=mixed_target_loss(criterion,logits.float(),y,paired_targets,lam,args.mixup_objective)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                for ev,mv in zip(ema.state_dict().values(),model.state_dict().values()):
                    if ev.is_floating_point(): ev.lerp_(mv,1-args.ema_decay)
                    else: ev.copy_(mv)
            total+=loss.item()
            if step%100==0:
                print(f'epoch={epoch+1} step={step}/{len(tl)} loss={loss.item():.4f}',flush=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        training_seconds=time.time()-t0
        training_peak_allocated=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None
        training_peak_reserved=torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None
        predictions=predict(ema,vl,device)
        modes=['mean','max','frontal','weighted'] if args.selection_aggregation=='auto' else [args.selection_aggregation]
        scores={}
        for mode in modes:
            _,mode_p,mode_y=aggregate(valid.df,predictions,mode)
            scores[mode]=evaluate(mode_y,mode_p)
        selected=max(scores,key=lambda k:scores[k]['macro_auc'])
        ids,p,y=aggregate(valid.df,predictions,selected)
        metrics=scores[selected]
        holdout_metrics=None
        if hl is not None:
            hp=predict(ema,hl,device)
            _,hprob,hy=aggregate(holdout.df,hp,args.holdout_aggregation)
            holdout_metrics=evaluate(hy,hprob)
        # A fixed patient holdout selects checkpoints without repeated official-valid tuning.
        selection_score=holdout_metrics['macro_auc'] if holdout_metrics else metrics['macro_auc']
        record={'epoch':epoch+1,'loss':total/len(tl),'seconds':time.time()-t0,
                'training_seconds':training_seconds,'validation_seconds':time.time()-t0-training_seconds,
                'train_images_per_second':len(tl)*args.batch_size/training_seconds,
                'training_peak_allocated_mib':training_peak_allocated,'training_peak_reserved_mib':training_peak_reserved,
                'optimizer_steps':len(tl),'backbone_lr_last':lr,
                'aggregation':selected,'aggregation_macro_auc':{k:v['macro_auc'] for k,v in scores.items()},
                'holdout':holdout_metrics,'selection_score':selection_score,**metrics}
        with (out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        save_validation_candidate(out, {'state_dict':ema.state_dict(),'epoch':epoch+1,
                                        'config':vars(args),'provenance':provenance}, record)
        if selection_score>best:
            best=selection_score
            stale_epochs=0
            torch.save({'state_dict':{k:v.cpu() for k,v in ema.state_dict().items()},'config':vars(args),'epoch':epoch+1,
                        'selection_aggregation':selected,'metrics':metrics,'holdout_metrics':holdout_metrics,'provenance':provenance},out/'best.pt')
            np.savez_compressed(out/'valid_predictions.npz',probabilities=predictions,study_probabilities=p,targets=y,study_ids=ids)
            (out/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        else:stale_epochs+=1
        torch.save({'state_dict':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch,'best':best,'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    'config':vars(args),'provenance':provenance,'stale_epochs':stale_epochs},out/'last.pt')
        if args.patience and epoch+1>=args.min_epochs and stale_epochs>=args.patience:
            print('EARLY_STOP selection split did not improve for '+str(stale_epochs)+' epochs',flush=True)
            break
    print('TRAINING_COMPLETE best_auc='+str(best),flush=True)
if __name__=='__main__': main()
