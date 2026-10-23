"""Train from scratch on the official training split, select on study-level AUC."""
import argparse
import copy
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

@torch.inference_mode()
def predict(model, loader, device, tta=False):
    model.eval()
    result = np.zeros((len(loader.dataset.df), 10), dtype=np.float32)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
            p = model(x).float().sigmoid()
            if tta:
                p = (p + model(x.flip(-1)).float().sigmoid())/2
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
    ap.add_argument('--loss', choices=['bce','asl'], default='bce')
    ap.add_argument('--selection-aggregation',choices=['mean','max','frontal','weighted','auto'],default='mean')
    ap.add_argument('--resume')
    ap.add_argument('--init',help='Initialize only from a checkpoint trained on official training data')
    ap.add_argument('--frontal-training',action='store_true')
    ap.add_argument('--regularize',action='store_true')
    ap.add_argument('--mixup',type=float,default=0.)
    ap.add_argument('--weight-decay',type=float,default=0.02)
    ap.add_argument('--positive-weight',action='store_true',help='Square-root inverse positive frequency, capped at 5')
    ap.add_argument('--patient-holdout',type=float,default=0.,help='Official training patients reserved for an independent selection split')
    args = ap.parse_args()
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
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    train = Images(args.root,args.cache,'train',args.size,train=True,
                   frontal_training=args.frontal_training,regularize=args.regularize)
    valid = Images(args.root,args.cache,'valid',args.size)
    holdout=None
    if args.patient_holdout:
        if args.init or args.resume:raise ValueError('Patient holdout must train from scratch to avoid prior checkpoint leakage')
        if not 0<args.patient_holdout<0.5:raise ValueError('Invalid holdout fraction')
        patients=np.sort(train.df.Subject_id.unique())
        rng=np.random.default_rng(args.seed)
        held=set(rng.choice(patients,int(len(patients)*args.patient_holdout),replace=False))
        holdout=Images(args.root,args.cache,'train',args.size)
        holdout.df=holdout.df[holdout.df.Subject_id.isin(held)].reset_index(drop=True)
        train.groups=[g for g in train.groups if train.df.iloc[g[0]].Subject_id not in held]
        assert not set(train.df.iloc[[g[0] for g in train.groups]].Subject_id)&set(holdout.df.Subject_id)
        (out/'holdout_patients.json').write_text(json.dumps(sorted(held)),encoding='utf-8')
    kwargs = dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    tl = DataLoader(train,shuffle=True,drop_last=True,**kwargs)
    vl = DataLoader(valid,shuffle=False,**kwargs)
    hl=DataLoader(holdout,shuffle=False,**kwargs) if holdout is not None else None
    model = build_model(args.model).to(device, memory_format=torch.channels_last)
    if args.init:
        init=torch.load(args.init,map_location='cpu',weights_only=False)
        state=init['state_dict']
        if args.model.endswith('_cap') and not init['config']['model'].endswith('_cap'):
            model.backbone.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')},strict=True)
            with torch.no_grad():
                model.weight.copy_(state['fc.weight']);model.bias.copy_(state['fc.bias'])
        else:
            model.load_state_dict(state)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    criterion = AsymmetricLoss() if args.loss=='asl' else torch.nn.BCEWithLogitsLoss()
    if args.positive_weight:
        if args.loss!='bce':raise ValueError('Positive weighting requires BCE')
        targets=np.stack(train.df.iloc[[g[0] for g in train.groups]].target)
        pos=targets.sum(0)
        weights=np.sqrt((len(targets)-pos)/np.maximum(pos,1)).clip(1,5)
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,dtype=torch.float32,device=device))
    best, start = -1., 0
    if args.resume:
        ck = torch.load(args.resume,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict']); ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start,best = ck['epoch']+1,ck['best']
        random.setstate(ck['random']); np.random.set_state(ck['numpy']); torch.set_rng_state(ck['torch'])
        if 'cuda_rng' in ck and device.type=='cuda': torch.cuda.set_rng_state_all(ck['cuda_rng'])
    print(json.dumps({'device':str(device),'model':args.model,'train_studies':len(train),'valid_images':len(valid),'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(start,args.epochs):
        t0=time.time(); model.train(); total=0
        for step,(x,y,_) in enumerate(tl):
            progress=(epoch+step/len(tl))
            lr=args.lr*min(1.,(progress+0.1)/2)*0.5*(1+math.cos(math.pi*progress/args.epochs))
            for group in optimizer.param_groups: group['lr']=lr
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            if args.mixup>0 and random.random()<0.5:
                lam=float(np.random.beta(args.mixup,args.mixup))
                order=torch.randperm(len(x),device=device)
                x=lam*x+(1-lam)*x[order]
                y=lam*y+(1-lam)*y[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(x)
                loss=criterion(logits.float(),y)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                for ev,mv in zip(ema.state_dict().values(),model.state_dict().values()):
                    if ev.is_floating_point(): ev.lerp_(mv,0.01)
                    else: ev.copy_(mv)
            total+=loss.item()
            if step%100==0:
                print(f'epoch={epoch+1} step={step}/{len(tl)} loss={loss.item():.4f}',flush=True)
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
            _,hprob,hy=aggregate(holdout.df,hp,'frontal')
            holdout_metrics=evaluate(hy,hprob)
        # A fixed patient holdout selects checkpoints without repeated official-valid tuning.
        selection_score=holdout_metrics['macro_auc'] if holdout_metrics else metrics['macro_auc']
        record={'epoch':epoch+1,'loss':total/len(tl),'seconds':time.time()-t0,
                'aggregation':selected,'aggregation_macro_auc':{k:v['macro_auc'] for k,v in scores.items()},
                'holdout':holdout_metrics,'selection_score':selection_score,**metrics}
        with (out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if selection_score>best:
            best=selection_score
            torch.save({'state_dict':{k:v.cpu() for k,v in ema.state_dict().items()},'config':vars(args),'epoch':epoch+1,
                        'selection_aggregation':selected,'metrics':metrics,'holdout_metrics':holdout_metrics,'provenance':provenance},out/'best.pt')
            np.savez_compressed(out/'valid_predictions.npz',probabilities=predictions,study_probabilities=p,targets=y,study_ids=ids)
            (out/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        torch.save({'state_dict':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch,'best':best,'random':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},out/'last.pt')
    print('TRAINING_COMPLETE best_auc='+str(best),flush=True)
if __name__=='__main__': main()
