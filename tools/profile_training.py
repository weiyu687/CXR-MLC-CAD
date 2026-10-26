"""Measure CUDA training throughput and allocator peaks; never save a model.

Each batch candidate runs in a fresh process. The complete model, EMA, AMP
backward pass and AdamW state are included. Synthetic inputs isolate compute;
the optional official-train loader additionally measures actual input waits.
This is a training resource probe, not a validation or inference score.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time


def probe(args):
    import torch
    from src.model import build_model
    torch.set_num_threads(4)
    torch.manual_seed(2041)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for resource profiling')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cudnn.benchmark = True
    model = build_model(args.model).cuda().to(memory_format=torch.channels_last).train()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss()
    if args.real_data:
        from src.data import Images
        dataset = Images(args.root,args.cache,'train',args.size,train=True,
                         frontal_training=True,regularize=True,normalization='imagenet',horizontal_flip=True)
        loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True,
                                            num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
        iterator = iter(loader)
    else:
        x = torch.randn(args.batch_size,3,args.size,args.size,device='cuda').to(memory_format=torch.channels_last)
        y = torch.randint(0,2,(args.batch_size,10),device='cuda').float()
    waits = []
    torch.cuda.reset_peak_memory_stats()
    started = None
    for index in range(args.warmup+args.steps):
        if index == args.warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        if args.real_data:
            waited = time.perf_counter()
            x,y,_ = next(iterator)
            if index >= args.warmup:
                waits.append(time.perf_counter()-waited)
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss = criterion(model(x).float(),y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for averaged,current in zip(ema.state_dict().values(),model.state_dict().values()):
                if averaged.is_floating_point():
                    averaged.lerp_(current,.001)
                else:
                    averaged.copy_(current)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    result = {'model':args.model,'size':args.size,'batch_size':args.batch_size,
              'real_data':args.real_data,'workers':args.workers,'steps':args.steps,'warmup':args.warmup,
              'images_per_second':args.batch_size*args.steps/elapsed,'mean_step_seconds':elapsed/args.steps,
              'loader_wait_seconds':sum(waits),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
              'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'device':torch.cuda.get_device_name(),
              'torch':str(torch.__version__),'python':sys.version.split()[0],
              'scope':'Full train stage including EMA and AdamW state; random initialization; no checkpoint saved'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--size',type=int,default=384)
    parser.add_argument('--batches',type=int,nargs='+',default=[24,32,48])
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--real-data',action='store_true')
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',default='data_cache/448')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--memory-fraction',type=float,default=.80,help='Allocator ceiling to avoid Windows shared-memory spill')
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    if args.steps<1 or args.warmup<1 or args.size<32 or args.workers<0 or not 0<args.memory_fraction<=1:
        raise ValueError('Invalid resource probe settings')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Preserve existing resource measurements: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.batch_size is not None:
        if args.batch_size<1:raise ValueError('Invalid batch size')
        result = probe(args)
        out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
        return
    results=[]
    for batch in args.batches:
        target=out.with_name(out.stem+'_b'+str(batch)+'.json')
        command=[sys.executable,'-u','-m','tools.profile_training','--model',args.model,
                 '--size',str(args.size),'--batch-size',str(batch),'--steps',str(args.steps),
                 '--warmup',str(args.warmup),'--workers',str(args.workers),'--root',args.root,
                 '--memory-fraction',str(args.memory_fraction),
                 '--cache',args.cache,'--out',str(target)]
        if args.real_data:command.append('--real-data')
        print('PROBING '+args.model+' batch='+str(batch),flush=True)
        completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if completed.returncode:
            log=target.with_suffix('.log');log.write_text(completed.stdout+completed.stderr,encoding='utf-8')
            result={'batch_size':batch,'failed':True,'exit_code':completed.returncode,'log':str(log)}
        else:
            result=json.loads(target.read_text(encoding='utf-8'))
        results.append(result)
        print(json.dumps(result),flush=True)
        out.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
