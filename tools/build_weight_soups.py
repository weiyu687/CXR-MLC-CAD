"""Build a small, explicit pairwise parameter-average experiment.

This is a checkpoint-averaging experiment inspired by model soups, not a
reproduction of that paper. Floating parameters AND floating buffers are
averaged; BatchNorm integer counters use max. No BN recalibration or data fitting
is performed. Validation scores must be recomputed, never copied from parents.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import torch


def average_checkpoints(first, second, fraction_first):
    if not 0 < fraction_first < 1:
        raise ValueError('Both source checkpoints must contribute')
    for key, default in [('model', None), ('size', None), ('normalization', 'legacy')]:
        if first['config'].get(key, default) != second['config'].get(key, default):
            raise ValueError('Checkpoint model/input settings differ: ' + key)
    upstream=lambda ck: ((ck.get('provenance') or {}).get('pretraining') or {}).get('sha256')
    if upstream(first) != upstream(second):
        raise ValueError('Generic pretraining sources differ')
    left=first['state_dict'];right=second['state_dict']
    if set(left)!=set(right):
        raise ValueError('State keys differ')
    combined={}
    for key, value in left.items():
        other=right[key]
        if value.shape!=other.shape or value.dtype!=other.dtype:
            raise ValueError('State shape/dtype differs: ' + key)
        if value.is_floating_point():
            if not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise ValueError('Nonfinite source tensor: ' + key)
            combined[key]=(value.double()*fraction_first+other.double()*(1-fraction_first)).to(value.dtype)
        elif key.endswith('num_batches_tracked'):
            combined[key]=torch.maximum(value,other).clone()
        elif torch.equal(value,other):
            combined[key]=value.clone()
        else:
            raise ValueError('Unequal nonfloating state cannot be averaged: ' + key)
    return {'state_dict':combined,'config':copy.deepcopy(first['config']),
            'epoch':None,'metrics':None,'holdout_metrics':None,
            'selection_aggregation':'frontal', 'selection_basis':'unmeasured_parameter_average',
            'provenance':copy.deepcopy(first.get('provenance'))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',nargs=2,required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--fractions-first',nargs='+',type=float,default=[.25,.5,.75])
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError('Existing soup experiment preserved: '+str(out))
    if len(set(args.fractions_first))!=len(args.fractions_first):raise ValueError('Duplicate fraction')
    source=[];parents=[]
    for name in args.checkpoint:
        path=Path(name).resolve()
        data=path.read_bytes()
        checkpoint=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
        if 'optimizer' in checkpoint or 'ema' in checkpoint:
            raise ValueError('Use deployed EMA snapshots, not raw optimizer/resume checkpoints')
        source.append(checkpoint)
        parents.append({'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),
                        'epoch':checkpoint.get('epoch'),'metrics':checkpoint.get('metrics')})
    # Validate all fractions and both sources before creating any outputs.
    if not args.fractions_first or any(not 0<value<1 for value in args.fractions_first):
        raise ValueError('Fractions must be strictly between zero and one')
    probe=average_checkpoints(*source,args.fractions_first[0]);del probe
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    dirty=bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip())
    builder_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True)
    runs=[]
    for i,checkpoint in enumerate(source):
        folder=out/('source_'+str(i+1));folder.mkdir()
        torch.save(checkpoint,folder/'best.pt')
        runs.append(str(folder))
    for index,fraction in enumerate(args.fractions_first):
        checkpoint=average_checkpoints(*source,fraction)
        checkpoint['construction']={'method':'pairwise_parameter_average',
            'source_commit':commit,'source_worktree_dirty':dirty,'builder_sha256':builder_hash,
            'sources':parents,'fraction_first':fraction,
            'floating_buffers':'arithmetic mean with the same fraction',
            'integer_batchnorm_counters':'maximum','batchnorm_recalibrated':False,
            'validation_recomputed':False,
            'paper':'https://proceedings.mlr.press/v162/wortsman22a.html'}
        folder=out/('mix_'+str(index+1));folder.mkdir()
        torch.save(checkpoint,folder/'best.pt')
        runs.append(str(folder))
    manifest={'source_commit':commit,'source_worktree_dirty':dirty,'builder_sha256':builder_hash,
              'sources':parents,'fractions_first':args.fractions_first,
              'runs':runs,'scope':'Unmeasured fixed-parameter candidates; no image/test access or copied performance claims'}
    (out/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
