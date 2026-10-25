"""Create fixed input-resolution candidates without training or changing weights."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import torch


SIZES=(320,384,448)
MODELS=('tv_densenet121','tv_efficientnet_v2_s','tv_resnet50')


def resolution_variant(source,size):
    if 'optimizer' in source or 'ema' in source:
        raise ValueError('Use a deployed EMA snapshot, not an optimizer/resume checkpoint')
    config=source['config']
    if config['model'] not in MODELS:raise ValueError('Only the audited adaptive-pooling CNN families are supported')
    if size not in SIZES or size==config['size']:
        raise ValueError('Choose a different fixed input size from 320/384/448')
    inference=copy.deepcopy(config);inference['size']=size
    return {'state_dict':{key:value.clone() for key,value in source['state_dict'].items()},
            'config':inference,'training_configuration':copy.deepcopy(source.get('training_configuration',config)),
            'epoch':None,'metrics':None,'holdout_metrics':None,'selection_aggregation':None,
            'selection_basis':'unmeasured_inference_resolution_variant',
            'provenance':copy.deepcopy(source.get('provenance')),
            'construction':{'method':'fixed_inference_resolution','source_input_size':config['size'],
                            'inference_input_size':size,'training_performed':False,
                            'weights_unchanged':True,'batchnorm_recalibrated':False,
                            'source_selected_epoch':source.get('epoch'),
                            'parent_construction':copy.deepcopy(source.get('construction')),
                            'validation_recomputed':False}}


def build_variants(checkpoint,sizes,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Existing resolution experiment preserved: '+str(out))
    if not sizes or len(set(sizes))!=len(sizes):raise ValueError('Provide distinct fixed sizes')
    path=Path(checkpoint).resolve();data=path.read_bytes()
    source=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    # Validate the entire requested grid before creating files.
    if (any(size not in SIZES or size==source['config']['size'] for size in sizes)
            or source['config']['model'] not in MODELS or 'optimizer' in source or 'ema' in source):
        raise ValueError('Unsupported source snapshot or input-size grid')
    project=Path(__file__).resolve().parents[1]
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=project,capture_output=True,text=True,check=True).stdout.strip()
    dirty=bool(subprocess.run(['git','status','--porcelain'],cwd=project,capture_output=True,text=True,check=True).stdout.strip())
    construction={'source_checkpoint':str(path),'source_checkpoint_sha256':hashlib.sha256(data).hexdigest(),
                  'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'source_commit':commit,'source_worktree_dirty':dirty}
    out.mkdir(parents=True)
    runs=[]
    for size in sizes:
        candidate=resolution_variant(source,size)
        candidate['construction'].update(construction)
        folder=out/str(size);folder.mkdir()
        candidate['config']['out']=str(folder.resolve())
        torch.save(candidate,folder/'best.pt')
        (folder/'config.json').write_text(json.dumps(candidate['config'],indent=2),encoding='utf-8')
        (folder/'construction.json').write_text(json.dumps(candidate['construction'],indent=2),encoding='utf-8')
        runs.append({'run':str(folder.resolve()),'size':size,
                     'checkpoint_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()})
        del candidate
    manifest={**construction,'sizes':list(sizes),'runs':runs,
              'scope':'Fixed inference inputs only; source tensors unchanged; no image/label/test access or inherited validation score'}
    (out/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--sizes',nargs='+',type=int,default=[384,448])
    parser.add_argument('--out',required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    print(json.dumps(build_variants(args.checkpoint,args.sizes,args.out),indent=2))


if __name__=='__main__':main()
