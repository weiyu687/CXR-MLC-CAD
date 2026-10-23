"""Print or execute predefined, fixed patient-holdout comparisons.

No experiment starts without --execute. Existing output directories are never
reused. Official validation is reported but does not select checkpoints here.
See docs/RESEARCH_REVIEW.md before expanding the experiment set.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

VARIANTS={
    'bce_baseline':['--loss','bce','--positive-weight','--freeze-backbone-epochs','0'],
    'bce_head_warmup':['--loss','bce','--positive-weight','--freeze-backbone-epochs','2'],
    'asl_head_warmup':['--loss','asl','--freeze-backbone-epochs','2'],
}

def build_commands(args):
    common=['--model','tv_densenet121','--pretrained','--size','320',
            '--batch-size','32','--epochs','12','--patience','4',
            '--lr','0.0001','--head-lr-mult','5','--weight-decay','0.0001',
            '--frontal-training','--regularize','--horizontal-flip',
            '--mixup','0','--mixup-objective','hard_loss',
            '--patient-holdout','0.12','--split-seed','2030','--seed','2036',
            '--selection-aggregation','frontal','--workers',str(args.workers),
            '--root',args.root,'--cache',args.cache]
    return [(variant,Path(args.output_root)/variant,
             [args.python,'-u','train.py',*common,*VARIANTS[variant],
              '--out',str(Path(args.output_root)/variant)]) for variant in args.variants]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',default='data_cache/448')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output-root',default='outputs/research_ablation_s2036')
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),
                        default=['bce_baseline','bce_head_warmup'])
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if len(set(args.variants))!=len(args.variants):raise ValueError('Duplicate variant')
    project=Path(__file__).resolve().parents[1]
    commands=build_commands(args)
    for variant,out,command in commands:
        print(json.dumps({'variant':variant,'out':str(out),'arguments':command},ensure_ascii=True),flush=True)
    if not args.execute:
        print('PRINT_ONLY: no training launched. Add --execute to run the selected fixed comparisons.')
        return
    for _,out,_ in commands:
        resolved=out if out.is_absolute() else project/out
        if resolved.exists():raise FileExistsError('Preserve existing experiment: '+str(resolved))
    for variant,_,command in commands:
        print('START '+variant,flush=True)
        subprocess.run(command,cwd=project,check=True)
        print('COMPLETE '+variant,flush=True)

if __name__=='__main__':main()
