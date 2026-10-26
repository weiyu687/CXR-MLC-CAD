"""Run a finite, frozen long-training queue and compare against the incumbent.

Only official train updates model parameters. Official valid selects snapshots
and fixed fusion; test is used only after model/configuration selection freezes.
The two recipes are not an isolated causal ablation of epoch count.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_commands(args):
    base=Path(args.work).resolve()
    common=['--root',str(Path(args.root).resolve()),'--cache',str(Path(args.cache).resolve()),
            '--batch-size',str(args.batch_size),'--workers',str(args.workers),
            '--epochs','36','--patience','0','--min-epochs','24',
            '--warmup-epochs','2','--min-lr-ratio','0.05','--ema-decay','0.999',
            '--no-decay-bias-norm','--cuda-memory-fraction','0.80',
            '--frontal-training','--regularize','--horizontal-flip',
            '--mixup','0.2','--mixup-objective','hard_loss','--positive-weight',
            '--loss','bce','--selection-aggregation','frontal']
    recipes=[('convnext_tiny_448_s2041',[
        '--model','tv_convnext_tiny','--pretrained','--size','448',
        '--lr','0.00005','--head-lr-mult','5','--weight-decay','0.05',
        '--freeze-backbone-epochs','2','--seed','2041']),
        ('effv2s_384_long_s2042',[
        '--model','tv_efficientnet_v2_s','--init',str(Path(args.efficientnet_init).resolve()),
        '--size','384','--lr','0.00001','--head-lr-mult','1','--weight-decay','0.01',
        '--freeze-backbone-epochs','0','--seed','2042'])]
    return [{'name':name,'out':str(base/name),
             'command':[args.train_python,'-u','train.py',*common,*specific,'--out',str(base/name)]}
            for name,specific in recipes]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-python',default=sys.executable)
    parser.add_argument('--target-python',required=True)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--ssd-root',required=True)
    parser.add_argument('--timing-root',required=True)
    parser.add_argument('--work',default='outputs/long_training_20261026')
    parser.add_argument('--out',default='submission/long_training_bundle')
    parser.add_argument('--incumbent',default='submission/round2_bundle')
    parser.add_argument('--efficientnet-init',default='outputs/effv2s_research_full_s2039/best.pt')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    work=Path(args.work).resolve()
    recipes=build_commands(args)
    if not args.execute:
        print(json.dumps(recipes,indent=2))
        return
    for path in [work/'plan.json',Path(args.out),*(Path(recipe['out']) for recipe in recipes)]:
        if path.exists():raise FileExistsError('Preserve existing work: '+str(path))
    work.mkdir(parents=True,exist_ok=True)
    incumbent=Path(args.incumbent).resolve()
    report=read(incumbent/'report.json')
    selection=read(incumbent/'selection.json')['selected']
    checkpoints=selection['checkpoints']
    if [digest(path) for path in checkpoints]!=report['checkpoint_sha256']:
        raise ValueError('Incumbent source checkpoints changed')
    specification=report['ensemble_weight_spec']
    weights_path=work/'incumbent_weights.json'
    weights_path.write_text(json.dumps(specification,indent=2),encoding='utf-8')
    preset={'schema_version':1,'ensemble_presets':[{
        'checkpoints':checkpoints,'checkpoint_sha256':report['checkpoint_sha256'],
        'weights_json':str(weights_path),'weights_json_sha256':digest(weights_path),
        'validation_csv_sha256':digest(Path(args.root)/'valid.csv'),
        'aggregation':report['arguments']['aggregation'],'tta':report['arguments']['tta'],
        'batch_size':1,'expected_validation_macro_auc':report['validation']['macro_auc']}]}
    registry=work/'incumbent_registry.json'
    registry.write_text(json.dumps(preset,indent=2),encoding='utf-8')
    from tools.ensemble_presets import read_presets
    read_presets(registry,args.root)
    source_files=[project/'train.py',*sorted((project/'src').glob('*.py')),Path(__file__)]
    source_hashes={str(path.relative_to(project)):digest(path) for path in source_files}
    source_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=project,capture_output=True,text=True,check=True).stdout.strip()
    profile_evidence={}
    for name in ['profile_conv448_real','profile_eff384_real']:
        path=work/(name+'_b'+str(args.batch_size)+'.json')
        measured=read(path)
        if not measured['real_data'] or measured['batch_size']!=args.batch_size or measured['peak_reserved_mib']>9830:
            raise ValueError('Missing or over-budget full-training resource probe')
        profile_evidence[path.name]={'sha256':digest(path),'measurement':measured}
    plan={'recipes':recipes,'epochs_each':36,'early_stopping':False,'batch_size':args.batch_size,
          'source_commit':source_commit,'source_sha256':source_hashes,'resource_probes':profile_evidence,
          'efficientnet_init_sha256':digest(args.efficientnet_init),
          'incumbent_submission_sha256':digest(incumbent/'submission.csv'),
          'incumbent_macro_auc':report['validation']['macro_auc'],
          'scope':'Two fixed long-training recipes; not an isolated epoch-count ablation; official train only',
          'official_website_submission_performed':False}
    (work/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    status={'stage':'PLAN_FROZEN','started_unix':time.time(),'completed_runs':[]}
    def save_status():
        temporary=work/'status.pending.json'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(work/'status.json')
        print(json.dumps(status),flush=True)
    def execute(command,log):
        with Path(log).open('w',encoding='utf-8') as stream:
            subprocess.run(command,cwd=project,stdout=stream,stderr=subprocess.STDOUT,check=True)
    save_status()
    try:
        for recipe in recipes:
            for name,expected in source_hashes.items():
                if digest(project/name)!=expected:raise ValueError('Frozen training source changed: '+name)
            status.update(stage='TRAINING',current_run=recipe['name']);save_status()
            execute(recipe['command'],work/(recipe['name']+'.log'))
            from tools.finish_research_ablation import completed_training
            audit=completed_training(recipe['out'])
            if audit['epochs_completed']!=36:raise ValueError('Long experiment did not complete 36 epochs')
            (Path(recipe['out'])/'completion_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
            status['completed_runs'].append(audit);save_status()
        status.update(stage='VALIDATION_SELECTION',current_run=None);save_status()
        runs=[str(Path(path).parent) for path in checkpoints]+[recipe['out'] for recipe in recipes]
        execute([args.target_python,'-u','tools/finalize.py','--runs',*runs,
                 '--root',args.root,'--cache',args.cache,'--ssd-root',args.ssd_root,
                 '--timing-root',args.timing_root,'--timing-split','valid',
                 '--samples','100','--latency-statistic','mean','--latency-ms','100',
                 '--batch-size','1','--study-timing','--max-members','3','--weight-grid','quarter',
                 '--ensemble-presets-registry',str(registry),
                 '--work',str(work/'deployment_selection'),'--out',args.out],work/'deployment_selection.log')
        final=read(work/'deployment_selection/complete.json')
        if final['validation']['macro_auc']+1e-12<plan['incumbent_macro_auc']:
            raise RuntimeError('Latency-screened winner is below the incumbent; preserve incumbent delivery')
        status.update(stage='BUNDLE_READY_FOR_REVIEW',bundle=final,
                      official_website_submission_performed=False,completed_unix=time.time())
        save_status()
    except Exception as error:
        status.update(stage='FAILED',error=repr(error));save_status()
        raise


if __name__=='__main__':main()
