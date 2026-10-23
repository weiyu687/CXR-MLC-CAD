"""Select, time, package and independently reproduce completed experiments.

Never submits to the competition website. Local timing is not T4 certification.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--cache',required=True)
    ap.add_argument('--ssd-root',required=True,help='Verified test-only original-image copy on local SSD')
    ap.add_argument('--work',default='outputs/final_selection')
    ap.add_argument('--out',default='submission/final_bundle')
    ap.add_argument('--samples',type=int,default=60)
    ap.add_argument('--latency-ms',type=float,default=100)
    ap.add_argument('--latency-statistic',choices=['max','mean'],default='max')
    ap.add_argument('--batch-size',type=int,default=64,help='Use 1 for exact alignment with the official serial cost protocol')
    ap.add_argument('--study-timing',action='store_true')
    ap.add_argument('--timing-root',help='Use a verified train/valid SSD copy to screen computation before final testing')
    ap.add_argument('--timing-split',choices=['train','valid','test'],default='test')
    args=ap.parse_args()
    project=Path(__file__).resolve().parents[1];os.chdir(project)
    if sys.version_info[:2]!=(3,12) or importlib.metadata.version('torch').split('+')[0]!='2.10.0':
        raise RuntimeError('Use the verified isolated Python 3.12 / PyTorch 2.10 runtime')
    if not 0<args.latency_ms<=100 or args.samples<10:raise ValueError('Invalid timing sample budget')
    work=Path(args.work).resolve();bundle=Path(args.out).resolve()
    if bundle.exists():raise FileExistsError('Existing bundle preserved; choose a new output path')
    work.mkdir(parents=True,exist_ok=False)
    root=Path(args.root).resolve();ssd=Path(args.ssd_root).resolve()
    if hashlib.sha256((root/'test.csv').read_bytes()).digest()!=hashlib.sha256((ssd/'test.csv').read_bytes()).digest():
        raise ValueError('SSD test metadata differs from the official dataset')
    runs=[Path(run).resolve() for run in args.runs]
    if len(set(runs))!=len(runs):raise ValueError('Duplicate experiment directories')
    for run in runs:
        if not (run/'best.pt').is_file():raise FileNotFoundError(run/'best.pt')
    def execute(arguments,log,cwd=project):
        print('RUN '+str(log),flush=True)
        with Path(log).open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,*map(str,arguments)],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Stage failed ({result.returncode}); see {log}')
    candidates=[]
    for tta in [False,True]:
        exports=[]
        for i,run in enumerate(runs):
            out=work/'exports'/('tta' if tta else 'plain')/f'{i:02d}_{run.name}'
            command=['predict.py','--checkpoint',run/'best.pt','--root',root,'--cache',args.cache,
                     '--out',out,'--workers','0','--batch-size',str(args.batch_size),'--export-images']
            if tta:command.append('--tta')
            execute(command,work/f'export_{int(tta)}_{i}.log')
            report=read(out/'report.json')
            if report['runtime']['device']!='cuda':raise RuntimeError('Final GPU export unexpectedly used CPU')
            exports.append(out/'predictions.npz')
        selection=work/f'ensembles_{int(tta)}.json'
        execute(['tools/select_ensemble.py','--runs',*runs,'--prediction-files',*exports,
                 '--root',root,'--out',selection],work/f'ensembles_{int(tta)}.log')
        candidates.extend(read(selection)['candidates'])
    candidates.sort(key=lambda c:(-c['metrics']['macro_auc'],-c['metrics']['macro_f1_tuned'],
                                  len(c['checkpoints'])*(2 if c['tta'] else 1),c['model_bytes']))
    (work/'ranked_candidates.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    timing_cache={};selected=None;chosen_timing=None
    for candidate in candidates:
        key=(tuple(candidate['checkpoints']),candidate['tta'],candidate['aggregation'] if args.study_timing else None)
        if key not in timing_cache:
            path=work/f'latency_{len(timing_cache):02d}.json'
            command=['tools/benchmark.py','--checkpoint',*candidate['checkpoints'],'--root',args.timing_root or ssd,
                     '--split',args.timing_split,
                     '--samples',str(args.samples),'--out',path]
            if candidate['tta']:command.append('--tta')
            if args.study_timing:command.extend(['--study-level','--aggregation',candidate['aggregation']])
            execute(command,path.with_suffix('.log'))
            timing_cache[key]=(path,read(path))
        path,timing=timing_cache[key]
        if timing[args.latency_statistic+'_ms']<=args.latency_ms:
            selected=dict(candidate);selected['checkpoint_sha256']=timing['checkpoint_sha256']
            chosen_timing=path;break
    if selected is None:raise RuntimeError('No candidate passed the local sampled latency limit; inspect timing reports')
    evidence={'selected':selected,'local_latency_report':str(chosen_timing),
              'local_sample_limit_ms':args.latency_ms,'local_timing_samples':args.samples,
              'local_limit_statistic':args.latency_statistic,'latency_selection_split':args.timing_split,
              'scope':'Validation-selected equal-weight subsets; local raw-image sample timing only',
              'T4_verified':False,'candidate_count':len(candidates)}
    evidence_path=work/'selected.json';evidence_path.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    predictions=work/'selected_predictions'
    command=['predict.py','--checkpoint',*selected['checkpoints'],'--aggregation',selected['aggregation'],
             '--root',root,'--cache',args.cache,'--out',predictions,'--workers','0','--batch-size',str(args.batch_size)]
    if selected['tta']:command.append('--tta')
    execute(command,work/'selected_predictions.log')
    actual=read(predictions/'report.json')
    if abs(actual['validation']['macro_auc']-selected['metrics']['macro_auc'])>1e-8:
        raise RuntimeError('Fresh validation does not match selection exports')
    execute(['tools/package.py','--checkpoint',*selected['checkpoints'],'--predictions',predictions,
             '--root',root,'--out',bundle,'--latency-report',chosen_timing,'--selection-report',evidence_path],work/'package.log')
    reproduced=work/'raw_reproduced'
    models=[bundle/'models'/f'model_{i+1}.pt' for i in range(len(selected['checkpoints']))]
    execute(['predict.py','--checkpoint',*models,'--calibration','report.json','--root',ssd,'--cache','__no_cache__',
             '--out',reproduced,'--workers','0'],work/'raw_reproduced.log',cwd=bundle)
    execute(['tools/verify_bundle.py','--bundle',bundle,'--root',ssd,'--reproduced',reproduced/'submission.csv',
             '--out',work/'verification.json'],work/'verification.log')
    verification=read(work/'verification.json')
    assert verification['independent_inference_reproduced'] and verification['local_latency_sample_limit_passed']
    summary={'bundle':str(bundle),'validation':actual['validation'],'verification':verification,'T4_verified':False}
    (work/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
