"""Run only the user-requested CheXNet experiments; paused queues stay paused."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--work',default='outputs/chexnet_20261027')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    root=Path(args.root).resolve()
    work=Path(args.work).resolve()
    cache=Path(args.cache).resolve()
    recipes=[]
    for name,rate in [('paper_lr001','.001'),('adapt_lr0001','.0001')]:
        command=[sys.executable,'-u','-m','tools.train_chexnet','--root',str(root),'--cache',str(cache),
            '--out',str(work/name),'--lr',rate,'--epochs','60','--min-epochs','20','--early-stopping','12',
            '--plateau-patience','2','--batch-size','16','--eval-batch-size','4','--workers','4','--seed','2050']
        recipes.append({'name':name,'initial_lr':float(rate),'command':command})
    if not args.execute:
        print(json.dumps(recipes,indent=2))
        return
    if work.exists():raise FileExistsError('Preserve previous CheXNet runs')
    git_status=subprocess.run(['git','status','--porcelain'],cwd=project,capture_output=True,text=True,check=True).stdout
    if git_status.strip():raise ValueError('Commit reviewed CheXNet code before freezing experiments')
    from tools.prepare_chexnet_cache import prepare
    cache_meta=[prepare(root,cache,split,6) for split in ['train','valid']]
    from src.model import build_model
    initialization=build_model('chexnet_densenet121',pretrained=True)
    pretrained=initialization.pretraining_provenance
    if pretrained['sha256']!='a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395':
        raise ValueError('ImageNet DenseNet121 initialization changed')
    del initialization
    work.mkdir(parents=True)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=project,capture_output=True,text=True,check=True).stdout.strip()
    files=[*sorted((project/'src').glob('*.py')),project/'tools/train_chexnet.py',
           project/'tools/predict_chexnet.py',Path(__file__).resolve()]
    source_hashes={str(path.relative_to(project)):sha(path) for path in files}
    author_files=list((project/'outputs/chexnet_reference_20261027').glob('*.py'))
    plan={'recipes':recipes,'source_commit':commit,'source_sha256':source_hashes,'cache':cache_meta,
        'reference_repository':'https://github.com/arnoweng/CheXNet','paper':'https://arxiv.org/abs/1711.05225',
        'reference_file_sha256':{path.name:sha(path) for path in author_files},
        'initialization':pretrained,'labels':10,'primary_validation':'Official 859 studies; frontal mean with fallback',
        'training_policy':'Official frontal images only; no patient overlap with valid; no validation gradients',
        'selection':'Save minimum validation BCE separately from maximum validation study AUC',
        'differences_from_reference':['14 to 10 labels, competition CSV and patient/study metadata',
            'Stable logits+BCE equivalent to sigmoid+BCE; mixed precision on local GPU',
            'Shared RGB short-side256 cache before train224 square resize; lossless PNG storage',
            'Study-level evaluation and loss scheduler; max60/min20 and plateau patience2 made explicit',
            'Second run changes initial learning rate only; no unpublished improved-model architecture claimed'],
        'prior_auc_dinov2_queue_paused':True,'medical_pretrained_weights_used':False,
        'official_website_submission_performed':False,'target_valid_auc':.9}
    (work/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    status={'stage':'PLAN_FROZEN','started_unix':time.time(),'source_commit':commit,'completed_runs':[]}
    def save():
        temporary=work/'status.pending.json'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(work/'status.json')
        print(json.dumps(status),flush=True)
    save()
    try:
        for recipe in recipes:
            for relative,expected in source_hashes.items():
                if sha(project/relative)!=expected:raise ValueError('Frozen CheXNet code changed: '+relative)
            status.update(stage='TRAINING',current_run=recipe['name'])
            save()
            with (work/(recipe['name']+'.log')).open('w',encoding='utf-8') as stream:
                subprocess.run(recipe['command'],cwd=project,stdout=stream,stderr=subprocess.STDOUT,check=True)
            folder=work/recipe['name']
            complete=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines()]
            if not 20<=len(history)<=60 or len(history)!=complete['epochs_completed']:
                raise ValueError('CheXNet experiment schedule is incomplete')
            status['completed_runs'].append({'name':recipe['name'],'epochs_completed':len(history),
                'selected_epoch':metrics['epoch'],'ten_crop_valid_auc':metrics['macro_auc'],
                'same_epoch_center_crop_valid_auc':metrics['center']['macro_auc'],
                'checkpoint_sha256':sha(folder/'best.pt'),'loss_selected_checkpoint_sha256':sha(folder/'loss_best.pt')})
            save()
        status.update(stage='TRAINING_COMPLETE_REVIEW_PENDING',current_run=None,completed_unix=time.time(),
            official_website_submission_performed=False)
        save()
    except Exception as error:
        status.update(stage='FAILED',error=repr(error))
        save()
        raise


if __name__=='__main__':main()
