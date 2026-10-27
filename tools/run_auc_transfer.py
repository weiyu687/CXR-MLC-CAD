"""Fixed sequential AUC/control and paper-based generic-transfer experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--work',default='outputs/auc_transfer_20261027')
    parser.add_argument('--init',default='outputs/resnet50_imagenet_hold_s2030/best.pt')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    work=Path(args.work).resolve()
    source=Path(args.init).resolve()
    root=str(Path(args.root).resolve())
    recipes=[]
    for objective in ['auc','bce']:
        name='resnet50_'+objective+'_paired'
        output=work/name
        command=[sys.executable,'-u','-m','tools.train_auc_reference','--init',str(source),
            '--objective',objective,'--root',root,'--cache',args.cache,'--out',str(output),
            '--epochs','8','--batch-size','32','--micro-batch','32','--workers','4','--seed','2049',
            '--lr','.01','--epoch-decay','.002','--margin','1']
        recipes.append({'name':name,'type':'metric_pair','max_epochs':8,'out':str(output),'command':command})
    for model,size,batch in [('dinov2_base_mean',392,24),('chexfusion_base22k_gap',384,16)]:
        name=model+'_weighted_bce'
        output=work/name
        command=[sys.executable,'-u','train.py','--model',model,'--pretrained','--root',root,'--cache',args.cache,
            '--out',str(output),'--size',str(size),'--batch-size',str(batch),'--epochs','100','--workers','4',
            '--lr','1e-5','--optimizer','adam','--schedule','constant','--loss','inverse_bce',
            '--normalization','unit_equalized','--augmentation','transfer','--image-training','--frontal-images',
            '--patient-holdout','.10','--split-seed','2030','--seed','2048','--selection-aggregation','frontal',
            '--holdout-aggregation','frontal','--head-lr-mult','1','--weight-decay','0','--ema-decay','0',
            '--warmup-epochs','0','--patience','12','--min-epochs','20','--cuda-memory-fraction','.8']
        recipes.append({'name':name,'type':'generic_transfer','max_epochs':100,'out':str(output),'command':command})
    if not args.execute:
        print(json.dumps(recipes,indent=2))
        return
    if work.exists():raise FileExistsError('Preserve earlier experimental queue')
    if subprocess.run(['git','status','--porcelain'],cwd=project,capture_output=True,text=True,check=True).stdout.strip():
        raise ValueError('Commit reviewed code before freezing experiments')
    work.mkdir(parents=True)
    files=[project/'train.py',*sorted((project/'src').glob('*.py')),Path(__file__).resolve(),
           project/'tools/train_auc_reference.py']
    code={str(path.relative_to(project)):sha(path) for path in files}
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=project,capture_output=True,text=True,check=True).stdout.strip()
    plan={'recipes':recipes,'source_commit':commit,'source_sha256':code,'initial_bce_checkpoint_sha256':sha(source),
        'training_csv_sha256':sha(Path(root)/'train.csv'),'validation_csv_sha256':sha(Path(root)/'valid.csv'),
        'auc_paper':'https://arxiv.org/abs/2012.03173',
        'transfer_paper':'https://doi.org/10.1038/s43856-026-01897-9',
        'sampling_adaptation':'20 label-specific positive/negative anchors plus random fill in a 32-study batch',
        'metric_pair_same_initialization_sampling_augmentation':True,
        'reference_initialization':'PESG v_ref initialized from pretrained v0 per paper Algorithm 1',
        'transfer_adaptations':['Official data only, 10 competition labels, existing canonical grayscale 448 cache',
            '392/384 inputs and batch24/16 for 12GB GPU, mixed precision instead of paper FP32',
            'Patient-level holdout; max100 epochs, holdout patience12 and minimum20 epochs',
            'Non-register DINOv2-base mean over CLS+patch tokens, matching author readout'],
        'note':'AUC pair inherits its own 12% pretraining holdout. Transfer uses the prior 10% holdout; do not directly compare these holdout scores.',
        'target_valid_auc':.90,'test_data_used_for_training_or_selection':False,'official_website_submission_performed':False}
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
            for name,expected in code.items():
                if sha(project/name)!=expected:raise ValueError('Frozen experiment code changed: '+name)
            status.update(stage='TRAINING',current_run=recipe['name'])
            save()
            with (work/(recipe['name']+'.log')).open('w',encoding='utf-8') as stream:
                subprocess.run(recipe['command'],cwd=project,stdout=stream,stderr=subprocess.STDOUT,check=True)
            folder=Path(recipe['out'])
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines()]
            metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            if [row['epoch'] for row in history]!=list(range(1,len(history)+1)):
                raise ValueError('Epoch history is incomplete')
            if recipe['type']=='metric_pair' and len(history)!=8:raise ValueError('Paired arm incomplete')
            if recipe['type']=='generic_transfer' and not 20<=len(history)<=100:raise ValueError('Transfer schedule incomplete')
            status['completed_runs'].append({'name':recipe['name'],'epochs_completed':len(history),
                'selected_epoch':metrics['epoch'],'selected_holdout_auc':metrics['selection_score'],
                'official_valid_auc_at_holdout_selection':metrics['macro_auc'],
                'max_official_valid_auc_over_epochs':max(row['macro_auc'] for row in history),
                'checkpoint_sha256':sha(folder/'best.pt')})
            if recipe['name']=='resnet50_bce_paired':
                identities=[json.loads((work/name/'provenance.json').read_text(encoding='utf-8'))['initial_classifier_sha256']
                            for name in ['resnet50_auc_paired','resnet50_bce_paired']]
                if identities[0]!=identities[1]:raise ValueError('Paired classification head initialization differs')
            save()
        status.update(stage='TRAINING_COMPLETE_REVIEW_PENDING',current_run=None,completed_unix=time.time(),
            official_website_submission_performed=False)
        save()
    except Exception as error:
        status.update(stage='FAILED',error=repr(error))
        save()
        raise


if __name__=='__main__':main()
