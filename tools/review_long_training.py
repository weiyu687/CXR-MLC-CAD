"""Review completed long experiments, verify the frozen bundle and archive it.

Publishes aggregate metrics only. No model reselection from test cost or test
statistics, no Git mutations, and no competition website submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild_valid(report, exports, root):
    """Bind every source probability to its model, CSV, image order and runtime."""
    from src.data import frame, aggregate
    from src.ensemble import combine_probabilities
    df=frame(root,'valid')
    expected_csv=sha(Path(root)/'valid.csv')
    pooled=[];ids=targets=None
    settings=report['arguments']
    for checkpoint_hash in report['checkpoint_sha256']:
        source=exports[(checkpoint_hash,settings['tta'])]
        with np.load(source) as saved:
            meta=json.loads(str(saved['image_prediction_metadata']))
            if (meta['checkpoint_sha256']!=checkpoint_hash or meta['validation_csv_sha256']!=expected_csv
                    or meta['batch_size']!=1 or meta['tta']!=settings['tta']
                    or not meta['deterministic_inference'] or meta['runtime']!=report['runtime']):
                raise ValueError('Validation export is not bound to the frozen deployment')
            np.testing.assert_array_equal(saved['valid_image_ids'],df.Dicom_id.to_numpy(dtype=str))
            current_ids,p,current_y=aggregate(df,saved['valid_images'],settings['aggregation'])
        if ids is not None:
            np.testing.assert_array_equal(current_ids,ids)
            np.testing.assert_array_equal(current_y,targets)
        ids,targets=current_ids,current_y
        pooled.append(p)
    patients=df.drop_duplicates('Study_id').set_index('Study_id').loc[ids].Subject_id.to_numpy(dtype=str)
    return ids,targets,patients,combine_probabilities(pooled,settings.get('ensemble_weights'))


def archive_bundle(bundle, archive):
    manifest=read(bundle/'manifest.json')
    paths=sorted([bundle/name for name in manifest['files']]+[bundle/'manifest.json'])
    for name,expected in manifest['files'].items():
        if sha(bundle/name)!=expected:raise ValueError('Package file changed: '+name)
    if archive.exists():raise FileExistsError('Preserve existing archive: '+str(archive))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as target:
        for path in paths:
            target.write(path,bundle.name+'/'+path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as saved:
        if saved.testzip() is not None or len(saved.namelist())!=len(paths):
            raise ValueError('Archive CRC or file count differs')
        for path in paths:
            stored=saved.read(bundle.name+'/'+path.relative_to(bundle).as_posix())
            if hashlib.sha256(stored).hexdigest()!=sha(path):
                raise ValueError('Archive file hash differs: '+str(path))
    digest=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    return {'name':archive.name,'bytes':archive.stat().st_size,'sha256':digest,
            'files':len(paths),'crc_and_all_file_hashes_verified':True}


def plot_histories(histories, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,6.8),sharex='col')
    for column,(name,history) in enumerate(histories.items()):
        x=[row['epoch'] for row in history]
        scores=[row['macro_auc'] for row in history]
        peak=int(np.argmax(scores))
        axes[0,column].plot(x,scores,color='#155E75',linewidth=2)
        axes[0,column].scatter(x[peak],scores[peak],color='#BE123C',zorder=3,label=f'Best epoch {x[peak]}: {scores[peak]:.6f}')
        axes[0,column].legend(loc='lower left',fontsize=8)
        axes[0,column].set_title(name.replace('_',' '))
        axes[0,column].set_ylabel('Validation Macro-AUC')
        axes[1,column].plot(x,[row['loss'] for row in history],color='#7C3AED',linewidth=2)
        axes[1,column].set_ylabel('Training objective')
        axes[1,column].set_xlabel('Completed epoch')
        for row in range(2):axes[row,column].grid(alpha=.2)
    fig.suptitle('36-epoch recipes: training loss and validation selection curves')
    fig.text(.5,.008,'PyTorch 2.8 training evaluation; repeated validation selection, not hidden-test results.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.96))
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',default='outputs/long_training_20261026_v2')
    parser.add_argument('--bundle',default='submission/long_training_bundle')
    parser.add_argument('--incumbent',default='submission/round2_bundle')
    parser.add_argument('--incumbent-work',default='outputs/research_delivery_classwise_s2038')
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--test-root',required=True)
    parser.add_argument('--archive',default='submission/CXR-MLC-CAD_long_training_20261026.zip')
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    import torch
    if sys.version_info[:2]!=(3,12) or str(torch.__version__).split('+')[0]!='2.10.0':
        raise RuntimeError('Use the verified Python 3.12 / PyTorch 2.10 runtime')
    project=Path(__file__).resolve().parents[1]
    work=Path(args.work).resolve();bundle=Path(args.bundle).resolve()
    incumbent=Path(args.incumbent).resolve();incumbent_work=Path(args.incumbent_work).resolve()
    review=work/'delivery_review'
    if review.exists():raise FileExistsError('Preserve existing completion review')
    status=read(work/'status.json')
    if status['stage']!='BUNDLE_READY_FOR_REVIEW':raise ValueError('Training and finalizer must complete before review')
    plan=read(work/'plan.json')
    if sha(incumbent/'submission.csv')!=plan['incumbent_submission_sha256']:
        raise ValueError('Incumbent submission changed after the training plan froze')
    for name,expected in plan['source_sha256'].items():
        if sha(project/name)!=expected:raise ValueError('Frozen training source changed: '+name)
    from tools.finish_research_ablation import completed_training
    histories={};training=[]
    for recipe in plan['recipes']:
        audited=completed_training(recipe['out'])
        if audited['epochs_completed']!=36:raise ValueError('Incomplete long-training recipe')
        training.append(audited)
        histories[recipe['name']]=[json.loads(line) for line in (Path(recipe['out'])/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    report=read(bundle/'report.json');old_report=read(incumbent/'report.json')
    selected=read(work/'deployment_selection/selected.json')
    if report['validation']['macro_auc']+1e-12<old_report['validation']['macro_auc']:
        raise ValueError('Selected deployment is below the incumbent')
    exports={}
    for path in (work/'deployment_selection/exports').glob('*/*/predictions.npz'):
        with np.load(path) as saved:
            meta=json.loads(str(saved['image_prediction_metadata']))
        key=(meta['checkpoint_sha256'],meta['tta'])
        if key in exports:raise ValueError('Duplicate source export')
        exports[key]=path
    old_ids,y,patients,old_p=rebuild_valid(old_report,exports,args.root)
    ids,new_y,new_patients,new_p=rebuild_valid(report,exports,args.root)
    np.testing.assert_array_equal(old_ids,ids);np.testing.assert_array_equal(y,new_y)
    np.testing.assert_array_equal(patients,new_patients)
    from src.metrics import evaluate
    for prediction,actual_report in [(old_p,old_report),(new_p,report)]:
        if evaluate(y,prediction)!=actual_report['validation']:
            raise ValueError('Rebuilt validation metrics differ from the frozen report')
    with np.load(work/'deployment_selection/selected_predictions/predictions.npz') as saved:
        np.testing.assert_array_equal(new_p,saved['valid']);np.testing.assert_array_equal(y,saved['targets'])
    with np.load(incumbent_work/'final_selection/selected_predictions/predictions.npz') as saved:
        np.testing.assert_array_equal(old_p,saved['valid']);np.testing.assert_array_equal(y,saved['targets'])
    from tools.patient_statistics import paired_cluster_bootstrap
    comparison=paired_cluster_bootstrap(y,{'round2_incumbent':old_p,'long_training_selected':new_p},patients,args.bootstrap,2044)
    comparison['scope']='Paired patient-cluster bootstrap after repeated validation selection; descriptive, not independent hidden-test confirmation'
    review.mkdir()
    (review/'patient_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    def execute(command,log):
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,*map(str,command)],cwd=project,stdout=stream,stderr=subprocess.STDOUT,check=True)
    execute(['tools/verify_bundle.py','--bundle',bundle,'--root',args.test_root,
             '--reproduced',work/'deployment_selection/raw_reproduced/submission.csv',
             '--out',review/'verification.json'],review/'verification.log')
    verification=read(review/'verification.json')
    if not verification['independent_inference_reproduced']:raise ValueError('Full original-image reproduction missing')
    from tools.gpu_idle import wait_for_idle_gpu
    idle=wait_for_idle_gpu(review/'cost_gpu_load.jsonl',max_wait_seconds=240,
                          on_wait=lambda state:print(json.dumps(state),flush=True))
    execute([bundle/'cost.py',bundle/'submission.csv',incumbent_work/'pictures_100',incumbent_work/'pictures_5',
             '--bundle',bundle,'--root',args.test_root,'--out',review/'cost'],review/'cost.log')
    cost=read(review/'cost/cost_report.json')
    if cost['runtime']!=report['runtime']:raise ValueError('Frozen cost runtime differs from deployment')
    if not all(cost[key] for key in ['official_sample_protocol','submission_predictions_match','local_mean_limit_passed']):
        raise ValueError('Frozen test reproduction/cost failed; preserve evidence without test-based reselection')
    archive=archive_bundle(bundle,Path(args.archive).resolve())
    plot=project/'docs/figures/long_training_20261026.png'
    plot_histories(histories,plot)
    training_public=[]
    for audit in training:
        history=histories[Path(audit['run']).name]
        cfg=audit['config']
        training_public.append({'name':Path(audit['run']).name,'model':cfg['model'],'size':cfg['size'],
            'epochs_completed':audit['epochs_completed'],'selected_epoch':audit['selected_epoch'],
            'best_training_runtime_valid_auc':audit['official_valid_macro_auc'],
            'last_training_runtime_valid_auc':history[-1]['macro_auc'],'last_training_loss':history[-1]['loss'],
            'peak_training_allocated_mib':max(row['training_peak_allocated_mib'] for row in history),
            'checkpoint_sha256':sha(Path(audit['run'])/'best.pt'),
            'history_aggregate':[{key:row[key] for key in ['epoch','loss','macro_auc','macro_f1_tuned','backbone_lr_last',
                                      'training_seconds','train_images_per_second','training_peak_allocated_mib']} for row in history]})
    model_info=read(bundle/'checkpoint_info.json')
    public={'version':'long_training_verified_20261026','training':training_public,
            'runtime':report['runtime'],'validation':report['validation'],
            'incumbent_validation':old_report['validation'],'patient_comparison':comparison,
            'checkpoint_sha256':report['checkpoint_sha256'],'models':[{'model':item['inference_configuration']['model'],
                'size':item['inference_configuration']['size'],'epoch':item['epoch']} for item in model_info],
            'aggregation':report['arguments']['aggregation'],'tta':report['arguments']['tta'],
            'ensemble_weights':report['arguments'].get('ensemble_weights'),'candidate_count':selected['candidate_count'],
            'validation_timing_ms':verification['local_latency_mean_ms'],
            'verification':{key:value for key,value in verification.items() if key!='folder'},
            'frozen_cost':{key:cost[key] for key in ['samples','warmup','rounds','mean_ms','max_probability_difference',
                'official_sample_protocol','submission_predictions_match','local_mean_limit_passed','runtime','T4_verified']},
            'submission_sha256':sha(bundle/'submission.csv'),'archive':archive,
            'source_commit':plan['source_commit'],'selection_bias':'Validation reuse affects model selection, refit metrics and fusion OOF; not independent hidden-test evidence.',
            'official_website_submission_performed':False,'new_hidden_test_result_known':False}
    if selected['selected'].get('classwise_analysis'):
        analysis=read(selected['selected']['classwise_analysis'])
        public['fusion_cv']={key:analysis[key] for key in ['selected_shrinkage','mean_oof_macro_auc','refit_validation']}
    public_path=project/'docs/research/long_training_verified_aggregate.json'
    if public_path.exists():raise FileExistsError('Preserve existing published result')
    public_path.write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipt={'bundle_and_archive_review_complete':True,'publication_pending':True,'bundle':str(bundle),'archive':archive,
             'submission_sha256':public['submission_sha256'],'public_result':str(public_path),
             'public_result_sha256':sha(public_path),'gpu_idle':idle,
             'official_website_submission_performed':False,'T4_verified':cost['T4_verified']}
    (review/'complete.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
