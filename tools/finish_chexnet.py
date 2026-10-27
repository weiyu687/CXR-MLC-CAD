"""Compare completed CheXNet experiments and conditionally prepare a new bundle.

Selection uses official validation only. A weaker candidate never triggers new
test inference. Test replay/cost failures stop delivery without test reselection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def table_for_studies(frame,ids,probabilities,gates):
    metadata=frame.drop_duplicates('Study_id').set_index('Study_id')
    table=pd.DataFrame({'Study_id':np.repeat(ids,10),'Predict_class':np.tile(np.arange(1,11),len(ids)),
        'Probability':probabilities.ravel(),'Gate':np.tile(gates,len(ids))})
    table.insert(0,'Subject_id',metadata.loc[table.Study_id].Subject_id.to_numpy())
    return table


def load_candidate(candidate,work,frame,expected_runtime,expected_csv):
    from src.data import aggregate
    from src.metrics import evaluate,submission_probabilities
    key=candidate['model_input_fingerprint']
    path=work/'target_runtime_review'/(key[:16]+'.npz')
    with np.load(path,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata']))
        if (metadata['model_input_fingerprint']!=key or metadata['runtime']!=expected_runtime
                or metadata['validation_csv_sha256']!=expected_csv or metadata['batch_size']!=1
                or not metadata.get('center_direct_forward')):
            raise ValueError('Candidate export is not bound to the deployed inputs')
        np.testing.assert_array_equal(saved['image_ids'],frame.Dicom_id.to_numpy(dtype=str))
        images=saved[candidate['crop']]
        if images.shape!=(len(frame),10) or not np.isfinite(images).all() or (images<0).any() or (images>1).any():
            raise ValueError('Invalid candidate image probabilities')
        ids,probabilities,targets=aggregate(frame,images,'frontal')
    probabilities=submission_probabilities(probabilities)
    if evaluate(targets,probabilities)!=candidate['validation']:
        raise ValueError('Candidate metrics differ from the reviewed export')
    return ids,probabilities,targets


def plot_histories(review,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(11,6.6),sharex='col')
    for column,summary in enumerate(review['training_summary']):
        name=summary['name']
        rows=summary['history_aggregate']
        x=[row['epoch'] for row in rows]
        axis=axes[0,column]
        axis.plot(x,[row['macro_auc'] for row in rows],color='#155e75',label='Ten-crop training-runtime evaluation')
        axis.axhline(review['incumbent_validation_auc'],color='#be123c',linestyle='--',label='Incumbent target-runtime AUC')
        axis.axhline(.9,color='#6b7280',linestyle=':',label='Requested validation target')
        axis.set_title(f"{name}: {len(rows)} completed epochs")
        axis.set_ylabel('Official valid Macro-AUC')
        axes[1,column].plot(x,[row['loss'] for row in rows],color='#7c3aed',label='Train frontal-image BCE')
        axes[1,column].plot(x,[row['validation_bce'] for row in rows],color='#ea580c',label='Valid study BCE')
        axes[1,column].set_ylabel('BCE summed across ten classes')
        axes[1,column].set_xlabel('Completed epoch')
        axes[1,column].legend(fontsize=8)
        for row in range(2):axes[row,column].grid(alpha=.2)
    figure.suptitle('CheXNet adaptation: fixed learning-rate comparison')
    handles,labels=axes[0,0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,fontsize=8)
    figure.text(.5,.01,'Curves: PyTorch 2.8. Final comparison: separate PyTorch 2.10 B=1 review. Validation is reused for selection.',ha='center',fontsize=8)
    figure.tight_layout(rect=(0,.025,1,.9))
    out.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(out,dpi=160)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',default='outputs/chexnet_20261027')
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--timing-root',required=True,help='SSD copy of valid.csv and original valid images')
    parser.add_argument('--test-root',required=True,help='Used only after a superior candidate is frozen')
    parser.add_argument('--incumbent',default='submission/round2_bundle')
    parser.add_argument('--incumbent-exports',default='outputs/research_delivery_classwise_s2038/final_selection/exports')
    parser.add_argument('--official-pictures',default='outputs/research_delivery_classwise_s2038/pictures_100')
    parser.add_argument('--official-warmup',default='outputs/research_delivery_classwise_s2038/pictures_5')
    parser.add_argument('--bundle',default='submission/chexnet_bundle_20261027')
    parser.add_argument('--archive',default='submission/CXR-MLC-CAD_chexnet_20261027.zip')
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    import torch
    from src.chexnet import CACHE_SIGNATURE
    from src.data import frame
    from src.metrics import evaluate,PROBABILITY_DECIMALS,CSV_DECIMAL_PLACES
    from tools.patient_statistics import paired_cluster_bootstrap
    from tools.review_long_training import rebuild_valid,archive_bundle
    from tools.package_chexnet import package,verify
    if sys.version_info[:2]!=(3,12) or str(torch.__version__).split('+')[0]!='2.10.0':
        raise RuntimeError('Use the target Python 3.12 / PyTorch 2.10 runtime')
    project=Path(__file__).resolve().parents[1]
    work=Path(args.work).resolve()
    review_path=project/'docs/research/chexnet_verified_20261027.json'
    review,receipt=read(review_path),read(work/'review_complete.json')
    if not receipt['review_complete'] or not review.get('center_direct_forward'):
        raise ValueError('Complete the direct-forward target-runtime review first')
    plan=read(work/'plan.json')
    for relative,expected in plan['source_sha256'].items():
        if sha(project/relative)!=expected:raise ValueError('Frozen training source changed: '+relative)
    completion_plan=read(work/'completion_plan.json')
    for relative,expected in completion_plan['source_sha256'].items():
        if sha(project/relative)!=expected:raise ValueError('Frozen completion source changed: '+relative)
    analysis=work/'completion_analysis'
    if analysis.exists():raise FileExistsError('Preserve earlier completion analysis')
    public_path=project/'docs/research/chexnet_completion_20261027.json'
    markdown=project/'docs/CHEXNET_RESULTS_20261027.md'
    plot=project/'docs/figures/chexnet_20261027.png'
    if any(path.exists() for path in [public_path,markdown,plot]):raise FileExistsError('Preserve published results')
    incumbent=Path(args.incumbent)
    old_report=read(incumbent/'report.json')
    if sha(incumbent/'submission.csv')!='ac8df6d748a57649153ba69234c1c1c4113b181908231d217fbbb41b9dfab48d':
        raise ValueError('User-reported incumbent submission changed')
    if old_report['runtime']!=review['runtime']:raise ValueError('Comparison runtimes differ')
    exports={}
    for path in Path(args.incumbent_exports).glob('*/*/predictions.npz'):
        with np.load(path,allow_pickle=False) as saved:metadata=json.loads(str(saved['image_prediction_metadata']))
        key=(metadata['checkpoint_sha256'],metadata['tta'])
        if key in exports:raise ValueError('Ambiguous incumbent validation export')
        exports[key]=path
    ids,targets,patients,old_predictions=rebuild_valid(old_report,exports,args.root)
    if evaluate(targets,old_predictions)!=old_report['validation']:raise ValueError('Incumbent metrics do not rebuild')
    validation_frame=frame(args.root,'valid')
    if len(ids)!=859 or len(np.unique(patients))!=228:raise ValueError('Official validation coverage differs')
    candidates=[];predictions={'round2_incumbent':old_predictions};seen=set()
    for candidate in sorted(review['experiments'],key=lambda value:value['validation']['macro_auc'],reverse=True):
        key=(candidate['model_input_fingerprint'],candidate['crop'])
        if key in seen:continue
        seen.add(key)
        current_ids,probabilities,current_targets=load_candidate(candidate,work,validation_frame,
            review['runtime'],sha(Path(args.root)/'valid.csv'))
        np.testing.assert_array_equal(current_ids,ids);np.testing.assert_array_equal(current_targets,targets)
        name=f"{candidate['run']}_{candidate['selection_basis']}_{candidate['crop']}"
        predictions[name]=probabilities
        candidates.append({**candidate,'comparison_name':name})
    comparison=paired_cluster_bootstrap(targets,predictions,patients,args.bootstrap,2051)
    analysis.mkdir()
    write(analysis/'patient_comparison.json',comparison)
    # Freeze the validation sample identities before any latency observation.
    csv_hash=sha(Path(args.root)/'valid.csv')
    if sha(Path(args.timing_root)/'valid.csv')!=csv_hash:raise ValueError('SSD timing copy differs')
    rng=np.random.default_rng(2052)
    indices=rng.choice(len(validation_frame),size=105,replace=False)
    for name,chosen in [('pictures_100',indices[:100]),('pictures_5',indices[100:])]:
        folder=analysis/name;folder.mkdir()
        for index in chosen:
            row=validation_frame.iloc[index]
            source=Path(args.timing_root)/'valid'/row.Subject_id/row.Study_id/(row.Dicom_id+'.jpg')
            shutil.copy2(source,folder/source.name)
    write(analysis/'validation_timing_sample.json',{'validation_csv_sha256':csv_hash,'seed':2052,
        'image_indices':indices.tolist(),'selection_used_labels':False,'test_images_used':False})
    def execute(arguments,log):
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',*map(str,arguments)],cwd=project,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
    old_auc=old_report['validation']['macro_auc']
    # Always measure the best candidate of each crop type for the experiment.
    # Also measure every AUC-improving candidate before freezing the selection.
    measured={};crops=set()
    for index,candidate in enumerate(candidates):
        crop=candidate['crop']
        if crop in crops and candidate['validation']['macro_auc']<=old_auc:continue
        crops.add(crop)
        folder=analysis/f'candidate_{index:02d}';folder.mkdir()
        checkpoint=work/candidate['run']/('best.pt' if candidate['selection_basis']=='official_valid_macro_auc' else 'loss_best.pt')
        if sha(checkpoint)!=candidate['checkpoint_sha256']:raise ValueError('Candidate checkpoint changed')
        calibration={'validation':candidate['validation'],'submission':None,
            'checkpoint_sha256':candidate['checkpoint_sha256'],'preprocessing':CACHE_SIGNATURE,
            'arguments':{'crop':crop,'batch_size':1,'workers':2},'runtime':review['runtime'],
            'probability_decimals':PROBABILITY_DECIMALS,'csv_decimal_places':CSV_DECIMAL_PLACES,
            'deterministic_inference':True,'dataset_csv_sha256':csv_hash,'validation_recomputed':True,
            'source_review_sha256':sha(review_path),'official_website_submission_performed':False}
        write(folder/'calibration.json',calibration)
        table_for_studies(validation_frame,ids,predictions[candidate['comparison_name']],
            candidate['validation']['gates']).to_csv(folder/'validation.csv',index=False,float_format='%.15f')
        execute(['-m','tools.cost_chexnet','--checkpoint',checkpoint,'--calibration',folder/'calibration.json',
            '--submission',folder/'validation.csv','--root',args.timing_root,'--split','valid',
            '--pictures',analysis/'pictures_100','--warmup-pictures',analysis/'pictures_5','--out',folder/'cost'],folder/'cost.log')
        cost=read(folder/'cost/cost_report.json')
        if (cost['runtime']!=review['runtime'] or not cost['submission_predictions_match'] or cost['split']!='valid'
                or cost['crop']!=crop or cost['batch_size']!=1 or cost['preprocessing']!=CACHE_SIGNATURE):
            raise ValueError('Validation cost identity/reproduction differs')
        measured[candidate['comparison_name']]={'checkpoint':str(checkpoint),'calibration':str(folder/'calibration.json'),
            'mean_ms':cost['mean_ms'],'local_mean_limit_passed':cost['local_mean_limit_passed'],
            'max_probability_difference':cost['max_probability_difference'],'T4_verified':False}
    eligible=[candidate for candidate in candidates if candidate['validation']['macro_auc']>old_auc
        and measured[candidate['comparison_name']]['local_mean_limit_passed']]
    selected=eligible[0] if eligible else None
    decision={'selection_basis':'Official valid AUC then full-study valid latency <=100ms; no test selection',
        'selected':selected,'incumbent_validation_auc':old_auc,'validation_cost':measured,
        'new_test_inference_authorized_by_selection':selected is not None,
        'prior_auc_dinov2_queue_paused':True,'official_website_submission_performed':False}
    write(analysis/'selection.json',decision)
    delivery=None
    if selected is not None:
        chosen=measured[selected['comparison_name']]
        tests=analysis/'test_predictions'
        execute(['-m','tools.predict_chexnet','--checkpoint',chosen['checkpoint'],'--calibration',chosen['calibration'],
            '--root',args.test_root,'--out',tests],analysis/'test_predictions.log')
        execute(['-m','tools.predict_chexnet','--checkpoint',chosen['checkpoint'],'--calibration',tests/'report.json',
            '--root',args.test_root,'--out',analysis/'raw_reproduced'],analysis/'raw_reproduced.log')
        execute(['-m','tools.cost_chexnet','--checkpoint',chosen['checkpoint'],'--calibration',tests/'report.json',
            '--submission',tests/'submission.csv','--root',args.test_root,'--pictures',args.official_pictures,
            '--warmup-pictures',args.official_warmup,'--out',analysis/'test_cost'],analysis/'test_cost.log')
        test_cost=read(analysis/'test_cost/cost_report.json')
        if not all(test_cost[key] for key in ['submission_predictions_match','local_mean_limit_passed','official_sample_protocol']):
            raise ValueError('Frozen test cost failed; no test-based fallback or reselection')
        package(chosen['checkpoint'],tests,args.bundle,args.test_root,analysis/'test_cost/cost_report.json',analysis/'selection.json')
        verification=verify(args.bundle,args.test_root,analysis/'raw_reproduced/submission.csv')
        write(analysis/'bundle_verification.json',verification)
        archive=archive_bundle(Path(args.bundle).resolve(),Path(args.archive).resolve())
        delivery={'bundle':Path(args.bundle).name,'submission_sha256':sha(Path(args.bundle)/'submission.csv'),
            'archive':archive,'verification':verification}
    plot_histories(review,plot)
    public={'source_commit':review['source_commit'],'completion_source_commit':completion_plan['source_commit'],
        'review_sha256':sha(review_path),'runtime':review['runtime'],
        'valid_studies':len(ids),'valid_patients':len(np.unique(patients)),
        'best_chexnet_candidate':candidates[0],'incumbent_validation':old_report['validation'],
        'incumbent_user_reported_test_auc':.7678,'patient_comparison':comparison,
        'validation_cost':{key:{field:value for field,value in entry.items() if field not in ['checkpoint','calibration']}
            for key,entry in measured.items()},'selected_new_chexnet_candidate':selected,'delivery':delivery,
        'target_valid_auc':.9,'target_met_in_validation':candidates[0]['validation']['macro_auc']>=.9,
        'test_images_or_metadata_read':selected is not None,'new_hidden_test_result_known':False,
        'official_website_submission_performed':False,'prior_auc_dinov2_queue_paused':True,
        'interpretation':'Repeated official-valid model/checkpoint/crop selection. Patient-cluster bootstrap is descriptive; neither independent confirmation nor a hidden-test score.'}
    write(public_path,public)
    rows=['# CheXNet 正式实验结果','',f"两组训练均已完成，代码冻结提交 `{review['source_commit']}`。目标环境为Python3.12/PyTorch2.10，batch1。",'',
        '| 实验 | 轮数 | 最佳十裁剪验证AUC（训练环境） |','|---|---:|---:|']
    for summary in review['training_summary']:
        rows.append(f"| {summary['name']} | {summary['epochs_completed']} | {max(row['macro_auc'] for row in summary['history_aggregate']):.6f} |")
    rows.extend(['',f"目标环境中最佳CheXNet验证AUC为 **{candidates[0]['validation']['macro_auc']:.6f}**，原交付为 **{old_auc:.6f}**。",
        '每种裁剪按实际部署形状独立前向；中心裁剪没有使用十裁剪批次中抽取的近似结果。',
        f"按228名验证患者配对重采样{args.bootstrap}次，各候选区间及逐类AUC见[汇总](research/chexnet_completion_20261027.json)。反复选型后的区间仍不是独立测试证据。",'',
        '![训练曲线](figures/chexnet_20261027.png)','',
        '已生成通过本机复核的新候选材料；隐藏测试分数未知。' if delivery else '没有验证AUC与本机耗时均符合替换条件的CheXNet单模型，保留原76.78分提交文件；没有为较弱候选重新读取测试集。',
        '验证AUC≥0.90目标'+('已达到。' if public['target_met_in_validation'] else '尚未达到。'),
        'T4未实测，未执行比赛网站提交。暂停的AUC/DINOv2方向没有恢复。',''])
    markdown.write_text('\n'.join(rows),encoding='utf-8')
    write(analysis/'complete.json',{'complete':True,'public_result':str(public_path),
        'bundle_created':delivery is not None,'publication_pending':True,'official_website_submission_performed':False})
    print(json.dumps({'best_chexnet_valid_auc':candidates[0]['validation']['macro_auc'],
        'bundle_created':delivery is not None,'public_result':str(public_path)},indent=2),flush=True)


if __name__=='__main__':main()
