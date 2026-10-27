"""Re-evaluate completed CheXNet candidates in the target runtime, valid only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.chexnet import CheXNetImages,load_checkpoint,collect_predictions
from src.data import aggregate
from src.inference import configure_deterministic_inference,runtime_information,model_input_fingerprint
from src.metrics import evaluate


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',default='outputs/chexnet_20261027')
    parser.add_argument('--root',default='raw_data/cxr_first')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--out',default='docs/research/chexnet_verified_20261027.json')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,12) or str(torch.__version__).split('+')[0]!='2.10.0':
        raise RuntimeError('Use the target Python 3.12 / PyTorch 2.10 runtime')
    work=Path(args.work)
    plan,status=read(work/'plan.json'),read(work/'status.json')
    if status['stage']!='TRAINING_COMPLETE_REVIEW_PENDING' or len(status['completed_runs'])!=2:
        raise ValueError('Both CheXNet experiments must finish before review')
    project=Path(__file__).resolve().parents[1]
    for relative,expected in plan['source_sha256'].items():
        if sha(project/relative)!=expected:raise ValueError('Frozen experiment source changed: '+relative)
    output=Path(args.out)
    review=work/'target_runtime_review'
    if output.exists() or review.exists():raise FileExistsError('Preserve earlier review artifacts')
    review.mkdir()
    configure_deterministic_inference()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu':torch.backends.mkldnn.enabled=False
    dataset=CheXNetImages(args.root,args.cache,'valid',mode='ten')
    if dataset.df.Study_id.nunique()!=859 or dataset.df.Subject_id.nunique()!=228:
        raise ValueError('Official validation population differs')
    history_by_run={}
    results=[]
    exports={}
    for completed in status['completed_runs']:
        folder=work/completed['name']
        history=[json.loads(line) for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(history)!=completed['epochs_completed'] or not 20<=len(history)<=60:
            raise ValueError('Incomplete planned experiment')
        history_by_run[completed['name']]=history
        for filename,selection in [('best.pt','official_valid_macro_auc'),('loss_best.pt','official_valid_bce')]:
            path=folder/filename
            digest=sha(path)
            expected=completed['checkpoint_sha256'] if filename=='best.pt' else completed['loss_selected_checkpoint_sha256']
            if digest!=expected:raise ValueError('Selected checkpoint changed')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            fingerprint=model_input_fingerprint(checkpoint)
            epoch=checkpoint['epoch']
            if fingerprint not in exports:
                model,_=load_checkpoint(path,device)
                loader=DataLoader(dataset,batch_size=1,num_workers=4,pin_memory=device.type=='cuda')
                probabilities=collect_predictions(model,loader,device)
                exports[fingerprint]=probabilities
                np.savez_compressed(review/(fingerprint[:16]+'.npz'),**probabilities,
                    image_ids=dataset.df.Dicom_id.to_numpy(dtype=str),metadata=np.array(json.dumps({
                        'model_input_fingerprint':fingerprint,'runtime':runtime_information(device),'batch_size':1,
                        'validation_csv_sha256':sha(Path(args.root)/'valid.csv')})))
                del model
            for crop,image_probabilities in exports[fingerprint].items():
                _,study,targets=aggregate(dataset.df,image_probabilities,'frontal')
                metrics=evaluate(targets,study)
                results.append({'run':completed['name'],'selection_basis':selection,'epoch':epoch,
                    'crop':crop,'checkpoint_sha256':digest,'model_input_fingerprint':fingerprint,'validation':metrics})
    best=max(results,key=lambda item:item['validation']['macro_auc'])
    report={'source_commit':plan['source_commit'],'runtime':runtime_information(device),'batch_size':1,
        'experiments':results,'best_validation_candidate':best,
        'incumbent_validation_auc':.8111219096556889,'incumbent_user_reported_test_auc':.7678,
        'target_valid_auc':.9,'target_met_in_validation':best['validation']['macro_auc']>=.9,
        'new_hidden_test_result_known':False,'official_website_submission_performed':False,
        'training_summary':[{'name':name,'epochs_completed':len(rows),
            'history_aggregate':[{key:row[key] for key in ['epoch','loss','validation_bce','macro_auc','lr_next','training_seconds']}
                                 for row in rows]} for name,rows in history_by_run.items()],
        'scope':'Official-valid selection among two learning rates, loss/AUC checkpoints and center/ten crops. Reuse is not independent hidden-test evidence.',
        'test_images_or_metadata_read':False,'submission_bundle_ready':False}
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    receipt={'review_complete':True,'public_aggregate':str(output),'best_validation_auc':best['validation']['macro_auc'],
        'publication_pending':True,'official_website_submission_performed':False}
    (work/'review_complete.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
