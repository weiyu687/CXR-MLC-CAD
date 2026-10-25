"""Synthetic check of finite classwise selection, TTA binding and member pruning."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from src.ensemble import combine_probabilities
from src.metrics import evaluate
from tools.automatic_fusion import fit_best_global_triples,prune_unused_members
from tools.fit_classwise_fusion import load_exports


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);ids=np.array([f'i{i}' for i in range(10)])
        y=np.repeat(np.array([0,1]*5)[:,None],10,axis=1)
        pd.DataFrame({'Dicom_id':ids,'Subject_id':[f'p{i}' for i in range(10)],
                      'Study_id':[f's{i}' for i in range(10)],'ViewPosition':['PA']*10,
                      'Predict_class':['1;2;3;4;5;6;7;8;9;10' if row[0] else '' for row in y]}).to_csv(root/'valid.csv',index=False)
        valid_hash=hashlib.sha256((root/'valid.csv').read_bytes()).hexdigest()
        checkpoints=[];hashes=[];exports={};members=[]
        for index in range(3):
            path=root/f'{index}.pt';path.write_bytes(f'synthetic source {index}'.encode())
            checkpoints.append(str(path));digest=hashlib.sha256(path.read_bytes()).hexdigest();hashes.append(digest)
            p=np.empty_like(y,dtype=float)
            for c in range(10):p[:,c]=np.where(y[:,c],.9,.1) if c%3==index else np.where(y[:,c],.1,.9)
            members.append(p)
            for tta in [False,True]:
                metadata={'checkpoint_sha256':digest,'validation_csv_sha256':valid_hash,'batch_size':1,
                          'tta':tta,'deterministic_inference':True,'runtime':{'scope':'synthetic only'}}
                exported=root/f'{index}_{int(tta)}.npz'
                np.savez(exported,valid_images=p,valid_image_ids=ids,image_prediction_metadata=json.dumps(metadata))
                exports[digest,tta]=exported
        base={'checkpoints':checkpoints,'aggregation':'frontal','model_bytes':sum(Path(p).stat().st_size for p in checkpoints),
              'ensemble_weights':[.5,.25,.25],
              'metrics':evaluate(y,combine_probabilities(members,[.5,.25,.25]))}
        inferior={**base,'tta':False,'aggregation':'mean',
                  'metrics':{**base['metrics'],'macro_auc':base['metrics']['macro_auc']-.05}}
        candidates=fit_best_global_triples([inferior,{**base,'tta':False},{**base,'tta':True}],exports,root,root/'fits')
        assert len(candidates)==2 and {c['tta'] for c in candidates}=={False,True}
        for candidate in candidates:
            assert candidate['metrics']['macro_auc']==1 and len(candidate['checkpoints'])==3
            assert candidate['aggregation']=='frontal' and candidate['global_source_validation_macro_auc']==base['metrics']['macro_auc']
            protocol=candidate['ensemble_weight_spec']['selection_protocol']
            assert protocol['shrinkage']>0 and protocol['tta']==candidate['tta']
            assert protocol['mean_oof_macro_auc'][str(protocol['shrinkage'])]>protocol['mean_oof_macro_auc']['0.0']
        try:
            load_exports(checkpoints,[exports[d,False] for d in hashes],root,'frontal',tta=True)
            raise AssertionError('Incorrect TTA source accepted')
        except ValueError:pass
        assert fit_best_global_triples([{**base,'checkpoints':checkpoints[:2],'tta':False}],exports,root,root/'no_triples')==[]
        # Every possible position of a zero-weight member can be removed with
        # bit-identical predictions, while retaining source ordering/hashes.
        rng=np.random.default_rng(81);p=rng.random((3,17,10))
        for zero_index in range(3):
            active=[i for i in range(3) if i!=zero_index]
            weights=np.zeros((3,10));weights[active[0]]=np.linspace(0.,1.,10);weights[active[1]]=1-weights[active[0]]
            spec={'checkpoint_sha256':hashes,'weights':weights.tolist(),'selection_protocol':{},'class_ids':list(range(1,11))}
            paths,pruned,kept=prune_unused_members(checkpoints,spec)
            assert kept==active and pruned['checkpoint_sha256']==[hashes[i] for i in active]
            assert paths==[str(Path(checkpoints[i]).resolve()) for i in active]
            np.testing.assert_array_equal(combine_probabilities(p,weights),combine_probabilities(p[kept],pruned['weights']))
        for broken in [{**spec,'checkpoint_sha256':hashes[::-1]},
                       {**spec,'class_ids':list(range(10,0,-1))}]:
            try:
                prune_unused_members(checkpoints,broken)
                raise AssertionError('Mismatched fitted model/class binding accepted')
            except ValueError:pass
        # When all members already rank every class perfectly, CV picks zero
        # strength; preserve the ordinary global candidates without duplicates.
        for digest in hashes:
            path=exports[digest,False]
            with np.load(path) as old:metadata=str(old['image_prediction_metadata'])
            np.savez(path,valid_images=np.where(y,.8,.2),valid_image_ids=ids,image_prediction_metadata=metadata)
        perfect={**base,'metrics':evaluate(y,np.where(y,.8,.2)),'tta':False}
        assert fit_best_global_triples([perfect],exports,root,root/'global_preferred')==[]
    print('PASS: at most one fit per TTA mode, patient-CV selection, export binding, sparse member removal and zero-strength fallback; synthetic scores only')


if __name__=='__main__':main()
