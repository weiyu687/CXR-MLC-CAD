"""Check frozen ensemble registry bindings and fresh score recomputation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from src.ensemble import combine_probabilities,read_weight_spec
from src.metrics import evaluate
from tools.ensemble_presets import read_presets,evaluate_presets,append_weight_arguments


def must_reject(function):
    try:
        function()
    except (ValueError,AssertionError):
        return
    raise AssertionError('Inconsistent ensemble evidence was accepted')


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        labels=['','','1;2;3;4;5;6;7;8;9;10','1;2;3;4;5;6;7;8;9;10']
        ids=np.array(['a','b','c','d'])
        pd.DataFrame({'Dicom_id':ids,'Subject_id':['p1','p2','p3','p4'],
                      'Study_id':['s1','s2','s3','s4'],'ViewPosition':['PA']*4,
                      'Predict_class':labels}).to_csv(root/'valid.csv',index=False)
        valid_hash=hashlib.sha256((root/'valid.csv').read_bytes()).hexdigest()
        paths=[];hashes=[];exports={};members=[]
        for index,values in enumerate([[.9,.1,.5,.5],[.2,.6,.5,.9]]):
            path=root/f'model_{index}.pt';path.write_bytes(f'synthetic {index}'.encode())
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            paths.append(str(path));hashes.append(digest)
            p=np.repeat(np.asarray(values)[:,None],10,axis=1);members.append(p)
            metadata={'checkpoint_sha256':digest,'validation_csv_sha256':valid_hash,
                      'batch_size':1,'tta':False,'deterministic_inference':True,
                      'runtime':{'python':'synthetic','torch':'synthetic'}}
            exported=root/f'export_{index}.npz'
            np.savez(exported,valid_images=p,valid_image_ids=ids,image_prediction_metadata=json.dumps(metadata))
            exports[digest,False]=exported
        weights=np.stack([np.linspace(.1,.4,10),np.linspace(.9,.6,10)]).tolist()
        spec={'schema_version':1,'checkpoint_sha256':hashes,'class_ids':list(range(1,11)),
              'weights':weights,'selection_protocol':{'scope':'synthetic check only'}}
        weights_path=root/'weights.json';weights_path.write_text(json.dumps(spec))
        y=np.repeat(np.array([0,0,1,1])[:,None],10,axis=1)
        score=evaluate(y,combine_probabilities(members,weights))['macro_auc']
        entry={'checkpoints':paths,'checkpoint_sha256':hashes,'weights_json':str(weights_path),
               'weights_json_sha256':hashlib.sha256(weights_path.read_bytes()).hexdigest(),
               'validation_csv_sha256':valid_hash,'aggregation':'frontal','tta':False,'batch_size':1,
               'expected_validation_macro_auc':score}
        registry=root/'registry.json'
        def write(value):
            registry.write_text(json.dumps({'schema_version':1,'ensemble_presets':[value]}))
        write(entry)
        presets=read_presets(registry,root)
        candidate=evaluate_presets(presets,exports,root,max_members=2)[0]
        assert candidate['metrics']['macro_auc']==score
        command=[];append_weight_arguments(command,candidate,root)
        assert command[0]=='--ensemble-weights-json'
        assert read_weight_spec(command[1],hashes)['weights']==candidate['ensemble_weights']
        assert read_weight_spec(command[1],hashes)['selection_protocol']==spec['selection_protocol']
        must_reject(lambda:evaluate_presets(presets,{},root))
        must_reject(lambda:evaluate_presets(presets,exports,root,max_members=1))
        broken=copy.deepcopy(presets);broken[0]['expected_validation_macro_auc']=.123
        must_reject(lambda:evaluate_presets(broken,exports,root))
        for field,value in [('checkpoint_sha256',hashes[::-1]),('weights_json_sha256','0'*64),
                            ('validation_csv_sha256','0'*64),('tta','false'),('batch_size',64)]:
            write({**entry,field:value})
            must_reject(lambda:read_presets(registry,root))
    print('PASS: fresh preset metrics, checkpoint/data/weight bindings, runtime settings, member limit and deployment arguments')


if __name__=='__main__':main()
