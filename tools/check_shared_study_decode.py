"""Compare shared decoding to independent member preprocessing on real JPEGs.

Uses only generated images and tiny synthetic networks; no competition score.
"""
from pathlib import Path
import copy
import shutil
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.data import aggregate
from src.deployment import StudyPredictor
from src.inference import forward_probabilities,model_input_fingerprint
from src.metrics import submission_probabilities
from src.ensemble import combine_probabilities
from src.preprocessing import decode_original_like_cache


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2041)
    device=torch.device('cpu')
    configs=[{'size':64,'normalization':'legacy'}, {'size':96,'normalization':'imagenet'},
             {'size':64,'normalization':'legacy'}]
    models=[torch.nn.Sequential(torch.nn.Conv2d(3,10,1),torch.nn.AdaptiveAvgPool2d(1),
                                torch.nn.Flatten()).eval() for _ in configs]
    def load(path,device):
        index=int(path)
        return models[index],{'config':configs[index]}
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        metadata=pd.DataFrame({'Dicom_id':['a','b','c'],'Subject_id':['p1','p1','p2'],
                               'Study_id':['s1','s1','s2'],'ViewPosition':['PA','LL','LL']})
        metadata.to_csv(root/'test.csv',index=False)
        for i,row in metadata.iterrows():
            path=root/'test'/row.Subject_id/row.Study_id/(row.Dicom_id+'.jpg')
            path.parent.mkdir(parents=True,exist_ok=True)
            x=np.arange((75+i*5)*(113+i*3)).reshape(75+i*5,113+i*3)
            Image.fromarray(((x*7+i*53)%256).astype(np.uint8)).save(path,quality=87)
        for mode in ['mean','max','frontal','weighted']:
            for tta in [False,True]:
                weights=np.stack([np.linspace(.125,.375,10),np.full(10,.25),np.linspace(.625,.375,10)]).tolist() if tta else None
                with patch('src.deployment.load_deployed_model',side_effect=load):
                    predictor=StudyPredictor(['0','1','2'],root,device,mode,tta,ensemble_weights=weights)
                with torch.inference_mode():
                    expected=[]
                    for model,ds in predictor.members:
                        image=np.stack([forward_probabilities(model,ds[i][0].unsqueeze(0),device,tta).numpy()[0]
                                        for i in range(len(ds))])
                        ids,p,_=aggregate(ds.df,image,mode)
                        expected.append(p)
                    expected=submission_probabilities(combine_probabilities(expected,weights))
                    for study_index,idx in enumerate([0,2]):
                        with patch('src.deployment.decode_original_like_cache',wraps=decode_original_like_cache) as decode:
                            study,actual=predictor(predictor.original_path(idx))
                            np.testing.assert_array_equal(actual,expected[study_index])
                            assert study==ids[study_index]
                            selected=1 if idx==2 or mode=='frontal' else 2
                            assert decode.call_count==selected
                            predictor(predictor.original_path(idx))
                            assert decode.call_count==2*selected,'Decoder cache leaked between calls'
                copied=root/'copied'/'a.jpg'
                copied.parent.mkdir(exist_ok=True)
                shutil.copy2(predictor.original_path(0),copied)
                with torch.inference_mode():
                    np.testing.assert_array_equal(predictor(copied)[1],expected[0])
                Image.new('L',(12,12),123).save(copied)
                try:
                    predictor(copied)
                    raise AssertionError('Altered timing input accepted')
                except ValueError as error:
                    assert 'differs from the original' in str(error)
        twins=[copy.deepcopy(models[0]),copy.deepcopy(models[0])]
        twin_config={'model':'synthetic_twin','size':64,'normalization':'legacy'}
        def load_twin(path,device):
            index=int(path)
            return twins[index],{'config':twin_config,'state_dict':twins[index].state_dict()}
        with patch('src.deployment.load_deployed_model',side_effect=load_twin):
            predictor=StudyPredictor(['0','1'],root,device,'weighted',True)
        assert predictor.unique_forward_groups==1
        changed={'config':{**twin_config,'size':96},'state_dict':twins[0].state_dict()}
        assert model_input_fingerprint(changed)!=predictor.forward_keys[0]
        calls=[0,0]
        for index,model in enumerate(twins):
            def hook(module,inputs,output,index=index):calls[index]+=1
            model.register_forward_hook(hook)
        with torch.inference_mode():
            first=predictor(predictor.original_path(0))[1]
            assert calls==[4,0] # Two images, each with fixed flip TTA.
            second=predictor(predictor.original_path(0))[1]
            assert calls==[8,0],'Neural outputs leaked across calls'
            np.testing.assert_array_equal(first,second)
            independent=[]
            for model,ds in predictor.members:
                p=np.stack([forward_probabilities(model,ds[i][0].unsqueeze(0),device,True).numpy()[0] for i in [0,1]])
                independent.append(aggregate(ds.df.iloc[:2].reset_index(drop=True),p,'weighted')[1][0])
            np.testing.assert_array_equal(first,submission_probabilities(combine_probabilities(independent)))
    print('PASS: shared vs independent decode exact for four pooling rules, TTA, mixed sizes/normalization, missing frontal view; no cross-call cache; copied-image hash check')


if __name__=='__main__':main()
