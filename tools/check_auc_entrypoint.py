"""Synthetic CPU end-to-end checks of both paired metric objectives."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from src.model import build_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('Preserve previous smoke check')
    out.mkdir(parents=True)
    source=out/'synthetic_initialization'
    source.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(2049)
    model=build_model('tv_resnet50')
    torch.save({'state_dict':model.state_dict(),'config':{'model':'tv_resnet50','size':64,
        'normalization':'imagenet','patient_holdout':.25,'loss':'bce'},
        'provenance':{'synthetic_random_fixture':True}},source/'best.pt')
    fixture=Path('outputs/reference_training_smoke_20261026_v2').resolve()
    held=(fixture/'chexfusion_tiny_gap/holdout_patients.json').read_text(encoding='utf-8')
    (source/'holdout_patients.json').write_text(held,encoding='utf-8')
    heads=[]
    for objective in ['auc','bce']:
        run=out/objective
        command=[sys.executable,'-u','-m','tools.train_auc_reference','--init',str(source/'best.pt'),
            '--objective',objective,'--root',str(fixture),'--cache',str(fixture/'cache'),
            '--out',str(run),'--epochs','2','--batch-size','32','--micro-batch','16','--workers','0']
        with (out/(objective+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==2 and checkpoint['optimizer']['steps']==2
        if objective=='auc':
            assert set(checkpoint['criterion'])=={'a','b','alpha'}
            assert (checkpoint['criterion']['alpha']>=0).all()
        heads.append(checkpoint['provenance']['initial_classifier_sha256'])
        history=[json.loads(line) for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(history)==2 and all(row['peak_allocated_mib'] is None for row in history)
        assert json.loads((run/'complete.json').read_text(encoding='utf-8'))['epochs_completed']==2
    assert heads[0]==heads[1]
    report={'paired_entrypoints_completed':True,'identical_initial_classification_head':True,
        'auc_auxiliary_and_optimizer_states_saved':True,'patient_holdout_preserved':True,
        'scope':'Synthetic CPU training from a random fixture; not real BCE pretraining or competition scores'}
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
