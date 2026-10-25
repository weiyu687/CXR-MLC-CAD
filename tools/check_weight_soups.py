"""Check parameter/buffer averaging and rejection of incompatible snapshots."""
import copy
import torch
from tools.build_weight_soups import average_checkpoints


def main():
    a={'config':{'model':'toy','size':32,'normalization':'imagenet'},
       'epoch':3,'metrics':{'macro_auc':.9},'provenance':{'pretraining':{'sha256':'same'}},
       'state_dict':{'weight':torch.tensor([1.,2.]),'running_mean':torch.tensor([3.]),
                     'bn.num_batches_tracked':torch.tensor(5),'unchanged':torch.tensor(True)}}
    b=copy.deepcopy(a)
    b['epoch']=4
    b['state_dict']['weight']=torch.tensor([5.,6.])
    b['state_dict']['running_mean']=torch.tensor([7.])
    b['state_dict']['bn.num_batches_tracked']=torch.tensor(9)
    result=average_checkpoints(a,b,.25)
    assert torch.equal(result['state_dict']['weight'],torch.tensor([4.,5.]))
    assert torch.equal(result['state_dict']['running_mean'],torch.tensor([6.]))
    assert result['state_dict']['bn.num_batches_tracked'].item()==9
    assert result['epoch'] is None and result['metrics'] is None and result['holdout_metrics'] is None
    assert torch.equal(a['state_dict']['weight'],torch.tensor([1.,2.]))
    for change in ['model','size','pretraining','dtype','integer','nan']:
        wrong=copy.deepcopy(b)
        if change=='model':wrong['config']['model']='other'
        elif change=='size':wrong['config']['size']=64
        elif change=='pretraining':wrong['provenance']['pretraining']['sha256']='different'
        elif change=='dtype':wrong['state_dict']['weight']=wrong['state_dict']['weight'].half()
        elif change=='integer':wrong['state_dict']['unchanged']=torch.tensor(False)
        elif change=='nan':wrong['state_dict']['weight'][0]=float('nan')
        try:
            average_checkpoints(a,wrong,.5)
            raise AssertionError('Incompatible source accepted: '+change)
        except ValueError:pass
    print('PASS: float parameters/buffers, integer BN counters, unchanged parents, no inherited scores/epoch, incompatible/nonfinite source rejection')


if __name__=='__main__':main()
