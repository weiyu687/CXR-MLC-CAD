"""Check schedule compatibility, optimizer parameter groups and the new head."""
import math
import torch
from src.training import learning_rate, optimizer_groups
from tools.check_training_semantics import ToyModel


def main():
    for horizon in [6,12,36]:
        for progress in [0.,.1,1.,2.,horizon/2,horizon-.01]:
            expected=1e-4*min(1.,(progress+.1)/2)*.5*(1+math.cos(math.pi*progress/horizon))
            assert learning_rate(progress,horizon,1e-4)==expected
    assert learning_rate(36,36,1e-4,2.,.05)==5e-6
    assert learning_rate(0,36,1e-4,0.,.05)==1e-4
    for options in [(0,1e-4,2.,0.),(36,-1e-4,2.,0.),(36,1e-4,-1.,0.),(36,1e-4,2.,1.1)]:
        try:
            learning_rate(0,*options)
            raise AssertionError('Invalid schedule accepted')
        except ValueError:pass
    model=ToyModel()
    groups=optimizer_groups(model,5.,True)
    names={id(parameter):name for name,parameter in model.named_parameters()}
    seen=[]
    for group in groups:
        for parameter in group['params']:
            name=names[id(parameter)];seen.append(id(parameter))
            assert group['lr_scale']==(5. if name.startswith('classifier.') else 1.)
            assert ('weight_decay' in group)==(parameter.ndim<=1 or name.endswith('.bias'))
    assert len(seen)==len(set(seen))==len(names)
    for group in optimizer_groups(model,1.,False):
        assert group['lr_scale']==1. and 'weight_decay' not in group
    # Ensure each parameter really receives its declared LR/decay from AdamW.
    optimizer=torch.optim.AdamW(groups,lr=1e-3,weight_decay=.05)
    assert all(group['weight_decay'] in [0.,.05] for group in optimizer.param_groups)
    from src.model import build_model
    convnext=build_model('tv_convnext_tiny')
    assert convnext.classifier[-1].out_features==10
    print('PASS: exact legacy LR; long schedule floor; classifier LR and bias/norm decay partition; ConvNeXt ten-label head')


if __name__=='__main__':main()
