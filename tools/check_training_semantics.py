"""Check loss interpolation and frozen-backbone buffers on a synthetic model."""
import copy
import json
from pathlib import Path
import tempfile
import torch
from src.model import AsymmetricLoss
from src.training import configure_train_stage, mixed_target_loss, save_validation_candidate


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.BatchNorm1d(8),torch.nn.Dropout(.5))
        self.classifier=torch.nn.Linear(8,10)
    def forward(self,x):
        return self.classifier(self.backbone(x))


def main():
    torch.manual_seed(2035)
    torch.set_num_threads(2)
    z=torch.linspace(-4,4,40).reshape(4,10).requires_grad_()
    y=(torch.arange(40).reshape(4,10)%3==0).float();paired=y.roll(1,0)
    lam=.3
    for criterion in [torch.nn.BCEWithLogitsLoss(),
                      torch.nn.BCEWithLogitsLoss(pos_weight=torch.linspace(1,5,10))]:
        hard=mixed_target_loss(criterion,z,y,paired,lam)
        soft=mixed_target_loss(criterion,z,y,paired,lam,'soft_target')
        assert torch.allclose(hard,soft)
        gh=torch.autograd.grad(hard,z,retain_graph=True)[0]
        gs=torch.autograd.grad(soft,z,retain_graph=True)[0]
        assert torch.allclose(gh,gs,atol=1e-7)
    asl=AsymmetricLoss()
    hard=mixed_target_loss(asl,z,y,paired,lam)
    expected=lam*asl(z,y)+(1-lam)*asl(z,paired)
    assert torch.equal(hard,expected)
    soft=mixed_target_loss(asl,z,y,paired,lam,'soft_target')
    assert (hard-soft).abs()>1.
    assert torch.isfinite(torch.autograd.grad(hard,z)[0]).all()
    for coefficient in [0.,1.]:
        result=mixed_target_loss(asl,z,y,paired,coefficient)
        assert torch.equal(result,asl(z,paired if coefficient==0 else y))

    model=ToyModel()
    # An originally frozen weight must remain frozen after the warmup.
    model.backbone[0].bias.requires_grad_(False)
    allowed={n for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    initial=copy.deepcopy(model.state_dict())
    configure_train_stage(model,True,allowed)
    assert model.classifier.training and not model.backbone.training
    x=torch.randn(6,4)
    optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    for name,value in model.state_dict().items():
        if name.startswith('backbone.'):
            assert torch.equal(value,initial[name]),name
    assert not torch.equal(model.classifier.weight,initial['classifier.weight'])
    configure_train_stage(model,False,allowed)
    assert model.backbone.training and not model.backbone[0].bias.requires_grad
    optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    assert not torch.equal(model.backbone[0].weight,initial['backbone.0.weight'])
    assert model.backbone[1].num_batches_tracked==1
    with tempfile.TemporaryDirectory() as temporary:
        record={'epoch':2,'macro_auc':.8,'auc':[.8]*10,'macro_f1_05':.4,'macro_f1_tuned':.5,
                'gates':[.5]*10,'aggregation':'frontal','holdout':{'macro_auc':.81},'selection_score':.81}
        checkpoint={'epoch':1,'config':{'model':'toy'},'state_dict':{'weight':torch.tensor([1.])},
                    'ema':{'weight':torch.tensor([2.])}}
        assert save_validation_candidate(temporary,checkpoint,record,resume_state=True)
        saved=torch.load(Path(temporary)/'official_valid_selected'/'best.pt',weights_only=False)
        assert saved['epoch']==2 and saved['state_dict']['weight'].item()==2
        assert saved['selection_basis']=='official_valid_macro_auc'
        assert not save_validation_candidate(temporary,checkpoint,{**record,'macro_auc':.79},resume_state=True)
        assert json.loads((Path(temporary)/'official_valid_selected'/'metrics.json').read_text())['macro_auc']==.8
        try:
            save_validation_candidate(temporary,checkpoint,{**record,'epoch':3},resume_state=True)
            raise AssertionError('Mismatched checkpoint/validation epoch accepted')
        except ValueError:pass
    print('PASS: BCE equivalence; ASL hard-label loss/gradient semantics; warmup freezes parameters, BN buffers and dropout; full stage restores original trainability')
    print('PASS: separate official-validation peaks use EMA, translate resume epoch correctly, reject stale scores and mismatched epochs')


if __name__=='__main__':main()
