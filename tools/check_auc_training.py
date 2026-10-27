"""Compare AUC losses, upstream PESG steps, coverage and checkpoint recovery."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

from src.auc_optimizer import PESG
from src.auc_training import MultiLabelAUCMV2, CoverageBatchSampler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2049)
    root=Path('outputs/research_review/repos/Optimization-AI/LibAUC/libauc')
    text=(root/'losses/auc.py').read_text(encoding='utf-8')
    # Execute only the audited, self-contained upstream multi-label loss class.
    source=text[text.index('class MultiLabelAUCMLoss('):text.index('class meanAveragePrecisionLoss(')]
    original_namespace={'torch':torch}
    exec(compile(source,'audited_libauc_multilabel','exec'),original_namespace)
    original=original_namespace['MultiLabelAUCMLoss'](num_labels=10,version='v2',device='cpu')
    loss=MultiLabelAUCMV2().double()
    targets=torch.randint(0,2,(32,10)).double()
    targets[0]=0
    targets[1]=1
    probabilities=torch.rand(32,10,dtype=torch.float64,requires_grad=True)
    other=probabilities.detach().clone().requires_grad_(True)
    actual=loss(probabilities,targets)
    expected=original(other,targets)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(probabilities.grad,other.grad,atol=1e-12,rtol=1e-10)
    for name in ['a','b','alpha']:
        torch.testing.assert_close(getattr(loss,name).grad,getattr(original,name).grad.double(),atol=1e-8,rtol=1e-6)
    split=MultiLabelAUCMV2().double()
    split.load_state_dict(loss.state_dict())
    chunked=probabilities.detach().clone().requires_grad_(True)
    counts=(targets.sum(0),(1-targets).sum(0))
    for begin in [0,16]:
        split(chunked[begin:begin+16],targets[begin:begin+16],counts,.5).backward()
    torch.testing.assert_close(chunked.grad,probabilities.grad,atol=1e-12,rtol=1e-10)
    for name in ['a','b','alpha']:
        torch.testing.assert_close(getattr(split,name).grad,getattr(loss,name).grad,atol=1e-12,rtol=1e-10)
    target_array=np.random.default_rng(2049).binomial(1,.02,(300,10))
    sampler=CoverageBatchSampler(target_array)
    first=list(sampler)
    assert first==list(sampler)
    for batch in first:
        assert (target_array[batch].sum(0)>0).all() and (target_array[batch].sum(0)<len(batch)).all()
    sampler.set_epoch(1)
    assert first!=list(sampler)
    reference_text=(root/'optimizers/pesg.py').read_text(encoding='utf-8').replace('from ..utils import is_distributed',
        'def is_distributed():\n    return False')
    reference={}
    exec(compile(reference_text,'audited_pesg','exec'),reference)
    vector=torch.nn.Parameter(torch.randn(10))
    reference_vector=torch.nn.Parameter(vector.detach().clone())
    criterion=MultiLabelAUCMV2()
    reference_criterion=copy.deepcopy(criterion)
    optimizer=PESG([vector],criterion,lr=.01,epoch_decay=.002,device='cpu',verbose=False)
    reference_optimizer=reference['PESG']([reference_vector],reference_criterion,lr=.01,
                                        epoch_decay=.002,device='cpu',verbose=False)
    # Initialize identically to the pretrained paper reference solution.
    for opt,parameter,objective in [(optimizer,vector,criterion),(reference_optimizer,reference_vector,reference_criterion)]:
        for ref,value in zip(opt.model_ref,[parameter,objective.a,objective.b]):ref.copy_(value.detach())
    for _ in range(3):
        for opt,parameter,objective in [(optimizer,vector,criterion),(reference_optimizer,reference_vector,reference_criterion)]:
            opt.zero_grad()
            objective.zero_grad()
            objective(parameter.expand(32,-1).sigmoid(),targets.float()).backward()
            opt.step()
        torch.testing.assert_close(vector,reference_vector,atol=0,rtol=0)
        for name in ['a','b','alpha']:torch.testing.assert_close(getattr(criterion,name),getattr(reference_criterion,name),atol=0,rtol=0)
    recovered_vector=torch.nn.Parameter(vector.detach().clone())
    recovered_criterion=MultiLabelAUCMV2()
    recovered_criterion.load_state_dict(criterion.state_dict())
    recovered_optimizer=PESG([recovered_vector],recovered_criterion,lr=.01,epoch_decay=.002,device='cpu',verbose=False)
    recovered_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for opt,parameter,objective in [(optimizer,vector,criterion),(recovered_optimizer,recovered_vector,recovered_criterion)]:
        opt.zero_grad()
        objective.zero_grad()
        objective(parameter.expand(32,-1).sigmoid(),targets.float()).backward()
        opt.step()
    torch.testing.assert_close(vector,recovered_vector,atol=0,rtol=0)
    for name in ['a','b','alpha']:torch.testing.assert_close(getattr(criterion,name),getattr(recovered_criterion,name),atol=0,rtol=0)
    assert optimizer.steps==recovered_optimizer.steps
    report={'torch':str(torch.__version__),'loss_and_gradient_match_author_v2':True,
        'microbatch_loss_gradient_equivalence':True,'all_labels_have_positive_and_negative_in_batch':True,
        'upstream_pesg_steps_match':True,'auxiliary_state_and_optimizer_resume_exact':True,
        'scope':'Synthetic CPU mathematical checks; no classification quality claim',
        'upstream_sha256':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [root/'losses/auc.py',root/'optimizers/pesg.py']}}
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
