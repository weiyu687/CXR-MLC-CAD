"""Check input-only variants preserve tensors, lineage and training claims."""
import hashlib
from pathlib import Path
import tempfile
import torch
from tools.build_inference_variants import build_variants,resolution_variant


def main():
    source={'config':{'model':'tv_densenet121','size':320,'normalization':'imagenet','epochs':12},
            'state_dict':{'weight':torch.arange(12,dtype=torch.float32).reshape(3,4),
                          'num_batches_tracked':torch.tensor(17)},
            'epoch':5,'metrics':{'macro_auc':.99},'provenance':{'scope':'synthetic'},
            'construction':{'method':'synthetic_parent'}}
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);path=root/'source.pt';torch.save(source,path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        result=build_variants(path,[384,448],root/'variants')
        assert len(result['runs'])==2
        for record in result['runs']:
            ck=torch.load(Path(record['run'])/'best.pt',map_location='cpu',weights_only=False)
            assert ck['config']['size']==record['size'] and ck['training_configuration']['size']==320
            assert ck['epoch'] is None and ck['metrics'] is None and ck['holdout_metrics'] is None
            assert ck['construction']['weights_unchanged'] and not ck['construction']['training_performed']
            assert ck['construction']['source_selected_epoch']==5 and ck['construction']['parent_construction']==source['construction']
            assert ck['construction']['source_checkpoint_sha256']==digest
            for key,value in source['state_dict'].items():torch.testing.assert_close(ck['state_dict'][key],value,rtol=0,atol=0)
            assert ck['provenance']==source['provenance']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest and source['config']['size']==320
        for sizes,out in [([512],root/'bad'),([384,384],root/'duplicate'),([320],root/'same'),([384],root/'variants')]:
            try:
                build_variants(path,sizes,out)
                raise AssertionError('Invalid or existing variant output accepted')
            except (ValueError,FileExistsError):pass
        for broken in [{**source,'optimizer':{}},{**source,'ema':{}},
                       {**source,'config':{**source['config'],'model':'unsupported'}}]:
            try:
                resolution_variant(broken,384)
                raise AssertionError('Unsupported source accepted')
            except ValueError:pass
    print('PASS: tensor/BN identity, source immutability, cleared inherited metrics, separate training/inference sizes, lineage and output guards')


if __name__=='__main__':main()
