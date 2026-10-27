"""Audit two public non-medical initializations used by comparable CXR work.

Only downloads the explicit upstream URLs below. Does not train on any medical
data, does not open competition test/valid images, and never downloads medical
pretrained models. The timm checkpoint converter is versioned in the audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import timm


CATALOG={
    'dinov2_base':{
        'model':'vit_base_patch14_dinov2.lvd142m',
        'url':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        'data':'LVD-142M; non-medical generic visual pretraining',
        'model_card':'https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md',
        'license':'Apache-2.0',
        'license_url':'https://github.com/facebookresearch/dinov2/blob/main/LICENSE',
        'kwargs':{'dynamic_img_size':True,'dynamic_img_pad':True},'sizes':[224,392]},
    'convnext_base_22k':{
        'model':'convnext_base.fb_in22k',
        'url':'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth',
        'data':'ImageNet-22K; non-medical natural images',
        'model_card':'https://huggingface.co/timm/convnext_base.fb_in22k',
        'license':'Apache-2.0 (weight model card); original implementation license supplied separately',
        'license_url':'https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE',
        'kwargs':{},'sizes':[224,448]},
    'dinov2_base_reg':{
        'model':'vit_base_patch14_reg4_dinov2.lvd142m',
        'url':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth',
        'data':'LVD-142M; non-medical generic visual pretraining',
        'model_card':'https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md',
        'license':'Apache-2.0',
        'license_url':'https://github.com/facebookresearch/dinov2/blob/main/LICENSE',
        'kwargs':{'dynamic_img_size':True,'dynamic_img_pad':True},'sizes':[224,392]},
}


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=list(CATALOG),default=list(CATALOG))
    parser.add_argument('--cache',default=str(Path(torch.hub.get_dir())/'checkpoints'))
    parser.add_argument('--out',default='outputs/reference_backbone_audit_20261026')
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2050)
    cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        metadata=CATALOG[name]
        source=cache/metadata['url'].rsplit('/',1)[-1]
        audit_path=out/(name+'.json')
        if audit_path.exists():
            prior=json.loads(audit_path.read_text(encoding='utf-8'))
            if sha(source)!=prior['sha256']:raise ValueError('Previously audited upstream weight changed')
            print('ALREADY_AUDITED '+name,flush=True)
            continue
        if not source.exists():
            temporary=source.with_suffix('.download')
            if temporary.exists():raise FileExistsError('Incomplete download preserved: '+str(temporary))
            print('DOWNLOAD '+metadata['url'],flush=True)
            torch.hub.download_url_to_file(metadata['url'],str(temporary),progress=True)
            temporary.replace(source)
        start=time.time()
        # Force this exact downloaded file; never let a model name silently
        # choose another HF weight or download executable remote model code.
        model=timm.create_model(metadata['model'],pretrained=True,num_classes=0,
            pretrained_cfg_overlay={'file':str(source),'hf_hub_id':None},**metadata['kwargs']).eval()
        checks=[]
        with torch.inference_mode():
            for size in metadata['sizes']:
                x=torch.randn(1,3,size,size)
                features=model.forward_features(x)
                embedding=model.forward_head(features,pre_logits=True)
                if not torch.isfinite(embedding).all() or embedding.ndim!=2:
                    raise ValueError('Invalid backbone features')
                checks.append({'size':size,'features_shape':list(features.shape),
                               'embedding_shape':list(embedding.shape),'finite':True})
        result={**{k:v for k,v in metadata.items() if k not in ['sizes']},
                'cached_file':str(source.resolve()),'sha256':sha(source),'upstream_bytes':source.stat().st_size,
                'feature_parameters':sum(parameter.numel() for parameter in model.parameters()),
                'torch':str(torch.__version__),'timm':timm.__version__,'device':'cpu',
                'forward_checks':checks,'seconds':time.time()-start,
                'competition_images_read':False,'medical_pretraining_performed':False,
                'scope':'Upstream provenance, local checkpoint conversion and finite synthetic features only; no competition accuracy result'}
        audit_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
        del model


if __name__=='__main__':main()
