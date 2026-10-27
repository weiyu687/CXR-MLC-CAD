"""Package a frozen CheXNet candidate and verify its test-only reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from src.chexnet import CACHE_SIGNATURE
from src.data import LABELS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_csv(path,root):
    table=pd.read_csv(path,float_precision='round_trip')
    metadata=pd.read_csv(Path(root)/'test.csv').drop_duplicates('Study_id').set_index('Study_id')
    if list(table.columns)!=['Subject_id','Study_id','Predict_class','Probability','Gate']:
        raise ValueError('Incorrect CSV columns')
    if set(table.Study_id)!=set(metadata.index) or len(table)!=10*len(metadata):
        raise ValueError('Incomplete study coverage')
    if table.duplicated(['Study_id','Predict_class']).any():raise ValueError('Duplicate study/class')
    if not table.groupby('Study_id').Predict_class.apply(lambda x:sorted(x)==list(range(1,11))).all():
        raise ValueError('Incorrect class order/coverage')
    np.testing.assert_array_equal(table.Subject_id,metadata.loc[table.Study_id].Subject_id)
    values=table[['Probability','Gate']].to_numpy()
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Invalid probabilities or gates')
    return table


def verify(bundle,root,reproduced=None):
    bundle=Path(bundle).resolve()
    manifest,report=read(bundle/'manifest.json'),read(bundle/'report.json')
    required={'predict.py','cost.py','tools/predict_chexnet.py','tools/cost_chexnet.py',
        'tools/gpu_idle.py','src/chexnet.py','src/model.py','models/model.pt','report.json',
        'README.md','labels.json','checkpoint_info.json','submission.csv',
        'requirements.txt','requirements-competition.txt'}
    if not required.issubset(manifest['files']):raise ValueError('Missing package files')
    for relative,expected in manifest['files'].items():
        path=(bundle/relative).resolve()
        if bundle not in path.parents or not path.is_file() or sha(path)!=expected:
            raise ValueError('Changed/missing/unsafe package path: '+relative)
    if read(bundle/'labels.json')!={str(i):name for i,name in enumerate(LABELS,1)}:
        raise ValueError('Incorrect label mapping')
    if report['preprocessing']!=CACHE_SIGNATURE or not report['deterministic_inference']:
        raise ValueError('Incorrect preprocessing or deterministic settings')
    if report['arguments']['batch_size']!=1:raise ValueError('Package must freeze batch=1')
    if report['validation_recomputed'] or report['dataset_csv_sha256']!=sha(Path(root)/'test.csv'):
        raise ValueError('Package is not a frozen export of this test CSV')
    if sha(bundle/'models/model.pt')!=report['checkpoint_sha256']:raise ValueError('Model identity differs')
    size=(bundle/'models/model.pt').stat().st_size
    deployed=size+sum((bundle/name).stat().st_size for name in ['report.json','labels.json','checkpoint_info.json'])
    if size!=manifest['model_bytes'] or deployed!=manifest['deployment_bytes'] or deployed>500_000_000:
        raise ValueError('Deployment size differs or exceeds 500MB')
    table=check_csv(bundle/'submission.csv',root)
    if not table.groupby('Predict_class').Gate.nunique().eq(1).all():raise ValueError('Variable class gates')
    np.testing.assert_allclose(table.groupby('Predict_class').Gate.first().sort_index(),
        report['validation']['gates'],rtol=0,atol=1e-15)
    information=read(bundle/'checkpoint_info.json')
    upstream=(information.get('provenance') or {}).get('pretraining')
    if upstream:
        if sha(bundle/'pretrain_model'/upstream['url'].rsplit('/',1)[-1])!=upstream['sha256']:
            raise ValueError('Generic pretraining audit file changed')
    result={'files_verified':len(manifest['files']),'model_bytes':size,'deployment_bytes':deployed,
        'submission_rows':len(table),'submission_studies':table.Study_id.nunique(),
        'model_hash_matches':True,'frozen_gates_match':True,'independent_inference_reproduced':False,
        'official_website_submission_performed':False,'T4_verified':False}
    if (bundle/'latency.json').is_file():
        cost=read(bundle/'latency.json')
        if (cost['checkpoint_sha256']!=report['checkpoint_sha256'] or cost['runtime']!=report['runtime']
                or cost.get('split')!='test' or cost['dataset_csv_sha256']!=sha(Path(root)/'test.csv')
                or cost.get('crop')!=report['arguments']['crop'] or cost.get('batch_size')!=1
                or cost.get('preprocessing')!=report['preprocessing']
                or not cost['submission_predictions_match']):raise ValueError('Incorrect bound latency evidence')
        result.update(local_cost_mean_ms=cost['mean_ms'],local_cost_limit_passed=cost['local_mean_limit_passed'],
            official_sample_protocol=cost['official_sample_protocol'])
    if (bundle/'selection.json').is_file():
        selected=read(bundle/'selection.json')['selected']
        if (selected['checkpoint_sha256']!=report['checkpoint_sha256'] or selected['crop']!=report['arguments']['crop']
                or selected['validation']!=report['validation']):raise ValueError('Frozen validation selection differs')
    if reproduced:
        other=check_csv(reproduced,root)
        rerun=read(Path(reproduced).with_name('report.json'))
        if (rerun['checkpoint_sha256']!=report['checkpoint_sha256'] or rerun['runtime']!=report['runtime']
                or rerun['validation_recomputed'] or rerun['preprocessing']!=report['preprocessing']):
            raise ValueError('Independent reproduction identity differs')
        for key in ['crop','batch_size']:
            if rerun['arguments'][key]!=report['arguments'][key]:raise ValueError('Reproduction recipe differs')
        table=table.sort_values(['Study_id','Predict_class']).reset_index(drop=True)
        other=other.sort_values(['Study_id','Predict_class']).reset_index(drop=True)
        pd.testing.assert_frame_equal(table[['Subject_id','Study_id','Predict_class','Gate']],
            other[['Subject_id','Study_id','Predict_class','Gate']])
        difference=float(np.max(np.abs(table.Probability-other.Probability)))
        if difference>1e-9:raise ValueError('Independent reproduction probabilities differ')
        result.update(independent_inference_reproduced=True,max_probability_difference=difference)
    return result


def package(checkpoint,predictions,out,root,latency=None,selection=None):
    checkpoint,predictions,out=Path(checkpoint),Path(predictions),Path(out)
    report=read(predictions/'report.json')
    check_csv(predictions/'submission.csv',root)
    if report['checkpoint_sha256']!=sha(checkpoint) or not report['arguments'].get('calibration'):
        raise ValueError('Only a hash-bound frozen test export can be packaged')
    if report['validation_recomputed']:raise ValueError('Freeze validation before test inference')
    if out.exists():raise FileExistsError('Preserve existing package: '+str(out))
    out.mkdir(parents=True)
    project=Path(__file__).resolve().parents[1]
    shutil.copytree(project/'src',out/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'tools').mkdir()
    (out/'tools/__init__.py').write_text('',encoding='utf-8')
    for name in ['predict_chexnet.py','cost_chexnet.py','gpu_idle.py','package_chexnet.py']:
        shutil.copy2(project/'tools'/name,out/'tools'/name)
    (out/'predict.py').write_text("from tools.predict_chexnet import main\n\nif __name__=='__main__':main()\n",encoding='utf-8')
    cost_entry='''import argparse
from pathlib import Path
import sys
from tools.cost_chexnet import main

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('submission')
    parser.add_argument('pictures')
    parser.add_argument('warmup_pictures')
    parser.add_argument('--root',required=True)
    parser.add_argument('--out',default='cost_results')
    args=parser.parse_args()
    bundle=Path(__file__).resolve().parent
    sys.argv=[sys.argv[0],'--checkpoint',str(bundle/'models/model.pt'),'--calibration',str(bundle/'report.json'),
        '--submission',args.submission,'--pictures',args.pictures,'--warmup-pictures',args.warmup_pictures,
        '--root',args.root,'--out',args.out]
    main()
'''
    (out/'cost.py').write_text(cost_entry,encoding='utf-8')
    (out/'models').mkdir()
    shutil.copy2(checkpoint,out/'models/model.pt')
    for name in ['requirements.txt','requirements-competition.txt']:
        shutil.copy2(project/name,out/name)
    for name in ['submission.csv','report.json']:
        shutil.copy2(predictions/name,out/name)
    checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    information={key:checkpoint_data.get(key) for key in ['epoch','config','provenance','selection_basis']}
    (out/'checkpoint_info.json').write_text(json.dumps(information,indent=2)+'\n',encoding='utf-8')
    (out/'labels.json').write_text(json.dumps(dict(enumerate(LABELS,1)),indent=2)+'\n',encoding='utf-8')
    upstream=(information.get('provenance') or {}).get('pretraining')
    if upstream:
        source=Path(upstream['cached_file'])
        if sha(source)!=upstream['sha256'] or 'ImageNet' not in upstream['training_data']:
            raise ValueError('Generic ImageNet provenance does not match')
        (out/'pretrain_model').mkdir()
        shutil.copy2(source,out/'pretrain_model'/source.name)
        (out/'licenses').mkdir()
        shutil.copy2(project/'docs/pretraining/torchvision-LICENSE.txt',out/'licenses/torchvision-LICENSE.txt')
        (out/'pretraining_sources.json').write_text(json.dumps(upstream,indent=2)+'\n',encoding='utf-8')
    if latency:shutil.copy2(latency,out/'latency.json')
    if selection:shutil.copy2(selection,out/'selection.json')
    text='''# CheXNet 比赛候选材料

初赛使用 `submission.csv`，本工具不执行网站提交。验证得分及实际推理环境在 `report.json`，隐藏测试成绩未知。

在Python3.12环境安装 `requirements-competition.txt`，然后运行：

```
python predict.py --checkpoint models/model.pt --root "<官方数据目录>" --calibration report.json --out reproduced
python cost.py submission.csv "<官方100张目录>" "<官方5张目录>" --root "<官方数据目录>"
```

复现只需 `test.csv` 和测试原图，不需要训练或验证标签、缓存、网络下载。RGB短边256后按冻结设置取224中心或十裁剪，使用ImageNet归一化；十裁剪逐裁剪sigmoid后均值。检查概率优先正位聚合，无正位时回退到全部已有图像。

测时连续三轮，计入读取原图、所需同检查配套图、预处理、前向、聚合和实际CSV写出；模型初始化不计。`latency.json` 是本机结果，T4尚未实测。`manifest.json` 绑定材料摘要和部署体积。`pretrain_model/` 仅在有公开ImageNet来源时供审计，不参与推理加载。

方法参考 [arnoweng/CheXNet](https://github.com/arnoweng/CheXNet) 与 [CheXNet论文](https://arxiv.org/abs/1711.05225)，适配为本赛十标签，只使用官方train训练和通用ImageNet初始化；未使用NIH医学模型。完整训练与实验记录见 [项目仓库](https://github.com/weiyu687/CXR-MLC-CAD)。
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    size=(out/'models/model.pt').stat().st_size
    deployed=size+sum((out/name).stat().st_size for name in ['report.json','labels.json','checkpoint_info.json'])
    if deployed>500_000_000:raise ValueError('Deployment exceeds 500MB')
    manifest={'model_bytes':size,'deployment_bytes':deployed,'official_website_submission_performed':False,
        'unused_upstream_weights_included_for_audit_only':bool(upstream),'files':{
            path.relative_to(out).as_posix():sha(path) for path in sorted(out.rglob('*')) if path.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return verify(out,root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--predictions')
    parser.add_argument('--root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--latency-report')
    parser.add_argument('--selection-report')
    parser.add_argument('--verify',help='Verify an existing bundle instead of creating one')
    parser.add_argument('--reproduced',help='Independent frozen test-only CSV')
    args=parser.parse_args()
    if args.verify:
        result=verify(args.verify,args.root,args.reproduced)
        target=Path(args.out)
        if target.exists():raise FileExistsError('Preserve existing verification report')
        target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:
        if not args.checkpoint or not args.predictions:parser.error('--checkpoint and --predictions are required')
        result=package(args.checkpoint,args.predictions,args.out,args.root,args.latency_report,args.selection_report)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
