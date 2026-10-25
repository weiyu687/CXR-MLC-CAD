"""Package code, selected checkpoints and actual predictions without raw images."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from predict import validate_submission
from src.data import LABELS
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',nargs='+',required=True)
    ap.add_argument('--predictions',default='submission')
    ap.add_argument('--out',default='submission/final')
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--latency-report',help='Optional measured report bound to the same models and runtime')
    ap.add_argument('--selection-report',help='Optional final selection evidence')
    args=ap.parse_args()
    checks=validate_submission(Path(args.predictions)/'submission.csv',args.root)
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new package directory; existing package preserved')
    out.mkdir(parents=True)
    for f in ['predict.py','train.py','cost.py','requirements.txt','requirements-competition.txt']:
        shutil.copy2(f,out/f)
    shutil.copytree('src',out/'src',ignore=shutil.ignore_patterns('__pycache__'))
    (out/'models').mkdir()
    total=0; paths=[];checkpoint_info=[]
    for i,f in enumerate(args.checkpoint):
        target=out/'models'/f'model_{i+1}.pt';shutil.copy2(f,target)
        total+=target.stat().st_size;paths.append('models/'+target.name)
        ck=torch.load(target,map_location='cpu',weights_only=False)
        checkpoint_info.append({'path':paths[-1],'training_configuration':ck['config'],
                                'epoch':ck.get('epoch'),'training_selection_aggregation':ck.get('selection_aggregation','mean'),
                                'training_selection_metrics':ck.get('metrics'),'holdout_selection_metrics':ck.get('holdout_metrics'),
                                'provenance':ck.get('provenance')})
        del ck
    upstream=[]
    for item in checkpoint_info:
        pretraining=(item.get('provenance') or {}).get('pretraining')
        if pretraining and pretraining['sha256'] not in [p['sha256'] for p in upstream]:
            source=Path(pretraining['cached_file'])
            assert hashlib.sha256(source.read_bytes()).hexdigest()==pretraining['sha256']
            folder=out/'pretrain_model';folder.mkdir(exist_ok=True)
            shutil.copy2(source,folder/source.name)
            upstream.append(pretraining)
    if upstream:
        shutil.copytree('docs/pretraining',out/'licenses')
        (out/'pretraining_sources.json').write_text(json.dumps(upstream,indent=2),encoding='utf-8')
    assert total<=500_000_000,'Model files exceed 500MB'
    for f in ['submission.csv','report.json']:
        shutil.copy2(Path(args.predictions)/f,out/f)
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    model_hashes=[hashlib.sha256((out/p).read_bytes()).hexdigest() for p in paths]
    if model_hashes!=report['checkpoint_sha256']:
        raise ValueError('Packaged weights differ from those used to generate predictions')
    if args.latency_report:
        latency=json.loads(Path(args.latency_report).read_text(encoding='utf-8'))
        assert latency['checkpoint_sha256']==model_hashes
        assert latency['checkpoint_bytes']==total and latency['models']==len(paths)
        assert latency['tta']==report['arguments']['tta']
        assert latency.get('ensemble_weights')==report['arguments'].get('ensemble_weights')
        assert latency['runtime']==report['runtime']
        timing_split=latency.get('split','test')
        assert latency.get('dataset_csv_sha256',latency['test_csv_sha256'])==hashlib.sha256((Path(args.root)/(timing_split+'.csv')).read_bytes()).hexdigest()
        shutil.copy2(args.latency_report,out/'latency.json')
    if args.selection_report:
        selection=json.loads(Path(args.selection_report).read_text(encoding='utf-8'))
        assert selection['selected']['checkpoint_sha256']==model_hashes
        assert selection['selected']['aggregation']==report['arguments']['aggregation']
        assert selection['selected']['tta']==report['arguments']['tta']
        assert selection['selected'].get('ensemble_weights')==report['arguments'].get('ensemble_weights')
        assert abs(selection['selected']['metrics']['macro_auc']-report['validation']['macro_auc'])<=1e-8
        shutil.copy2(args.selection_report,out/'selection.json')
    (out/'checkpoint_info.json').write_text(json.dumps(checkpoint_info,indent=2),encoding='utf-8')
    (out/'labels.json').write_text(json.dumps(dict(enumerate(LABELS,1)),indent=2),encoding='utf-8')
    switches=' --calibration report.json'
    command='python predict.py --checkpoint '+' '.join(paths)+' --root "<official_data_root>" --cache "<cache_root>" --out reproduced --workers 2'+switches
    readme=f'''# 提交材料

初赛上传 `submission.csv` 即可，不需要上传此整个目录。

复赛复现：安装 requirements.txt 后在此目录运行。官方 Python 3.12 容器可使用 requirements-competition.txt 固定 PyTorch 2.10 / torchvision 0.25；此目标环境仍需目标机复验。

```
{command}
```

将占位路径替换为官方数据目录与缩放缓存目录。没有缓存时可以直接读取原图，但更慢。此命令只需 test.csv 与测试图像，使用已冻结的验证集阈值与推理设置，不需要验证集标签。

`labels.json` 记录类别映射，`checkpoint_info.json` 记录权重训练配置、选中轮次和来源信息，`manifest.json` 记录文件摘要。旧基线未记录的来源字段为 null，不据此声称已验证对应环境。

实际生成时的运行环境见 report.json 的 runtime 字段。若包含 latency.json / selection.json，它们分别记录同一模型与环境的本机样本测时及验证集选择依据；本机样本通过不等同于 T4 或整个测试集最坏延迟达标。

模型仅以官方 train 划分训练，valid 用于模型和阈值选择。验证得分不是隐藏测试成绩。T4 推理延迟与官方环境仍需目标机复验。未执行网站提交。

若有 pretrain_model/ 和 licenses/，它们是官网 2026-10-23 规则第 1 条允许的非医学 ImageNet 初始化及其核验材料；不参与推理加载。医学数据仅来自比赛训练集。

新版官方测时入口：`python cost.py submission.csv <pictures_100> <pictures_5> --root <official_data_root>`。使用 5 张预热、100 张输入、连续 3 轮，并计入实际结果写出。为与检查级 CSV 一致，脚本对每张计时输入重复其完整检查推理，必要时从 --root 读取同检查配套图像；这是比单张前向更保守的口径。其结果不能等同于官方 T4 复测。样本数不足时仅可用 --samples / --warmup 做脚本检查，不算官方协议测时。
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    deployment_bytes=total+sum((out/name).stat().st_size for name in ['report.json','labels.json','checkpoint_info.json'])
    assert deployment_bytes<=500_000_000,'Deployed weights and necessary calibration exceed 500MB'
    manifest={'checks':checks,'model_bytes':total,'deployment_bytes':deployment_bytes,
              'unused_upstream_weights_included_for_audit_only':bool(upstream),'files':{}}
    for p in sorted(out.rglob('*')):
        if p.is_file():manifest['files'][p.relative_to(out).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'folder':str(out.resolve()),**checks,'model_bytes':total}))
if __name__=='__main__':main()
