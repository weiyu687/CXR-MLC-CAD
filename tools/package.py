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
    args=ap.parse_args()
    checks=validate_submission(Path(args.predictions)/'submission.csv',args.root)
    out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new package directory; existing package preserved')
    out.mkdir(parents=True)
    for f in ['predict.py','train.py','requirements.txt','requirements-competition.txt']:
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
                                'training_selection_metrics':ck.get('metrics'),'provenance':ck.get('provenance')})
        del ck
    assert total<=500_000_000,'Model files exceed 500MB'
    for f in ['submission.csv','report.json']:
        shutil.copy2(Path(args.predictions)/f,out/f)
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    model_hashes=[hashlib.sha256((out/p).read_bytes()).hexdigest() for p in paths]
    if model_hashes!=report['checkpoint_sha256']:
        raise ValueError('Packaged weights differ from those used to generate predictions')
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

模型仅以官方 train 划分训练，valid 用于模型和阈值选择。验证得分不是隐藏测试成绩。T4 推理延迟与官方环境仍需目标机复验。未执行网站提交。
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    manifest={'checks':checks,'model_bytes':total,'files':{}}
    for p in sorted(out.rglob('*')):
        if p.is_file():manifest['files'][p.relative_to(out).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'folder':str(out.resolve()),**checks,'model_bytes':total}))
if __name__=='__main__':main()
