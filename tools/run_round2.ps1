param([string]$Python = 'C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$cache='C:/Users/Administrator/.cache/cxr-mlc-cad/448'
if (Test-Path outputs/densenet121_front_mix_s2026) { throw 'Preserve existing experiment' }
& $Python -u train.py --model densenet121 --size 320 --batch-size 32 --epochs 45 --lr 0.0008 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --patient-holdout 0.12 --selection-aggregation frontal --seed 2026 --workers 2 --cache $cache --out outputs/densenet121_front_mix_s2026
if ($LASTEXITCODE -ne 0) { throw 'DenseNet training failed' }
if (Test-Path outputs/efficientnetv2_front_mix_s2027) { throw 'Preserve existing experiment' }
& $Python -u train.py --model tf_efficientnetv2_b0 --init outputs/efficientnetv2_b0_s123/best.pt --size 384 --batch-size 32 --epochs 18 --lr 0.00012 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2027 --workers 2 --cache $cache --out outputs/efficientnetv2_front_mix_s2027
if ($LASTEXITCODE -ne 0) { throw 'EfficientNet refinement failed' }
'ROUND2_TRAINING_COMPLETE' | Set-Content outputs/round2_training_complete.txt
