param([string]$Python='C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$cache='C:/Users/Administrator/.cache/cxr-mlc-cad/448'
foreach ($directory in @('outputs/resnet50_imagenet_hold_s2030','outputs/resnet50_imagenet_full_s2031','outputs/effv2s_imagenet_s2032')) {
    if (Test-Path $directory) { throw "Preserve existing experiment: $directory" }
}
& $Python -u train.py --model tv_resnet50 --pretrained --size 320 --batch-size 32 --epochs 16 --lr 0.0002 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --patient-holdout 0.12 --selection-aggregation frontal --seed 2030 --workers 2 --cache $cache --out outputs/resnet50_imagenet_hold_s2030
if ($LASTEXITCODE -ne 0) { throw 'ImageNet ResNet50 patient-holdout training failed' }
& $Python -u train.py --model tv_resnet50 --init outputs/resnet50_imagenet_hold_s2030/best.pt --size 384 --batch-size 24 --epochs 8 --lr 0.00005 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2031 --workers 2 --cache $cache --out outputs/resnet50_imagenet_full_s2031
if ($LASTEXITCODE -ne 0) { throw 'ImageNet ResNet50 full-data refinement failed' }
& $Python -u train.py --model tv_efficientnet_v2_s --pretrained --size 320 --batch-size 24 --epochs 18 --lr 0.00015 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2032 --workers 2 --cache $cache --out outputs/effv2s_imagenet_s2032
if ($LASTEXITCODE -ne 0) { throw 'ImageNet EfficientNetV2-S training failed' }
'IMAGENET_ROUND2_TRAINING_COMPLETE' | Set-Content outputs/imagenet_round2_training_complete.txt
