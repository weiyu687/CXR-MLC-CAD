param([string]$Python='C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe',[switch]$SkipHoldout)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$cache='C:/Users/Administrator/.cache/cxr-mlc-cad/448'
foreach ($directory in @('outputs/resnet50_imagenet_full_s2031','outputs/effv2s_imagenet_s2032','outputs/densenet121_imagenet_s2034')) {
    if (Test-Path $directory) { throw "Preserve existing experiment: $directory" }
}
if (!$SkipHoldout) {
    if (Test-Path outputs/resnet50_imagenet_hold_s2030) { throw 'Preserve existing holdout experiment; use -SkipHoldout to continue from its checkpoint' }
    & $Python -u train.py --model tv_resnet50 --pretrained --head-lr-mult 1 --size 320 --batch-size 32 --epochs 16 --patience 3 --lr 0.0002 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --patient-holdout 0.12 --selection-aggregation frontal --seed 2030 --workers 2 --cache $cache --out outputs/resnet50_imagenet_hold_s2030
    if ($LASTEXITCODE -ne 0) { throw 'ImageNet ResNet50 patient-holdout training failed' }
} elseif (!(Test-Path outputs/resnet50_imagenet_hold_s2030/best.pt)) { throw 'Completed patient-holdout checkpoint is required' }
& $Python -u train.py --model tv_resnet50 --init outputs/resnet50_imagenet_hold_s2030/best.pt --size 384 --batch-size 24 --epochs 8 --patience 3 --lr 0.00005 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2031 --workers 2 --cache $cache --out outputs/resnet50_imagenet_full_s2031
if ($LASTEXITCODE -ne 0) { throw 'ImageNet ResNet50 full-data refinement failed' }
& $Python -u train.py --model tv_efficientnet_v2_s --pretrained --head-lr-mult 5 --size 320 --batch-size 24 --epochs 18 --patience 4 --lr 0.00015 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2032 --workers 2 --cache $cache --out outputs/effv2s_imagenet_s2032
if ($LASTEXITCODE -ne 0) { throw 'ImageNet EfficientNetV2-S training failed' }
& $Python -u train.py --model tv_densenet121 --pretrained --head-lr-mult 5 --size 320 --batch-size 32 --epochs 12 --patience 4 --lr 0.0001 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2034 --workers 2 --cache $cache --out outputs/densenet121_imagenet_s2034
if ($LASTEXITCODE -ne 0) { throw 'ImageNet DenseNet121 training failed' }
'IMAGENET_ROUND2_TRAINING_COMPLETE' | Set-Content outputs/imagenet_round2_training_complete.txt
