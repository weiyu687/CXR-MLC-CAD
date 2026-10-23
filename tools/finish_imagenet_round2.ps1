param([int]$TrainingProcessId=59232)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
if (!(Test-Path outputs/imagenet_round2_training_complete.txt)) { throw 'ImageNet experiments did not finish successfully' }
if (Test-Path outputs/effv2s_imagenet_448_s2033) { throw 'Preserve existing high-resolution experiment' }
& C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe -u train.py --model tv_efficientnet_v2_s --init outputs/effv2s_imagenet_s2032/best.pt --size 448 --batch-size 16 --epochs 6 --lr 0.00003 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2033 --workers 2 --cache C:/Users/Administrator/.cache/cxr-mlc-cad/448 --out outputs/effv2s_imagenet_448_s2033
if ($LASTEXITCODE -ne 0) { throw 'High-resolution ImageNet refinement failed' }
& C:/Users/Administrator/.cache/cxr-mlc-cad/runtime_target312/Scripts/python.exe -u tools/finalize.py --runs outputs/resnet18_cap_asl_s44 outputs/resnet50_imagenet_full_s2031 outputs/effv2s_imagenet_s2032 outputs/effv2s_imagenet_448_s2033 --cache C:/Users/Administrator/.cache/cxr-mlc-cad/448 --ssd-root C:/Users/Administrator/.cache/cxr-mlc-cad/test_ssd --timing-root C:/Users/Administrator/.cache/cxr-mlc-cad/valid_ssd --timing-split valid --study-timing --batch-size 1 --latency-statistic mean --latency-ms 100 --samples 100 --work outputs/final_selection_v2 --out submission/round2_bundle
if ($LASTEXITCODE -ne 0) { throw 'ImageNet final delivery failed; inspect logs' }
'IMAGENET_ROUND2_FINALIZATION_COMPLETE'
