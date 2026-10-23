param([int]$TrainingProcessId=59232)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
if (!(Test-Path outputs/imagenet_round2_training_complete.txt)) { throw 'ImageNet experiments did not finish successfully' }
& C:/Users/Administrator/.cache/cxr-mlc-cad/runtime_target312/Scripts/python.exe -u tools/finalize.py --runs outputs/resnet18_cap_asl_s44 outputs/resnet50_imagenet_hold_s2030 outputs/resnet50_imagenet_full_s2031 outputs/effv2s_imagenet_s2032 --cache C:/Users/Administrator/.cache/cxr-mlc-cad/448 --ssd-root C:/Users/Administrator/.cache/cxr-mlc-cad/test_ssd --timing-root C:/Users/Administrator/.cache/cxr-mlc-cad/valid_ssd --timing-split valid --batch-size 1 --latency-statistic mean --latency-ms 85 --samples 100 --work outputs/final_selection_v2 --out submission/round2_bundle
if ($LASTEXITCODE -ne 0) { throw 'ImageNet final delivery failed; inspect logs' }
'IMAGENET_ROUND2_FINALIZATION_COMPLETE'
