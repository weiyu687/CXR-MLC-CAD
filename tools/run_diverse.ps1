param(
    [int]$PrerequisiteProcessId = 0,
    [string]$Python = 'C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe',
    [string]$PackedCache = 'C:/Users/Administrator/.cache/cxr-mlc-cad/448'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($PrerequisiteProcessId -gt 0) {
    Write-Output 'Waiting for the live attention/ASL pipeline to finish.'
    Wait-Process -Id $PrerequisiteProcessId -ErrorAction SilentlyContinue
}
if (!(Test-Path outputs/improvements_complete.txt)) { throw 'Prerequisite experiments did not complete successfully' }
if (Test-Path outputs/efficientnetv2_b0_s123) { throw 'Experiment already exists; resume explicitly' }
& $Python -u train.py --model tf_efficientnetv2_b0 --epochs 50 --lr 0.001 --loss asl --seed 123 --size 384 --batch-size 32 --workers 2 --selection-aggregation auto --cache $PackedCache --out outputs/efficientnetv2_b0_s123
if ($LASTEXITCODE -ne 0) { throw 'Independent EfficientNetV2 training failed' }
& $Python -u tools/select_aggregation.py --predictions outputs/efficientnetv2_b0_s123/valid_predictions.npz --out outputs/efficientnetv2_b0_s123/aggregation.json
if ($LASTEXITCODE -ne 0) { throw 'EfficientNetV2 aggregation comparison failed' }
$aggregation=(Get-Content outputs/efficientnetv2_b0_s123/aggregation.json -Raw | ConvertFrom-Json).best
& $Python -u predict.py --checkpoint outputs/efficientnetv2_b0_s123/best.pt --aggregation $aggregation --cache $PackedCache --out submission/efficientnetv2 --workers 2
if ($LASTEXITCODE -ne 0) { throw 'EfficientNetV2 inference failed' }
'DIVERSE_CANDIDATE_COMPLETE' | Set-Content outputs/diverse_complete.txt
