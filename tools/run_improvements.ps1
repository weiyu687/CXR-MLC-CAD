param(
    [int]$BaselineProcessId = 0,
    [string]$Python = 'C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe',
    [string]$PackedCache = 'C:/Users/Administrator/.cache/cxr-mlc-cad/448'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($BaselineProcessId -gt 0) {
    Write-Output 'Waiting for the live baseline pipeline to finish.'
    Wait-Process -Id $BaselineProcessId -ErrorAction SilentlyContinue
}
if (!(Test-Path outputs/baseline_complete.txt)) { throw 'Baseline did not complete successfully' }
if ((Test-Path outputs/resnet18_cap_s43) -or (Test-Path outputs/resnet18_cap_asl_s44)) {
    throw 'Candidate directories already exist; preserve previous experiments and resume explicitly'
}
& $Python -u train.py --model resnet18_cap --init outputs/resnet18_s42/best.pt --epochs 12 --lr 0.0002 --seed 43 --size 384 --batch-size 32 --workers 2 --cache $PackedCache --out outputs/resnet18_cap_s43
if ($LASTEXITCODE -ne 0) { throw 'Attention fine-tuning failed' }
& $Python -u predict.py --checkpoint outputs/resnet18_cap_s43/best.pt --cache $PackedCache --out submission/attention --workers 2
if ($LASTEXITCODE -ne 0) { throw 'Attention inference failed' }
& $Python -u tools/select_aggregation.py --predictions outputs/resnet18_cap_s43/valid_predictions.npz --out outputs/resnet18_cap_s43/aggregation.json
if ($LASTEXITCODE -ne 0) { throw 'Attention aggregation comparison failed' }
& $Python -u train.py --model resnet18_cap --init outputs/resnet18_cap_s43/best.pt --epochs 10 --lr 0.0001 --loss asl --seed 44 --size 384 --batch-size 32 --workers 2 --cache $PackedCache --out outputs/resnet18_cap_asl_s44
if ($LASTEXITCODE -ne 0) { throw 'Asymmetric-loss fine-tuning failed' }
& $Python -u predict.py --checkpoint outputs/resnet18_cap_asl_s44/best.pt --cache $PackedCache --out submission/attention_asl --workers 2
if ($LASTEXITCODE -ne 0) { throw 'Asymmetric-loss inference failed' }
& $Python -u tools/select_aggregation.py --predictions outputs/resnet18_cap_asl_s44/valid_predictions.npz --out outputs/resnet18_cap_asl_s44/aggregation.json
if ($LASTEXITCODE -ne 0) { throw 'Asymmetric-loss aggregation comparison failed' }
'CANDIDATES_COMPLETE' | Set-Content outputs/improvements_complete.txt
