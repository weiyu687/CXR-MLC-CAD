param([int]$TrainingProcessId = 59564)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
if (!(Test-Path outputs/round2_training_complete.txt)) { throw 'Round 2 training did not finish successfully' }
$python='C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe'
if (!(Test-Path outputs/densenet121_front_full_s2028/best.pt)) {
    if (Test-Path outputs/densenet121_front_full_s2028) { throw 'Incomplete full-training experiment; inspect before resuming' }
    & $python -u train.py --model densenet121 --init outputs/densenet121_front_mix_s2026/best.pt --size 320 --batch-size 32 --epochs 16 --lr 0.0001 --weight-decay 0.0001 --loss bce --positive-weight --frontal-training --regularize --mixup 0.2 --selection-aggregation frontal --seed 2028 --workers 2 --cache C:/Users/Administrator/.cache/cxr-mlc-cad/448 --out outputs/densenet121_front_full_s2028
    if ($LASTEXITCODE -ne 0) { throw 'Full-training refinement failed' }
}
& C:/Users/Administrator/.cache/cxr-mlc-cad/runtime_target312/Scripts/python.exe -u tools/finalize.py --runs outputs/resnet18_cap_asl_s44 outputs/efficientnetv2_b0_s123 outputs/densenet121_front_full_s2028 outputs/efficientnetv2_front_mix_s2027 --cache C:/Users/Administrator/.cache/cxr-mlc-cad/448 --ssd-root C:/Users/Administrator/.cache/cxr-mlc-cad/test_ssd --work outputs/final_selection_v2 --out submission/round2_bundle --samples 60
if ($LASTEXITCODE -ne 0) { throw 'Round 2 finalization failed' }
'ROUND2_FINALIZATION_COMPLETE'
