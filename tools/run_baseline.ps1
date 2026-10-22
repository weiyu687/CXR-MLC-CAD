param(
    [int]$PrepareProcessId = 0,
    [string]$Python = 'C:/ProgramData/Anaconda3/envs/cxr-mlc-cad/python.exe',
    [string]$PackedCache = 'C:/Users/Administrator/.cache/cxr-mlc-cad/448'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($PrepareProcessId -gt 0) {
    Write-Output 'Waiting for private resized image cache to finish.'
    Wait-Process -Id $PrepareProcessId -ErrorAction SilentlyContinue
}
& $Python -u tools/pack_cache.py --out $PackedCache
if ($LASTEXITCODE -ne 0) { throw 'Cache packing failed' }
& $Python -u tools/check_pipeline.py
if ($LASTEXITCODE -ne 0) { throw 'Pipeline checks failed' }
& $Python -u train.py --model resnet18 --epochs 35 --size 320 --batch-size 48 --workers 2 --cache $PackedCache --out outputs/resnet18_s42
if ($LASTEXITCODE -ne 0) { throw 'Baseline training failed' }
& $Python -u predict.py --checkpoint outputs/resnet18_s42/best.pt --cache $PackedCache --out submission/baseline --workers 2
if ($LASTEXITCODE -ne 0) { throw 'Prediction failed' }
& $Python -u tools/benchmark.py --checkpoint outputs/resnet18_s42/best.pt --out outputs/resnet18_s42/latency.json
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed' }
& $Python -u tools/package.py --checkpoint outputs/resnet18_s42/best.pt --predictions submission/baseline --out submission/baseline_bundle
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed' }
'BASELINE_BUNDLE_COMPLETE' | Set-Content outputs/baseline_complete.txt
