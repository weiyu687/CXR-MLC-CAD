param(
    [int]$PrerequisiteProcessId = 0,
    [string]$Python = 'C:/Users/Administrator/.cache/cxr-mlc-cad/runtime_target312/Scripts/python.exe',
    [string]$PackedCache = 'C:/Users/Administrator/.cache/cxr-mlc-cad/448',
    [string]$SsdRoot = 'C:/Users/Administrator/.cache/cxr-mlc-cad/test_ssd'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($PrerequisiteProcessId -gt 0) {
    Write-Output 'Waiting for the live independent-model pipeline to finish.'
    Wait-Process -Id $PrerequisiteProcessId -ErrorAction SilentlyContinue
}
if (!(Test-Path outputs/diverse_complete.txt)) { throw 'Independent model did not complete successfully' }
Start-Sleep -Seconds 10
& $Python -u tools/finalize.py --runs outputs/resnet18_s42 outputs/resnet18_cap_s43 outputs/resnet18_cap_asl_s44 outputs/efficientnetv2_b0_s123 --cache $PackedCache --ssd-root $SsdRoot --work outputs/final_selection_v1 --out submission/final_bundle --samples 60 --latency-ms 100
if ($LASTEXITCODE -ne 0) { throw 'Final selection or qualification failed; inspect stage logs' }
