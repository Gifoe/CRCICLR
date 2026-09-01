$ErrorActionPreference = 'Continue'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET'
$py = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
$exp = Join-Path $repo 'experiments\persist_eeg_cde_seed0_pilot'
$runtime = Join-Path $exp 'runtime'
$stdout = Join-Path $runtime 'PILOT_RUN.log'
$stderr = Join-Path $runtime 'PILOT_RUN.err.log'
$exitPath = Join-Path $runtime 'PILOT_RUN.exit.json'
$env:CDE_REPO = $repo
$env:CANONICAL_REPO = $repo
$env:PERSIST_STAGE0_REPO = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full'
$env:PERSIST_WBCIC_CACHE = 'D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache'
Set-Location $repo
$script = Join-Path $exp 'code\cde_seed0_pilot.py'
& $py -u $script --datasets OpenBMI WBCIC --device cuda 1>> $stdout 2>> $stderr
$code = $LASTEXITCODE
$payload = @{complete = ($code -eq 0); exit_code = $code; finished_at = (Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json
Set-Content -LiteralPath $exitPath -Value $payload -Encoding UTF8
exit $code
