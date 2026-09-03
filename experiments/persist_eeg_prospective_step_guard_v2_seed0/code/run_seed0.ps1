$ErrorActionPreference = 'Continue'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET'
$exp = Join-Path $repo 'experiments\persist_eeg_prospective_step_guard_v2_seed0'
$runtime = Join-Path $exp 'runtime'
$py = 'E:\Anaconda\envs\benchmark_tta_win\python.exe'
$script = Join-Path $exp 'code\persist_au_seed0.py'
$stdout = Join-Path $runtime 'SEED0_RUN.log'
$stderr = Join-Path $runtime 'SEED0_RUN.err.log'
$exitPath = Join-Path $runtime 'SEED0_RUN.exit.json'

New-Item -ItemType Directory -Force -Path $runtime | Out-Null
Remove-Item -LiteralPath $stdout, $stderr, $exitPath -Force -ErrorAction SilentlyContinue
$env:CANONICAL_REPO = $repo
$env:PERSIST_STAGE0_REPO = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full'
$env:PERSIST_WBCIC_CACHE = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_LAUNCH_BLOCKING = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PERSIST_TORCH_THREADS = '1'
Set-Location $repo
& $py -u $script --device cuda 1>> $stdout 2>> $stderr
$code = $LASTEXITCODE
$payload = @{ complete = ($code -eq 0); exit_code = $code; finished_at = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json
Set-Content -LiteralPath $exitPath -Value $payload -Encoding UTF8
exit $code
