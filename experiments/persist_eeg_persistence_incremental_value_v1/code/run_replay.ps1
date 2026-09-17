$ErrorActionPreference = 'Stop'
$env:PEEH_RUNTIME = 'D:\nips-temp\TotalP\P1\persist_incremental_value_runtime\seed0_replay'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$runtime = 'D:\nips-temp\TotalP\P1\persist_incremental_value_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_persistence_incremental_value_v1\code\replay_seed0_isolated.py'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
& $python -u $script *>> (Join-Path $runtime 'seed0_isolated.log')
$status = $LASTEXITCODE
[IO.File]::WriteAllText((Join-Path $runtime 'seed0_isolated.exit'), [string]$status)
exit $status
