$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$runtime = 'D:\nips-temp\TotalP\P1\persist_incremental_value_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_persistence_incremental_value_v1\code\replay_pswa_from_frozen_q.py'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
& $python -u $script *>> (Join-Path $runtime 'pswa_replay.log')
$status = $LASTEXITCODE
[IO.File]::WriteAllText((Join-Path $runtime 'pswa_replay.exit'), [string]$status)
exit $status
