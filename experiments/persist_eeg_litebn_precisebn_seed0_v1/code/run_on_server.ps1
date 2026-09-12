$ErrorActionPreference = 'Stop'
$taskRepo = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$taskPython = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$taskScript = "$taskRepo\experiments\persist_eeg_litebn_precisebn_seed0_v1\code\run_precisebn.py"
# Keep this command/session alive. Do not terminate other training processes.
& $taskPython -u $taskScript --repo $taskRepo --recovered 'D:\nips-temp\TotalP\P1\precisebn_recovered_historical' --cache 'D:\nips-temp\TotalP\P1\srgeo_combined_cache' --runtime 'D:\nips-temp\TotalP\P1\precisebn_seed0_v2_runtime' --phase all
if ($LASTEXITCODE -ne 0) { throw "Precise-BN stopped at a failed invariant; inspect log before resuming." }
