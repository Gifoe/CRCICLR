$ErrorActionPreference = 'Stop'
$taskRepo = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u "$taskRepo\experiments\persist_eeg_litebn_lighttail_seed0_v1\code\run_lighttail.py" --repo $taskRepo --recovered 'D:\nips-temp\TotalP\P1\precisebn_recovered_historical' --cache 'D:\nips-temp\TotalP\P1\srgeo_combined_cache' --historical-runtime 'D:\nips-temp\TotalP\P1\carrier_5fold_multiseed_stability_runtime' --runtime 'D:\nips-temp\TotalP\P1\lighttail_seed0_runtime' --phase all
if ($LASTEXITCODE -ne 0) { throw 'LightTail invariant failed; do not bypass a preflight gate.' }
