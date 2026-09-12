$ErrorActionPreference = 'Stop'
& 'E:\Anaconda\envs\persist_stable_251\python.exe' -u 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK\experiments\persist_eeg_litebn_localstats_seed0_v1\code\run_localstats.py' --repo 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK' --recovered 'D:\nips-temp\TotalP\P1\precisebn_recovered_historical' --cache 'D:\nips-temp\TotalP\P1\srgeo_combined_cache' --historical-runtime 'D:\nips-temp\TotalP\P1\carrier_5fold_multiseed_stability_runtime' --runtime 'D:\nips-temp\TotalP\P1\localstats_seed0_runtime' --phase all
exit $LASTEXITCODE
