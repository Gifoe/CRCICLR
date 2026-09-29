$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_SEVEN_BACKBONE_HELDOUT_WORK'
$experiment = Join-Path $root 'experiments\persist_eeg_token_reliability_pooling_audit_seed0_v1'
$entry = Join-Path $experiment 'code\run.py'
$expected = 'A2EB3E2D43DBE4D2A153AD878D9AAB1FCD9B3BB3401E81CC4AAD5AE51A391E31'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry).Hash -ne $expected) { throw 'preflight entry hash mismatch' }
$runtime = 'D:\nips-temp\TotalP\P1\token_reliability_pooling_runtime'
if (Test-Path -LiteralPath (Join-Path $runtime 'preflight_fold0_v1.log')) { throw 'preflight log already exists' }
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$env:TOKEN_AUDIT_RUNTIME = $runtime
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$env:OFFICIAL_BACKBONE_ROOT = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime\official'
$env:MODERN_REPO = $root
$py = 'E:\Anaconda\envs\persist_stable_251\python.exe'
& $py $entry --fold 0 --preflight *> (Join-Path $runtime 'preflight_fold0_v1.log')
if ($LASTEXITCODE -ne 0) { throw "preflight failed with exit $LASTEXITCODE" }
