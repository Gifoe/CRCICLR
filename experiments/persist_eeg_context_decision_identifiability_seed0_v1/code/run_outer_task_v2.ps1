param([Parameter(Mandatory=$true)][ValidateRange(0,4)][int]$Fold)
$ErrorActionPreference = 'Stop'
$runtime = 'D:\nips-temp\TotalP\P1\pc_context_decision_identifiability_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $runtime 'code\outer_v2.py'
$env:CID_RUNTIME = $runtime
$env:CID_UPSTREAM_TRAIN = 'D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime\train'
$env:CID_CORE_CODE = Join-Path $runtime 'code\run_v2.py'
$env:CID_SDG_CODE = 'D:\nips-temp\TotalP\P1\shared_decision_geometry_full_runtime\code\run_full_v1.py'
$env:SEVEN_REPO = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$env:SEVEN_RUNTIME = 'D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$stdout = Join-Path $runtime ("outer_fold{0}_v2.stdout.log" -f $Fold)
$stderr = Join-Path $runtime ("outer_fold{0}_v2.stderr.log" -f $Fold)
$exit = Join-Path $runtime ("outer_fold{0}_v2.exit" -f $Fold)
for ($i=0; $i -lt 5; $i++) {
    if (-not (Test-Path -LiteralPath (Join-Path $runtime ("fold{0}\TRAIN_SEAL.json" -f $i)))) {
        throw "TRAIN fold $i not sealed"
    }
}
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr) -or
    (Test-Path -LiteralPath $exit) -or
    (Test-Path -LiteralPath (Join-Path $runtime ("fold{0}\outer_results.json" -f $Fold)))) {
    throw 'Existing OUTER attempt; refuse duplicate'
}
$freeRam = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
if ($freeRam -lt 2GB) { throw "RAM reserve gate: $freeRam" }
$gpu = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
if (-not $gpu -or [int]($gpu | Select-Object -First 1) -lt 16000) {
    throw "GPU free-memory gate: $gpu"
}
$p = Start-Process -FilePath $python -ArgumentList @('-u', ('"'+$script+'"'), '--fold', "$Fold") -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT $($p.ExitCode) $([DateTime]::UtcNow.ToString('o'))" | Set-Content -LiteralPath $exit -Encoding UTF8
if ($p.ExitCode -ne 0) { throw "OUTER evaluation failed: $($p.ExitCode)" }
