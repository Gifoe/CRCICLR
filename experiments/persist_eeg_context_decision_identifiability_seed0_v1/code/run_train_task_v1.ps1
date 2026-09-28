param([Parameter(Mandatory=$true)][ValidateRange(0,4)][int]$Fold)
$ErrorActionPreference = 'Stop'
$runtime = 'D:\nips-temp\TotalP\P1\pc_context_decision_identifiability_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $runtime 'code\run.py'
$env:CID_RUNTIME = $runtime
$env:CID_UPSTREAM_TRAIN = 'D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime\train'
$log = Join-Path $runtime ("train_fold{0}_v1.log" -f $Fold)
$exit = Join-Path $runtime ("train_fold{0}_v1.exit" -f $Fold)
if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $exit) -or
    (Test-Path -LiteralPath (Join-Path $runtime ("fold{0}\TRAIN_SEAL.json" -f $Fold)))) {
    throw 'Existing train attempt; refuse duplicate'
}
$stamp = [DateTime]::UtcNow.ToString('o')
"START $stamp fold=$Fold" | Set-Content -LiteralPath $log -Encoding UTF8
& $python -u $script train --fold $Fold *>> $log
$code = $LASTEXITCODE
"EXIT $code $([DateTime]::UtcNow.ToString('o'))" | Set-Content -LiteralPath $exit -Encoding UTF8
if ($code -ne 0) { throw "Train analysis failed: $code" }
