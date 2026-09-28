$ErrorActionPreference = 'Stop'
$runtime = 'D:\nips-temp\TotalP\P1\pc_context_decision_identifiability_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $runtime 'code\aggregate_v1.py'
$env:CID_RUNTIME = $runtime
$env:CID_UPSTREAM_TRAIN = 'D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime\train'
$env:CID_CORE_CODE = Join-Path $runtime 'code\run_v2.py'
$env:CID_OUTER_CODE = Join-Path $runtime 'code\outer_v2.py'
$stdout = Join-Path $runtime 'aggregate_v1.stdout.log'
$stderr = Join-Path $runtime 'aggregate_v1.stderr.log'
$exit = Join-Path $runtime 'aggregate_v1.exit'
for ($i=0; $i -lt 5; $i++) {
    if (-not (Test-Path -LiteralPath (Join-Path $runtime ("fold{0}\outer_results.json" -f $i)))) {
        throw "OUTER fold $i not complete"
    }
}
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr) -or
    (Test-Path -LiteralPath $exit) -or
    (Test-Path -LiteralPath (Join-Path $runtime 'outputs\DECISION_SUMMARY.json'))) {
    throw 'Existing aggregate attempt; refuse duplicate'
}
$p = Start-Process -FilePath $python -ArgumentList @('-u', ('"'+$script+'"')) -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT $($p.ExitCode) $([DateTime]::UtcNow.ToString('o'))" | Set-Content -LiteralPath $exit -Encoding UTF8
if ($p.ExitCode -ne 0) { throw "Aggregate failed: $($p.ExitCode)" }
