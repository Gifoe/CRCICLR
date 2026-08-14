$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full'
$python = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
$env:PYTHONPATH = "$repo\src"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$jobs = @(
    @(0, 0),
    @(0, 1),
    @(1, 0),
    @(1, 1),
    @(2, 0),
    @(2, 1)
)
foreach ($job in $jobs) {
    $fold = $job[0]
    $seed = $job[1]
    Write-Output "START SI_V2 fold=$fold seed=$seed"
    & $python "$repo\p4_selective_invariance.py" --mode development --version SI_V2 --fold $fold --seed $seed
    if ($LASTEXITCODE -ne 0) {
        throw "SI_V2 fold=$fold seed=$seed failed with exit code $LASTEXITCODE"
    }
    Write-Output "DONE SI_V2 fold=$fold seed=$seed"
}
Write-Output 'PANEL_RUNS_COMPLETE'
