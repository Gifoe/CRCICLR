<# Run the separately provenanced prediction repair on the original server. #>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][ValidateSet('cell','queue','aggregate')][string]$Mode,
  [ValidateSet('EEGNet','EEGConformer')][string]$Model,
  [ValidateSet('OpenBMI_MI','OpenBMI_SSVEP')][string]$Task,
  [int]$Fold=-1
)
$ErrorActionPreference='Stop'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$entry=Join-Path $PSScriptRoot 'repair_error_predictability_v2.py'
$runtime='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:PERSIST_SOURCE_REPO='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$env:COUPLING_RUNTIME=$runtime
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONUNBUFFERED='1'
if($Mode -eq 'cell' -and ([string]::IsNullOrWhiteSpace($Model) -or [string]::IsNullOrWhiteSpace($Task) -or $Fold -notin 0..4)) { throw 'cell mode requires Model, Task, Fold 0..4' }
$log=Join-Path $runtime ("scheduled_prediction_repair_v2_${Mode}.log")
try {
  "PREDICTION_REPAIR_START mode=$Mode utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  if($Mode -eq 'queue') {
    foreach($m in @('EEGNet','EEGConformer')) {
      foreach($t in @('OpenBMI_MI','OpenBMI_SSVEP')) {
        foreach($f in 0..4) {
          do {
            $freeGB=[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)
            $gpuMiB=[int](((((& nvidia-smi --query-gpu=memory.free --format=csv,noheader) -join ',') -split ' ')[0]))
            if($freeGB -lt 40 -or $gpuMiB -lt 16000) { Start-Sleep -Seconds 30 }
          } while($freeGB -lt 40 -or $gpuMiB -lt 16000)
          "CELL_START $m $t $f freeGB=$freeGB gpuMiB=$gpuMiB" | Out-File -FilePath $log -Append -Encoding utf8
          $arguments="-u `"$entry`" --mode cell --model $m --task $t --fold $f"
          $command='"'+$python+'" '+$arguments+' >> "'+$log+'" 2>&1'
          & cmd.exe /d /c $command
          if($LASTEXITCODE -ne 0) { throw "prediction repair cell $m $t $f exited $LASTEXITCODE" }
        }
      }
    }
  } else {
    $arguments=if($Mode -eq 'cell') { "-u `"$entry`" --mode cell --model $Model --task $Task --fold $Fold" } else { "-u `"$entry`" --mode aggregate" }
    $command='"'+$python+'" '+$arguments+' >> "'+$log+'" 2>&1'
    & cmd.exe /d /c $command
    if($LASTEXITCODE -ne 0) { throw "prediction repair Python exited $LASTEXITCODE" }
  }
  "PREDICTION_REPAIR_END mode=$Mode exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
  exit 0
} catch {
  "PREDICTION_REPAIR_EXCEPTION mode=$Mode $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
