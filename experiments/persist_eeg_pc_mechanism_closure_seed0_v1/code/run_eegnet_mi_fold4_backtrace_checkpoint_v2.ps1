<# Wait for six native P_t audits, then run final-P backtrace and checkpoint mechanisms. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold4_seed0'
$backtraceEntry=Join-Path $experiment 'code\audit_final_p_backtrace_eegnet_v2.py'
$mechanismEntry=Join-Path $experiment 'code\audit_checkpoint_mechanism_eegnet_v2.py'
$mainLog=Join-Path $runtime 'scheduled_backtrace_checkpoint_EEGNet_OpenBMI_MI_f4_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
function Wait-Resources([string]$Phase){while($true){$free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB;$gpu=[int](([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim());if($free-ge40-and$gpu-ge16000){"$Phase RESOURCE_READY free_gb=$([math]::Round($free,2)) gpu_mib=$gpu utc=$([datetime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8;return};Start-Sleep 30}}
function Run-Python([string]$Phase,[string]$Args,[string]$LogName){Wait-Resources $Phase;$log=Join-Path $runtime $LogName;if(Test-Path $log){throw "$Phase log already exists"};& cmd.exe /d /c ('"'+$python+'" -u '+$Args+' >> "'+$log+'" 2>&1');if($LASTEXITCODE-ne0){throw "$Phase exited $LASTEXITCODE"}}
try{
 if(Test-Path $mainLog){throw 'pipeline log already exists'}
 if((Get-FileHash $backtraceEntry -Algorithm SHA256).Hash.ToLowerInvariant()-ne'8410b848359f5b6b8afa5df1cd30b5b70aefeaadd57ad2ce7f096cdf511d7149'){throw 'backtrace source SHA mismatch'}
 if((Get-FileHash $mechanismEntry -Algorithm SHA256).Hash.ToLowerInvariant()-ne'097f6c25e200392359d630add0bfa7c36431bc3813234ade48771e481096a520'){throw 'mechanism source SHA mismatch'}
 if((Get-FileHash (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant()-ne'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'){throw 'analysis lock SHA mismatch'}
 if((Get-FileHash (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant()-ne'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'){throw 'amendment SHA mismatch'}
 "BACKTRACE_CHECKPOINT_F4_START utc=$([datetime]::UtcNow.ToString('o'))"|Out-File $mainLog -Encoding utf8
 $endpointPath=Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json';while(-not(Test-Path $endpointPath)){Start-Sleep 15}
 $endpoint=Get-Content $endpointPath -Raw|ConvertFrom-Json;$epochs=@($endpoint.schedule|ForEach-Object{[int]$_.epoch}|Sort-Object -Unique);if($epochs.Count-ne6){throw "expected six endpoint epochs, got $($epochs.Count)"}
 while($true){$valid=0;foreach($epoch in $epochs){$p=Join-Path $cell ("checkpoint_native_pt_v3\epoch_{0:d3}.json"-f$epoch);if(Test-Path $p){$j=Get-Content $p -Raw|ConvertFrom-Json;if($j.status-eq'CHECKPOINT_NATIVE_PT_COMPLETE_REPLICA_ONLY'-and$j.final_heldout_accessed-eq$false){$valid++}}};if($valid-eq6){break};Start-Sleep 20}
 $backtrace=Join-Path $cell 'FINAL_P_BACKTRACE_V1.json'
 if(-not(Test-Path $backtrace)){Run-Python 'FINAL_P_BACKTRACE' ('"'+$backtraceEntry+'" --task OpenBMI_MI --fold 4') 'scheduled_final_p_backtrace_EEGNet_OpenBMI_MI_f4_v2.log'}
 $bj=Get-Content $backtrace -Raw|ConvertFrom-Json;if($bj.status-ne'FINAL_P_BACKTRACE_COMPLETE_REPLICA_ONLY'-or$bj.final_heldout_accessed-ne$false){throw 'backtrace invalid'}
 foreach($epoch in $epochs){$tag='{0:d3}'-f$epoch;$out=Join-Path $cell "checkpoint_mechanism_v1\epoch_$tag.json";if(-not(Test-Path $out)){Run-Python "CHECKPOINT_$tag" ('"'+$mechanismEntry+'" --task OpenBMI_MI --fold 4 --epoch '+$epoch) "scheduled_checkpoint_mechanism_EEGNet_OpenBMI_MI_f4_e${tag}_v2.log"};$j=Get-Content $out -Raw|ConvertFrom-Json;if($j.status-ne'CHECKPOINT_MECHANISM_COMPLETE_REPLICA_ONLY'-or$j.final_heldout_accessed-ne$false){throw "checkpoint $epoch invalid"}}
 "BACKTRACE_CHECKPOINT_F4_COMPLETE utc=$([datetime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8;exit 0
}catch{if(Test-Path $mainLog){"BACKTRACE_CHECKPOINT_F4_EXCEPTION $_ utc=$([datetime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8};exit 1}
