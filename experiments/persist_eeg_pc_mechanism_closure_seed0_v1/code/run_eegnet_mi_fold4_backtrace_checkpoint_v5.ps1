<# Corrected V5: accept the frozen native-P_t status emitted by the audited V5 producer. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1';$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1';$rt='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime';$cell=Join-Path $rt 'cells\eegnet\openbmi_mi\fold4_seed0';$py='E:\Anaconda\envs\persist_stable_251\python.exe';$back=Join-Path $exp 'code\audit_final_p_backtrace_eegnet_v2.py';$mech=Join-Path $exp 'code\audit_checkpoint_mechanism_eegnet_v2.py';$log=Join-Path $rt 'scheduled_backtrace_checkpoint_EEGNet_OpenBMI_MI_f4_v5.log'
$env:PERSIST_SOURCE_REPO=$repo;$env:MECHANISM_RUNTIME=$rt;$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime';$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime';$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official';$env:PYTHONUNBUFFERED='1'
function Gate{while($true){$f=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB;$g=[int](([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim());if($f-ge40-and$g-ge16000){return};Start-Sleep 30}}
function Run([string]$phase,[string]$Arguments,[string]$name){Gate;$pl=Join-Path $rt $name;if(Test-Path $pl){throw "$phase log already exists"};& cmd.exe /d /c ('"'+$py+'" -u '+$Arguments+' >> "'+$pl+'" 2>&1');if($LASTEXITCODE-ne0){throw "$phase exited $LASTEXITCODE"}}
try{
 if(Test-Path $log){throw 'V5 log already exists'}
 if((Get-FileHash $back -Algorithm SHA256).Hash.ToLowerInvariant()-ne'8410b848359f5b6b8afa5df1cd30b5b70aefeaadd57ad2ce7f096cdf511d7149'){throw 'backtrace SHA mismatch'}
 if((Get-FileHash $mech -Algorithm SHA256).Hash.ToLowerInvariant()-ne'097f6c25e200392359d630add0bfa7c36431bc3813234ade48771e481096a520'){throw 'mechanism SHA mismatch'}
 if((Get-FileHash (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant()-ne'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'){throw 'lock mismatch'}
 if((Get-FileHash (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant()-ne'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'){throw 'amendment mismatch'}
 'BACKTRACE_CHECKPOINT_F4_V5_START'|Out-File $log -Encoding utf8
 $ep=Get-Content (Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json') -Raw|ConvertFrom-Json;$epochs=@($ep.schedule|ForEach-Object{[int]$_.epoch}|Sort-Object -Unique);if($epochs.Count-ne6){throw 'schedule is not six unique epochs'}
 foreach($e in $epochs){$p=Join-Path $cell ("checkpoint_native_pt_v5\epoch_{0:d3}.json"-f$e);if(-not(Test-Path $p)){throw "native P_t missing $e"};$j=Get-Content $p -Raw|ConvertFrom-Json;if($j.status-ne'NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM'-or$j.final_heldout_accessed-ne$false){throw "native P_t invalid $e"}}
 $out=Join-Path $cell 'FINAL_P_BACKTRACE_V1.json';if(-not(Test-Path $out)){Run 'BACKTRACE' ('"'+$back+'" --task OpenBMI_MI --fold 4') 'scheduled_final_p_backtrace_EEGNet_OpenBMI_MI_f4_v5.log'};$j=Get-Content $out -Raw|ConvertFrom-Json;if($j.status-ne'FINAL_P_BACKTRACE_COMPLETE_REPLICA_ONLY'-or$j.final_heldout_accessed-ne$false){throw 'backtrace invalid'}
 foreach($e in $epochs){$tag='{0:d3}'-f$e;$out=Join-Path $cell "checkpoint_mechanism_v1\epoch_$tag.json";if(-not(Test-Path $out)){Run "MECH_$tag" ('"'+$mech+'" --task OpenBMI_MI --fold 4 --epoch '+$e) "scheduled_checkpoint_mechanism_EEGNet_OpenBMI_MI_f4_e${tag}_v5.log"};$j=Get-Content $out -Raw|ConvertFrom-Json;if($j.status-ne'CHECKPOINT_MECHANISM_COMPLETE_REPLICA_ONLY'-or$j.final_heldout_accessed-ne$false){throw "mechanism invalid $e"}}
 'BACKTRACE_CHECKPOINT_F4_V5_COMPLETE'|Out-File $log -Append -Encoding utf8;exit 0
}catch{"BACKTRACE_CHECKPOINT_F4_V5_EXCEPTION $_"|Out-File $log -Append -Encoding utf8;exit 1}




