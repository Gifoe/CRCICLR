$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1';$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1';$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime';$cell=Join-Path $runtime 'cells\eegnet\openbmi_ssvep\fold4_seed0'
$entry=Join-Path $exp 'code\build_cell_compact_preview_eegnet_v9.py';$python='E:\Anaconda\envs\persist_stable_251\python.exe';$log=Join-Path $runtime 'scheduled_compact_recovery_EEGNet_OpenBMI_SSVEP_f4_v1.log'
$env:PERSIST_SOURCE_REPO=$repo;$env:MECHANISM_RUNTIME=$runtime;$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime';$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime';$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official';$env:PYTHONUNBUFFERED='1'
try{
 if(Test-Path $log){throw 'recovery log exists'};"COMPACT_RECOVERY_F4_V1_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Encoding utf8
 if((Get-FileHash $entry -Algorithm SHA256).Hash.ToLowerInvariant()-ne'0d9bbfccd4a8efe7ad657f7757ccad50b870d3f9c0f6575ffd1a3e96047eb2c9'){throw 'compact V9 source SHA mismatch'}
 $failed=Join-Path $runtime 'compact_preview\eegnet_openbmi_ssvep_fold4_seed0_v7\FAIL_CLOSED.json';$fa=Get-Content $failed -Raw|ConvertFrom-Json;if($fa.status-ne'FAIL_CLOSED'-or$fa.final_heldout_accessed-ne$false-or$fa.reason-notmatch'FINAL_P_BACKTRACE_V1.json'){throw 'preserved V7 failure invalid'}
 $s2=Join-Path $cell 'SUBJECT_SESSION_AUDIT_V2.json';$sa=Get-Content $s2 -Raw|ConvertFrom-Json;if($sa.status-ne'SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE'-or[int]$sa.fold-ne4-or@($sa.subject_session_signature_rows).Count-ne68-or$sa.final_heldout_accessed-ne$false){throw 'subject V2 invalid'};$sha=(Get-FileHash $s2 -Algorithm SHA256).Hash.ToLowerInvariant()
 if(@(Get-ChildItem (Join-Path $cell 'train_random_subject_v1') -Filter 'draw_*.json').Count-ne20){throw 'TRAIN controls incomplete'}
 foreach($p in @((Join-Path $cell 'FINAL_P_BACKTRACE_V2.json'),(Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json'),(Join-Path $cell 'DIRECTIONAL_V4.json'))){if(-not(Test-Path $p)){throw "required input missing $p"}}
 $out=Join-Path $runtime 'compact_preview\eegnet_openbmi_ssvep_fold4_seed0_v9\PROVENANCE.json';if(Test-Path $out){throw 'V9 output exists'}
 $cmd='"'+$python+'" -u "'+$entry+'" --task OpenBMI_SSVEP --fold 4 --subject-v2-sha '+$sha+' >> "'+$log+'" 2>&1';&cmd.exe /d /c $cmd;if($LASTEXITCODE-ne0){throw "compact V9 exited $LASTEXITCODE"}
 $p=Get-Content $out -Raw|ConvertFrom-Json;if($p.status-ne'CELL_COMPACT_PREVIEW_COMPLETE_NOT_20_CELL_RESULT'-or$p.final_heldout_accessed-ne$false-or[int]$p.fold-ne4){throw 'compact V9 invalid'}
 "COMPACT_RECOVERY_F4_V1_COMPLETE sha=$((Get-FileHash $out -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8;exit 0
}catch{if(Test-Path $log){"COMPACT_RECOVERY_F4_V1_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $log -Append -Encoding utf8};exit 1}
