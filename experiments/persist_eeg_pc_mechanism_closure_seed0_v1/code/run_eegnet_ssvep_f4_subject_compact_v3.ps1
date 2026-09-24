$ErrorActionPreference='Stop'
$base='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\run_eegnet_ssvep_f1_subject_compact_v3.ps1'
$expected='1d76b0ae355268f9a19610770cab6a5a409bda32e7d5e5a52e2c8e5b4d386e8a'
if((Get-FileHash $base -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){throw 'pinned F1 subject/compact source SHA mismatch'}
$source=Get-Content $base -Raw
$source=$source.Replace("scheduled_subject_compact_EEGNet_OpenBMI_SSVEP_f1_v3.log","scheduled_subject_compact_EEGNet_OpenBMI_SSVEP_f4_v3.log")
$source=$source.Replace("scheduled_backtrace_checkpoint_EEGNet_OpenBMI_SSVEP_f1_v3.log","scheduled_backtrace_checkpoint_EEGNet_OpenBMI_SSVEP_f4_v1.log")
$source=$source.Replace("scheduled_compact_EEGNet_OpenBMI_SSVEP_f1_v3.log","scheduled_compact_EEGNet_OpenBMI_SSVEP_f4_v3.log")
$source=$source.Replace('fold1_seed0','fold4_seed0').Replace('OpenBMI_SSVEP_f1','OpenBMI_SSVEP_f4').Replace('SSVEP_F1','SSVEP_F4').Replace('--fold 1','--fold 4')
$source=$source.Replace('[int]$upstream.fold -ne 1','[int]$upstream.fold -ne 4').Replace('[int]$directional.fold -ne 1','[int]$directional.fold -ne 4')
$source=$source.Replace('[int]$replay.fold -ne 1','[int]$replay.fold -ne 4').Replace('[int]$endpoint.fold -ne 1','[int]$endpoint.fold -ne 4')
$source=$source.Replace('[int]$native.fold -ne 1','[int]$native.fold -ne 4').Replace('[int]$mechanism.fold -ne 1','[int]$mechanism.fold -ne 4')
$source=$source.Replace('[int]$backtrace.fold -ne 1','[int]$backtrace.fold -ne 4').Replace('[int]$sv1.fold -ne 1','[int]$sv1.fold -ne 4')
$source=$source.Replace('[int]$sv2.fold -ne 1','[int]$sv2.fold -ne 4').Replace('[int]$preview.fold -ne 1','[int]$preview.fold -ne 4')
$source=$source.Replace('PERSIST_EEG_PC_PIPELINE_EEGNET_SSVEP_F4_BACKTRACE_CHECKPOINT_V3','PERSIST_EEG_PC_PIPELINE_EEGNET_SSVEP_F4_BACKTRACE_CHECKPOINT_V1')
$source=$source.Replace('BACKTRACE_CHECKPOINT_SSVEP_F4_V3_COMPLETE','BACKTRACE_CHECKPOINT_SSVEP_F4_V1_COMPLETE').Replace('BACKTRACE_CHECKPOINT_SSVEP_F4_V3_EXCEPTION','BACKTRACE_CHECKPOINT_SSVEP_F4_V1_EXCEPTION')
$source=$source.Replace('checkpoint_mechanism_v1','checkpoint_mechanism_v2').Replace('FINAL_P_BACKTRACE_V1.json','FINAL_P_BACKTRACE_V2.json')
& ([ScriptBlock]::Create($source))
exit $LASTEXITCODE
