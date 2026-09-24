$ErrorActionPreference='Stop'
$base='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\run_eegnet_ssvep_f4_endpoint_native_v5.ps1'
$expected='93d3ffb9aa5850143dff2d61e6585786db2aa3ecf1b72a8dc4d8fad35b6d24ad'
if((Get-FileHash -LiteralPath $base -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){throw 'pinned V5 source SHA mismatch'}
$source=Get-Content -LiteralPath $base -Raw
$source=$source.Replace('function Invoke-Python([string]$args,[string]$phaseLog)','function Invoke-Python([string]$arguments,[string]$phaseLog)')
$source=$source.Replace("`$cmd='`"'+`$python+'`" -u '+`$args+' >> `"'+`$p+'`" 2>&1'","`$cmd='`"'+`$python+'`" -u '+`$arguments+' >> `"'+`$p+'`" 2>&1'")
$source=$source.Replace('_f4_v5.log','_f4_v6.log')
$source=$source.Replace('_f4_v5_e','_f4_v6_e')
$source=$source.Replace('F4_V5','F4_V6')
$source=$source.Replace('endpoint/native V5 log exists','endpoint/native V6 log exists')
$source=$source.Replace('scheduled_endpoint_native_EEGNet_OpenBMI_SSVEP_f4_v5.log','scheduled_endpoint_native_EEGNet_OpenBMI_SSVEP_f4_v6.log')
& ([ScriptBlock]::Create($source))
exit $LASTEXITCODE
