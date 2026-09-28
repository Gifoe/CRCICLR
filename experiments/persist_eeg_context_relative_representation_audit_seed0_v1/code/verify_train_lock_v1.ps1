param([string]$Root='D:\nips-temp\TotalP\P1\context_relative_representation_runtime')
$ErrorActionPreference='Stop'
$path=Join-Path $Root 'source\protocol\TRAIN_SELECTION_LOCK.json'
if(-not (Test-Path -LiteralPath $path)){throw 'frozen TRAIN_SELECTION_LOCK missing'}
$book=Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
if($book.role -ne 'TRAIN_GEOMETRY' -or $book.outer_selection -ne $false -or $book.formal_final_heldout_eeg_reads -ne 0){throw 'invalid selection lock'}
$proto=Join-Path $Root 'source\protocol\PROTOCOL_LOCK.json'
$amend=Join-Path $Root 'source\protocol\PROTOCOL_AMENDMENT_V2.json'
if((Get-FileHash -Algorithm SHA256 -LiteralPath $proto).Hash.ToLowerInvariant() -ne $book.protocol_lock_sha256){throw 'protocol drift'}
if((Get-FileHash -Algorithm SHA256 -LiteralPath $amend).Hash.ToLowerInvariant() -ne $book.protocol_amendment_sha256){throw 'amendment drift'}
foreach($i in 0..4){
  $p=Join-Path $Root "train_lock\fold${i}.json"
  if(-not (Test-Path -LiteralPath $p)){throw "missing fold$i TRAIN selection"}
  $row=$book.folds.PSObject.Properties["$i"].Value
  $m=Get-Content -Raw -LiteralPath $p | ConvertFrom-Json
  if((Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant() -ne $row.train_selection_sha256){throw "TRAIN selection SHA drift fold$i"}
  if($m.role -ne 'TRAIN_GEOMETRY' -or $m.fold -ne $i -or $m.B -ne $row.B -or $m.C_absolute -ne $row.C_absolute -or $m.C_relative -ne $row.C_relative -or $m.lambda_shrink -ne $row.lambda_shrink -or $m.lambda_z -ne $row.lambda_z -or $m.pca_k -ne $row.pca_k){throw "TRAIN selection values drift fold$i"}
}
"TRAIN_SELECTION_LOCK_VERIFIED sha=$((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant())"
