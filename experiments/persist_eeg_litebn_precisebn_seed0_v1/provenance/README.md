# Recovered historical sources

The user supplied a second SSH source server after the first Windows-server
source audit found incompatible ERP/SSVEP checkpoints. This snapshot resolves
that earlier blocker; old architecture audit files describe the rejected first
source and are not the current recovered checkpoint manifest.

`TRANSFER_RECEIPT.json` lists 161 source artifacts with byte counts and SHA256:
101 historical source/protocol/result files and 60 selected checkpoints across
four tasks, five folds and three seeds. Every transfer was verified on the
destination server. Checkpoints are stored outside Git at
`D:/nips-temp/TotalP/P1/precisebn_recovered_historical/checkpoints`.
Only seed0 enters the diagnostic. No EEG files were transferred.

The snapshot is copied byte-for-byte, with Git newline conversion disabled.
The source repository inspection commit is not assumed to be the training
commit. ERP/SSVEP protocol-lock git commit and historical checkpoint SHA256
records provide their recorded provenance; MI records do not name the original
training commit, so that limitation is explicit.

To reconstruct the external recovered root, put this `snapshot` folder and
receipt alongside `checkpoints/<task>/fold<fold>_seed<seed>/selected_best.pt`
downloaded from the authorized source paths in the receipt. The runner verifies
all receipt hashes and original provenance hashes before EEG evaluation.

The active runner uses only original CompactLite weights. SRGEO/EMA, SBTR and
the smaller seven-backbone LiteBN were not used as baseline substitutes.
