# First-five compact previews (partial release)

Scope: EEGNet × OpenBMI MI × seed0 × folds 0–4. These are five cell-specific compact previews, not Phase-2 completion or a cross-task/model conclusion.

Frozen identity hashes are recorded in each `PROVENANCE.json`: upstream gate `7cdc2c0fe5b5e32dcb98b917a344d0eaaf0cfca75eec3a504d56650f2223df5a`, protocol `3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c`, implementation `4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee`, final-projector amendment `aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474`, and analysis lock `ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88`.

All five compact directories carry per-file output hashes, source-manifest hashes, row counts, trajectory provenance and a cell-specific report. `VALIDATION_SUMMARY.json` records the independent structural/hash audit; `SHA256SUMS.txt` indexes the packaged files (excluding itself). Recheck with:

```powershell
python ..\..\code\validate_firstfive_compacts.py
```

The original-server source audit JSONs are not included; they were independently transferred and hash-verified during release validation. If retained locally, they can be rechecked with `--source-audit-root <path-to-phase2_firstfive_sourceaudit>`.

Interpretation constraints: historical selected/final endpoint audits are distinct from the missing intermediate snapshots. The six-point timecourse is `RETRAINED_REPLICA_TRAJECTORY`, not the historical trajectory. Outer-development values are evaluation-only, post-outcome descriptions. There is no final-heldout access, prospective performance model, or causal claim. Intermediate layers use TRAIN-only orthogonal projectors; the final layer retains the amended frozen historical canonical oblique projector.
