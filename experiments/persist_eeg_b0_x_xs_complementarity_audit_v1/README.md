# B0/X/XS development-outer complementarity audit

This experiment replays exact historical LiteBN B0, X, and XS checkpoints and measures trial-level rescue/harm overlap plus label-informed oracle headroom. It is analysis-only: no checkpoint is trained, fine-tuned, selected, or altered.

Scope is restricted to the development outer subjects and evaluation sessions recorded by:

- `persist_eeg_litebn_x_singlemodel_seed0_v1` for B0 versus X, seed 0;
- `persist_eeg_xs_full_multiseed_finaltest_v1` development artifacts for B0 versus XS, seeds 0/1/2.

Final heldout, internal heldout, sealed-test membership, labels, and result artifacts are outside scope and are not accessed.

The pipeline is fail-closed:

1. `replay_models.py` verifies exact checkpoint/normalizer SHA256 values and reproduces historical subject BA, macro-F1, and accuracy at tolerance `1e-8`.
2. `paired_error_audit.py` requires one-to-one stable trial keys and produces CC/RESCUE/HARM/WW states.
3. `oracle_headroom.py` computes subject-level pairwise oracle bounds and the seed-0 B0+X+XS union oracle.
4. `aggregate_audit.py` performs 10,000 paired subject bootstraps and writes the final report.

The seed-0 union uses the X experiment's exact B0 as the common baseline. The exact XS checkpoint and normalizer are shared with the seed-0 XS replay, so X and XS predictions remain trial-aligned while avoiding an invalid comparison between two different historical B0 checkpoint grids.

Example:

```bash
python code/replay_models.py --repo /path/to/repo --runtime /path/to/audit-runtime
python code/paired_error_audit.py --repo /path/to/repo --runtime /path/to/audit-runtime
python code/oracle_headroom.py --repo /path/to/repo
python code/aggregate_audit.py --repo /path/to/repo
```
