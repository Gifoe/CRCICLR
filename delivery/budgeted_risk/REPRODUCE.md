# Reproduce the B-HiCER Stage-0 run

From `/root/autodl-tmp/hsc_tta_eeg/repo` on the audited server:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hsc_gpu
python scripts/budgeted_risk/run_all.py \
  --project-root /root/autodl-tmp/hsc_tta_eeg \
  --device cuda
python scripts/budgeted_risk/validate_stage0.py
pytest -q
```

The run requires the frozen CBraMod checkpoint, token embeddings, master cohorts,
and contextual-risk episodes at the paths recorded in `STAGE0_METHOD_FREEZE.json`.
On the completed server run, `RUN_STATE.json` is terminal (`STOPPED_NO_GO`); use a
fresh output directory or archive the existing run before a deliberate rerun.

