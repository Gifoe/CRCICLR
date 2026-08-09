# Reproduce HSC-TTA v2

Environment: `/root/miniconda3/envs/hsc_gpu`; project root: `/root/autodl-tmp/hsc_tta_eeg`; repository: `/root/autodl-tmp/hsc_tta_eeg/repo`.

```bash
cd /root/autodl-tmp/hsc_tta_eeg/repo
bash scripts/run_v2_full_development.sh --resume --device cuda --batch-size 128
```

Use `--start-stage`, `--stop-after-stage`, `--datasets`, `--seeds`, or `--dry-run` for bounded reruns. Stages write independent logs, JSON states, and SHA-256 manifests under `outputs/v2_joint_certified`. Do not run stage 18 before stage 17 creates the method freeze.

The repository does not contain EEG data, token embeddings, model checkpoints, or large parquet counterfactuals. Those remain in the server data/output roots.
