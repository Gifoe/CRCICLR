#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT=/root/autodl-tmp/hsc_tta_eeg
PY=/root/miniconda3/envs/hsc_cpu/bin/python
PYTEST=/root/miniconda3/envs/hsc_cpu/bin/pytest

cd "$ROOT/repo"

"$PY" scripts/check_environment.py --config configs/storage.yaml --device cpu
"$PY" scripts/download_eegmmidb.py --config configs/datasets/eegmmidb.yaml --verify-only --resume --num-workers 8 --device cpu > "$ROOT/logs/verify_eegmmidb.log" 2>&1
"$PY" scripts/download_hmc.py --config configs/datasets/hmc.yaml --verify-only --resume --num-workers 8 --device cpu > "$ROOT/logs/verify_hmc.log" 2>&1
"$PY" scripts/download_cap.py --config configs/datasets/cap.yaml --verify-only --resume --num-workers 8 --device cpu > "$ROOT/logs/verify_cap.log" 2>&1
"$PY" scripts/verify_downloads.py --config configs/storage.yaml --device cpu > "$ROOT/logs/verify_all.log" 2>&1
"$PY" scripts/audit_datasets.py --config configs/storage.yaml --device cpu > "$ROOT/logs/audit_full.log" 2>&1

"$PY" scripts/preprocess_dataset.py --config configs/datasets/eegmmidb.yaml --num-workers 4 --resume --device cpu > "$ROOT/logs/preprocess_full_eegmmidb.log" 2>&1
"$PY" scripts/preprocess_dataset.py --config configs/datasets/hmc.yaml --num-workers 1 --resume --device cpu > "$ROOT/logs/preprocess_full_hmc.log" 2>&1

# CAP records are isolated in fresh processes because this AutoDL container is
# capped at 2 GiB. The pre-read resume check makes earlier batches inexpensive.
CAP_SUBJECTS=$("$PY" - <<'PY'
import pandas as pd
p = "/root/autodl-tmp/hsc_tta_eeg/data/manifests/subjects.parquet"
frame = pd.read_parquet(p)
print(int(((frame.dataset == "cap") & frame.eligible).sum()))
PY
)
for ((limit=1; limit<=CAP_SUBJECTS; limit++)); do
  "$PY" scripts/preprocess_dataset.py --config configs/datasets/cap.yaml --limit-subjects "$limit" --num-workers 1 --resume --device cpu >> "$ROOT/logs/preprocess_full_cap.log" 2>&1
done

"$PY" scripts/make_splits.py --config configs/storage.yaml --device cpu > "$ROOT/logs/make_splits.log" 2>&1
for dataset in eegmmidb hmc cap; do
  for seed in 0 1 2 3 4; do
    "$PY" scripts/build_episodes.py --config "configs/datasets/${dataset}.yaml" --seed "$seed" --resume --device cpu >> "$ROOT/logs/build_episodes.log" 2>&1
  done
done

"$PY" scripts/validate_cpu_artifacts.py --config configs/storage.yaml --device cpu > "$ROOT/logs/validate_cpu_artifacts_final.log" 2>&1
bash scripts/run_critical_index_cpu_phase.sh
