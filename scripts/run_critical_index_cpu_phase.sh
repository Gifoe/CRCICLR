#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/hsc_tta_eeg/repo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hsc_cpu
export CUDA_VISIBLE_DEVICES=""

mkdir -p /root/autodl-tmp/hsc_tta_eeg/logs
python scripts/make_internal_splits.py
for dataset in hmc cap eegmmidb; do
  for seed in 0 1 2 3 4; do
    python scripts/build_main120_episodes.py \
      --config "configs/datasets/${dataset}.yaml" --seed "${seed}" --device cpu
  done
done
python scripts/freeze_channel_protocol.py
python scripts/validate_critical_index_cpu_artifacts.py
python scripts/run_mock_pipeline.py --config configs/method/hsc_tta.yaml --seed 2027 --device cpu
pytest -q | tee /root/autodl-tmp/hsc_tta_eeg/logs/cpu_critical_index_pytest.log
pytest --cov=src/hsc_tta \
  --cov-report=term-missing \
  --cov-report=json:/root/autodl-tmp/hsc_tta_eeg/outputs/cpu_critical_index_coverage.json \
  -q | tee /root/autodl-tmp/hsc_tta_eeg/logs/cpu_critical_index_coverage.log
python scripts/generate_critical_index_cpu_report.py
