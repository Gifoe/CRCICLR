#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
python scripts/check_environment.py --config configs/storage.yaml --device cpu
pytest -q
python scripts/run_mock_pipeline.py --config configs/method/hsc_tta.yaml --device cpu --resume

