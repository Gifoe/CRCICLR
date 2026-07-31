#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
pytest -q
pytest --cov=src/hsc_tta --cov-report=term-missing

