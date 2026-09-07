#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/hsc_tta_eeg
REPO=${ROOT}/repo
PY=/root/miniconda3/envs/hsc_gpu/bin/python
RESUME=""
DATASETS=hmc,cap,eegmmidb
SEEDS=0,1,2,3,4
DEVICE=cuda
BATCH_SIZE=64
NUM_WORKERS=0
START_STAGE=0
STOP_AFTER_STAGE=16
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=--resume; shift ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --start-stage) START_STAGE="$2"; shift 2 ;;
    --stop-after-stage) STOP_AFTER_STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "${REPO}"
mkdir -p "${ROOT}/outputs/full_experiment/logs"

run_stage() {
  local number="$1"
  local name="$2"
  shift 2
  if (( number < START_STAGE || number > STOP_AFTER_STAGE )); then return 0; fi
  echo "[$(date --iso-8601=seconds)] GPU-${number} ${name}"
  if (( DRY_RUN == 1 )); then printf 'DRY RUN:'; printf ' %q' "$@"; printf '\n'; return 0; fi
  "$@" 2>&1 | tee "${ROOT}/outputs/full_experiment/logs/GPU-${number}-${name}.log"
  "${PY}" scripts/update_gpu_state.py "GPU-${number}-${name}"
}

run_stage 0 cpu-and-cap-gate /root/miniconda3/envs/hsc_cpu/bin/python scripts/validate_critical_index_cpu_artifacts.py
run_stage 1 environment-audit "${PY}" scripts/audit_gpu_environment.py
run_stage 2 preflight "${PY}" scripts/run_gpu_preflight.py
run_stage 3 embeddings "${PY}" scripts/extract_frozen_embeddings.py --datasets "${DATASETS}" --device "${DEVICE}" --batch-size "${BATCH_SIZE}" ${RESUME}
run_stage 4 hmc-heads "${PY}" scripts/train_task_heads.py --datasets hmc --seeds "${SEEDS}" --device "${DEVICE}" ${RESUME}
run_stage 5 eegmmidb-heads "${PY}" scripts/train_task_heads.py --datasets eegmmidb --seeds "${SEEDS}" --device "${DEVICE}" ${RESUME}
run_stage 6 action-hyperparameters "${PY}" scripts/run_formal_gpu_pipeline.py history --datasets "${DATASETS}" --seeds "${SEEDS}" --device "${DEVICE}" ${RESUME}
run_stage 8 predictors-and-calibration "${PY}" scripts/run_formal_gpu_pipeline.py predictors --datasets "${DATASETS}" --seeds "${SEEDS}" --device "${DEVICE}" ${RESUME}
run_stage 11 freeze "${PY}" scripts/run_formal_gpu_pipeline.py freeze
run_stage 12 decisions "${PY}" scripts/run_formal_gpu_pipeline.py decisions --datasets "${DATASETS}" --seeds "${SEEDS}" --device "${DEVICE}"
run_stage 13 decisions-frozen "${PY}" scripts/run_formal_gpu_pipeline.py decision-freeze
run_stage 14 outcomes "${PY}" scripts/run_formal_gpu_pipeline.py outcomes --datasets "${DATASETS}" --seeds "${SEEDS}" --device "${DEVICE}"
run_stage 15 reports "${PY}" scripts/run_formal_gpu_pipeline.py reports
run_stage 16 complete "${PY}" -c "print('GPU full experiment complete')"
