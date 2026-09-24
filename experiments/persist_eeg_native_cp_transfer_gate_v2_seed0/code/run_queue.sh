#!/usr/bin/env bash
set -euo pipefail
stage=${1:?stage required}
queue=${2:?queue A or B required}
cd "$(dirname "$0")/../../.."
script=experiments/persist_eeg_native_cp_transfer_gate_v2_seed0/code/run.py
runtime=${NATIVE_GATE_RUNTIME:-/root/rivermind-data/native_gate_v2_runtime}
run_cell() {
  local task=$1 fold=$2
  if [[ $stage == refit ]]; then
    local status="$runtime/discovery/${task,,}/fold${fold}_seed0/COMPLETE.json"
    for attempt in {1..360}; do
      [[ -f $status ]] && break
      sleep 10
    done
    [[ -f $status ]] || { echo "Discovery missing: $task fold $fold" >&2; exit 1; }
  fi
  python "$script" "$stage" --task "$task" --fold "$fold"
}
if [[ $queue == A ]]; then
  for task in OpenBMI_MI OpenBMI_ERP; do
    for fold in 0 1 2 3 4; do
      run_cell "$task" "$fold"
    done
  done
elif [[ $queue == B ]]; then
  for task in OpenBMI_SSVEP WBCIC_MI; do
    for fold in 0 1 2 3 4; do
      run_cell "$task" "$fold"
    done
  done
else
  echo "queue must be A or B" >&2
  exit 2
fi
