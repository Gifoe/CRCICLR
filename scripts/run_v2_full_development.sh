#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/hsc_tta_eeg
REPO="$ROOT/repo"
OUT="$ROOT/outputs/v2_joint_certified"
PYTHON=${HSC_V2_PYTHON:-/root/miniconda3/envs/hsc_gpu/bin/python}
START=0
STOP=19
RESUME=0
DRY_RUN=0
DATASETS=hmc,eegmmidb
SEEDS=0,1,2,3,4
DEVICE=cuda
BATCH_SIZE=128

usage() {
  echo "Usage: $0 [--resume] [--start-stage N] [--stop-after-stage N] [--datasets CSV] [--seeds CSV] [--device DEVICE] [--batch-size N] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --start-stage) START=$2; shift 2 ;;
    --stop-after-stage) STOP=$2; shift 2 ;;
    --datasets) DATASETS=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if (( START < 0 || STOP > 19 || START > STOP )); then
  echo "Invalid stage interval $START..$STOP" >&2
  exit 2
fi

mkdir -p "$OUT/logs" "$OUT/state" "$OUT/hashes"
export HSC_V2_DATASETS="$DATASETS" HSC_V2_SEEDS="$SEEDS" HSC_V2_DEVICE="$DEVICE" HSC_V2_BATCH_SIZE="$BATCH_SIZE"

NAMES=(
  repository_data_audit
  v1_oracle_diagnostics
  v2_split_generation
  token_embedding_extraction
  source_head_qualification
  action_candidate_evaluation
  action_library_gate
  risk_predictor_nested_oof
  benefit_predictor_nested_oof
  joint_calibration
  nested_development_decisions
  decision_freeze
  nested_development_outcomes
  external_baselines
  ablations
  simulations_theory_audit
  certifiability_analysis
  method_freeze
  old_final_exploratory_replication
  reports
)

COMMANDS=(
  "$PYTHON scripts/audit_v2_repository.py"
  "$PYTHON scripts/analyze_v1_oracle_headroom.py"
  "$PYTHON scripts/generate_v2_splits.py"
  "$PYTHON scripts/extract_token_embeddings_v2.py --resume --device $DEVICE --batch-size $BATCH_SIZE"
  "$PYTHON scripts/qualify_source_models_v2.py"
  "$PYTHON scripts/build_v2_development_surfaces.py"
  "$PYTHON scripts/analyze_v2_action_library.py"
  "$PYTHON scripts/run_v2_nested_development.py"
  "$PYTHON scripts/verify_v2_nested_stage.py --stage benefit"
  "$PYTHON scripts/verify_v2_nested_stage.py --stage calibration"
  "$PYTHON scripts/verify_v2_nested_stage.py --stage decisions"
  "$PYTHON scripts/verify_v2_nested_stage.py --stage freeze"
  "$PYTHON scripts/verify_v2_nested_stage.py --stage outcomes"
  "$PYTHON scripts/run_v2_baselines.py"
  "$PYTHON scripts/run_v2_ablations.py"
  "$PYTHON scripts/run_v2_simulations.py"
  "$PYTHON scripts/run_v2_certifiability.py"
  "$PYTHON scripts/freeze_v2_method.py"
  "$PYTHON scripts/run_v2_exploratory_replication.py --device $DEVICE"
  "$PYTHON scripts/generate_v2_reports.py"
)

cd "$REPO"
for ((stage=START; stage<=STOP; stage++)); do
  name=${NAMES[$stage]}
  state="$OUT/state/stage_$(printf '%02d' "$stage")_${name}.json"
  log="$OUT/logs/stage_$(printf '%02d' "$stage")_${name}.log"
  if (( RESUME == 1 )) && [[ -f "$state" ]] && grep -q '"status": "completed"' "$state"; then
    echo "skip completed stage $stage $name"
    continue
  fi
  echo "stage $stage $name: ${COMMANDS[$stage]}"
  if (( DRY_RUN == 1 )); then
    continue
  fi
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  "$PYTHON" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"stage":int(sys.argv[2]),"name":sys.argv[3],"status":"running","started_utc":sys.argv[4]},indent=2)+"\n")' "$state" "$stage" "$name" "$started"
  if bash -lc "${COMMANDS[$stage]}" > >(tee "$log") 2>&1; then
    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    hash_file="$OUT/hashes/stage_$(printf '%02d' "$stage")_${name}.sha256"
    find "$OUT" -type f ! -path "$OUT/logs/*" ! -path "$OUT/state/*" ! -path "$OUT/hashes/*" -print0 | sort -z | xargs -0 sha256sum > "$hash_file"
    digest=$(sha256sum "$hash_file" | awk '{print $1}')
    "$PYTHON" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"stage":int(sys.argv[2]),"name":sys.argv[3],"status":"completed","started_utc":sys.argv[4],"finished_utc":sys.argv[5],"output_hash_manifest_sha256":sys.argv[6]},indent=2)+"\n")' "$state" "$stage" "$name" "$started" "$finished" "$digest"
  else
    code=$?
    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    "$PYTHON" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"stage":int(sys.argv[2]),"name":sys.argv[3],"status":"failed","started_utc":sys.argv[4],"finished_utc":sys.argv[5],"exit_code":int(sys.argv[6]),"log":sys.argv[7]},indent=2)+"\n")' "$state" "$stage" "$name" "$started" "$finished" "$code" "$log"
    exit "$code"
  fi
done
