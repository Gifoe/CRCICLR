from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze_v2_method(root: str | Path) -> Path:
    root = Path(root)
    repo = root / "repo"
    base = root / "outputs/v2_joint_certified"
    required = [
        base / "nested_dev/ALL_DEV_DECISIONS.parquet",
        base / "nested_dev/ALL_DEV_CALIBRATION_SCORES.parquet",
        base / "nested_dev/DEV_RESULTS_WITH_CI.csv",
        base / "baselines/EXTERNAL_BASELINE_RESULTS.csv",
        base / "ablations/ABLATION_RESULTS.csv",
        base / "theory/SIMULATION_V2_RESULTS.csv",
        base / "certifiability/CERTIFIABILITY_SAMPLE_SIZE.csv",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"development is incomplete; refusing method freeze: {missing}")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != "v2-joint-risk-benefit":
        raise RuntimeError("method freeze must be created on v2-joint-risk-benefit")
    features = pd.read_parquet(base / "actions/DEVELOPMENT_CONTEXT_FEATURES.parquet")
    feature_columns = [c for c in features.columns if c not in {"dataset", "seed", "subject_id", "alpha", "action_available"}]
    bounds = pd.read_parquet(base / "nested_dev/ALL_DEV_JOINT_BOUNDS.parquet")
    source_hashes: dict[str, str] = {}
    for selected in sorted((base / "source_models").glob("*/seed_*/selected.json")):
        payload = json.loads(selected.read_text())
        model_path = Path(payload["model_path"])
        source_hashes[str(selected.relative_to(base))] = sha256(model_path)
    old_test_ids: dict[str, dict[str, list[str]]] = {}
    for dataset in ("hmc", "eegmmidb", "cap"):
        old_test_ids[dataset] = {}
        for split_path in sorted((root / "data/splits" / dataset).glob("seed_*.json")):
            split = json.loads(split_path.read_text())
            role = "external_final_test" if dataset == "cap" else "final_test"
            old_test_ids[dataset][split_path.stem] = split["roles"].get(role, [])
    config_hashes = {str(path.relative_to(repo)): sha256(path) for path in sorted((repo / "configs").rglob("*")) if path.is_file()}
    payload: dict[str, object] = {
        "schema_version": "hsc_tta_v2_method_freeze_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "branch": "v2-joint-risk-benefit",
        "development_complete": True,
        "methods_frozen": True,
        "old_final_outcomes_used_for_method_selection": False,
        "source_model_hashes": source_hashes,
        "action_library": ["no_tta", "official_t3a", "robust_residual_adapter"],
        "action_hyperparameters": {
            "official_t3a": {"filter_k": 20, "confidence": None, "rule": "author_equivalent"},
            "robust_residual_adapter": {"bottleneck": 64, "steps": 3, "learning_rate": 5e-5,
                                         "beta": 0.5, "gamma": 0.1, "eta": 1e-3,
                                         "reliability_quantile": 0.2, "w2_initialization": "zero"},
        },
        "predictors": {"risk_candidates": ["elastic_net", "hist_gradient_boosting"],
                       "benefit_candidates": ["elastic_net", "hist_gradient_boosting", "pairwise_ridge"],
                       "selection": "subject-grouped meta OOF primary metric then simpler model"},
        "predictor_features": feature_columns,
        "alpha": [0.1, 0.2],
        "delta": 0.1,
        "lambda_grid": np.r_[np.linspace(0.50, 0.99, 20), 1.0].tolist(),
        "scale_rule": "median absolute meta-OOF residual with positive floor, separately for risk and benefit",
        "observed_development_scale_ranges": {"c_j": [float(bounds.c_j.min()), float(bounds.c_j.max())],
                                                "c_delta": [float(bounds.c_delta.min()), float(bounds.c_delta.max())]},
        "joint_score": "per-subject maximum over every available action and both normalized risk-underestimation and benefit-overestimation residuals",
        "calibration_quantile": "ceil((m+1)*(1-delta))-th order statistic, clipped to m",
        "selector": "available non-No-TTA actions require upper critical index <20 and lower benefit >0; maximize lower benefit, then lower set size/cost; otherwise No-TTA with full-set allowed",
        "seeds": [0, 1, 2, 3, 4],
        "development_protocol": "5x5 original-seed by outer-fold subject-disjoint nested development CV",
        "metrics": ["marginal_violation", "certified_only_violation", "joint_validity", "nonharm_violation",
                    "csr", "full_set_fallback", "average_set_size", "singleton_rate", "argmax_error",
                    "macro_f1", "balanced_accuracy", "cohen_kappa", "selected_vs_no_tta_gain", "nar", "her",
                    "tta_selection_rate", "safe_beneficial_selection_precision"],
        "baseline_definitions": {
            "B1": "No-TTA plus standard global policy CRC",
            "B2": "meta-selected best fixed action plus policy CRC",
            "B3": "meta-selected entropy gate plus policy CRC",
            "B4": "custom U-only agreement heuristic plus policy CRC; not claimed as official TTALine",
            "B5": "proposed simultaneous joint risk-benefit certificate",
        },
        "tainted_exploratory_test_subject_ids": old_test_ids,
        "config_sha256": config_hashes,
        "development_artifact_sha256": {str(path.relative_to(base)): sha256(path) for path in required},
    }
    freeze_path = base / "freeze/V2_METHOD_FREEZE.json"
    _atomic_json(payload, freeze_path)
    _atomic_json(payload, base / "provenance/V2_METHOD_FREEZE.json")
    return freeze_path
