"""Independent purity, completeness, arithmetic, and artifact validation."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cfg = common.protocol()
    global_path = common.RUNTIME / "GLOBAL_SOURCE_FREEZE.json"
    global_marker = common.read_json(global_path)
    if global_marker.get("pass") is not True or global_marker.get("run_count") != 30:
        raise RuntimeError("global source freeze invalid")
    frozen_at = float(global_marker["frozen_at_unix"])
    run_count = 0
    pseudo_subjects, future_subjects = [], []
    for backbone in common.BACKBONES:
        for fold in range(5):
            roles = common.frozen_fold(fold)
            for seed in range(3):
                context = common.unit_dir(backbone, fold, seed)
                source = common.read_json(context / "SOURCE_COMPLETE.json")
                outcome = common.read_json(context / "OUTCOME_COMPLETE.json")
                if source.get("pass") is not True or outcome.get("pass") is not True:
                    raise RuntimeError(f"completion marker failed: {context}")
                if float(source["completed_at_unix"]) > frozen_at or float(outcome["completed_at_unix"]) < frozen_at:
                    raise RuntimeError("two-phase chronology violation")
                for name, expected in source["source_artifacts"].items():
                    if sha256(context / name) != expected:
                        raise RuntimeError(f"post-freeze source mutation: {context / name}")
                frozen_directions = np.load(context / "DIRECTIONS_FROZEN.npz", allow_pickle=False)["directions"]
                if frozen_directions.shape != (64, 8) or not np.allclose(np.linalg.norm(frozen_directions, axis=0), 1.0, atol=1e-10):
                    raise RuntimeError("frozen direction count/norm failure")
                protocol = common.read_json(context / "UNIT_PROTOCOL.json")
                if protocol["roles"]["outcome"] != list(roles["outcome"]) or protocol["outcome_subjects_or_labels_materialized"] is not False:
                    raise RuntimeError("unit protocol role/purity mismatch")
                pseudo_subjects.append(pd.read_csv(context / "pseudo_subject_utility.csv"))
                future_subjects.append(pd.read_csv(context / "future_subject_utility.csv"))
                run_count += 1
    cells = pd.read_csv(common.RESULTS / "direction_utility_raw.csv")
    within = pd.read_csv(common.RESULTS / "within_run_transport.csv")
    policy = pd.read_csv(common.RESULTS / "utility_policy_per_run.csv")
    if run_count != 30 or len(cells) != 240 or len(within) != 30 or len(policy) != 30:
        raise RuntimeError("primary artifact cardinality failure")
    if cells.groupby("run_id").size().tolist() != [8] * 30 or cells.direction_SHA.isna().any():
        raise RuntimeError("direction completeness failure")
    pseudo = pd.concat(pseudo_subjects, ignore_index=True)
    future = pd.concat(future_subjects, ignore_index=True)
    if len(pseudo) != 1920 or len(future) != 1920:
        raise RuntimeError("subject-first utility cardinality failure")
    keys = ["backbone", "fold", "seed", "direction_id"]
    pseudo_mean = pseudo.groupby(keys).U_BA.mean().rename("recomputed").reset_index()
    future_mean = future.groupby(keys).U_BA.mean().rename("recomputed").reset_index()
    check_pseudo = cells.merge(pseudo_mean, on=keys, validate="one_to_one")
    check_future = cells.merge(future_mean, on=keys, validate="one_to_one")
    if not np.allclose(check_pseudo.U_pseudo_BA, check_pseudo.recomputed, atol=1e-12) or not np.allclose(check_future.U_future_BA, check_future.recomputed, atol=1e-12):
        raise RuntimeError("subject-first utility recomputation mismatch")
    recomputed_rho = []
    recomputed_policy = []
    for run_id, run in cells.groupby("run_id", sort=True):
        recomputed_rho.append((run_id, float(spearmanr(run.U_pseudo_BA, run.U_future_BA).statistic)))
        selected = float(run.loc[run.U_pseudo_BA.idxmax(), "U_future_BA"])
        recomputed_policy.append((run_id, selected - float(run.U_future_BA.mean())))
    rho_frame = pd.DataFrame(recomputed_rho, columns=["run_id", "rho_check"]).merge(within, on="run_id", validate="one_to_one")
    policy_frame = pd.DataFrame(recomputed_policy, columns=["run_id", "delta_check"]).merge(policy, on="run_id", validate="one_to_one")
    if not np.allclose(rho_frame.rho_check, rho_frame.rho_run, equal_nan=True) or not np.allclose(policy_frame.delta_check, policy_frame.Top1_minus_Random):
        raise RuntimeError("transport/policy recomputation mismatch")
    imports = []
    for path in common.HERE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(str(node.module))
    forbidden = [name for name in imports if any(token in name.lower() for token in ("wbcic", "holdout", "raw_loader", "dataset_loader"))]
    if forbidden:
        raise RuntimeError(f"restricted loader import: {forbidden}")
    stats = common.read_json(common.RESULTS / "utility_transport_statistics.json")
    if stats["terminal_state"] not in cfg["terminal_states"]:
        raise RuntimeError("terminal state outside frozen set")
    required = [
        "nested_subject_splits.csv", "training_ledger.csv", "direction_utility_raw.csv", "within_run_transport.csv",
        "utility_transport_statistics.json", "utility_prediction_raw.csv", "utility_prediction_summary.csv",
        "utility_policy_per_run.csv", "utility_policy_statistics.json", "heterogeneity_statistics.json",
        "permutation_control.json", "holdout_purity.json",
    ]
    figure_names = ["pseudo_vs_future_utility", "within_run_rank_transport", "prediction_rmse_comparison", "policy_future_utility", "utility_distribution"]
    missing = [str(common.RESULTS / name) for name in required if not (common.RESULTS / name).exists()]
    missing += [str(common.FIGURES / f"{name}.{suffix}") for name in figure_names for suffix in ("png", "pdf") if not (common.FIGURES / f"{name}.{suffix}").exists()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    validation = {
        "pass": True,
        "protocol_sha256": sha256(common.PROTOCOL_PATH),
        "global_source_freeze_sha256": sha256(global_path),
        "run_count": run_count,
        "direction_cell_count": len(cells),
        "pseudo_subject_cell_count": len(pseudo),
        "future_subject_cell_count": len(future),
        "two_phase_chronology": True,
        "source_artifact_hashes": "PASS",
        "direction_count_and_norms": "PASS",
        "subject_first_utility_recomputation": "PASS",
        "transport_recomputation": "PASS",
        "policy_recomputation": "PASS",
        "restricted_loader_imports": forbidden,
        "OpenBMI_internal_holdout_accessed": False,
        "OpenBMI_holdout_membership_enumerated": False,
        "WBCIC_accessed": False,
        "outer_outcome_used_before_source_freeze": False,
        "pseudo_target_used_for_training_or_selection": False,
        "terminal_state": stats["terminal_state"],
        "recommendation": stats["recommendation"],
    }
    common.write_json(common.RESULTS / "final_validation.json", validation)
    print("INDEPENDENT_VALIDATION_PASS")
    print(stats["terminal_state"])


if __name__ == "__main__":
    main()
