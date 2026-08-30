"""Create the immutable V3 data-access and training locks before S3."""
from __future__ import annotations

import hashlib
import json

import common as c


CODE_FILES = ("common.py", "bures.py", "source_v3.py", "source_gate.py", "forensic_v2.py", "preflight_geometry.py", "freeze_protocol.py", "official_confirmation.py", "finalize_v3.py", "validate_v3.py")


def _hash_bytes(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts: digest.update(part)
    return digest.hexdigest()


def main() -> None:
    c.ensure_dirs()
    gate_path = c.RESULTS / "SOURCE_GATE.json"
    gate = c.read_json(gate_path) if gate_path.is_file() else {"source_gate_pass": False, "terminal_if_stop": "SOURCE_GATE_MISSING", "selected": None}
    split_hashes = {}; crossfit_hashes = {}; roots = {}
    for dataset in c.DATASETS:
        _, _, root = c.load_development_data(dataset); c.reject_reserved_path(root); roots[dataset] = str(root)
        for fold in c.FOLDS:
            role = c.roles(dataset, fold)
            for name, subjects in role.items():
                split_hashes[f"{dataset}/fold-{fold}/{name}"] = hashlib.sha256("\n".join(subjects).encode()).hexdigest()
            for seed in c.SEEDS:
                train, _ = c.source_indices(dataset, fold)
                assignment = [int(c.stable_seed(dataset, fold, seed, str(s), int(y), int(r)) % 2) for s, y, r in zip(*[c.load_feature_cache(dataset, fold, seed, "train")[key] for key in ("subjects", "labels", "indices")])]
                crossfit_hashes[f"{dataset}/fold-{fold}/seed-{seed}"] = c.array_sha256(__import__("numpy").asarray(assignment, dtype="int8"))
                del train
    existing = [c.CODE / name for name in CODE_FILES if (c.CODE / name).is_file()]
    code_hash = c.code_tree_sha256(existing)
    data_lock = {
        "schema": "BURES_SCST_DATA_ACCESS_LOCK_V1", "created_at_git_sha": c.git_head(), "dataset_roots": roots,
        "authorized_development": {"OpenBMI": [1, 2], "WBCIC": [0, 1]}, "prospective_discovery": {"WBCIC": [2]},
        "forbidden": ["WBCIC outer 10", "OpenBMI sealed holdout", "any reserved outer subject"],
        "split_hashes": split_hashes, "crossfit_assignment_hashes": crossfit_hashes, "s3_opened_before_lock": False, "outer_or_sealed_opened": False,
    }
    source_lock = {
        "schema": "BURES_SCST_SOURCE_DEVELOPMENT_LOCK_V1", "created_at_git_sha": c.git_head(),
        "source_gate_sha256": c.sha256_path(gate_path) if gate_path.is_file() else None, "selected_recipe": gate.get("selected"),
        "search_grid": {"q": [0.25, 0.50], "lambda_T": [0.25, 0.50, 1.00]}, "source_datasets": ["OpenBMI", "WBCIC S1->S2"],
        "selection_complete_before_s3": True, "s3_opened": False,
    }
    v3_lock = {
        "schema": "BURES_SCST_V3_PROTOCOL_LOCK_V1", "git_sha": c.git_head(), "code_files": CODE_FILES, "code_tree_sha256": code_hash,
        "dataset_roots": roots, "split_hashes": split_hashes, "crossfit_assignment_hashes": crossfit_hashes,
        "method": "Class-Centered Second-Order Bures Subject-Conditional Semantic Transport", "representation": "detached trusted final feature block, identity adapter then trainable final adapter in Stage-2",
        "class_centering": "global class centroid; equal class weighting across labels", "covariance": "within-class subject covariance averaged equally across classes",
        "shrinkage": "lambda_s=d/(n_s+d); C_s=(1-lambda_s)C_s_raw+lambda_s C_pool", "eigenvalue_floor": "1e-4 times median positive pooled eigenvalue",
        "cross_fitting": "stable hash(dataset,fold,seed,subject,class,row_id) two-way; anchor uses opposite half; no anchor/duplicate row",
        "K_targets": 8, "alpha_grid": [0.25, 0.50, 0.75, 1.00], "kNN": 5,
        "validity_gates": ["G1 >=4/5 anchor-excluded neighbors class y", "G2 local 5NN radius <= class 95th percentile", "G3 target same-class 5NN distance decreases", "G4 target Gaussian NLL decreases", "G5 EMA teacher class and positive margin", "G6 displacement/local radius <=1"],
        "q": [0.25, 0.50], "lambda_T": [0.25, 0.50, 1.00], "relative_margin_coefficient": 0.5,
        "training": {"warmup_epochs": 3, "stage2_epochs": 15, "head_lr": 1e-4, "adapter_lr": 1e-5, "weight_decay": 1e-3, "gradient_clip": 3.0, "EMA_decay": 0.99, "batch_size": 256, "sampler": "subject-balanced"},
        "controls": ["subject-balanced ERM", "Mixup", "V2-ME-HardSCST", "same-class Manifold-Mixup", "Bures-Uniform", "Bures-HardRandom", "Bures-HardSCST"],
        "folds": list(c.FOLDS), "seeds": list(c.SEEDS), "bootstrap": {"unit": "biological subject", "draws": 10000, "CI": 0.95},
        "success_criteria": {"source": ["OpenBMI delta >=0.002", "WBCIC S1->S2 delta >=0.002", "paired CI lower >0 versus ERM, HardRandom, Manifold", "transport gates pass", "clean degradation <=0.001"], "official": ["delta >=0.005", "CI lower >0", ">=3/5 folds positive", "beats matched random and Manifold", "no material fidelity degradation"]},
        "stop_criteria": ["BURES_SCST_TRANSPORT_NOT_REALIZED", "BURES_SCST_SOURCE_GATE_FAILED", "BURES_SCST_NOT_SUPPORTED_ON_OFFICIAL_ATCNET"],
        "source_gate_pass": bool(gate.get("source_gate_pass", False)), "s3_authorized": False, "outer_or_sealed_opened": False,
    }
    c.write_json(c.PROTOCOL / "DATA_ACCESS_LOCK.json", data_lock); c.write_json(c.PROTOCOL / "BURES_SCST_SOURCE_LOCK.json", source_lock); c.write_json(c.PROTOCOL / "BURES_SCST_V3_LOCK.json", v3_lock)
    c.write_json(c.PROTOCOL / "OUTER_CONFIRMATION_TEMPLATE.json", {"schema": "BURES_SCST_OUTER_CONFIRMATION_TEMPLATE_V1", "status": "TEMPLATE_ONLY_NOT_AUTHORIZED", "outer_opened": False, "required_prior_terminal": "BURES_SCST_CROSS_ARCH_SUPPORTED"})
    print(json.dumps(v3_lock, indent=2))


if __name__ == "__main__": main()
