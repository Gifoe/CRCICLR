"""Create immutable pre-S3 protocol locks after source-only selection."""
from __future__ import annotations

import hashlib
import json

import v2_common as c


CODE_FILES = (
    "candidate_engine.py", "discovery.py", "freeze_protocol.py", "mixed_effects.py",
    "source_search.py", "training_components.py", "v2_common.py",
)


def code_hash() -> str:
    digest = hashlib.sha256()
    for name in sorted(CODE_FILES):
        path = c.CODE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    decision = c.read_json(c.RESULTS / "SOURCE_DECISION.json")
    if decision.get("source_gate_pass") is not True or decision.get("selected") is None:
        raise RuntimeError("SOURCE_GATE_NOT_PASSED")
    selected = decision["selected"]
    split_hashes = {}
    roots = {}
    for dataset in c.DATASETS:
        _, _, root = c.load_development_data(dataset)
        roots[dataset] = str(root)
        c.reject_reserved_path(root)
        for fold in c.FOLDS:
            role = c.roles(dataset, fold)
            for name, subjects in role.items():
                if name == "outcome":
                    # Record subject IDs only. No samples or labels are opened.
                    pass
                payload = "\n".join(subjects).encode()
                split_hashes[f"{dataset}/fold-{fold}/{name}"] = hashlib.sha256(payload).hexdigest()
    data_lock = {
        "schema": "ME_HARD_SCST_DATA_ACCESS_LOCK_V1",
        "created_at_git_sha": c.git_head(),
        "dataset_roots": roots,
        "authorized_development": {"OpenBMI": [1, 2], "WBCIC": [0, 1]},
        "prospective_discovery": {"WBCIC": [2]},
        "forbidden": ["WBCIC outer 10", "OpenBMI sealed/outer", "any reserved outer"],
        "split_hashes": split_hashes,
        "s3_sample_opened_before_lock": False,
        "outer_or_sealed_opened": False,
    }
    source_lock = {
        "schema": "ME_HARD_SCST_SOURCE_DEVELOPMENT_LOCK_V1",
        "created_at_git_sha": c.git_head(),
        "source_decision_sha256": c.sha256(c.RESULTS / "SOURCE_DECISION.json"),
        "source_grid_sha256": c.sha256(c.RESULTS / "SOURCE_RECIPE_SEARCH.csv"),
        "selected_recipe": selected,
        "selection_complete_before_s3": True,
        "s3_opened": False,
    }
    lock = {
        "schema": "ME_HARD_SCST_V2_PROTOCOL_LOCK_V1",
        "git_sha": c.git_head(),
        "code_files": list(CODE_FILES),
        "code_tree_sha256": code_hash(),
        "dataset_roots": roots,
        "split_hashes": split_hashes,
        "model": "ATCNet-CleanRoom",
        "scope": selected["scope"],
        "bank_refresh_rule": "Scope A fixed coordinates; Scope B EMA teacher bank exactly once at epoch start",
        "cross_fitting_rule": "exclude current anchor from subject-class and class means",
        "shrinkage_rule": "eta=n/(n+rho_y), rho_y=median positive subject-class count",
        "K": c.K_TARGETS,
        "alpha_grid": list(c.ALPHAS),
        "gates": ["G1 valid source/target and LOO", "G2 source 95th-percentile local support", "G3 2-of-3 clean nearest labels", "G4 EMA teacher correct with margin>0", "G5 residual-SVD whitened norm <= training 95th percentile"],
        "q": selected["q"],
        "lambda_H": selected["lambda_H"],
        "EMA_decay": c.EMA_DECAY,
        "optimizer": {"name": "AdamW", "learning_rate": c.LEARNING_RATE, "weight_decay": c.WEIGHT_DECAY, "gradient_clip": 3.0},
        "scheduler": None,
        "epochs": c.EPOCHS,
        "batch_size": c.BATCH_SIZE,
        "seeds": list(c.SEEDS),
        "folds": list(c.FOLDS),
        "controls": ["ERM", "Mixup", "V1-RandomTransport", "Dynamic-ClassConditional-Uniform-NoKL", "Factorized-Uniform-NoKL", "Factorized-HardRandom"],
        "metrics": ["biological-subject-balanced BA", "macro-F1"],
        "bootstrap": {"unit": "biological subject", "paired": True, "draws": 10000, "CI": 0.95},
        "success_gates": ["delta ERM >0", "CI lower ERM >0", "positive folds >=3/5", "CI lower vs HardRandom >0", "CI lower vs Factorized-Uniform >0", "diagnostic: coverage>=0.50, median valid>=2, semantic pass no more than 0.05 below selected source minimum, finite bank stability"],
        "stop_rules": {"source_fail": "ME_HARD_SCST_MECHANISM_NOT_REALIZED", "discovery_fail": "ME_HARD_SCST_NOT_SUPPORTED"},
        "s3_opened_before_lock": False,
        "outer_or_sealed_opened": False,
    }
    c.write_json(c.PROTOCOL / "DATA_ACCESS_LOCK.json", data_lock)
    c.write_json(c.PROTOCOL / "SOURCE_DEVELOPMENT_LOCK.json", source_lock)
    c.write_json(c.PROTOCOL / "ME_HARD_SCST_V2_LOCK.json", lock)
    c.write_json(c.PROTOCOL / "OUTER_CONFIRMATION_LOCK_TEMPLATE.json", {
        "status": "TEMPLATE_ONLY_NOT_AUTHORIZED",
        "required_prior_terminal": "ME_HARD_SCST_CROSS_ARCH_SUPPORTED",
        "outer_opened": False,
        "note": "A separate one-shot protocol is required; this experiment never opens outer resources.",
    })
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
