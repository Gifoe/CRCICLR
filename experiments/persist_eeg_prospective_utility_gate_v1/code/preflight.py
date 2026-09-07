"""Independent preflight for data boundary, splits, and Phase-2 equivalence."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

import common


def load_phase2_common():
    path = common.REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1" / "code" / "common.py"
    spec = importlib.util.spec_from_file_location("phase2_reference_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase-2 reference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    common.ensure_dirs()
    cfg = common.protocol()
    data = common.load_data(label_subjects=None)
    checks: dict[str, object] = {
        "schema": cfg["schema"],
        "repository_start_sha": cfg["repository_start_sha"],
        "cache_root": str(data.cache_root),
        "signal_shape": list(data.x.shape),
        "metadata_rows": len(data.metadata),
        "labels_materialized": int((data.metadata.label >= 0).sum()),
        "subject_count": data.metadata.subject_id.nunique(),
        "folds": {},
    }
    if checks["labels_materialized"] != 0:
        raise RuntimeError("preflight materialized labels")
    split_rows = []
    for fold in range(5):
        roles = common.frozen_fold(fold)
        role_sets = {key: set(value) for key, value in roles.items() if key != "source"}
        if any(role_sets[a] & role_sets[b] for i, a in enumerate(role_sets) for b in list(role_sets)[i + 1:]):
            raise RuntimeError(f"fold {fold} overlaps")
        checks["folds"][str(fold)] = {key: list(value) for key, value in roles.items()}
        for role in ("fit_train", "fit_validation", "pseudo_target", "outcome"):
            for subject in roles[role]:
                split_rows.append({"fold": fold, "role": role, "subject_id": subject})

    reference = load_phase2_common()
    equivalence = {}
    for backbone in common.BACKBONES:
        seed = common.stable_seed("equivalence", backbone)
        current_model = common.build_model(backbone, seed)
        reference_model = reference.build_model(backbone, seed)
        equivalence[backbone] = {
            "current_state_sha": common.state_sha256(current_model),
            "phase2_state_sha": reference.state_sha256(reference_model),
        }
        equivalence[backbone]["pass"] = equivalence[backbone]["current_state_sha"] == equivalence[backbone]["phase2_state_sha"]
        if not equivalence[backbone]["pass"]:
            raise RuntimeError(f"{backbone} architecture/initialization differs from Phase 2")
    rng = np.random.default_rng(17)
    x = rng.normal(size=(80, 64))
    subjects = np.repeat(np.asarray([str(i) for i in range(1, 21)]), 4)
    sessions = np.tile(np.asarray([1, 1, 2, 2]), 20)
    center, basis, meta = common.persistent_directions(x, subjects, sessions, 8)
    r_center, r_basis, r_meta = reference.persistent_directions(x, subjects, sessions, 8)
    direction_equal = np.array_equal(center, r_center) and np.array_equal(basis, r_basis) and meta == r_meta
    erased_equal = np.array_equal(common.erase_direction(x, center, basis[:, 0]), reference.erase_direction(x, r_center, r_basis[:, 0]))
    if not direction_equal or not erased_equal:
        raise RuntimeError("Phase-2 direction/intervention equivalence failed")
    checks["equivalence"] = {"backbones": equivalence, "direction": direction_equal, "erasure": erased_equal}

    imports = []
    for path in sorted(common.HERE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(str(node.module))
    forbidden_imports = [name for name in imports if any(token in name.lower() for token in ("wbcic", "holdout", "raw_loader", "dataset_loader"))]
    if forbidden_imports:
        raise RuntimeError(f"restricted/historical loader import found: {forbidden_imports}")
    checks["imports_sha256"] = hashlib.sha256("\n".join(sorted(imports)).encode()).hexdigest()
    checks["restricted_loader_imports"] = forbidden_imports
    checks["pass"] = True
    common.write_csv(common.RESULTS / "nested_subject_splits.csv", __import__("pandas").DataFrame(split_rows))
    common.write_json(common.RUNTIME / "PREFLIGHT.json", checks)
    print("PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
