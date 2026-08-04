from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .episodes import validate_three_way


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_episode_artifacts(root: str | Path, datasets: tuple[str, ...]) -> dict[str, object]:
    root = Path(root); records = []
    for dataset in datasets:
        for seed in range(5):
            old_path = root / "data/episodes_main120" / dataset / f"seed_{seed}.parquet"
            new_path = root / "data/episodes_v3" / dataset / f"seed_{seed}.parquet"
            old = pd.read_parquet(old_path).set_index("subject_id"); new = pd.read_parquet(new_path).set_index("subject_id")
            if set(old.index) != set(new.index): raise AssertionError(f"subject mismatch: {dataset}/seed_{seed}")
            for subject, row in new.iterrows():
                adapt = np.asarray(row.adapt_indices, int); probe = np.asarray(row.probe_indices, int); future = np.asarray(row.future_indices, int)
                validate_three_way(adapt, probe, future)
                if not np.array_equal(future, np.asarray(old.loc[subject].future_indices, int)):
                    raise AssertionError(f"Future changed: {dataset}/seed_{seed}/{subject}")
                if not np.array_equal(np.r_[adapt, probe], np.asarray(old.loc[subject].context_indices, int)):
                    raise AssertionError(f"A/P does not reconstruct context: {dataset}/seed_{seed}/{subject}")
            records.append({"dataset": dataset, "seed": seed, "subjects": len(new), "episode_sha256": sha256_file(new_path),
                            "future_preserved": True, "context_reconstructed": True, "three_way_valid": True})
    return {"validated_files": len(records), "subject_rows": sum(x["subjects"] for x in records), "files": records}


def source_model_manifest(root: str | Path, datasets: tuple[str, ...] = ("hmc", "eegmmidb")) -> dict[str, object]:
    root = Path(root); models = []
    for dataset in datasets:
        for seed in range(5):
            selected_path = root / "outputs/v2_joint_certified/source_models" / dataset / f"seed_{seed}" / "selected.json"
            selected = json.loads(selected_path.read_text()); checkpoint = Path(selected["model_path"])
            split = root / "data/splits" / dataset / f"seed_{seed}.json"
            models.append({"dataset": dataset, "seed": seed, "checkpoint_path": str(checkpoint),
                           "checkpoint_sha256": sha256_file(checkpoint), "selected_manifest_sha256": sha256_file(selected_path),
                           "training_split_path": str(split), "training_split_sha256": sha256_file(split),
                           "architecture": selected["architecture"], "n_classes": 5 if dataset == "hmc" else 4,
                           "qualification_macro_f1": selected["macro_f1"],
                           "qualification_balanced_accuracy": selected["balanced_accuracy"],
                           "qualification_state_hash": selected["state_hash"], "reused_from": "v2-joint-risk-benefit"})
    return {"policy": "all V3 methods and baselines share these frozen V2-qualified source models", "models": models}

