"""Locked implementation runner for the user-authorized seed-0 SEARCH pilot."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from task_datasets import (PROTOCOL, RUNTIME, TASKS, RawGPUCache, build_model, class_weights,
                           load_bundle, normalizer, set_seed, write_json)
from train_task_carriers import lock_sha, train_one


def main() -> int:
    amendment = json.loads((PROTOCOL / "SEED0_PRELIMINARY_AMENDMENT.json").read_text(encoding="utf-8"))
    if not amendment.get("pass") or amendment.get("seeds") != [0] or not amendment.get("heldout_evaluation_forbidden"):
        raise RuntimeError("invalid seed0 amendment")
    lock = lock_sha(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((PROTOCOL / "SUBJECT_SPLIT_REFERENCE.json").read_text(encoding="utf-8"))
    records = []
    for task in ("ERP", "SSVEP"):
        bundle = load_bundle(task, split["search_subjects"]); cache = RawGPUCache(bundle, device)
        for source_fold in split["folds"]:
            fold = dict(source_fold); fold["_seed"] = 0
            mean, std, norm = normalizer(bundle, fold["inner_train_subjects"])
            weights, weight = class_weights(bundle, fold["inner_train_subjects"])
            for model_name in ("EEGNet", "LiteBN"):
                set_seed(0); model = build_model(model_name, task).to(device); set_seed(100000)
                info = train_one(model, model_name, task, fold, bundle, cache, mean, std, weights, {**weight, "_normalizer": norm}, device, lock)
                info["initialization_seed"] = 0; info["training_rng_seed"] = 100000
                records.append({k: v for k, v in info.items() if k not in ("history", "elapsed_seconds")})
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
        del cache
        if device.type == "cuda": torch.cuda.empty_cache()
    if len(records) != 20 or not all(Path(r["checkpoint_path"]).is_file() for r in records):
        raise RuntimeError("seed0 carrier grid incomplete")
    write_json(PROTOCOL / "SEED0_CHECKPOINT_PROVENANCE.json", {"pass": True, "expected_trainings": 20, "seed": 0, "records": records, "heldout_labels_opened": False})
    print("OPENBMI_TASK_SEED0_CARRIER_GRID_COMPLETE")
    return 0


if __name__ == "__main__": raise SystemExit(main())