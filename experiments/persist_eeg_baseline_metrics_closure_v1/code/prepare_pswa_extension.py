"""Freeze published PSWA implementation, PEEH assignments, and random probes."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
SOURCE = EXP / "code" / "published_run_pswa_recovery.py"
LOCK_PATH = EXP / "protocol" / "PSWA_EXTENSION_LOCK.json"
CSGD_LOCK = EXP / "protocol" / "FROZEN_INFERENCE_LOCK.json"
PSWA_REF = EXP / "protocol" / "PSWA_PUBLISHED_REFERENCE.csv"
PSWA_AGG = EXP / "outputs" / "crossbackbone_pswa_v1" / "CROSSBACKBONE_PSWA_SEED0.csv"
PEEH_CODE = (P1 / "CRCICLR_CROSSBACKBONE_PEEH_WORK" / "experiments"
             / "persist_eeg_crossbackbone_peeh_v1" / "code" / "run_crossbackbone_peeh.py")
PEEH_RUNTIME = P1 / "crossbackbone_peeh_runtime" / "cells"
MODELS = ("ModernTCN", "Medformer")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if LOCK_PATH.exists():
        raise RuntimeError("PSWA extension lock already exists")
    if sha(SOURCE) != "ca977e5a3fd30bd3c0d8bb601a694de2760415d800b71cb24d9aeed5e289e84f":
        raise RuntimeError("published PSWA source script differs")
    spec = importlib.util.spec_from_file_location("closure_pswa_lock_peeh", PEEH_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(PEEH_CODE)
    peeh = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = peeh
    spec.loader.exec_module(peeh)
    frozen = json.loads(CSGD_LOCK.read_text(encoding="utf-8"))
    ckpt = {(row["Model"], row["Task"], int(row["fold"])): row
            for row in frozen["evaluation_checkpoints"] if int(row["seed"]) == 0}
    reference = {(row["model"], row["task"]): row for row in read_csv(PSWA_REF)
                 if row["model"] in ("EEGNet", "TeCh")}
    observed = {(row["Model"], row["Task"]): row for row in read_csv(PSWA_AGG)
                if row["Model"] in ("EEGNet", "TeCh")}
    regression = []
    for key, expected in sorted(reference.items()):
        actual = observed[key]
        fields = (("PSWA_pp", "PSWA mean"), ("CI_low_pp", "PSWA CI low"),
                  ("CI_high_pp", "PSWA CI high"))
        error = max(abs(float(expected[left]) - float(actual[right])) for left, right in fields)
        if error > 1e-10 or expected["status"] != actual["Status"] or expected["coverage"] != actual["Protected coverage"]:
            raise RuntimeError(f"published PSWA regression failed: {key} error={error}")
        regression.append({"model": key[0], "task": key[1], "status": "PASS", "max_abs_error": error})
    if len(regression) != 8:
        raise RuntimeError("expected eight completed published EEGNet/TeCh rows")

    cells = []
    for model in MODELS:
        for task in TASKS:
            for fold in range(5):
                path = PEEH_RUNTIME / model.lower() / task.lower() / f"fold{fold}_seed0.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                stored = json.loads(path.read_text(encoding="utf-8"))
                if (stored["Model"], stored["Task"], int(stored["fold"]), int(stored["seed"])) != (model, task, fold, 0):
                    raise RuntimeError(f"PEEH assignment identity mismatch: {path}")
                checkpoint = ckpt[model, task, fold]
                if stored["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
                    raise RuntimeError(f"PEEH checkpoint differs from frozen inference: {path}")
                protected = sorted(set(map(int, stored["protected_blocks"])))
                rank = int(stored["rank"])
                if any(index < 0 or index >= rank for index in protected):
                    raise RuntimeError(f"invalid Protected coordinate: {path}")
                controls = [np.random.default_rng(peeh.stable_seed("final-random", model, task, fold, draw))
                            .choice(np.arange(rank), size=len(protected), replace=False).astype(int).tolist()
                            for draw in range(100)] if protected else []
                controls_sha = hashlib.sha256(json.dumps(controls, separators=(",", ":")).encode()).hexdigest()
                cells.append({"model": model, "task": task, "fold": fold, "seed": 0,
                              "peeh_cell_path": str(path), "peeh_cell_sha256": sha(path),
                              "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                              "normalizer_sha256": checkpoint["normalizer_sha256"],
                              "active_rank": rank, "protected_indices": protected,
                              "protected_rank": len(protected), "random_draws": len(controls),
                              "random_controls_sha256": controls_sha,
                              "status": "NONEMPTY" if protected else "EMPTY_UNDEFINED_PSWA"})
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "PERSIST_EEG_PSWA_EXTENSION_LOCK_V1",
        "published_pswa_commit": "3fd6aad50c08144c95a58454ca76d58f21aedc25",
        "published_pswa_script_sha256": sha(SOURCE),
        "peeh_source_commit": "bd2cc4346041d16f544963703e02bf93e262fbd3",
        "peeh_script_sha256": sha(PEEH_CODE),
        "published_regression": regression,
        "published_regression_pass": True,
        "random_control_rule": "exact PEEH final-random stable_seed(model,task,fold,draw), 100 draws, equal Protected rank",
        "ridge_alpha": 0.01,
        "bootstrap_draws": 20000,
        "cells": cells,
        "selection_rerun": False,
        "training_performed": False,
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOCK_PATH.with_suffix(".sha256").write_text(sha(LOCK_PATH) + "\n", encoding="utf-8")
    print(f"PSWA_EXTENSION_LOCKED cells={len(cells)} nonempty={sum(row['status']=='NONEMPTY' for row in cells)} "
          f"published_regression={len(regression)}/8 sha256={sha(LOCK_PATH)}", flush=True)


if __name__ == "__main__":
    main()
