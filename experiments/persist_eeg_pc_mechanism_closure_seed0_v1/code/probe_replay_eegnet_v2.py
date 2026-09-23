"""TRAIN/inner-only preflight for historical EEGNet baseline replay."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
os.environ.setdefault("SEVEN_RUNTIME", str(ROOT.parent / "seven_backbone_fourtask_3seed_runtime"))
os.environ.setdefault("OFFICIAL_BACKBONE_ROOT", str(Path(os.environ["SEVEN_RUNTIME"]) / "official"))
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
sys.path.insert(0, str(C.UP.SEVEN_CODE))
import run_search as S  # noqa: E402
from run_mediation_cell_v2 import verified_context  # noqa: E402


def main() -> None:
    output = EXP / "probe" / "EEGNet_OpenBMI_MI_fold0_replay_preflight_v2.json"
    if output.exists():
        raise RuntimeError("replay preflight evidence already exists")
    model_name, task, fold, seed = "EEGNet", "OpenBMI_MI", 0, 0
    gate_path, _, source, protocol_sha = verified_context(model_name, task, fold)
    record = source["record"]
    freeze = S._freeze_hash()
    data = S.load_search_fold(task, fold)
    recipe = S._frozen_recipe(task, model_name)
    if (record["protocol_freeze_sha256"] != freeze or data["split_sha256"] != record["split_sha256"]
            or data["normalizer"] != record["normalizer"] or recipe != record["recipe"]):
        raise RuntimeError("historical replay data/recipe/freeze mismatch")
    S._set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S.build_model(model_name, dataset=data["dataset"], channels=data["channels"],
                          samples=data["samples"], classes=data["classes"], tech_recipe=None).to(device)
    initial = S._state_sha(model)
    if initial != record["initial_state_sha256"]:
        raise RuntimeError("historical initialization identity mismatch")
    invariant = hashlib.sha256(json.dumps({
        "task": task, "model": model_name, "fold": fold, "seed": seed,
        "initial_state": initial, "split": data["split_sha256"],
        "normalizer": data["normalizer"], "recipe": recipe,
        "freeze": freeze, "adapter": S.MODEL_NATIVE_RESAMPLING[model_name],
    }, sort_keys=True).encode()).hexdigest()
    if invariant != record["invariant_sha256"]:
        raise RuntimeError("historical replay invariant mismatch")
    latest = Path(record["checkpoint_path"]).with_name("latest.pt")
    if C.digest(source["checkpoint"]) != source["hashes"]["checkpoint_sha256"] or not latest.is_file():
        raise RuntimeError("historical selected/final endpoint unavailable or changed")
    if not int(recipe["epochs"]) > 1:
        raise RuntimeError("invalid baseline epoch schedule")
    C.write_json(output, {
        "status": "REPLAY_PREFLIGHT_PASSED", "scope": "TRAIN_INNER_ONLY",
        "model": model_name, "task": task, "fold": fold, "seed": seed,
        "epochs": int(recipe["epochs"]), "initial_state_sha256": initial,
        "invariant_sha256": invariant, "selected_sha256": C.digest(source["checkpoint"]),
        "historical_latest_sha256": C.digest(latest), "historical_latest_bytes": latest.stat().st_size,
        "upstream_gate_sha256": C.digest(gate_path),
        "upstream_protocol_sha256": protocol_sha,
        "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
    })
    print("REPLAY_PREFLIGHT_PASSED", recipe["epochs"], flush=True)


if __name__ == "__main__":
    main()
