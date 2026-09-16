#!/usr/bin/env python3
"""Freeze source provenance before either bridge computation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/persist_eeg_sire_diagnostic_design_bridge_v1"
OLD = ROOT / "experiments/persist_eeg_litebn_ablation_v2_seed0"
MULTI = ROOT / "experiments/persist_eeg_litebn_ablation_v2_multiseed_v1"
sys.path.insert(0, str(OLD / "code"))
import run_peeh_bridge as peeh  # noqa: E402
import run_pswa_bridge as pswa  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    runtime = peeh.load_runtime()
    _, folds, _ = runtime.base.load_folds()
    entries = []
    for task in peeh.TASKS:
        dataset = runtime.base.TASKS[task]["dataset"]
        for fold in range(5):
            split = folds[dataset][fold]
            normalizer = peeh.normalizer_path(task, fold)
            if not normalizer.is_file():
                raise FileNotFoundError(normalizer)
            for model in peeh.MODELS:
                checkpoint = peeh.checkpoint_path(model, task, fold, 0)
                embedding = pswa.peeh_embedding_path(task, model, fold, 0)
                protected = pswa.peeh_result_path(task, model, fold, 0)
                session_q = pswa.session_cache_path(task, model, fold, 0)
                for p in (checkpoint, embedding, protected, session_q):
                    if not p.is_file():
                        raise FileNotFoundError(p)
                record = json.loads(protected.read_text())
                if sha(checkpoint) != record["checkpoint_sha256"] or sha(normalizer) != record["normalizer_sha256"]:
                    raise RuntimeError(f"frozen checkpoint/normalizer drift: {task} {model} f{fold}")
                entries.append({
                    "part": "A", "task": task, "fold": fold, "seed": 0, "variant": model,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                    "normalizer_path": str(normalizer), "normalizer_sha256": sha(normalizer),
                    "embedding_path": str(embedding), "embedding_sha256": sha(embedding),
                    "protected_path": str(protected), "protected_sha256": sha(protected),
                    "session_q_path": str(session_q), "session_q_sha256": sha(session_q),
                    "protected_coordinates": record["protected_coordinates"],
                    "random_control_source": "run_pswa_bridge.py::random_sets, final-random deterministic 100 draws",
                    "subject_split": {k: split[k] for k in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects")},
                    "eval_subjects": list(pswa.subjects_sessions(task)[0]),
                    "embedding_definition": "64D pre-classifier hidden representation, frozen eval mode",
                    "new_neural_training": False,
                    "final_heldout_used_for_diagnostic_evaluation": task != "WBCIC_MI",
                    "true_outer_used_for_diagnostic_evaluation": task == "WBCIC_MI",
                    "outer_development_used_for_part_a": False,
                })
    training = json.loads((MULTI / "runtime/TRAINING_LOGS.json").read_text())
    training_map = {(r["task"], r["variant"], int(r["fold"]), int(r["seed"])): r for r in training}
    carrier_protocol = ROOT / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol"
    carrier_splits = json.loads((carrier_protocol / "FIVEFOLD_SPLIT.json").read_text())["folds"]["OpenBMI"]
    carrier_manifest = {int(r["fold"]): r for r in json.loads((carrier_protocol / "MANIFEST_HASHES.json").read_text())
                        if r["dataset"] == "OpenBMI"}
    candidates = ("B0_FULL_EXISTING", "B1_SAME_SCALE_63", "B2_SCALE_COLLAPSE", "B4_ONE_STAGE_BACKEND")
    for fold in range(5):
        split = folds["OpenBMI"][fold]
        normalizer = peeh.normalizer_path("OpenBMI_MI", fold)
        for role in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects"):
            if set(split[role]) != set(carrier_splits[fold][role]):
                raise RuntimeError(f"historical Full subject split mismatch: f{fold} {role}")
        _, _, norm_meta = runtime.base.load_tensor_pair(normalizer)
        if norm_meta["mean_std_sha256"] != carrier_manifest[fold]["normalizer"]["mean_std_sha256"]:
            raise RuntimeError(f"historical Full normalizer mismatch: f{fold}")
        for seed in range(3):
            for variant in candidates:
                if variant == "B0_FULL_EXISTING":
                    checkpoint = peeh.litebn_path("OpenBMI_MI", fold, seed)
                    source = "official historical frozen Full; NOT protocol-matched to B1/B2/B4"
                    training_record = None
                else:
                    training_record = training_map[("OpenBMI_MI", variant, fold, seed)]
                    checkpoint = Path(training_record["checkpoint"])
                    source = "ablation-v2 seed0 or multiseed-v1 selected checkpoint"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                if training_record is not None and training_record.get("checkpoint_sha256") != sha(checkpoint):
                    raise RuntimeError(f"ablation checkpoint hash differs from training record: f{fold} s{seed} {variant}")
                entries.append({
                    "part": "B", "task": "OpenBMI_MI", "fold": fold, "seed": seed, "variant": variant,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                    "normalizer_path": str(normalizer), "normalizer_sha256": sha(normalizer),
                    "normalizer_mean_std_sha256": norm_meta["mean_std_sha256"],
                    "historical_carrier_split_and_normalizer_match": True,
                    "subject_split": {k: split[k] for k in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects")},
                    "embedding_definition": "64D pre-classifier hidden representation, frozen eval mode",
                    "protected_assignment_source": "Appendix-L full-64D train-only PEEH selection; new inner-train fit",
                    "random_control_source": "100 deterministic equal-union-rank active-coordinate sets",
                    "new_neural_training": False, "source": source,
                    "training_record": {k: training_record.get(k) for k in ("selected_epoch", "inner_val_BA", "seed", "fold")}
                    if training_record else None,
                    "final_heldout_used_in_part_b": False,
                    "outer_development_used_for_selection": False,
                })
    assert len(entries) == 100
    outer = [folds["OpenBMI"][f]["outer_dev_subjects"] for f in range(5)]
    flat = [s for group in outer for s in group]
    if len(flat) != 40 or len(set(flat)) != 40:
        raise RuntimeError("outer-development folds do not partition 40 subjects")
    manifest = {"schema": "SIRE_DIAGNOSTIC_DESIGN_BRIDGE_V1", "entries": entries,
                "part_a_cells": 40, "part_b_checkpoint_cells": 60,
                "part_b_B0_matched": False, "new_neural_training": 0,
                "outer_dev_subjects_unique": sorted(set(flat), key=int),
                "formal_model_name": "SIRE-EEG",
                "historical_names": "LiteBN/CompactLite are artifact provenance only"}
    write(EXP / "protocol/SOURCE_MANIFEST.json", manifest)
    lines = ["# Source audit", "", "100 frozen checkpoint cells verified by SHA256: 40 Part A and 60 Part B.",
             "Part A reuses the exact seed0 Full/B1 checkpoint, normalizer, embedding, Protected union and PSWA random-control rule.",
             "Part B uses server's historical Full as B0 at the user's explicit direction; split and train-only normalizer match but training protocol is NOT matched to B1/B2/B4.",
             "No neural retraining is planned. The Part B comparison is descriptive and cannot isolate architecture effects.",
             "The 40 outer-development subjects form a non-overlapping five-fold partition (8 per fold).",
             "Part A evaluates previously exposed 14-person internal-heldout or WBCIC 10-person true-outer cohorts.",
             "Part B must not read outer-development outcomes until SELECTION_FREEZE.json exists.",
             "Checkpoint and normalizer paths/hashes, splits and exact source details are in SOURCE_MANIFEST.json."]
    (EXP / "protocol/SOURCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SOURCE_AUDIT_OK entries={len(entries)}", flush=True)


if __name__ == "__main__":
    main()
