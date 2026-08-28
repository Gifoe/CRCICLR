from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import torch

import common as c


def verify_protocol_lock() -> dict:
    lock_path = c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("SCAA_STAGE0_PROTOCOL_LOCK.json is absent")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(lock_path.relative_to(c.REPO))],
        cwd=c.REPO,
        check=True,
        capture_output=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(lock_path.relative_to(c.REPO))],
        cwd=c.REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("protocol lock must be committed and clean")
    lock = c.read_json(lock_path)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["created_at_code_commit"], "HEAD"],
        cwd=c.REPO,
        check=True,
    )
    for relative, expected in lock["code_hashes"].items():
        path = c.EXP / relative
        if c.sha256(path) != expected:
            raise RuntimeError(f"post-freeze code modification: {relative}")
    if c.sha256(c.PROTOCOL / "DATA_ACCESS_LOCK.json") != lock["data_access_lock_sha256"]:
        raise RuntimeError("DATA_ACCESS_LOCK changed after freeze")
    if c.sha256(c.PROTOCOL / "ADAPTATION_RECIPE_SELECTION.json") != lock["adaptation_recipe_selection_sha256"]:
        raise RuntimeError("adaptation recipe changed after freeze")
    if lock["sealed_outer"]["identifiers_present"] is not False:
        raise RuntimeError("outer identifiers must not be present")
    return lock


def main() -> None:
    c.ensure_dirs()
    lock = verify_protocol_lock()
    recipe = c.read_json(c.PROTOCOL / "ADAPTATION_RECIPE_SELECTION.json")
    lr = float(recipe["selected_lr"])
    if recipe["adapter"] != "classifier_head_only_supervised" or not recipe["competence_gate_pass"]:
        raise RuntimeError("only the frozen competent head-only recipe is implemented")

    data = c.load_data()
    if tuple(c.subject_sort(data.metadata.subject_id.astype(str).unique())) != tuple(lock["development_subjects"]):
        raise RuntimeError("cache subject set differs from frozen development subjects")
    if set(data.metadata.session_id.astype(int).unique()) != {0, 1, 2}:
        raise RuntimeError("unexpected session set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("utility evaluation requires the server GPU")
    raw = torch.from_numpy(np.asarray(data.x)).to(device)

    rows: list[dict] = []
    trial_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("backbone", "fold", "seed", "subject_id", "session_id", "row_index", "label", "anchor_pred", "adapted_pred")
    }
    for backbone in c.BACKBONES:
        for fold in c.FOLDS:
            target_subjects = c.roles(fold)["outcome"]
            all_indices = c.row_indices(data.metadata, target_subjects, (0, 1, 2))
            for seed in c.SEEDS:
                model, mean, std, _ = c.load_anchor(backbone, fold, seed, device)
                extracted = c.extract(model, raw, data.metadata, all_indices, mean, std)
                initial_weight = model.head.weight.detach().float().cpu().numpy()
                initial_bias = model.head.bias.detach().float().cpu().numpy()
                for subject in target_subjects:
                    subject_mask = extracted["subjects"].astype(str) == subject
                    features = extracted["features"][subject_mask]
                    labels = extracted["labels"][subject_mask]
                    sessions = extracted["sessions"][subject_mask]
                    anchor_logits = extracted["logits"][subject_mask]
                    row_indices = extracted["indices"][subject_mask]
                    s1_all = np.flatnonzero(sessions == 0)
                    relative_train, relative_val = c.chronological_class_split(labels[s1_all])
                    train_idx = s1_all[relative_train]
                    val_idx = s1_all[relative_val]
                    adapted = c.adapt_head(
                        features,
                        labels,
                        train_idx,
                        val_idx,
                        initial_weight,
                        initial_bias,
                        lr,
                        c.stable_seed("SCAA-frozen-head", backbone, fold, seed, subject, lr),
                    )
                    record = {
                        "backbone": backbone,
                        "fold": fold,
                        "seed": seed,
                        "subject_id": subject,
                        "target_seen_by_anchor": False,
                        "adaptation_session": "S1",
                        "S2_or_S3_used_for_training_or_selection": False,
                        "selected_lr": lr,
                        "best_epoch_S1_validation": adapted["best_epoch"],
                        "parameter_relative_change": adapted["parameter_relative_change"],
                        "anchor_S1_validation_confidence": float(c.softmax_np(anchor_logits[val_idx]).max(axis=1).mean()),
                        "anchor_S1_validation_BA": c.metrics(labels[val_idx], anchor_logits[val_idx])["BA"],
                        "adapted_S1_validation_BA": c.metrics(labels[val_idx], adapted["logits"][val_idx])["BA"],
                    }
                    for session_id, label in ((1, "S2"), (2, "S3")):
                        idx = np.flatnonzero(sessions == session_id)
                        anchor_metric = c.metrics(labels[idx], anchor_logits[idx])
                        adapted_metric = c.metrics(labels[idx], adapted["logits"][idx])
                        record[f"n_{label}"] = int(len(idx))
                        record[f"anchor_{label}_BA"] = anchor_metric["BA"]
                        record[f"adapted_{label}_BA"] = adapted_metric["BA"]
                        record[f"Delta_{label}_BA"] = adapted_metric["BA"] - anchor_metric["BA"]
                        record[f"anchor_{label}_macro_F1"] = anchor_metric["macro_F1"]
                        record[f"adapted_{label}_macro_F1"] = adapted_metric["macro_F1"]
                        record[f"Delta_{label}_macro_F1"] = adapted_metric["macro_F1"] - anchor_metric["macro_F1"]

                        n = len(idx)
                        trial_parts["backbone"].append(np.full(n, backbone, dtype="U16"))
                        trial_parts["fold"].append(np.full(n, fold, dtype=np.int8))
                        trial_parts["seed"].append(np.full(n, seed, dtype=np.int8))
                        trial_parts["subject_id"].append(np.full(n, int(subject), dtype=np.int16))
                        trial_parts["session_id"].append(np.full(n, session_id, dtype=np.int8))
                        trial_parts["row_index"].append(row_indices[idx].astype(np.int32))
                        trial_parts["label"].append(labels[idx].astype(np.int8))
                        trial_parts["anchor_pred"].append(anchor_logits[idx].argmax(1).astype(np.int8))
                        trial_parts["adapted_pred"].append(adapted["logits"][idx].argmax(1).astype(np.int8))
                    rows.append(record)
                del model
                torch.cuda.empty_cache()
                print(f"[utility] {backbone} fold={fold} seed={seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame["_subject_sort"] = frame.subject_id.astype(int)
    frame = frame.sort_values(["backbone", "_subject_sort", "seed"]).drop(columns="_subject_sort")
    if len(frame) != 41 * 2 * 3:
        raise RuntimeError(f"expected 246 subject/backbone/seed rows, found {len(frame)}")
    if frame.groupby(["backbone", "subject_id"]).seed.nunique().ne(3).any():
        raise RuntimeError("incomplete seed coverage")
    c.write_csv(c.RESULTS / "PER_SUBJECT_SEED_UTILITY.csv", frame)
    np.savez_compressed(c.RUNTIME / "TRIAL_PREDICTIONS.npz", **{key: np.concatenate(value) for key, value in trial_parts.items()})
    c.write_json(
        c.RUNTIME / "UTILITY_EXECUTION.json",
        {
            "schema": "SCAA_STAGE0_UTILITY_EXECUTION_V1",
            "protocol_lock_sha256": c.sha256(c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json"),
            "execution_commit": c.git_head(),
            "rows": len(frame),
            "trial_prediction_rows": int(sum(len(part) for part in trial_parts["label"])),
            "S2_or_S3_used_for_training_or_selection": False,
            "outer_accessed": False,
            "complete": True,
        },
    )
    print("SCAA_STAGE0_FROZEN_UTILITY_COMPLETE rows=246", flush=True)


if __name__ == "__main__":
    main()
