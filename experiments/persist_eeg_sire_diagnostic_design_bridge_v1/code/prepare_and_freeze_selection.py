#!/usr/bin/env python3
"""Part B prepare phase. This program never constructs an outer-dev bundle."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from source_audit import EXP, MULTI, OLD, sha, write

sys.path.insert(0, str(OLD / "code"))
import run_peeh_bridge as peeh  # noqa: E402
import run_pswa_bridge as pswa  # noqa: E402
sys.path.insert(0, str(MULTI / "code"))
import run_multiseed as arch  # noqa: E402

OUT = EXP / "outputs/prospective_selection"
RUN = EXP / "runtime/part_b"
CANDIDATES = ("B0_FULL_EXISTING", "B1_SAME_SCALE_63", "B2_SCALE_COLLAPSE", "B4_ONE_STAGE_BACKEND")
TASK = "OpenBMI_MI"
SESSIONS = (1, 2)


def manifest():
    value = json.loads((EXP / "protocol/SOURCE_MANIFEST.json").read_text())
    entries = {(e["fold"], e["seed"], e["variant"]): e for e in value["entries"] if e["part"] == "B"}
    if len(entries) != 60:
        raise RuntimeError("missing Part-B frozen checkpoint manifest")
    return entries


def write_protocol() -> None:
    spec = {"schema": "SIRE_P3_DIAGNOSTIC_SELECTOR_V1", "candidate_order_for_exact_ties": list(CANDIDATES),
            "task": TASK, "folds": 5, "seeds": [0, 1, 2], "outer_labels_used_to_design_rule": False,
            "admissible_if": "Protected nonempty AND inner-validation PEEH_pp > 0 AND inner-validation PSWA_pp > 0",
            "primary_choice": "Among admissible, maximum inner-validation complement_retained_WSBA",
            "fallback": "If no admissible candidate, maximum inner-validation intact_probe_WSBA",
            "ties": "exact numerical ties use candidate_order_for_exact_ties",
            "evaluation_unit": "inner-validation biological subject; average seeds within subject first",
            "no_significance_gate": True, "random_policy": "uniform exact expectation across four candidates",
            "B0_status": "historical frozen Full, user-directed; not matched training protocol",
            "evaluation_label": "pre-specified development replay, not strictly prospective"}
    target = EXP / "protocol/SELECTOR_SPEC.json"
    if target.is_file() and json.loads(target.read_text()) != spec:
        raise RuntimeError("SELECTOR_SPEC changed after initial freeze")
    write(target, spec)
    audit = {"schema": "SIRE_P3_PROSPECTIVE_AUDIT_V1", "task": TASK,
             "candidate_training_uses_outer_dev_subjects": False,
             "candidate_early_stopping_uses_outer_dev_labels": False,
             "selector_uses_outer_dev_labels_before_freeze": False,
             "final_heldout_14_used_in_part_b": False,
             "historical_Full_B0_protocol_matched": False,
             "historical_Full_B0_reason": "existing multiseed protocol explicitly states B0_trained=false and official_reference_is_matched=false",
             "prior_exposure": "CONFIRMED: persist_eeg_carrier_5fold_multiseed_stability_v1/code/train_grid.py evaluates the historical Full/LiteBN checkpoint on these outer_dev_subjects and records selected_outer_BA and per-subject results",
             "classification": "pre-specified development replay, not strictly prospective",
             "claim_limit": "policy results do not isolate causal architecture effect and do not establish independent prospective confirmation",
             "future_P4_requirement": "apply the identical frozen selector to an independent cohort without retuning"}
    target = EXP / "protocol/PROSPECTIVE_AUDIT.json"
    if target.is_file() and json.loads(target.read_text()) != audit:
        raise RuntimeError("PROSPECTIVE_AUDIT changed")
    write(target, audit)


def load_model(runtime, variant, checkpoint, seed, device):
    if variant == "B0_FULL_EXISTING":
        model = runtime.base.build_model("LiteBN_BASELINE", TASK)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    else:
        model = arch.make_model(TASK, variant, seed)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device).eval()
    if any(m.training for m in model.modules()):
        raise RuntimeError("model not eval-locked")
    return model


def infer(model, bundle, indices, mean, std, device):
    hh, yy, predictions = [], [], []
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None]
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None]
    with torch.inference_mode():
        for start in range(0, len(indices), 64):
            ids = indices[start:start + 64]
            raw = bundle.signal_batch(ids)
            x = torch.as_tensor(np.ascontiguousarray(raw), dtype=torch.float32, device=device)
            logits, h = model((x - mean_t) / torch.clamp(std_t, min=1e-6))
            if h.ndim != 2 or h.shape[1] != 64:
                raise RuntimeError("not a 64D pre-classifier embedding")
            hh.append(h.float().cpu().numpy())
            predictions.append(logits.float().cpu().numpy())
            yy.append(bundle.labels(ids))
    return np.concatenate(hh), np.concatenate(yy), np.concatenate(predictions)


def emb_path(fold, seed, variant):
    return RUN / "development_embeddings" / variant / f"fold{fold}_seed{seed}.npz"


def cached_embeddings(runtime, bundle, fold, seed, variant, entry, mean, std, device):
    path = emb_path(fold, seed, variant)
    if path.is_file():
        cached = pswa.load_npz(path)
        if cached["metadata"]["checkpoint_sha256"] != entry["checkpoint_sha256"]:
            raise RuntimeError("stale development embedding")
        return cached
    train_subjects = entry["subject_split"]["inner_train_subjects"]
    val_subjects = entry["subject_split"]["inner_val_subjects"]
    train_ids = bundle.indices(train_subjects, SESSIONS)
    val_ids = bundle.indices(val_subjects, SESSIONS)
    model = load_model(runtime, variant, Path(entry["checkpoint_path"]), seed, device)
    train_h, train_y, _ = infer(model, bundle, train_ids, mean, std, device)
    val_h, val_y, val_logits = infer(model, bundle, val_ids, mean, std, device)
    rows = bundle.rows
    result = {"train_h": train_h, "train_y": train_y,
              "train_subject": np.asarray([str(rows[int(i)].subject) for i in train_ids], dtype="U16"),
              "train_session": np.asarray([int(rows[int(i)].session) for i in train_ids]),
              "val_h": val_h, "val_y": val_y, "val_logits": val_logits,
              "val_subject": np.asarray([str(rows[int(i)].subject) for i in val_ids], dtype="U16"),
              "val_session": np.asarray([int(rows[int(i)].session) for i in val_ids]),
              "metadata": {"checkpoint_sha256": entry["checkpoint_sha256"],
                           "normalizer_sha256": entry["normalizer_sha256"],
                           "train_subjects": train_subjects, "validation_subjects": val_subjects,
                           "outer_dev_loaded": False}}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in result.items() if k != "metadata"},
                        metadata=json.dumps(result["metadata"], sort_keys=True))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def session_ba(y, pred, subjects, sessions, target, classes):
    values = {}
    for s in target:
        values[str(s)] = {}
        for t in SESSIONS:
            m = (subjects.astype(str) == str(s)) & (sessions == t)
            if not m.any():
                raise RuntimeError(f"missing validation {s} S{t}")
            values[str(s)][t] = pswa.ba(y[m], pred[m], classes)
    return values


def probe_scores(train_x, train_y, val_x, val_y, val_subject, val_session, subjects):
    classes = len(np.unique(train_y))
    pack = pswa.ridge_fit(train_x, train_y, classes)
    prediction = pswa.ridge_predict(val_x, pack)
    return session_ba(val_y, prediction, val_subject, val_session, subjects, classes)


def evaluate_cell(data, fold, seed, variant, entry):
    train_h, train_y = data["train_h"], data["train_y"]
    val_h, val_y = data["val_h"], data["val_y"]
    val_subject, val_session = data["val_subject"].astype(str), data["val_session"].astype(int)
    subjects = entry["subject_split"]["inner_val_subjects"]
    classes = len(np.unique(train_y))
    meta = pd.DataFrame({"subject_id": data["train_subject"].astype(str),
                         "session_id": data["train_session"].astype(int), "label": train_y.astype(int)})
    run_key = ("PEEH-v1", TASK, variant, fold, seed)
    spectrum = peeh.build_spectrum(meta, train_h, run_key)
    protected, selected_blocks = peeh.select_protected(train_h, meta, spectrum, run_key, classes)
    erasures = peeh.evaluate_erasure(train_h, train_y, val_h, val_y, val_subject,
                                    spectrum, protected, run_key, classes)
    q_train, q_val = peeh.coordinates(train_h, spectrum), peeh.coordinates(val_h, spectrum)
    random_sets = pswa.random_sets(TASK, variant, fold, seed, len(protected), len(spectrum["rho"]))
    intact = probe_scores(train_h, train_y, val_h, val_y, val_subject, val_session, subjects)
    complement = probe_scores(peeh.erase(train_h, spectrum, protected), train_y,
                              peeh.erase(val_h, spectrum, protected), val_y, val_subject, val_session, subjects)
    ponly = probe_scores(q_train[:, protected], train_y, q_val[:, protected], val_y, val_subject, val_session, subjects)
    random_only = {str(s): [] for s in subjects}
    for subset in random_sets:
        current = probe_scores(q_train[:, subset], train_y, q_val[:, subset], val_y, val_subject, val_session, subjects)
        for s in subjects:
            random_only[str(s)].append(min(current[str(s)].values()))
    decoder = session_ba(val_y, data["val_logits"].argmax(1), val_subject, val_session, subjects, classes)
    erasure_by_subject = {str(x["subject_id"]): x for x in erasures}
    rows = []
    for s in subjects:
        s = str(s)
        po = min(ponly[s].values())
        ro = float(np.mean(random_only[s]))
        rows.append({"task": TASK, "fold": fold, "seed": seed, "variant": variant, "subject_id": s,
                     "protected_rank": len(protected), "active_rank": len(spectrum["rho"]),
                     "protected_coverage": int(bool(protected)), "PEEH_pp": erasure_by_subject[s]["PEEH_pp"],
                     "PSWA_pp": 100 * (po - ro), "protected_only_WSBA": po,
                     "random_only_WSBA": ro, "complement_retained_WSBA": min(complement[s].values()),
                     "intact_probe_WSBA": min(intact[s].values()),
                     "validation_full_model_future_BA": decoder[s][2],
                     "checkpoint_sha256": entry["checkpoint_sha256"]})
    return rows, {"fold": fold, "seed": seed, "variant": variant, "protected_coordinates": protected,
                  "active_rank": len(spectrum["rho"]), "selected_blocks": sum(x["protected"] for x in selected_blocks),
                  "random_sets_sha256": pswa.sha256_json(random_sets),
                  "checkpoint_sha256": entry["checkpoint_sha256"],
                  "normalizer_sha256": entry["normalizer_sha256"],
                  "training_subjects": entry["subject_split"]["inner_train_subjects"],
                  "validation_subjects": subjects,
                  "outer_dev_used": False}


def select(profile):
    result = []
    order = {v: i for i, v in enumerate(CANDIDATES)}
    for fold in range(5):
        f = profile[profile.fold == fold].copy()
        admissible = f[(f.protected_coverage > 0) & (f.PEEH_pp > 0) & (f.PSWA_pp > 0)]
        if len(admissible):
            selected = sorted(admissible.to_dict("records"),
                              key=lambda r: (-r["complement_retained_WSBA"], order[r["variant"]]))[0]
            reason = "admissible; highest complement-retained WSBA"
        else:
            selected = sorted(f.to_dict("records"),
                              key=lambda r: (-r["intact_probe_WSBA"], order[r["variant"]]))[0]
            reason = "no admissible candidate; highest intact-probe WSBA"
        val_selected = sorted(f.to_dict("records"),
                              key=lambda r: (-r["validation_full_model_future_BA"], order[r["variant"]]))[0]
        result.append({"fold": fold, "diagnostic_choice": selected["variant"], "validation_BA_choice": val_selected["variant"],
                       "reason": reason, "admissible_candidates": admissible.variant.tolist(),
                       "chosen_PEEH_pp": selected["PEEH_pp"], "chosen_PSWA_pp": selected["PSWA_pp"],
                       "chosen_complement_retained_WSBA": selected["complement_retained_WSBA"]})
    return result


def main() -> None:
    if (OUT / "OUTER_SUBJECT_RESULTS.csv").exists():
        raise RuntimeError("outer outcomes already opened; prepare cannot rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    write_protocol()
    entries = manifest()
    runtime = peeh.load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audit_rows = [], []
    row_path = OUT / "DEVELOPMENT_DIAGNOSTICS.csv"
    if row_path.is_file():
        rows = pd.read_csv(row_path).to_dict("records")
        for row in rows:
            row["subject_id"] = str(row["subject_id"])
    result_path = RUN / "development_cell_audit.json"
    if result_path.is_file():
        audit_rows = json.loads(result_path.read_text())
    done = {(int(x["fold"]), int(x["seed"]), x["variant"]) for x in audit_rows}
    for fold in range(5):
        split = entries[(fold, 0, CANDIDATES[0])]["subject_split"]
        expected = split["inner_train_subjects"] + split["inner_val_subjects"]
        if len(expected) != 32 or len(set(expected)) != 32:
            raise RuntimeError("development split mismatch")
        bundle = runtime.base.build_bundle(TASK, expected)
        mean, std, _ = runtime.base.load_tensor_pair(peeh.normalizer_path(TASK, fold))
        for variant in CANDIDATES:
            for seed in range(3):
                if (fold, seed, variant) in done:
                    continue
                entry = entries[(fold, seed, variant)]
                if sha(Path(entry["checkpoint_path"])) != entry["checkpoint_sha256"]:
                    raise RuntimeError("checkpoint drift")
                data = cached_embeddings(runtime, bundle, fold, seed, variant, entry, mean, std, device)
                current, audit = evaluate_cell(data, fold, seed, variant, entry)
                rows.extend(current)
                audit_rows.append(audit)
                pswa.atomic_csv(row_path, rows)
                write(result_path, audit_rows)
                done.add((fold, seed, variant))
                print(f"PART_B_DEV f{fold} s{seed} {variant} P={len(audit['protected_coordinates'])}", flush=True)
        del bundle
        gc.collect()
    if len(done) != 60 or len(rows) != 60 * 6:
        raise RuntimeError(f"development diagnostics incomplete: cells={len(done)} rows={len(rows)}")
    df = pd.DataFrame(rows)
    # First average seeds within each fold/candidate/biological subject.
    subject = df.groupby(["fold", "variant", "subject_id"], as_index=False)[["protected_coverage", "PEEH_pp", "PSWA_pp",
        "protected_only_WSBA", "random_only_WSBA", "complement_retained_WSBA", "intact_probe_WSBA",
        "validation_full_model_future_BA"]].mean()
    profile = subject.groupby(["fold", "variant"], as_index=False)[["protected_coverage", "PEEH_pp", "PSWA_pp",
        "protected_only_WSBA", "random_only_WSBA", "complement_retained_WSBA", "intact_probe_WSBA",
        "validation_full_model_future_BA"]].mean()
    profile["nonempty_seed_runs"] = df.groupby(["fold", "variant"]).protected_coverage.sum().to_numpy()
    pswa.atomic_csv(OUT / "PER_FOLD_CANDIDATE_PROFILE.csv", profile)
    choices = select(profile)
    committed_cell_audit = OUT / "DEVELOPMENT_CELL_AUDIT.json"
    write(committed_cell_audit, audit_rows)
    spec_path = EXP / "protocol/SELECTOR_SPEC.json"
    source_path = EXP / "protocol/SOURCE_MANIFEST.json"
    input_hashes = {str(p): sha(p) for p in (spec_path, source_path, EXP / "protocol/SOURCE_AUDIT.md",
                                                 EXP / "protocol/PROSPECTIVE_AUDIT.json",
                                                 row_path, OUT / "PER_FOLD_CANDIDATE_PROFILE.csv", committed_cell_audit)}
    freeze = {"schema": "SIRE_P3_SELECTION_FREEZE_V1", "status": "FROZEN_BEFORE_OUTER_EVALUATION",
              "classification": "pre-specified development replay (historical B0 nonmatched; prior exposure cannot be excluded)",
              "outer_development_labels_read": False, "choices": choices,
              "selector_spec_sha256": sha(spec_path), "selection_input_hashes": input_hashes,
              "checkpoint_sha256": {f"f{f}_s{s}_{v}": entries[(f, s, v)]["checkpoint_sha256"]
                                    for f in range(5) for s in range(3) for v in CANDIDATES}}
    write(OUT / "SELECTION_FREEZE.json", freeze)
    write(OUT / "PROTOCOL_AUDIT.json", {"development_rows": len(rows), "candidate_seed_cells": len(done),
          "inner_train_subjects_per_fold": 26, "inner_val_subjects_per_fold": 6,
          "outer_dev_used_in_selection": False, "final_heldout_used": False,
          "B0_protocol_matched": False, "selection_freeze_sha256": sha(OUT / "SELECTION_FREEZE.json")})
    pswa.atomic_csv(OUT / "MATCHED_CHECKPOINT_AUDIT.csv", [{"fold": f, "seed": s, "variant": v,
        "checkpoint": entries[(f,s,v)]["checkpoint_path"], "sha256": entries[(f,s,v)]["checkpoint_sha256"],
        "protocol_matched_to_B1_B2_B4": v != "B0_FULL_EXISTING"} for f in range(5) for s in range(3) for v in CANDIDATES])
    print("PART_B_SELECTION_FROZEN", flush=True)


if __name__ == "__main__":
    main()
