"""One-time OUTER evaluation after all five TRAIN seals exist. No final heldout."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
CORE_PATH = Path(os.environ.get("CID_CORE_CODE", str(Path(__file__).with_name("run.py"))))
core_spec = importlib.util.spec_from_file_location("cid_core_outer", CORE_PATH)
assert core_spec and core_spec.loader
cid = importlib.util.module_from_spec(core_spec)
sys.modules[core_spec.name] = cid
core_spec.loader.exec_module(cid)

SDG_PATH = Path(os.environ.get("CID_SDG_CODE", r"D:\nips-temp\TotalP\P1\shared_decision_geometry_full_runtime\code\run_full_v1.py"))


def load_sdg():
    spec = importlib.util.spec_from_file_location("cid_pinned_sdg", SDG_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_outer(fold, proof):
    sdg = load_sdg()
    ids, sessions, split_sha, cache_name = sdg.split_role("OpenBMI_MI", fold)
    if split_sha != proof["split_sha256"] or {k: list(v) for k, v in ids.items()} != proof["role_subjects"]:
        raise RuntimeError("frozen split or role drift")
    inner = sdg.module(f"cid_inner_{fold}", sdg.SEVEN_CODE / "tech_recipe_selection.py")
    raw, _, _, mapping = inner._openbmi_rows(sdg.CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[0],), cache_name)
    mean, std, norm_sha = sdg.historical_normalizer(raw)
    shape = (raw.shape[1], raw.shape[2]); del raw
    if norm_sha != proof["normalizer_sha256"]: raise RuntimeError("historical input normalizer drift")
    model, head, model_proof = sdg.checkpoint("EEGNet", "OpenBMI_MI", fold, split_sha, norm_sha, shape)
    if model_proof["checkpoint_sha256"] != proof["checkpoint_sha256"] or \
       model_proof["model_source_sha256"] != proof["model_source_sha256"]:
        raise RuntimeError("checkpoint or model-source drift")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); before = sdg.state_sha(model)
    if before != proof["model_state_sha256"]: raise RuntimeError("neural state mismatch")
    pieces = []
    for session in sessions:
        raw, session_labels, owner, new_mapping = inner._openbmi_rows(
            sdg.CACHE, list(ids["OUTER_DEVELOPMENT"]), (session,), cache_name, mapping)
        if new_mapping != mapping: raise RuntimeError("outer class mapping drift")
        x = ((raw - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
        del raw
        h = sdg.extract(model, head, x, device)
        del x
        # Labels are carried to an isolated oracle/outcome branch below. They
        # are never inspected for S1 context selection or feature construction.
        pieces.append({"h": h, "y": session_labels.astype(np.int64),
                       "subject": owner.astype(str),
                       "session": np.full(len(h), session, dtype=np.int64)})
    outer = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    if sdg.state_sha(model) != before: raise RuntimeError("neural weights/BN buffers mutated")
    if set(outer["subject"]) != set(ids["OUTER_DEVELOPMENT"]): raise RuntimeError("OUTER subject mismatch")
    if set(outer["subject"]) & set(ids["TRAIN_GEOMETRY"]): raise RuntimeError("subject leakage")
    w = head.weight.detach().cpu().numpy(); b = head.bias.detach().cpu().numpy()
    if sdg.digest_array(w, b) != proof["head_weight_sha256"]: raise RuntimeError("native head drift")
    return outer, {"split_sha256": split_sha, "normalizer_sha256": norm_sha,
                   "checkpoint_sha256": model_proof["checkpoint_sha256"],
                   "model_state_sha256_before": before, "model_state_sha256_after": sdg.state_sha(model),
                   "model_source_sha256": proof["model_source_sha256"],
                   "outer_subjects": list(ids["OUTER_DEVELOPMENT"]), "heldout_eeg_reads": 0,
                   "native_logits_identity_verified_by_extraction": True}


def descriptors_for_source(a, subjects, coord, pop, axes, budget, q):
    vals = {}
    for s in subjects:
        h, _ = cid.get_rows(a, s, 1)
        if budget != "ALL": h = h[:int(budget)]
        if len(h) < 8: raise RuntimeError("TRAIN context budget unavailable")
        z = coord.transform(h)
        vals[s] = {fam: cid.descriptor(z, pop, axes, fam, q) for fam in cid.FAMILIES}
        vals[s]["CTX_COMBINED_PLUS_PC"] = cid.descriptor(z, pop, axes, "CTX_COMBINED_PLUS_PC", q)
    return vals


def one_predict(mat, target, delta, pop, rank, alpha):
    _, singular, vt = np.linalg.svd(delta, full_matrices=False)
    basis = vt[:rank].T
    mm, vv = cid.descriptor_transform(mat, target)
    reg = Ridge(alpha=alpha).fit(mm, delta @ basis)
    correction = np.atleast_1d(reg.predict(vv)[0]) @ basis.T
    return pop + correction, correction, singular, float(reg.score(mm, delta @ basis))


def correction_quality(pred, actual, anchors):
    n = np.linalg.norm(pred) * np.linalg.norm(actual)
    cos = float(pred @ actual / n) if n > 1e-12 else None
    err = float(np.linalg.norm(pred - actual) / (np.linalg.norm(actual) + 1e-12))
    xx = np.c_[anchors, np.ones(len(anchors))]
    pr, ac = xx @ pred, xx @ actual
    fn = np.linalg.norm(pr) * np.linalg.norm(ac)
    fcos = float(pr @ ac / fn) if fn > 1e-12 else None
    fr2 = float(1 - np.sum((pr - ac) ** 2) / (np.sum((ac - ac.mean()) ** 2) + 1e-12))
    return {"correction_cosine": cos, "normalized_error": err,
            "function_cosine": fcos, "function_R2": fr2}


def evaluate_fold(fold):
    for k in range(5):
        if not (cid.RUNTIME / f"fold{k}/TRAIN_SEAL.json").is_file():
            raise RuntimeError("all five TRAIN selections must be sealed before OUTER")
    root = cid.RUNTIME / f"fold{fold}"
    seal_path = root / "TRAIN_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["outer_accessed"] or seal["final_heldout_eeg_reads"]:
        raise RuntimeError("TRAIN seal integrity")
    if seal["protocol_sha256"] != cid.sha(cid.EXP / "protocol/PROTOCOL_LOCK.json") or \
       seal["analysis_lock_sha256"] != cid.sha(cid.EXP / "protocol/ANALYSIS_LOCK.json"):
        raise RuntimeError("lock changed after TRAIN selection")
    a, _, proof = cid.load_train(fold)
    if proof != seal["representation_provenance"]: raise RuntimeError("TRAIN provenance drift")
    subjects = cid.ordered_subjects(a)
    with np.load(root / "train_seal.npz", allow_pickle=False) as f: frozen = {k: f[k] for k in f.files}
    coord = StandardScaler(); coord.mean_ = frozen["scaler_mean"]; coord.scale_ = frozen["scaler_scale"]
    coord.var_ = coord.scale_ ** 2; coord.n_features_in_ = len(coord.mean_)
    if cid.digest(coord.mean_, coord.scale_) != seal["scaler_sha256"]: raise RuntimeError("scaler SHA drift")
    choice = seal["best_global_type"]; config = seal["final_config"]
    pop = frozen["pop"]
    if not np.allclose(pop, cid.pop_head(a, subjects, coord, choice), atol=1e-8): raise RuntimeError("head drift")
    oracle_train = {s: frozen["oracle"][i] for i, s in enumerate(subjects)}
    delta = np.stack([oracle_train[s] - pop for s in subjects])
    anchors = frozen["anchors"]
    axes = cid.covariance_reference(a, subjects, coord)
    q = cid.pc_in_coord(a, coord, subjects, None)
    outer, outer_proof = extract_outer(fold, proof)
    outsubjects = sorted(set(outer["subject"]), key=int)
    rows = []; predictions = []; budgets = []; shuffles = []
    source_all = {B: descriptors_for_source(a, subjects, coord, pop, axes, B, q) for B in cid.LOCK["context_budgets_trials"]}
    for s in outsubjects:
        # Deployable predictions are computed from h only. S1 labels are not
        # consulted until the separate oracle-only branch below.
        cx = np.asarray(outer["h"][(outer["subject"] == s) & (outer["session"] == 1)], np.float64)
        ex = np.asarray(outer["h"][(outer["subject"] == s) & (outer["session"] == 2)], np.float64)
        ey = np.asarray(outer["y"][(outer["subject"] == s) & (outer["session"] == 2)], np.int64)
        z2 = coord.transform(ex)
        if len(cx) < 64: raise RuntimeError("OUTER context budget unavailable")
        method_heads = {"NATIVE_POPULATION_HEAD": frozen["native"],
                        "GLOBAL_REFIT_HEAD": frozen["global_refit"],
                        "BEST_GLOBAL": pop, "MEAN_CORRECTION": pop + delta.mean(axis=0)}
        for B in cid.LOCK["context_budgets_trials"]:
            x = cx if B == "ALL" else cx[:B]
            z = coord.transform(x)
            fam = config["descriptor"]
            target = cid.descriptor(z, pop, axes, fam, q)
            mat = np.stack([source_all[B][t][fam] for t in subjects])
            rule, pred_delta, singular, train_r2 = one_predict(mat, target, delta, pop, config["rank"], config["ridge_alpha"])
            budgets.append({"fold": fold, "subject": str(s), "budget": str(B), "trials": len(x),
                            **cid.scores(rule, z2, ey)})
            if B == "ALL":
                method_heads["CONTEXT_PREDICTED"] = rule
                chosen_prediction, chosen_train_r2 = pred_delta, train_r2
                near_mat, near_vec = cid.descriptor_transform(mat, target)
                near = np.argmin(np.sum((near_mat - near_vec) ** 2, axis=1))
                method_heads["NEAREST_CONTEXT"] = pop + delta[near]
                # Secondary P/C descriptor: same rank/regularization, same
                # source/OUTER subjects and budget, no classifier P/C operation.
                pcm = np.stack([source_all[B][t]["CTX_COMBINED_PLUS_PC"] for t in subjects])
                pct = cid.descriptor(z, pop, axes, "CTX_COMBINED_PLUS_PC", q)
                pc_rule, _, _, _ = one_predict(pcm, pct, delta, pop, config["rank"], config["ridge_alpha"])
                method_heads["CTX_COMBINED_PLUS_PC"] = pc_rule
                # Matched generic combined comparator at same rank/alpha.
                gm = np.stack([source_all[B][t]["CTX_COMBINED"] for t in subjects])
                gt = cid.descriptor(z, pop, axes, "CTX_COMBINED", q)
                gc_rule, _, _, _ = one_predict(gm, gt, delta, pop, config["rank"], config["ridge_alpha"])
                method_heads["CTX_COMBINED_MATCHED"] = gc_rule
        # Freeze deployable outputs before any S1 label read.
        for method, rule in method_heads.items():
            rows.append({"fold": fold, "subject": str(s), "method": method,
                         "label_assisted": False, **cid.scores(rule, z2, ey)})
        # Evaluation-only labelled S1 oracle; impossible deployable input.
        cy = np.asarray(outer["y"][(outer["subject"] == s) & (outer["session"] == 1)], np.int64)
        oracle = cid.logistic(coord.transform(cx), cy)
        rows.append({"fold": fold, "subject": str(s), "method": "LABEL_ASSISTED_ORACLE_S1_HEAD",
                     "label_assisted": True, **cid.scores(oracle, z2, ey)})
        _, _, vt = np.linalg.svd(delta, full_matrices=False)
        basis = vt[:config["rank"]].T
        actual_coeff = (oracle - pop) @ basis
        predicted_coeff = chosen_prediction @ basis
        source_coeff_mean = np.mean(delta @ basis, axis=0)
        coeff_r2 = float(1 - np.sum((predicted_coeff - actual_coeff) ** 2) /
                         (np.sum((actual_coeff - source_coeff_mean) ** 2) + 1e-12))
        predictions.append({"fold": fold, "subject": str(s), "descriptor": config["descriptor"],
                            "rank": config["rank"], "ridge_alpha": config["ridge_alpha"],
                            "coefficient_train_R2": chosen_train_r2, "coefficient_OUTER_R2": coeff_r2,
                            **correction_quality(chosen_prediction, oracle - pop, anchors)})
        # Control C: deterministic TRAIN-subject permutation, no OUTER fitting.
        target = cid.descriptor(coord.transform(cx), pop, axes, config["descriptor"], q)
        mat = np.stack([source_all["ALL"][t][config["descriptor"]] for t in subjects])
        for draw in range(cid.LOCK["shuffle_permutations"]):
            seed = int.from_bytes(bytes.fromhex(cid.digest(np.array([fold, draw], np.int64)))[:8], "little")
            perm = np.random.default_rng(seed).permutation(len(subjects))
            rule, _, _, _ = one_predict(mat[perm], target, delta, pop, config["rank"], config["ridge_alpha"])
            shuffles.append({"fold": fold, "subject": str(s), "draw": draw, **cid.scores(rule, z2, ey)})
        print("OUTER_SUBJECT", fold, s, flush=True)
    save = root / "outer_results.json"
    cid.save_json_new(save, {"schema": "CID_OUTER_V1", "fold": fold, "train_seal_sha256": cid.sha(seal_path),
        "outer_proof": outer_proof, "rows": rows, "prediction_quality": predictions,
        "budget_rows": budgets, "shuffle_rows": shuffles,
        "subject_id_as_feature": False, "outer_labels_in_descriptor": False,
        "outer_hyperparameter_selection": False, "formal_final_heldout_eeg_reads": 0})
    print("OUTER_COMPLETE", fold, cid.sha(save), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, required=True, choices=range(5))
    evaluate_fold(p.parse_args().fold)
