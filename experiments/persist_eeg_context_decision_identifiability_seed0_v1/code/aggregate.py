"""Aggregate frozen TRAIN/OUTER audits into compact subject-level evidence."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CORE_PATH = Path(os.environ.get("CID_CORE_CODE", str(Path(__file__).with_name("run.py"))))
core_spec = importlib.util.spec_from_file_location("cid_core_aggregate", CORE_PATH)
assert core_spec and core_spec.loader
cid = importlib.util.module_from_spec(core_spec)
sys.modules[core_spec.name] = cid
core_spec.loader.exec_module(cid)

OUT = cid.EXP / "outputs"
METHODS = ("NATIVE_POPULATION_HEAD", "GLOBAL_REFIT_HEAD", "BEST_GLOBAL", "MEAN_CORRECTION", "NEAREST_CONTEXT", "CONTEXT_PREDICTED",
           "LABEL_ASSISTED_ORACLE_S1_HEAD", "CTX_COMBINED_MATCHED", "CTX_COMBINED_PLUS_PC")


def write_csv(name, rows):
    if not rows: raise RuntimeError(f"empty output {name}")
    path = OUT / name
    if path.exists(): raise FileExistsError(path)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def avg(rows, field="BA"):
    return float(np.mean([float(r[field]) for r in rows]))


def method_map(rows):
    return {(int(r["fold"]), str(r["subject"]), r["method"]): r for r in rows}


def subject_paired_effect(rows, method_a, method_b, field="BA"):
    lookup = method_map(rows)
    by = defaultdict(list)
    for (fold, subject, method), row in lookup.items():
        if method == method_a:
            other = lookup[(fold, subject, method_b)]
            by[subject].append(float(row[field]) - float(other[field]))
    return {s: float(np.mean(v)) for s, v in by.items()}


def bootstrap(effects, name, draws=20000):
    keys = sorted(effects, key=int); values = np.array([effects[k] for k in keys], np.float64)
    if len(keys) < 2: raise RuntimeError("too few biological subjects for bootstrap")
    seed = int.from_bytes(bytes.fromhex(cid.digest(np.array([len(keys), len(name)], np.int64)))[:8], "little")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(keys), size=(draws, len(keys)))
    dist = values[ix].mean(axis=1)
    lo, hi = np.quantile(dist, [.025, .975])
    return {"contrast": name, "n_unique_subjects": len(keys), "n_draws": draws,
            "estimate": float(values.mean()), "CI95_low": float(lo), "CI95_high": float(hi)}


def fold_mean(rows, method):
    return {f: avg([r for r in rows if int(r["fold"]) == f and r["method"] == method]) for f in range(5)}


def exact_hash_inventory():
    return {p.name: cid.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "HASH_INDEX.json"}


def train_pc_crossfit(fold, a, audit, config, choice):
    out = []; subjects = cid.ordered_subjects(a)
    halves = [set(map(str, block)) for block in audit["crossfit_half_subjects"]]
    for held in subjects:
        half = 0 if held in halves[0] else 1
        if held not in halves[half] or held in set(map(str, audit["crossfit"][half]["fit_subjects"])):
            raise RuntimeError("TRAIN P/C half crossfit leakage")
        source = [s for s in subjects if s != held]
        coord = cid.fit_coord(a, source); pop = cid.pop_head(a, source, coord, choice)
        oracles = cid.subject_oracles(a, source, coord)
        delta = np.stack([oracles[s] - pop for s in source])
        axes = cid.covariance_reference(a, source, coord)
        raw = np.asarray(a[f"p_half{half}"], np.float64)
        q = np.linalg.qr(coord.scale_[:, None] * raw)[0] if raw.shape[1] else raw
        desc = {}
        for s in source + [held]:
            z = coord.transform(cid.get_rows(a, s, 1)[0])
            desc[s] = {fam: cid.descriptor(z, pop, axes, fam, q)
                       for fam in ("CTX_COMBINED", "CTX_COMBINED_PLUS_PC")}
        z2 = coord.transform(cid.get_rows(a, held, 2)[0]); y2 = cid.get_rows(a, held, 2)[1]
        for fam in ("CTX_COMBINED", "CTX_COMBINED_PLUS_PC"):
            mat = np.stack([desc[s][fam] for s in source]); vec = desc[held][fam]
            basis = np.linalg.svd(delta, full_matrices=False)[2][:config["rank"]].T
            mm, vv = cid.descriptor_transform(mat, vec)
            from sklearn.linear_model import Ridge
            pred = Ridge(alpha=config["ridge_alpha"]).fit(mm, delta @ basis).predict(vv)
            correction = np.atleast_1d(pred[0]) @ basis.T
            sc = cid.scores(pop + correction, z2, y2)
            out.append({"role": "TRAIN_LOSO_HALF_CROSSFIT", "fold": fold, "subject": held,
                        "descriptor": fam, "P_half": half, "held_excluded_from_P_fit": True, **sc})
    return out


def aggregate():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()): raise RuntimeError("output directory not empty; no overwrite")
    seals = []; results = []
    for fold in range(5):
        root = cid.RUNTIME / f"fold{fold}"
        s = json.loads((root / "TRAIN_SEAL.json").read_text())
        r = json.loads((root / "outer_results.json").read_text())
        if r["train_seal_sha256"] != cid.sha(root / "TRAIN_SEAL.json"):
            raise RuntimeError("TRAIN seal mismatch")
        if r["formal_final_heldout_eeg_reads"] or r["outer_hyperparameter_selection"] or r["outer_labels_in_descriptor"]:
            raise RuntimeError("integrity failure")
        if s["protocol_sha256"] != cid.sha(cid.EXP / "protocol/PROTOCOL_LOCK.json"):
            raise RuntimeError("protocol drift")
        seals.append(s); results.append(r)
    cid.save_json_new(cid.EXP / "protocol/SOURCE_PROVENANCE.json", {
        "schema": "CID_SOURCE_PROVENANCE_V1", "basis_commit": "c79fbb73b8dc7756371ad479af0041e7ff176e82",
        "recent_mechanism_trajectory_read": ["8a8b708b30e19a20a83902d272fbe53fe8279f12",
            "d5184c5423806693a3b218f2d0329cd6b29d5d9a", "378f71f46fcfc7f8644e8885f15a95af169038d0",
            "7ed3ab94508b718519b4a39f1b375f491a874d52", "c79fbb73b8dc7756371ad479af0041e7ff176e82"],
        "protocol_sha256": cid.sha(cid.EXP / "protocol/PROTOCOL_LOCK.json"),
        "analysis_lock_sha256": cid.sha(cid.EXP / "protocol/ANALYSIS_LOCK.json"),
        "run_code_sha256": cid.sha(Path(cid.__file__)),
        "outer_code_sha256": cid.sha(Path(os.environ.get("CID_OUTER_CODE", str(cid.EXP / "code/outer.py")))),
        "aggregate_code_sha256": cid.sha(Path(__file__)),
        "train_seal_sha256": [cid.sha(cid.RUNTIME / f"fold{f}/TRAIN_SEAL.json") for f in range(5)],
        "outer_sha256": [cid.sha(cid.RUNTIME / f"fold{f}/outer_results.json") for f in range(5)],
        "formal_final_heldout_reads": 0})
    cid.save_json_new(cid.EXP / "protocol/REPRESENTATION_PROVENANCE.json", {
        "schema": "CID_REPRESENTATION_PROVENANCE_V1", "backbone": "EEGNet", "task": "OpenBMI_MI",
        "seed": 0, "embedding_dim": 64, "folds": [s["representation_provenance"] for s in seals],
        "outer_extraction": [r["outer_proof"] for r in results],
        "head_logits_identity_verified_on_upstream_batches": True,
        "common_scaler_sha256_by_fold": [s["scaler_sha256"] for s in seals],
        "neural_weights_and_BN_buffers_unchanged": all(r["outer_proof"]["model_state_sha256_before"] ==
            r["outer_proof"]["model_state_sha256_after"] for r in results),
        "formal_final_heldout_eeg_reads": 0})
    outer = [row for r in results for row in r["rows"]]
    quality = [row for r in results for row in r["prediction_quality"]]
    budget = [row for r in results for row in r["budget_rows"]]
    shuffle = [row for r in results for row in r["shuffle_rows"]]
    if len(shuffle) != 200 * len(quality): raise RuntimeError("shuffle count")

    global_rows = []; correction = []; function = []; spectrum = []; descriptor = []; pairwise_correction_cosines = []
    loso = [row | {"fold": fold} for fold, s in enumerate(seals) for row in s["train_loso_rows"]]
    for fold in range(5):
        a, audit, proof = cid.load_train(fold)
        s = seals[fold]
        train_subjects = cid.ordered_subjects(a)
        choice_by_subject = {str(x["subject"]): x for x in s["nested_loso_choices"]}
        for held in train_subjects:
            source = [x for x in train_subjects if x != held]
            ep = cid.build_episode(a, source, held, s["best_global_type"])
            cfg = choice_by_subject[held]
            fam = cfg["descriptor"]
            mat = np.stack([ep["desc"][x][fam] for x in source])
            vec = ep["desc"][held][fam]
            mm, vv = cid.descriptor_transform(mat, vec)
            near = int(np.argmin(np.sum((mm - vv) ** 2, axis=1)))
            oracle_held = cid.logistic(ep["coord"].transform(cid.get_rows(a, held, 1)[0]),
                                       cid.get_rows(a, held, 1)[1])
            for method, theta in (("NEAREST_CONTEXT", ep["oracle"][source[near]]),
                                  ("LABEL_ASSISTED_ORACLE_S1_HEAD", oracle_held)):
                loso.append({"fold": fold, "subject": held, "method": method,
                    "descriptor": fam, "rank": cfg["rank"], "ridge_alpha": cfg["ridge_alpha"],
                    "label_assisted": method.startswith("LABEL_ASSISTED"),
                    **cid.scores(theta, ep["z2"], ep["y2"])})
        with np.load(cid.RUNTIME / f"fold{fold}/train_seal.npz", allow_pickle=False) as z:
            frozen = {k: z[k] for k in z.files}
        subs = list(map(str, frozen["subjects"])); pop = frozen["pop"]
        anchors = frozen["anchors"]; anchor_subjects = []
        for t in subs:
            for cls in (0, 1):
                anchor_subjects.extend([(t, cls, i) for i in range(4)])
        if len(anchor_subjects) != len(anchors): raise RuntimeError("anchor inventory")
        for row in s["global_loso_rows"]: global_rows.append({"fold": fold, "role": "TRAIN_LOSO", **row})
        for row in results[fold]["rows"]:
            if row["method"] in ("NATIVE_POPULATION_HEAD", "GLOBAL_REFIT_HEAD"):
                global_rows.append({"fold": fold, "role": "OUTER_DEVELOPMENT", "subject": row["subject"],
                                    "head_type": row["method"], "BA": row["BA"], "macro_F1": row["macro_F1"], "NLL": row["NLL"]})
        for i, subject in enumerate(subs):
            theta = frozen["oracle"][i]; delta = theta - pop
            n = np.linalg.norm(theta[:-1]) * np.linalg.norm(pop[:-1])
            cosine = float(theta[:-1] @ pop[:-1] / n) if n > 1e-12 else None
            correction.append({"fold": fold, "subject": subject, "correction_norm": float(np.linalg.norm(delta)),
                "head_cosine": cosine, "boundary_angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))) if cosine is not None else None,
                "intercept_difference": float(delta[-1]), "oracle_S1_label_assisted": True})
            rr = cid.margin(delta, anchors)
            for j, (anchor_sub, cls, anchor_ix) in enumerate(anchor_subjects):
                function.append({"fold": fold, "subject": subject, "anchor_subject": anchor_sub,
                    "anchor_class": cls, "anchor_trial_index": anchor_ix, "margin_correction": float(rr[j])})
        singular = frozen["singular"]
        delta_mat = frozen["oracle"] - pop[None, :]
        norms = np.linalg.norm(delta_mat, axis=1)
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                if norms[i] * norms[j] > 1e-12:
                    pairwise_correction_cosines.append(float(delta_mat[i] @ delta_mat[j] / (norms[i] * norms[j])))
        for i, value in enumerate(singular[:min(26, len(singular))]):
            spectrum.append({"fold": fold, "component": i + 1, "singular_value": float(value),
                "cumulative_energy": float(np.sum(singular[:i+1] ** 2) / np.sum(singular ** 2)),
                "selected_rank": s["final_config"]["rank"]})
        for fam in cid.FAMILIES:
            vals = [r for r in s["grid"] if r["descriptor"] == fam]
            best = sorted(vals, key=lambda r: (-r["mean_BA"], r["mean_NLL"], r["rank"], -r["ridge_alpha"]))[0]
            descriptor.append({"fold": fold, "descriptor": fam, "train_loso_best_BA": best["mean_BA"],
                               "train_loso_best_rank": best["rank"], "train_loso_best_alpha": best["ridge_alpha"],
                               "unlabeled_S1_only": True, "transform_source": "TRAIN-only", "PCA_cap": 8})
    write_csv("GLOBAL_HEAD_AUDIT.csv", global_rows)
    write_csv("SUBJECT_DECISION_CORRECTIONS.csv", correction)
    write_csv("FUNCTION_CORRECTION_TARGETS.csv", function)
    write_csv("PERSONALIZATION_HEADROOM.csv", [dict(r, headroom_BA=float(r["BA"]) - float(method_map(outer)[(int(r["fold"]), str(r["subject"]), "BEST_GLOBAL")]["BA"]))
        for r in outer if r["method"] == "LABEL_ASSISTED_ORACLE_S1_HEAD"])
    write_csv("CORRECTION_SPECTRUM.csv", spectrum)
    write_csv("CONTEXT_DESCRIPTOR_AUDIT.csv", descriptor)
    write_csv("TRAIN_LOSO_CONTEXT_RESULTS.csv", loso)
    write_csv("OUTER_CONTEXT_RESULTS.csv", outer)
    write_csv("CONTEXT_CORRECTION_PREDICTION.csv", quality)
    write_csv("CONTEXT_BUDGET_CURVE.csv", budget)
    write_csv("CONTEXT_SHUFFLE_NULL.csv", shuffle)

    baseline = []
    for f in range(5):
        for m in METHODS:
            group = [r for r in outer if int(r["fold"]) == f and r["method"] == m]
            baseline.append({"fold": f, "method": m, "n_subjects": len(group),
                             "BA": avg(group), "macro_F1": avg(group, "macro_F1"), "NLL": avg(group, "NLL")})
    write_csv("CONTEXT_BASELINE_COMPARISON.csv", baseline)
    pc = []; outer_pc = {}
    for f in range(5):
        gen = next(r for r in baseline if r["fold"] == f and r["method"] == "CTX_COMBINED_MATCHED")
        aug = next(r for r in baseline if r["fold"] == f and r["method"] == "CTX_COMBINED_PLUS_PC")
        outer_pc[f] = {"role": "OUTER_DEVELOPMENT", "fold": f, "generic_combined_BA": gen["BA"],
                       "combined_plus_pc_BA": aug["BA"], "PC_gain_BA": aug["BA"] - gen["BA"],
                       "scope": "same primary rank/alpha, source-only P"}
        pc.append(outer_pc[f])
        aa, audit, _ = cid.load_train(f)
        pc.extend(train_pc_crossfit(f, aa, audit, seals[f]["final_config"], seals[f]["best_global_type"]))
    write_csv("PC_CONTEXT_ABLATION.csv", pc)
    lookup = method_map(outer)
    subject_effects = []
    for f, sub, m in lookup:
        if m != "BEST_GLOBAL": continue
        gl = lookup[f, sub, "BEST_GLOBAL"]["BA"]
        mean = lookup[f, sub, "MEAN_CORRECTION"]["BA"]
        ctx = lookup[f, sub, "CONTEXT_PREDICTED"]["BA"]
        oracle = lookup[f, sub, "LABEL_ASSISTED_ORACLE_S1_HEAD"]["BA"]
        subject_effects.append({"fold": f, "subject": sub, "oracle_minus_global": oracle - gl,
            "context_minus_global": ctx - gl, "context_minus_mean": ctx - mean,
            "per_subject_recovery": (ctx - gl) / (oracle - gl) if oracle > gl else None})
    write_csv("SUBJECT_LEVEL_EFFECTS.csv", subject_effects)
    contrasts = [bootstrap(subject_paired_effect(outer, a, b), name) for a, b, name in (
        ("LABEL_ASSISTED_ORACLE_S1_HEAD", "BEST_GLOBAL", "oracle_minus_global"),
        ("CONTEXT_PREDICTED", "BEST_GLOBAL", "context_minus_global"),
        ("CONTEXT_PREDICTED", "MEAN_CORRECTION", "context_minus_mean"),
        ("LABEL_ASSISTED_ORACLE_S1_HEAD", "MEAN_CORRECTION", "oracle_minus_mean"))]
    write_csv("BOOTSTRAP_CONTRASTS.csv", contrasts)
    cb = {r["contrast"]: r for r in contrasts}
    bmean = {m: avg([r for r in outer if r["method"] == m]) for m in METHODS}
    headroom = bmean["LABEL_ASSISTED_ORACLE_S1_HEAD"] - bmean["BEST_GLOBAL"]
    gain = bmean["CONTEXT_PREDICTED"] - bmean["BEST_GLOBAL"]
    mean_gain = bmean["CONTEXT_PREDICTED"] - bmean["MEAN_CORRECTION"]
    recovery = gain / headroom if headroom > 0 else None
    fm = {m: fold_mean(outer, m) for m in METHODS}
    null_p95 = {}
    for f in range(5):
        by = defaultdict(list)
        for r in shuffle:
            if int(r["fold"]) == f: by[int(r["draw"])].append(r["BA"])
        if len(by) != 200: raise RuntimeError("200 shuffled draws missing")
        null_p95[f] = float(np.quantile([np.mean(v) for v in by.values()], .95))
    gate_a = headroom >= .015 and sum(fm["LABEL_ASSISTED_ORACLE_S1_HEAD"][f] > fm["BEST_GLOBAL"][f] for f in range(5)) >= 4 and cb["oracle_minus_global"]["CI95_low"] > 0
    gate_b = (gain >= .005 and mean_gain >= .003 and recovery is not None and recovery >= .25 and
              sum(fm["CONTEXT_PREDICTED"][f] > fm["BEST_GLOBAL"][f] for f in range(5)) >= 4 and
              sum(fm["CONTEXT_PREDICTED"][f] > null_p95[f] for f in range(5)) >= 4 and
              cb["context_minus_global"]["CI95_low"] > -.005)
    pc_gain = bmean["CTX_COMBINED_PLUS_PC"] - bmean["CTX_COMBINED_MATCHED"]
    if not gate_a: case, action = "NO_MEANINGFUL_PERSONALIZATION_HEADROOM", "STOP_CONTEXT_MODEL_DIRECTION"
    elif bmean["LABEL_ASSISTED_ORACLE_S1_HEAD"] - bmean["MEAN_CORRECTION"] < .003:
        case, action = "GLOBAL_CORRECTION_EXPLAINS_PERSONALIZATION", "PURSUE_GLOBAL_CALIBRATION_NOT_CONTEXT_MODEL"
    elif not gate_b: case, action = "PERSONALIZATION_HEADROOM_EXISTS_BUT_CONTEXT_NOT_IDENTIFIABLE", "DO_NOT_BUILD_ZERO_LABEL_CONTEXT_MODEL"
    elif pc_gain >= .003 and sum(outer_pc[f]["PC_gain_BA"] > 0 for f in range(5)) >= 4:
        case, action = "PC_CONTEXT_FEATURES_HELP_BUT_GENERAL_CONTEXT_ALREADY_WORKS", "PROCEED_CONTEXT_REPROGRAMMING_MODEL_WITH_PC_ABLATION"
    else: case, action = "PC_STRUCTURE_NOT_REQUIRED_FOR_CONTEXT_IDENTIFICATION", "PROCEED_CONTEXT_REPROGRAMMING_MODEL_WITHOUT_PC"
    summary = {"schema": "CID_DECISION_V1", "Gate_A": bool(gate_a), "Gate_B": bool(gate_b),
        "pooled_oracle_headroom_BA": headroom, "pooled_context_gain_BA": gain,
        "pooled_context_minus_mean_BA": mean_gain, "pooled_recovery": recovery,
        "PC_augmentation_gain_BA": pc_gain, "fold_shuffled_p95_BA": null_p95,
        "primary_interpretation": case, "exact_next_action": action,
        "formal_final_heldout_eeg_reads": 0, "branch": cid.LOCK["branch"],
        "outer_historically_exposed": True,
        "train_seal_sha256": [cid.sha(cid.RUNTIME / f"fold{f}/TRAIN_SEAL.json") for f in range(5)],
        "outer_audit_sha256": [cid.sha(cid.RUNTIME / f"fold{f}/outer_results.json") for f in range(5)]}
    cid.save_json_new(OUT / "DECISION_SUMMARY.json", summary)
    cid.save_json_new(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {"formal_final_heldout_eeg_reads": 0,
        "population_roles": cid.LOCK["population_roles"], "OUTER_only_after_all_train_seals": True,
        "OUTER_s1_labels_oracle_only": True, "OUTER_s2_labels_outcome_only": True,
        "all_train_seals_exist": True, "historical_outer_exposure_disclosed": True})
    report = ["# Context decision identifiability, seed 0", "",
        "This is a frozen-backbone direction-qualification test, not a Context Reprogramming model. OUTER_DEVELOPMENT was historically exposed in earlier mechanism work; formal final-heldout EEG reads: 0.", ""]
    headroom_values = np.array(list(subject_paired_effect(
        outer, "LABEL_ASSISTED_ORACLE_S1_HEAD", "BEST_GLOBAL").values()))
    oracle_rows = [r for r in outer if r["method"] == "LABEL_ASSISTED_ORACLE_S1_HEAD"]
    global_eval_rows = [r for r in outer if r["method"] == "BEST_GLOBAL"]
    family_counts = {fam: sum(s["final_config"]["descriptor"] == fam for s in seals) for fam in cid.FAMILIES}
    answers = [
        f"Q1. Labelled S1 oracle minus strongest TRAIN-chosen global head: {headroom:+.4f} BA, "
        f"macro-F1 {avg(oracle_rows, 'macro_F1')-avg(global_eval_rows, 'macro_F1'):+.4f}, "
        f"NLL {avg(oracle_rows, 'NLL')-avg(global_eval_rows, 'NLL'):+.4f}; "
        f">+1pp {np.mean(headroom_values > .01):.1%}, <-1pp {np.mean(headroom_values < -.01):.1%}, "
        f"median {np.median(headroom_values):+.4f}, worst-quartile mean {np.mean(np.sort(headroom_values)[:max(1,len(headroom_values)//4)]):+.4f}; "
        f"Gate A {'PASS' if gate_a else 'FAIL'}.",
        f"Q2. Subject corrections: median norm {np.median([x['correction_norm'] for x in correction]):.4f}; "
        f"median head cosine {np.median([x['head_cosine'] for x in correction if x['head_cosine'] is not None]):.4f}; "
        f"median pairwise correction cosine {np.median(pairwise_correction_cosines):.4f} "
        f"(IQR {np.quantile(pairwise_correction_cosines,.25):.4f}–{np.quantile(pairwise_correction_cosines,.75):.4f}). "
        "Numerical heterogeneity alone does not establish transferable context information.",
        f"Q3. Oracle minus mean correction: {bmean['LABEL_ASSISTED_ORACLE_S1_HEAD'] - bmean['MEAN_CORRECTION']:+.4f} BA.",
        f"Q4. Parameter cosine mean {np.mean([x['correction_cosine'] for x in quality if x['correction_cosine'] is not None]):.4f}; function-response cosine mean {np.mean([x['function_cosine'] for x in quality if x['function_cosine'] is not None]):.4f}. Future-session BA remains primary.",
        f"Q5. Context minus global: {gain:+.4f} BA; context minus mean correction: {mean_gain:+.4f} BA; Gate B {'PASS' if gate_b else 'FAIL'}.",
        f"Q6. Pooled labelled-oracle headroom recovery: {recovery if recovery is not None else 'undefined (nonpositive headroom)'}.",
        f"Q7. Fold-level context exceeds shuffled-context p95 in {sum(fm['CONTEXT_PREDICTED'][f] > null_p95[f] for f in range(5))}/5 folds.",
        f"Q8. Final TRAIN-selected descriptor counts: {family_counts}. TRAIN LOSO family audits are in CONTEXT_DESCRIPTOR_AUDIT.csv; OUTER never selects a family.",
        f"Q9. P/C augmentation minus matched generic combined descriptor: {pc_gain:+.4f} BA (secondary; no P/C classifier intervention).",
        f"Q10. {action}. Primary interpretation: {case}.",
        "Unseen subject IDs have no learned embedding and are not usable as a transductive input; no subject ID enters the context mapper.", ""]
    report += answers
    report += ["## Fold table", "", "|fold|TRAIN subjects|OUTER subjects|dim|global type|global BA|oracle BA|headroom|mean BA|nearest BA|context BA|context gain|vs mean|recovery|parameter cosine|function cosine|rank|ridge alpha|descriptor|full trials|B8|B16|B32|B64|shuffle p95|PC gain|interpretation|",
               "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for f in range(5):
        cfg = seals[f]["final_config"]
        gf = fm["BEST_GLOBAL"][f]; of = fm["LABEL_ASSISTED_ORACLE_S1_HEAD"][f]
        cf = fm["CONTEXT_PREDICTED"][f]; mf = fm["MEAN_CORRECTION"][f]
        nsub = len({str(r["subject"]) for r in outer if int(r["fold"]) == f})
        bb = {str(B): avg([r for r in budget if int(r["fold"]) == f and r["budget"] == str(B)]) for B in cid.LOCK["context_budgets_trials"]}
        full = [r["trials"] for r in budget if int(r["fold"]) == f and r["budget"] == "ALL"]
        qq = [r for r in quality if int(r["fold"]) == f]
        ccos = np.mean([r["correction_cosine"] for r in qq if r["correction_cosine"] is not None])
        fcos = np.mean([r["function_cosine"] for r in qq if r["function_cosine"] is not None])
        fold_case = "headroom_negative" if of <= gf else ("context_gain" if cf > gf else "context_no_gain")
        fields = [f, 26, nsub, 64, seals[f]["best_global_type"], f"{gf:.4f}", f"{of:.4f}", f"{of-gf:+.4f}",
                  f"{mf:.4f}", f"{fm['NEAREST_CONTEXT'][f]:.4f}", f"{cf:.4f}", f"{cf-gf:+.4f}",
                  f"{cf-mf:+.4f}", f"{(cf-gf)/(of-gf):.3f}" if of > gf else "undefined", f"{ccos:.3f}", f"{fcos:.3f}",
                  cfg["rank"], cfg["ridge_alpha"], cfg["descriptor"], int(np.median(full)),
                  *(f"{bb[str(B)]:.4f}" for B in (8,16,32,64)), f"{null_p95[f]:.4f}",
                  f"{outer_pc[f]['PC_gain_BA']:+.4f}", fold_case]
        report.append("|" + "|".join(map(str, fields)) + "|")
    report += ["", f"Pooled oracle headroom: {headroom:+.4f}; pooled context gain: {gain:+.4f}; pooled recovery: {recovery}.",
               f"Gate A: {'PASS' if gate_a else 'FAIL'}; Gate B: {'PASS' if gate_b else 'FAIL'}.",
               f"Primary interpretation: `{case}`. Exact next action: `{action}`.",
               f"Formal final-heldout EEG reads: 0. Branch: `{cid.LOCK['branch']}`.",
               "Final commit SHA: populated after reviewed GitHub push in the accompanying commit record.", ""]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    cid.save_json_new(OUT / "HASH_INDEX.json", exact_hash_inventory())
    print("AGGREGATE_COMPLETE", case, action, len(subject_effects), flush=True)


if __name__ == "__main__": aggregate()
