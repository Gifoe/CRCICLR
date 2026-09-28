"""Aggregate locked five-fold audits without fitting or reading EEG arrays."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import aggregate as core

EXP = Path(__file__).resolve().parents[1]
OUT = core.ROOT / "final_outputs"
AMEND = json.loads((EXP / "protocol/PROTOCOL_AMENDMENT_V2.json").read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def gather(subdir: str, table: str) -> list[dict]:
    rows = []
    for fold in range(5):
        folder = core.ROOT / subdir / f"fold{fold}"
        audit = json.loads((folder / "AUDIT.json").read_text(encoding="utf-8"))
        path = folder / f"{table}.csv"
        if audit["files"][path.name] != core.sha(path) or audit["final_heldout_eeg_reads"] != 0:
            raise RuntimeError(f"provenance mismatch: {path}")
        rows.extend(read_csv(path))
    return rows


def arm_map(rows: list[dict], metric: str = "BA"):
    return {(int(r["fold"]), r["subject"], r["arm"]): float(r[metric]) for r in rows
            if r.get(metric, "") != ""}


def biological_contrast(rows: list[dict], a: str, b: str) -> list[tuple[str, float]]:
    m = arm_map(rows)
    by_sub = defaultdict(list)
    for fold, subject, arm in m:
        if arm == a and (fold, subject, b) in m:
            by_sub[subject].append(m[(fold, subject, a)] - m[(fold, subject, b)])
    return [(s, float(np.mean(v))) for s, v in sorted(by_sub.items(), key=lambda x: int(x[0]))]


def bootstrap(values: list[tuple[str, float]], label: str) -> dict:
    if len(values) < 8:
        raise RuntimeError(f"too few biological subjects for {label}")
    data = np.asarray([x[1] for x in values], np.float64)
    rng = np.random.default_rng(core.seed("BIOLOGICAL_BOOTSTRAP", label, 0))
    ix = rng.integers(0, len(data), size=(core.LOCK["paired_bootstrap_draws"], len(data)))
    draws = data[ix].mean(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"contrast": label, "biological_subjects": len(data), "draws": len(draws),
            "mean": float(data.mean()), "ci95_low": float(low), "ci95_high": float(high),
            "resampling_unit": "biological_subject"}


def safe_mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def one(rows, fold, arm):
    matches = [r for r in rows if int(r["fold"]) == fold and r["arm"] == arm]
    if len(matches) != 1:
        raise RuntimeError(f"nonunique fold/arm: {fold} {arm}")
    return matches[0]


def report(summary, fold_rows, bootstrap_rows, all_rows):
    b = {r["contrast"]: r for r in bootstrap_rows}
    f = summary["folds"]
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)
    lines = ["# Frozen EEGNet recording-context representation audit, seed 0", "",
             "OUTER_DEVELOPMENT was historically exposed and was evaluation-only here. CHECKPOINT_VALIDATION was inventory-only. Formal final-heldout EEG reads: 0. No neural parameter was trained or changed.", "",
             "The predeclared REL_MEAN class-relation gate is algebraically impossible: (c1-r)-(c0-r)=c1-c0 within each recording. The equality was checked numerically. Thus geometry cleanliness in the same-class centroid metric cannot be substituted for Gate A.", ""]
    q = summary["answers"]
    for n in range(1, 13):
        lines.append(f"Q{n}. {q[f'Q{n}']}")
    lines += ["", "## Locked fold table", "",
              "|" + "|".join(fold_rows[0].keys()) + "|",
              "|" + "|".join("---" for _ in fold_rows[0]) + "|"]
    for row in fold_rows:
        lines.append("|" + "|".join(fmt(v) for v in row.values()) + "|")
    lines += ["", "## Pooled summaries", "",
              f"Mean Absolute BA: {f['absolute_BA']:.6f}; mean REL_MEAN BA: {f['relative_BA']:.6f}; mean REL_SHRINK BA: {f['shrink_BA']:.6f}.",
              f"Pooled REL-ABS BA: {b['REL_MEAN_MINUS_ABSOLUTE']['mean']:.6f}, 95% biological-subject paired CI [{b['REL_MEAN_MINUS_ABSOLUTE']['ci95_low']:.6f}, {b['REL_MEAN_MINUS_ABSOLUTE']['ci95_high']:.6f}].",
              f"Class-relation cosine gain: {f['relation_gain']:.9f}; session-variance reduction: {f['session_variance_reduction']:.6f}; class-information change (TRAIN probe BA): {f['class_probe_BA_change']:.6f}.",
              "Gates: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in summary["gates"].items()) + ".",
              f"Primary interpretation: `{summary['primary_interpretation']}`. Exact next action: `{summary['next_action']}`.",
              "Formal final-heldout EEG reads: 0.",
              "Branch: `codex/persist-eeg-context-relative-representation-audit-seed0-v1`.",
              "Final commit SHA: provided in the reviewed GitHub commit/delivery record after push; a commit cannot contain its own SHA.", ""]
    return "\n".join(lines)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    provenance_by_fold = {}
    for fold in range(5):
        lock = core.ROOT / "train_lock" / f"fold{fold}.json"
        if not lock.is_file():
            raise RuntimeError(f"missing TRAIN lock fold{fold}")
        provenance_by_fold[str(fold)] = {}
        for role in ("TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"):
            suffix = "_v2" if fold == 0 and role == "TRAIN_GEOMETRY" else ""
            p = core.ROOT / "features" / f"fold{fold}_{role.lower()}{suffix}" / "PROVENANCE.json"
            meta = json.loads(p.read_text(encoding="utf-8"))
            if meta["final_heldout_eeg_reads"] or meta["model_state_before"] != meta["model_state_after"]:
                raise RuntimeError(f"feature provenance fail {p}")
            provenance_by_fold[str(fold)][role] = {"provenance_sha256": core.sha(p),
                "checkpoint_sha256": meta["checkpoint_sha256"], "checkpoint_record_sha256": meta["checkpoint_record_sha256"],
                "model_source_sha256": meta["model_source_sha256"], "split_sha256": meta["split_sha256"],
                "normalizer_sha256": meta["normalizer_sha256"], "model_state_before": meta["model_state_before"],
                "model_state_after": meta["model_state_after"], "role_subjects": meta["role_subjects"],
                "stage_files": meta["stage_files"], "stage_audit": meta["stage_audit"]}
    tables = {}
    for table in ("CONTEXT_REFERENCE_AUDIT", "REPRESENTATION_VARIANCE_DECOMPOSITION", "CROSS_SESSION_GEOMETRY",
                  "LAYER_LOCALIZATION", "SUBJECT_SESSION_CLASS_PROBES", "INFORMATION_RETENTION"):
        tables[table] = gather("geometry", table)
    for table in ("OUTER_TRANSFER_RESULTS", "SUBJECT_LEVEL_EFFECTS", "WRONG_SUBJECT_REFERENCE_NULL",
                  "RANDOM_REFERENCE_NULL", "CONTEXT_BUDGET_CURVE", "CONTEXT_SHRINKAGE_RESULTS",
                  "RELATIVE_Z_RESULTS", "PCA_REMOVAL_CONTROL", "CONTEXT_CLASS_BALANCE_AUDIT", "TRIAL_ORDER_AUDIT",
                  "FULL_SESSION_LOO_DIAGNOSTIC"):
        tables[table] = gather("outer", table)
    tables["CONTEXT_REFERENCE_AUDIT"].extend(gather("localization", "CONTEXT_REFERENCE_OUTER"))
    tables["CROSS_SESSION_GEOMETRY"].extend(gather("localization", "CROSS_SESSION_GEOMETRY_OUTER"))
    local_outer = gather("localization", "LAYER_LOCALIZATION_OUTER")
    local_map = {(int(r["fold"]), r["stage"], r["arm"]): r for r in local_outer}
    for row in tables["LAYER_LOCALIZATION"]:
        fold, stage = int(row["fold"]), row["stage"]
        row["OUTER_absolute_BA_secondary"] = local_map[(fold, stage, "ABSOLUTE")]["OUTER_subject_equal_BA"]
        row["OUTER_relative_BA_secondary"] = local_map[(fold, stage, "REL_MEAN")]["OUTER_subject_equal_BA"]
        row["OUTER_REL_minus_ABS_BA_secondary"] = float(row["OUTER_relative_BA_secondary"]) - float(row["OUTER_absolute_BA_secondary"])
        row["secondary_decoder_note"] = local_map[(fold, stage, "ABSOLUTE")]["decoder"]
    locks = [json.loads((core.ROOT / "train_lock" / f"fold{fold}.json").read_text(encoding="utf-8")) for fold in range(5)]
    tables["TRAIN_CONTEXT_SELECTION"] = [{"fold": l["fold"], "B": l["B"], "C_absolute": l["C_absolute"],
                                           "C_relative": l["C_relative"], "lambda_shrink": l["lambda_shrink"],
                                           "lambda_z": l["lambda_z"], "pca_k": l["pca_k"],
                                           "scope": "TRAIN_GEOMETRY_ONLY"} for l in locks]
    tables["GLOBAL_CENTER_CONTROL"] = [r for r in tables["OUTER_TRANSFER_RESULTS"]
                                       if r["arm"] in ("ABSOLUTE", "GLOBAL_SOURCE_CENTER", "FEATUREWISE_STANDARDIZATION", "REL_MEAN")]
    effects = tables["SUBJECT_LEVEL_EFFECTS"]
    contrasts = (("REL_MEAN", "ABSOLUTE"), ("REL_SHRINK", "ABSOLUTE"),
                 ("REL_MEAN", "GLOBAL_SOURCE_CENTER"), ("REL_MEAN", "WRONG_SUBJECT_REFERENCE_MEAN"),
                 ("REL_MEAN", "PCA_REMOVAL"))
    boots = [bootstrap(biological_contrast(effects, a, b), f"{a}_MINUS_{b}") for a, b in contrasts]
    relation = defaultdict(list)
    for row in tables["CROSS_SESSION_GEOMETRY"]:
        if row["stage"] == "EMBEDDING" and row["role"] == "OUTER_DEVELOPMENT":
            relation[row["subject"]].append(float(row["class_relation_cosine_delta"]))
    boots.append(bootstrap([(s, float(np.mean(v))) for s, v in relation.items()], "CLASS_RELATION_COSINE_GAIN"))
    tables["BOOTSTRAP_CONTRASTS"] = boots
    bootmap = {r["contrast"]: r for r in boots}
    outer = tables["OUTER_TRANSFER_RESULTS"]
    wrong = tables["WRONG_SUBJECT_REFERENCE_NULL"]
    random = tables["RANDOM_REFERENCE_NULL"]
    geom = [r for r in tables["CROSS_SESSION_GEOMETRY"] if r["stage"] == "EMBEDDING" and r["role"] == "OUTER_DEVELOPMENT"]
    if max(abs(float(r["class_relation_cosine_delta"])) for r in geom) > 1e-7 or max(
        abs(float(r["class_relation_drift_abs"]) - float(r["class_relation_drift_rel"])) for r in geom) > 1e-7:
        raise RuntimeError("session-constant translation unexpectedly changed class relation")
    var = [r for r in tables["REPRESENTATION_VARIANCE_DECOMPOSITION"] if r["stage"] == "EMBEDDING"]
    probes = [r for r in tables["SUBJECT_SESSION_CLASS_PROBES"] if r["stage"] == "EMBEDDING"]
    folds = []
    for fold in range(5):
        pick = lambda arm: one(outer, fold, arm)
        g = [r for r in geom if int(r["fold"]) == fold]
        vabs = one(var, fold, "ABSOLUTE"); vrel = one(var, fold, "REL_MEAN")
        pabs = one(probes, fold, "ABSOLUTE"); prel = one(probes, fold, "REL_MEAN")
        ca = [r for r in tables["CONTEXT_REFERENCE_AUDIT"] if int(r["fold"]) == fold and r["stage"] == "EMBEDDING" and r["role"] == "OUTER_DEVELOPMENT"]
        cb = [r for r in tables["CONTEXT_CLASS_BALANCE_AUDIT"] if int(r["fold"]) == fold]
        wn = np.asarray([float(r["BA"]) for r in wrong if int(r["fold"]) == fold])
        rn = np.asarray([float(r["BA"]) for r in random if int(r["fold"]) == fold])
        rel_ba, abs_ba = float(pick("REL_MEAN")["subject_equal_BA"]), float(pick("ABSOLUTE")["subject_equal_BA"])
        folds.append({"fold": fold, "TRAIN_subjects": 26, "OUTER_subjects": 8,
                      "selected_B": locks[fold]["B"], "selected_lambda": locks[fold]["lambda_shrink"],
                      "absolute_BA": abs_ba, "REL_MEAN_BA": rel_ba,
                      "REL_SHRINK_BA": float(pick("REL_SHRINK")["subject_equal_BA"]),
                      "REL_Z_BA": float(pick("REL_Z")["subject_equal_BA"]),
                      "REL_minus_ABS_BA": rel_ba - abs_ba,
                      "global_center_BA": float(pick("GLOBAL_SOURCE_CENTER")["subject_equal_BA"]),
                      "S1_reference_for_S2_BA": float(pick("S1_REFERENCE_FOR_S2")["subject_equal_BA"]),
                      "wrong_subject_p95_BA": float(np.quantile(wn, 0.95)),
                      "random_reference_p95_BA": float(np.quantile(rn, 0.95)),
                      "PCA_removal_BA": float(pick("PCA_REMOVAL")["subject_equal_BA"]),
                      "absolute_relation_cosine": safe_mean(g, "class_relation_cosine_abs"),
                      "relative_relation_cosine": safe_mean(g, "class_relation_cosine_rel"),
                      "relation_improvement": safe_mean(g, "class_relation_cosine_delta"),
                      "absolute_class_variance": float(vabs["class_variance"]),
                      "relative_class_variance": float(vrel["class_variance"]),
                      "absolute_session_variance": float(vabs["session_within_subject_variance"]),
                      "relative_session_variance": float(vrel["session_within_subject_variance"]),
                      "absolute_session_probe_BA": float(pabs["session_BA_train_only"]),
                      "relative_session_probe_BA": float(prel["session_BA_train_only"]),
                      "absolute_subject_ID_BA": float(pabs["subject_ID_BA_train_only"]),
                      "relative_subject_ID_BA": float(prel["subject_ID_BA_train_only"]),
                      "context_class_proportion": safe_mean(cb, "natural_context_class_fraction"),
                      "reference_norm": safe_mean(ca, "session2_reference_norm"),
                      "transform_norm": safe_mean(ca, "session2_reference_norm"),
                      "fold_interpretation": "REL_GAIN" if rel_ba > abs_ba else "NO_REL_GAIN"})
    delta = bootmap["REL_MEAN_MINUS_ABSOLUTE"]
    rel_global = bootmap["REL_MEAN_MINUS_GLOBAL_SOURCE_CENTER"]
    rel_wrong = bootmap["REL_MEAN_MINUS_WRONG_SUBJECT_REFERENCE_MEAN"]
    relation_gain = float(np.mean([r["relation_improvement"] for r in folds]))
    drift = [float(r["class_relation_drift_abs"]) - float(r["class_relation_drift_rel"]) for r in geom]
    same_class_gain = float(np.mean([(float(r["same_class_cosine_rel_0"]) + float(r["same_class_cosine_rel_1"]) -
                                      float(r["same_class_cosine_abs_0"]) - float(r["same_class_cosine_abs_1"])) / 2 for r in geom]))
    # Exact within-session relation invariance makes the specified Gate A
    # impossible for session-constant REL_MEAN. Never replace it by the easier
    # same-class-centroid cosine or a numerical roundoff fluctuation.
    gate_a = False
    gate_b = bool(delta["mean"] >= .010 and sum(r["REL_minus_ABS_BA"] > 0 for r in folds) >= 4 and delta["ci95_low"] > 0)
    std_ba = float(np.mean([float(one(outer, f, "FEATUREWISE_STANDARDIZATION")["subject_equal_BA"]) for f in range(5)]))
    gate_c = bool(rel_global["mean"] > 0 and np.mean([r["REL_MEAN_BA"] for r in folds]) > std_ba)
    gate_d = bool(sum(r["REL_MEAN_BA"] > r["wrong_subject_p95_BA"] for r in folds) >= 4 or rel_wrong["ci95_low"] > 0)
    probe_change = float(np.mean([float(one(probes, f, "REL_MEAN")["class_BA_train_only"]) -
                                  float(one(probes, f, "ABSOLUTE")["class_BA_train_only"]) for f in range(5)]))
    class_ratio = float(np.mean([r["relative_class_variance"] / max(r["absolute_class_variance"], 1e-12) for r in folds]))
    gate_e = bool(probe_change >= -.010 and class_ratio >= .90)
    balance = tables["CONTEXT_CLASS_BALANCE_AUDIT"]
    frac = np.asarray([float(r["natural_context_class_fraction"]) for r in balance])
    gain = np.asarray([float(r["natural_REL_minus_ABS_BA"]) for r in balance])
    corr = float(np.corrcoef(frac, gain)[0, 1]) if np.std(frac) > 0 and np.std(gain) > 0 else 0.0
    balance_gain = float(np.mean([float(r["label_assisted_balanced_context_BA"]) -
                                  float(r["natural_reference_matched_outcome_BA"]) for r in balance]))
    if all((gate_a, gate_b, gate_c, gate_d, gate_e)) and not (abs(corr) >= .5 and balance_gain >= .01):
        case, action = "RECORDING_CONTEXT_RELATIVE_ENCODING_SUPPORTED", "PROCEED_REFERENCE_CONDITIONED_ENCODING_MODEL"
    elif delta["mean"] <= 0 and balance_gain >= .01 and abs(corr) >= .5:
        case, action = "CLASS_BALANCE_EXPLAINS_CONTEXT_REFERENCE", "STOP_CONTEXT_RELATIVE_MODEL_DIRECTION"
    elif delta["mean"] > 0 and rel_global["mean"] <= 0:
        case, action = "GLOBAL_CENTERING_EXPLAINS_THE_EFFECT", "PURSUE_SIMPLE_GLOBAL_NORMALIZATION"
    elif same_class_gain >= .05 and delta["mean"] <= 0:
        case, action = "GEOMETRY_STABILIZES_WITHOUT_PREDICTIVE_GAIN", "STOP_CONTEXT_RELATIVE_MODEL_DIRECTION"
    elif 0 < delta["mean"] < .010 and np.mean([r["REL_MEAN_BA"] - r["S1_reference_for_S2_BA"] for r in folds]) > 0 and rel_wrong["mean"] > 0:
        case, action = "CURRENT_SESSION_REFERENCE_REQUIRED_BUT_GAIN_SMALL", "STOP_ARCHITECTURE_DIRECTION"
    else:
        case, action = "NO_CONTEXT_RELATIVE_ADVANTAGE", "STOP_CONTEXT_RELATIVE_MODEL_DIRECTION"
    gates = {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d, "E": gate_e}
    fsum = {"absolute_BA": float(np.mean([r["absolute_BA"] for r in folds])),
            "relative_BA": float(np.mean([r["REL_MEAN_BA"] for r in folds])),
            "shrink_BA": float(np.mean([r["REL_SHRINK_BA"] for r in folds])),
            "relation_gain": relation_gain,
            "session_variance_reduction": float(np.mean([r["absolute_session_variance"] - r["relative_session_variance"] for r in folds])),
            "class_probe_BA_change": probe_change}
    context = [r for r in tables["CONTEXT_REFERENCE_AUDIT"] if r["stage"] == "EMBEDDING" and r["role"] == "OUTER_DEVELOPMENT"]
    shift_ratio = safe_mean(context, "mean_shift_to_within_rms")
    stage_gains = {stage: float(np.mean([float(r["OUTER_REL_minus_ABS_BA_secondary"]) for r in tables["LAYER_LOCALIZATION"]
                                         if r["stage"] == stage])) for stage in core.LOCK["stages"]}
    stage_max = max(stage_gains, key=stage_gains.get)
    answers = {
        "Q1": f"TRAIN recording-mean shift / within-session RMS = {shift_ratio:.4f} (EMBEDDING).",
        "Q2": f"TRAIN session-within-subject variance reduction = {fsum['session_variance_reduction']:.6f}; see decomposition for class/residual tradeoff.",
        "Q3": f"Class-relation cosine gain = {relation_gain:.9f}; session-constant subtraction cannot alter this relation. Same-class centroid cosine gain = {same_class_gain:.6f}.",
        "Q4": f"TRAIN class probe BA change = {probe_change:.6f}; relative/absolute class variance ratio = {class_ratio:.6f}; Gate E {'passes' if gate_e else 'fails'}.",
        "Q5": f"OUTER subject-equal REL-ABS BA = {delta['mean']:+.6f}, 95% subject CI [{delta['ci95_low']:.6f},{delta['ci95_high']:.6f}].",
        "Q6": f"REL-global-center BA contrast = {rel_global['mean']:+.6f}; feature-standardization BA = {std_ba:.6f}.",
        "Q7": f"Real minus wrong-reference mean BA = {rel_wrong['mean']:+.6f}; p95 fold controls and 200 draws/fold retained.",
        "Q8": f"Current REL minus S1-reference BA = {np.mean([r['REL_MEAN_BA']-r['S1_reference_for_S2_BA'] for r in folds]):+.6f}.",
        "Q9": f"Context class fraction versus subject gain correlation = {corr:+.4f}; matched label-assisted diagnostic gain = {balance_gain:+.6f}; labels never form deployed references.",
        "Q10": f"Secondary largest descriptive OUTER stage gain: {stage_max} ({stage_gains[stage_max]:+.6f} BA); EMBEDDING remains predeclared primary and no OUTER stage selects the claim.",
        "Q11": f"TRAIN session probe BA change = {np.mean([r['relative_session_probe_BA']-r['absolute_session_probe_BA'] for r in folds]):+.6f}; class probe change = {probe_change:+.6f}.",
        "Q12": f"{case}; exact next action {action}. Gate A is algebraically unattainable under REL_MEAN."
    }
    summary = {"schema": "CONTEXT_RELATIVE_DECISION_V1", "gates": gates,
               "primary_interpretation": case, "next_action": action,
               "folds": fsum, "answers": answers, "formal_final_heldout_eeg_reads": 0,
               "outer_historically_exposed": True, "outer_selection": False,
               "class_relation_translation_invariant": True,
               "context_class_fraction_gain_correlation": corr,
               "label_assisted_matched_gain": balance_gain}
    OUT.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        core.write_csv_new(OUT / f"{name}.csv", rows)
    core.save_json_new(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {
        "formal_final_heldout_eeg_reads": 0, "allowed_roles": ["TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"],
        "checkpoint_validation": "inventory only; no EEG read",
        "outer_development": "historically exposed, evaluation-only after all TRAIN locks",
        "all_train_locks_sha256": {f"fold{f}": core.sha(core.ROOT / "train_lock" / f"fold{f}.json") for f in range(5)}})
    core.save_json_new(OUT / "DECISION_SUMMARY.json", summary)
    with (OUT / "FINAL_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report(summary, folds, boots, tables))
    proto = core.ROOT / "final_protocol"
    proto.mkdir(parents=True, exist_ok=False)
    source_dir = core.ROOT / "source"
    core.save_json_new(proto / "SOURCE_PROVENANCE.json", {
        "schema": "CONTEXT_RELATIVE_SOURCE_PROVENANCE_V1", "folds": provenance_by_fold,
        "protocol_lock_sha256": core.sha(source_dir / "protocol/PROTOCOL_LOCK.json"),
        "protocol_amendment_sha256": core.sha(source_dir / "protocol/PROTOCOL_AMENDMENT_V2.json"),
        "analysis_code_sha256": {p.name: core.sha(p) for p in sorted((source_dir / "code").glob("*.py"))},
        "formal_final_heldout_eeg_reads": 0})
    core.save_json_new(proto / "REPRESENTATION_STAGE_AUDIT.json", {
        "schema": "CONTEXT_RELATIVE_REPRESENTATION_STAGES_V1", "primary_stage": "EMBEDDING",
        "canonical_stage_names": core.LOCK["stages"], "stage_by_fold_role": {
            f: {role: p["stage_audit"] for role, p in roles.items()} for f, roles in provenance_by_fold.items()},
        "stage_feature_sha256_by_fold_role": {
            f: {role: {name: detail["feature_sha256"] for name, detail in p["stage_files"].items()}
                for role, p in roles.items()} for f, roles in provenance_by_fold.items()},
        "temporal_pool_disclosure": "canonical temporal_bn boundary, deterministic adaptive time average to 16 bins before flatten; secondary stage only",
        "extraction_hook": {"TEMPORAL": "EEGNet.bn1(temporal) output in exact manual forward",
                            "SPATIAL": "EEGNet.drop1(pool1(elu(bn2(spatial)))) output in exact manual forward",
                            "SHARED": "EEGNet.drop2(pool2(elu(bn3(point(depth))))) output in exact manual forward",
                            "EMBEDDING": "EEGNet.embedding output, exact final affine head input"},
        "native_forward_check": "every batch manual-stage logits torch.allclose to model(batch), rtol=1e-5 atol=1e-6",
        "all_other_stage_operations": "flatten canonical tensor; EMBEDDING exact native head input",
        "formal_final_heldout_eeg_reads": 0})
    print(json.dumps({"decision": case, "action": action, "outputs": len(list(OUT.iterdir())),
                      "summary_sha256": core.sha(OUT / "DECISION_SUMMARY.json")}))


if __name__ == "__main__":
    run()
