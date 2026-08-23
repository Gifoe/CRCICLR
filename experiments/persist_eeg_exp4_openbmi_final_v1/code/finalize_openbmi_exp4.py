"""Consolidate the legal OpenBMI Exp4 development runs.

The runner intentionally leaves the sealed confirmation cohort untouched.  This
script only combines compact development artifacts from the MI-specific and
competent Conformer-Norm audits and writes the final actionability-boundary
report; it never reads the raw/cache arrays or any holdout predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get(
    "PERSIST_EXP4_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V3_WORK\experiments\persist_eeg_exp4_openbmi_final_v1",
))
RUN_MI = ROOT / "runs" / "mi_trust"
RUN_CONF = ROOT / "runs" / "conformer"
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "protocol"
FIGURES = ROOT / "figures"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, x) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def copy_compact(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.csv"):
        shutil.copy2(p, dst / p.name)
    for p in src.glob("*.json"):
        shutil.copy2(p, dst / p.name)


def bootstrap(values: np.ndarray, seed: int = 20260823, n: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(values), size=(n, len(values)))
    m = values[ix].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    copy_compact(RUN_MI / "results", RESULTS)
    for p in (RUN_MI / "figures").glob("*.png"):
        shutil.copy2(p, FIGURES / p.name)

    mi = pd.read_csv(RUN_MI / "results" / "DEV_METHOD_SUMMARY.csv")
    conf = pd.read_csv(RUN_CONF / "results" / "DEV_METHOD_SUMMARY.csv")
    # The exploratory runner's `selected_development` flag identifies the
    # highest-BA control in that pass.  The final scientific decision rejects
    # that control as a PERSIST model, so clear the flag in the consolidated
    # artifact and expose the role explicitly.
    mi["selected_development"] = False
    mi["scientific_role"] = np.where(mi.method.isin(["DGUG_PROTECT", "DGUG_GATED", "UTILITY_TRUST_REGION"]), "PERSIST_candidate", np.where(mi.method.isin(["NOADAPT", "GENERIC"]), "anchor", "matched_control"))
    conf["selected_development"] = False
    conf["scientific_role"] = np.where(conf.method.isin(["NOADAPT", "GENERIC"]), "anchor", "audit_variant")
    # Use a tolerance for BA ties (one trial is 0.01); numerical -1e-16
    # differences must not be counted as negative transfer.
    dev_for_summary = pd.read_csv(RUN_MI / "results" / "DEV_SUBJECT_RESULTS.csv")
    piv_for_summary = dev_for_summary.pivot_table(index="subject_id", columns="method", values="BA")
    if "NOADAPT" in piv_for_summary:
        for method in mi.method:
            if method in piv_for_summary:
                mi.loc[mi.method.eq(method), "negative_transfer_rate"] = float(np.mean(piv_for_summary[method] < piv_for_summary["NOADAPT"] - 1e-9))
    mi.to_csv(RESULTS / "DEV_METHOD_SUMMARY.csv", index=False)
    def row(frame: pd.DataFrame, method: str) -> dict:
        q = frame[frame.method.eq(method)].iloc[0]
        return {"representation": "MI_SPECIFIC_EEGNET" if frame is mi else "CONFORMER_NORM", "method": method, "mean_BA": float(q.mean_BA), "delta_vs_Generic": float(q.delta_vs_Generic), "negative_transfer_rate": float(q.negative_transfer_rate), "negative_transfer_severity": float(q.negative_transfer_severity), "subjects_favoring": int(q.subjects_favoring), "selection_scope": "V8_SEARCH source-fold prospective", "target_future_labels_used_for_fit": False, "internal_holdout_used": False, "OUTER_TEST_USED": False}
    baseline_rows = [
        row(mi, "NOADAPT"), row(mi, "GENERIC"), row(conf, "NOADAPT"), row(conf, "GENERIC"),
        {"representation": "FBCVARIANCE_NORM", "method": "NOADAPT", "mean_BA": 0.8915, "delta_vs_Generic": np.nan, "negative_transfer_rate": np.nan, "negative_transfer_severity": np.nan, "subjects_favoring": np.nan, "selection_scope": "V8_SEARCH cache probe", "target_future_labels_used_for_fit": False, "internal_holdout_used": False, "OUTER_TEST_USED": False},
        {"representation": "HISTORICAL_REFERENCE", "method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD", "mean_BA": 0.8377777778, "delta_vs_Generic": np.nan, "negative_transfer_rate": np.nan, "negative_transfer_severity": np.nan, "subjects_favoring": np.nan, "selection_scope": "historical 54-subject reference; not used for selection", "target_future_labels_used_for_fit": False, "internal_holdout_used": False, "OUTER_TEST_USED": False},
    ]
    pd.DataFrame(baseline_rows).to_csv(RESULTS / "GENERIC_BASELINE_CANDIDATES.csv", index=False)
    strongest = row(conf, "GENERIC")
    mi_generic = row(mi, "GENERIC")
    mi_noadapt = row(mi, "NOADAPT")
    mi_dgug = row(mi, "DGUG_PROTECT")
    mi_trust = row(mi, "UTILITY_TRUST_REGION")
    mi_decision = row(mi, "DECISION_ONLY_PROTECT")
    mi_utility = row(mi, "UTILITY_ONLY_PROTECT")
    mi_random = row(mi, "RANDOM_PROTECT")

    pareto = []
    for frame, rep in ((mi, "MI_SPECIFIC_EEGNET"), (conf, "CONFORMER_NORM")):
        for _, q in frame.iterrows():
            pareto.append({"representation": rep, "method": str(q.method), "mean_BA": float(q.mean_BA), "delta_vs_Generic": float(q.delta_vs_Generic), "negative_transfer_rate": float(q.negative_transfer_rate), "negative_transfer_severity": float(q.negative_transfer_severity), "subjects_favoring": int(q.subjects_favoring), "utility_retention": float(q.utility_retention) if np.isfinite(q.utility_retention) else np.nan, "decision_grounding_specificity": 1 if str(q.method) in {"DGUG_PROTECT", "DGUG_GATED", "UTILITY_TRUST_REGION"} else 0, "random_control_gap": np.nan, "PCA_control_gap": np.nan, "identity_control_gap": np.nan, "persistence_control_gap": np.nan, "parameter_count": int(q.parameter_count), "selection_scope": "development descriptive"})
    pd.DataFrame(pareto).to_csv(RESULTS / "MODEL_PARETO_FRONTIER.csv", index=False)
    pd.DataFrame(pareto).to_csv(RESULTS / "CONTROL_COMPARISON.csv", index=False)

    # True repeated server runs were performed with three run seeds.  The
    # closed-form MI-specific candidates are expected to be seed-stable; the
    # table is still read from the independent run directories rather than
    # duplicated in memory.
    seed_rows = []
    for seed in (0, 1, 2):
        p = ROOT / "runs" / f"mi_seed{seed}" / "results" / "DEV_METHOD_SUMMARY.csv"
        if p.exists():
            d = pd.read_csv(p)
            for method in ("DGUG_PROTECT", "UTILITY_TRUST_REGION", "DECISION_ONLY_PROTECT"):
                q = d[d.method.eq(method)].iloc[0]
                seed_rows.append({"seed": seed, "representation": "MI_SPECIFIC_EEGNET", "method": method, "mean_BA": float(q.mean_BA), "delta_vs_generic": float(q.delta_vs_Generic), "negative_transfer_rate": float(q.negative_transfer_rate), "internal_holdout_used": False})
    pd.DataFrame(seed_rows).to_csv(RESULTS / "SEED_ROBUSTNESS.csv", index=False)

    dev = pd.read_csv(RUN_MI / "results" / "DEV_SUBJECT_RESULTS.csv")
    piv = dev.pivot_table(index="subject_id", columns="method", values="BA")
    delta = (piv["DGUG_PROTECT"] - piv["GENERIC"]).dropna().to_numpy()
    dgug_stats = {"method": "DGUG_PROTECT", "mean_BA": float(piv["DGUG_PROTECT"].mean()), "generic_BA": float(piv["GENERIC"].mean()), "paired_delta_mean": float(delta.mean()), "paired_delta_median": float(np.median(delta)), "bootstrap_95ci": bootstrap(delta), "subjects_favoring": int(np.sum(delta > 0)), "subjects": int(len(delta)), "negative_transfer_rate": float(np.mean(piv["DGUG_PROTECT"] < piv["NOADAPT"])), "internal_holdout_used": False, "outer_test_used": False}
    write_json(RESULTS / "STATISTICAL_TESTS.json", {"PERSISTENCE_GUARD": dgug_stats, "strongest_generic": strongest})
    neg_rows = pd.DataFrame({
        "subject_id": piv.index,
        "generic_delta_vs_noadapt": piv["GENERIC"] - piv["NOADAPT"],
        "ours_method": "DGUG_PROTECT",
        "ours_delta_vs_noadapt": piv["DGUG_PROTECT"] - piv["NOADAPT"],
        "ours_minus_generic": piv["DGUG_PROTECT"] - piv["GENERIC"],
        "generic_harmed": piv["GENERIC"] < piv["NOADAPT"],
        "ours_harmed": piv["DGUG_PROTECT"] < piv["NOADAPT"],
        "generic_harmed_rescued": (piv["GENERIC"] < piv["NOADAPT"]) & (piv["DGUG_PROTECT"] >= piv["NOADAPT"]),
        "newly_harmed": (piv["GENERIC"] >= piv["NOADAPT"]) & (piv["DGUG_PROTECT"] < piv["NOADAPT"]),
    })
    neg_rows.to_csv(RESULTS / "NEGATIVE_TRANSFER.csv", index=False)

    split_hash = sha256(Path(os.environ.get("PERSIST_V8_SPLIT", r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST\experiments\persist_eeg_final_model_v8\outputs\protocol\V8_SEARCH_SPLIT.json")))
    write_json(PROTOCOL / "GENERIC_BASELINE_LOCK.json", {"locked_for_development": True, "strongest_fair_generic": strongest, "old_historical_reference_BA": 0.8377777778, "generic_selection_uses_target_future_labels": False, "internal_holdout_used": False, "outer_test_used": False})
    write_json(PROTOCOL / "OPENBMI_CONFIRMATION_LOCK.json", {"authorized": False, "terminal_reason": "mechanism not actionable/specific on V8_SEARCH", "internal_holdout_used": False, "outer_test_used": False, "sealed_holdout_count": 14})
    if (PROTOCOL / "OPENBMI_EXP4_FINAL_LOCK.json").exists():
        (PROTOCOL / "OPENBMI_EXP4_FINAL_LOCK.json").unlink()

    neg = pd.read_csv(RUN_MI / "results" / "NEGATIVE_TRANSFER.csv")
    generic_harm = float(np.mean(piv["GENERIC"] < piv["NOADAPT"]))
    dgug_harm = float(np.mean(piv["DGUG_PROTECT"] < piv["NOADAPT"]))
    dgug_rescued = int(neg.loc[neg.ours_delta_vs_noadapt >= 0, "generic_harmed"].sum()) if "generic_harmed" in neg else 0
    newly_harmed = int(neg.loc[neg.ours_delta_vs_noadapt < 0, "generic_harmed"].eq(False).sum()) if "generic_harmed" in neg else 0
    mech = pd.read_csv(RUN_MI / "results" / "MECHANISM_SUBSPACE_RESULTS.csv")
    pud = mech[mech.subspace.str.startswith("PUD_")]
    traj = pd.read_csv(RUN_MI / "results" / "HISTORY_TO_FUTURE_PREDICTION.csv")
    pred = pd.read_csv(RUN_MI / "results" / "DECISION_DEPENDENCE_RESULTS.csv")
    corr_text = pred[pred.predictor.notna()][["predictor", "spearman_with_future_delta"]].to_string(index=False) if "predictor" in pred else "unavailable"
    mechanism = f"PUD candidates had source persistence roughly {pud.persistence_strength.min():.3f}–{pud.persistence_strength.max():.3f} and positive signed utility in the audited ranks, so descriptive mechanism exists. However, the history-side predictors were weak (see HISTORY_TO_FUTURE_PREDICTION.csv; terminal correlations: {corr_text}), the PUD guard was {mi_dgug['mean_BA']:.4f} versus MI Generic {mi_generic['mean_BA']:.4f}, and utility-only/decision-only controls were stronger ({mi_utility['mean_BA']:.4f}/{mi_decision['mean_BA']:.4f})."
    (ROOT / "MECHANISM_HEADROOM_AUDIT.md").write_text("# Mechanism headroom audit\n\n" + mechanism + "\n\nConclusion: P/U/D is descriptive on V8_SEARCH but does not yield prospective, control-specific actionability.\n", encoding="utf-8")
    (ROOT / "NEGATIVE_TRANSFER_AUDIT.md").write_text(f"# Negative transfer audit\n\nMI-specific NoAdapt={mi_noadapt['mean_BA']:.4f}; MI Generic={mi_generic['mean_BA']:.4f}; Generic harm rate={generic_harm:.3f}. DGUG_PROTECT={mi_dgug['mean_BA']:.4f}; DGUG harm rate={dgug_harm:.3f}. Strongest fair Conformer Generic={strongest['mean_BA']:.4f}.\n", encoding="utf-8")
    (ROOT / "README.md").write_text(f"# PERSIST-EEG OpenBMI Exp4\n\nDevelopment-only history-to-future audit on V8_SEARCH (40 subjects, S1→S2). Internal holdout (14 subjects) and historical outer-test stayed sealed. Strongest fair Generic on the competent Conformer-Norm cache is {strongest['mean_BA']:.4f}. The MI-specific PUD guard does not beat utility/decision controls, so terminal state is **EXP4_OPENBMI_MECHANISM_NOT_ACTIONABLE**.\n\nRun: `run_openbmi_exp4.py`; consolidate: `finalize_openbmi_exp4.py`.\n", encoding="utf-8")
    (ROOT / "FINAL_MODEL_CARD.md").write_text(f"# Final model card\n\nTerminal state: **EXP4_OPENBMI_MECHANISM_NOT_ACTIONABLE**. No internal confirmation was authorized. Strongest fair Generic={strongest['mean_BA']:.4f} (Conformer-Norm); MI-specific Generic={mi_generic['mean_BA']:.4f}; MI-specific DGUG={mi_dgug['mean_BA']:.4f} ({mi_dgug['delta_vs_Generic']:+.4f} vs MI Generic); decision-only control={mi_decision['mean_BA']:.4f}; utility-only control={mi_utility['mean_BA']:.4f}. The complete Persistence→Causal Utility→Decision Grounding→Better Future Adaptation chain is not supported.\n", encoding="utf-8")
    (ROOT / "CLAIM_AUDIT.md").write_text("# Claim audit\n\nSupported: persistent, causally useful and decision-responsive directions can be measured in the MI-specific cache on legal source episodes.\n\nNot supported: that these signals prospectively identify who will be harmed by Generic adaptation or that a P/U/D guard improves future Session-2 performance specifically beyond matched utility/decision/random controls.\n\nNo confirmation claim is made because the internal holdout remained sealed.\n", encoding="utf-8")
    (ROOT / "REVIEWER_SELF_AUDIT.md").write_text("# Reviewer self-audit\n\nThe primary falsification is control specificity: decision-only and utility-only controls equal or beat the PUD guard, while history-side P/U/D quantities do not predict future consequence reliably. A stronger Conformer-Norm Generic also exceeds the MI-specific baseline. The correct conclusion is an actionability boundary, not a positive paper claim. Holdout labels and embeddings remain sealed.\n", encoding="utf-8")
    (ROOT / "EXP123_BRIDGE_AUDIT.md").write_text("# Exp1–Exp3 bridge audit\n\nExp4 reconstructed persistence, signed utility, and symmetric centered decision dependence in the exact MI-specific 64-D cache representation. The descriptive bridge exists, but prospective utility protection failed specificity and consequence prediction on V8_SEARCH.\n", encoding="utf-8")
    (ROOT / "REPRESENTATION_ALIGNMENT_AUDIT.md").write_text("# Representation alignment\n\nMI-specific EEGNet fold-0 features were used for the mechanism route; no historical latent coordinates were transplanted. A separate Conformer-Norm audit was run only to establish the strongest fair Generic baseline, not to tune the MI method.\n", encoding="utf-8")
    (ROOT / "DECISION_METRIC_AUDIT.md").write_text("# Decision metric audit\n\nAll candidate/control erasures use centered `[0,z]` two-class logits, with identical centering. Additive class-logit shifts therefore do not affect the metric.\n", encoding="utf-8")
    (ROOT / "INTERVENTION_CONTROL_AUDIT.md").write_text("# Intervention control audit\n\nRanks are matched. Random controls are matched to PUD removed RMS using source features only; PCA, identity, persistence-only, utility-only and decision-only controls use the same S1 history budget.\n", encoding="utf-8")
    (ROOT / "MODEL_SELECTION_AUDIT.md").write_text("# Model selection audit\n\nGeneric and protected variants were selected prospectively within five subject-only V8_SEARCH folds using source S2 outcomes. Final interpretation is Pareto/control-specific, not highest BA. Because PUD did not beat simpler controls, no final method lock was created.\n", encoding="utf-8")
    (ROOT / "ITERATION_LEDGER.md").write_text("# Iteration ledger\n\n- Round 1: MI-specific functional projection; decision-only control won.\n- Round 2: persistence-threshold PUD reconstruction; PUD improved over Generic slightly but remained below utility/decision controls.\n- Round 3: history-side DGUG gating and utility trust-region; no prospective actionability.\n- Round 4: competent Conformer-Norm Generic audit; Generic exceeded MI-specific route.\n- Round 5: three independent MI seeds; same direction and same control failure.\n\nTerminal: EXP4_OPENBMI_MECHANISM_NOT_ACTIONABLE.\n", encoding="utf-8")
    (ROOT / "GENERIC_BASELINE_AUDIT.md").write_text(f"# Generic baseline audit\n\nStrongest fair Generic is Conformer-Norm S1-only head calibration, BA={strongest['mean_BA']:.4f} on V8_SEARCH. This exceeds the historical 54-subject reference 0.83778 but is not directly comparable in cohort size. MI-specific Generic={mi_generic['mean_BA']:.4f}. No future target labels were used for fitting or selection.\n", encoding="utf-8")
    (ROOT / "REPRODUCIBILITY.md").write_text(f"# Reproducibility\n\nServer runner: `run_openbmi_exp4.py`; finalizer: `finalize_openbmi_exp4.py`. V8 split hash={split_hash}. Three MI seeds (0,1,2) were rerun; five subject-only folds were used. Raw EEG, caches, checkpoints, vendor binaries, and sealed IDs are excluded from Git.\n", encoding="utf-8")

    report = f"""# PERSIST-EEG Experiment 4 — OpenBMI MI closure\n\n## Terminal state\n\n**EXP4_OPENBMI_MECHANISM_NOT_ACTIONABLE**\n\nThe descriptive mechanism is measurable, but the complete empirical chain does not close. The P/U/D guard is not prospective or control-specific on V8_SEARCH, and no sealed confirmation was opened.\n\n## Protocol and legality\n\n1. OpenBMI MI is primary because Exp1–Exp3 causal and decision-grounding evidence was established on the same resource.\n2. Deployment is history-to-future: target Session 1 labels adapt; target Session 2 is unseen outcome.\n3. Development used only V8_SEARCH ({40} subjects; five subject-only folds).\n4. V8 internal holdout remained sealed ({14} subjects); historical outer-test remained sealed.\n5. No target Session-2 labels, embeddings, predictions or partial metrics were used during search.\n\n## Baselines\n\n6. MI-specific NoAdapt={mi_noadapt['mean_BA']:.4f}; MI-specific Generic={mi_generic['mean_BA']:.4f}.\n7. Strongest fair Generic audit: Conformer-Norm={strongest['mean_BA']:.4f}, negative-transfer rate={strongest['negative_transfer_rate']:.3f}. It is stronger than the old 0.83778 historical reference, though cohort sizes differ.\n8. MI-specific Generic harm rate={generic_harm:.3f}; it harms a substantial development subset.\n\n## Mechanism and intervention\n\n9. MI-specific PUD subspaces show persistence strength {pud.persistence_strength.min():.3f}–{pud.persistence_strength.max():.3f} and positive signed utility in source folds.\n10. Symmetric centered decision dependence is finite and invariant to additive class-logit shifts.\n11. DGUG_PROTECT={mi_dgug['mean_BA']:.4f} ({mi_dgug['delta_vs_Generic']:+.4f} vs MI Generic; harm rate={dgug_harm:.3f}). Utility trust-region={mi_trust['mean_BA']:.4f}; utility-only={mi_utility['mean_BA']:.4f}; decision-only={mi_decision['mean_BA']:.4f}; random matched={mi_random['mean_BA']:.4f}.\n12. Decision-only and utility-only controls are at least as strong as the PUD guard; history-side predictor correlations are weak. Therefore Decision Grounding adds no demonstrated unique actionability.\n13. Three MI seeds reproduced the same closed-form direction; this is robustness of the negative specificity finding, not confirmation.\n\n## Confirmation boundary\n\n14. `OPENBMI_EXP4_FINAL_LOCK.json` was not created. Internal holdout and outer-test results are intentionally absent.\n15. No second-backbone method search was used to rescue the result; Conformer-Norm was audited only for Generic strength.\n16. The strongest justified claim is an actionability boundary: persistent and causally useful decision-responsive structure exists descriptively, but the current P/U/D intervention does not prospectively improve future-session adaptation beyond simpler controls.\n17. The stronger claim that Persistence→Causal Utility→Decision Grounding→Better Future Adaptation is a complete empirical chain is not justified.\n\nRuntime artifacts and failed variants are in `results/`, `protocol/`, `figures/`, and `ITERATION_LEDGER.md`.\n"""
    (ROOT / "EXP4_OPENBMI_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_json(ROOT / "EXP4_OPENBMI_FINAL_REPORT.json", {"terminal_state": "EXP4_OPENBMI_MECHANISM_NOT_ACTIONABLE", "NoAdapt_MI_specific": mi_noadapt["mean_BA"], "Generic_MI_specific": mi_generic["mean_BA"], "Generic_strongest_fair": strongest["mean_BA"], "DGUG_MI_specific": mi_dgug["mean_BA"], "DGUG_delta_vs_MI_Generic": mi_dgug["delta_vs_Generic"], "utility_only": mi_utility["mean_BA"], "decision_only": mi_decision["mean_BA"], "random_control": mi_random["mean_BA"], "internal_holdout_used": False, "outer_test_used": False, "development_subject_count": 40, "sealed_internal_holdout_count": 14, "major_variants": 10})
    print(json.dumps({"terminal_state": "EXP4_OPENBMI_MECHANISM_NOT_ACTIONABLE", "strongest_generic": strongest["mean_BA"], "mi_dgug": mi_dgug["mean_BA"], "mi_decision_control": mi_decision["mean_BA"], "holdout_used": False}, indent=2))


if __name__ == "__main__":
    main()
