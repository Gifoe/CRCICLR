"""Render the preregistered Q1-Q12 report from compact validated outputs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"


def table(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def m(values): return float(np.mean(list(values)))
def fmt(v): return f"{v:+.4f}" if v is not None else "NA"


def main():
    summary = json.loads((OUT / "DECISION_SUMMARY.json").read_text(encoding="utf-8"))
    effects = table("SUBJECT_LEVEL_EFFECTS.csv")
    b = table("CONTEXT_CLASS_BALANCE_AUDIT.csv")
    sense = table("CONTEXT_SENSITIVITY.csv")
    pairing = table("TRAIN_CONTEXT_PAIRING_NULL.csv")
    folds = summary["fold_table"]
    stat = summary["contrasts"]
    g = summary["gates"]
    arm = lambda name: m(float(x[name+"_BA"]) for x in folds)
    pair_gain = m(float(x["true_minus_null"]) for x in pairing)
    benefit = np.asarray([float(x["interaction_minus_noninteraction"]) for x in effects])
    fraction = np.asarray([float(x["first_B_class_fraction1"]) for x in b])
    imbalance = np.abs(fraction-.5)
    corr = float(np.corrcoef(imbalance, benefit)[0,1]) if np.std(imbalance) and np.std(benefit) else None
    norm = np.asarray([float(x["context_norm"]) for x in b])
    norm_corr = float(np.corrcoef(fraction, norm)[0,1]) if np.std(fraction) and np.std(norm) else None
    available = [float(x["label_assisted_delta"]) for x in b if x["label_assisted_delta"] not in (None, "")]
    positive = sum(benefit > 0); negative = sum(benefit < 0)
    lowrank = arm("LOWRANK_BILINEAR")
    elem = arm("ELEMENTWISE_INTERACTION")
    full = arm("FULL_SIMPLE_INTERACTION")
    q_by_case = defaultdict(dict)
    for row in table("INTERACTION_COMPATIBILITY.csv"):
        q_by_case[(row["fold"], row["subject"], row["session"])][int(row["class"])] = np.asarray(json.loads(row["mean_q"]))
    q_sep = defaultdict(list)
    for (_, _, session), classes in q_by_case.items():
        q_sep[session].append(float(np.linalg.norm(classes[1]-classes[0])))
    ci = stat["interaction_minus_noninteraction"]
    columns = ["fold", "train_subject_count", "outer_subject_count", "B", "TRIAL_ONLY_BA",
               "RELATIVE_SUBTRACTION_BA", "ADDITIVE_CONTEXT_BA", "ADDITIVE_RELATIVE_BA",
               "ELEMENTWISE_INTERACTION_BA", "LOWRANK_BILINEAR_BA", "selected_bilinear_rank",
               "best_noninteraction", "best_noninteraction_BA", "best_interaction", "best_interaction_BA",
               "interaction_minus_noninteraction", "interaction_ablation_BA", "interaction_term_gain",
               "wrong_mean_BA", "wrong_p95_BA", "current_minus_wrong", "historical_BA",
               "current_minus_historical", "random_matched_BA", "h_only_expanded_BA",
               "context_only_BA", "improved_subject_fraction", "harmed_subject_fraction", "fold_interpretation"]
    lines = [
        "# Trial-context interaction qualification — frozen EEGNet / OpenBMI_MI / seed 0",
        "",
        "Primary stage: EMBEDDING (64D); OUTER_DEVELOPMENT was historically exposed and is evaluation-only here. Formal final-heldout EEG reads: **0**. All B, C, K and family choices were frozen using TRAIN_GEOMETRY before new OUTER interaction evaluation. The low-rank interaction uses a supervised TRAIN-only moment-SVD factorization and a regularized linear readout; it is not a jointly optimized neural bilinear layer.",
        "",
        "## Predeclared answers",
        "",
        f"Q1. Additive current context minus trial-only pooled BA: {fmt(summary['additive_minus_trial_only'])}. This is a descriptive OUTER contrast, not a selection criterion.",
        f"Q2. Selected interaction minus trial-only pooled BA: {fmt(m(float(x['interaction_BA'])-float(x['trial_only_BA']) for x in effects))}.",
        f"Q3. Selected interaction minus h-r pooled BA: {fmt(m(float(x['interaction_BA'])-float(next(y['RELATIVE_SUBTRACTION_BA'] for y in folds if str(y['fold'])==x['fold'])) for x in effects))}.",
        f"Q4. Selected interaction minus [h,r,h-r] pooled BA: {fmt(m(float(x['interaction_BA'])-float(next(y['ADDITIVE_RELATIVE_BA'] for y in folds if str(y['fold'])==x['fold'])) for x in effects))}; primary best-noninteraction contrast {fmt(ci['point_BA'])}, subject-cluster 95% CI [{fmt(ci['ci95_low'])}, {fmt(ci['ci95_high'])}].",
        f"Q5. Interaction-term ablation gain: {fmt(stat['interaction_term_gain']['point_BA'])}; Gate B {'PASS' if g['B'] else 'FAIL'}.",
        f"Q6. Versus 100 matched random linear expansions: {fmt(stat['interaction_minus_random']['point_BA'])}; versus H-only expanded: {fmt(stat['interaction_minus_honly']['point_BA'])}; Gate D {'PASS' if g['D'] else 'FAIL'}. Random linear expansion changes L2 geometry but does not add multiplicative coupling.",
        f"Q7. True minus wrong-subject current-context BA: {fmt(stat['true_minus_wrong']['point_BA'])}, subject-cluster 95% CI [{fmt(stat['true_minus_wrong']['ci95_low'])}, {fmt(stat['true_minus_wrong']['ci95_high'])}]; Gate C {'PASS' if g['C'] else 'FAIL'}. Wrong-context mean prediction-flip fraction {m(float(x['wrong_flip_fraction_mean']) for x in sense):.4f}, mean absolute logit change {m(float(x['wrong_mean_abs_logit_change']) for x in sense):.4f}, wrong-minus-true NLL {fmt(m(float(x['wrong_minus_true_NLL']) for x in sense))}. TRAIN true-pairing minus 200 shuffled-pairing null mean: {fmt(pair_gain)}.",
        f"Q8. Current S2 minus historical S1 context BA: {fmt(stat['current_minus_historical']['point_BA'])}; Gate E {'PASS' if g['E'] else 'FAIL'}; strong >=.005: {'YES' if g['E_strong'] else 'NO'}.",
        f"Q9. Absolute first-B class-imbalance vs interaction benefit correlation (descriptive): {fmt(corr)}; first-B class-1 fraction vs raw context norm correlation: {fmt(norm_corr)}. Label-assisted balanced first-B diagnostic mean delta: {fmt(m(available) if available else None)} (available n={len(available)}); this diagnostic is not deployable, and no target label enters the primary reference.",
        f"Q10. Across fold-subject appearances, improved {positive}/{len(effects)}, harmed {negative}/{len(effects)}, tied {len(effects)-positive-negative}; bootstrap clusters repeated appearances by biological subject ID.",
        f"Q11. Low-rank BA {lowrank:.4f} at selected ranks {[x['selected_bilinear_rank'] for x in folds]}, elementwise BA {elem:.4f}, full simple BA {full:.4f}. Mean norm of between-class q separation: current {m(q_sep['current_S2']):.4f}, historical {m(q_sep['historical_S1']):.4f}, wrong-reference mean {m(q_sep['wrong_S2_mean_200']):.4f}; this is a coordinate-level descriptive diagnostic, not predictive or biological evidence. Ranks/regularization were TRAIN-selected, not OUTER-tuned.",
        f"Q12. Build a plug-and-play block? {'YES' if summary['next_action']=='PROCEED_CONTEXT_INTERACTION_BLOCK' else 'NO'}. Exact next action: `{summary['next_action']}`.",
        "",
        "## Decision gates and interpretation",
        "",
        f"Pooled trial-only BA {arm('TRIAL_ONLY'):.4f}; TRAIN-selected best noninteraction BA {m(float(x['best_noninteraction_BA']) for x in folds):.4f}; TRAIN-selected best interaction BA {m(float(x['best_interaction_BA']) for x in folds):.4f}. Primary ΔBA {fmt(ci['point_BA'])} with subject-cluster 95% CI [{fmt(ci['ci95_low'])}, {fmt(ci['ci95_high'])}]. Interaction ablation gain {fmt(stat['interaction_term_gain']['point_BA'])}; wrong-context penalty {fmt(stat['true_minus_wrong']['point_BA'])}; historical-context penalty {fmt(stat['current_minus_historical']['point_BA'])}; matched-capacity random/H-only contrasts {fmt(stat['interaction_minus_random']['point_BA'])}/{fmt(stat['interaction_minus_honly']['point_BA'])}.",
        "",
        f"Case: `{summary['case']}`. Gates A/B/C/D/E: " + "/".join("PASS" if g[k] else "FAIL" for k in "ABCDE") + ". A requires >=.010 BA, >=4/5 positive folds, and lower CI>0. B requires >=.005 BA and >=4/5 positive folds. C requires >=.010 and lower CI>0. D requires positive matched-control contrasts. E requires current >= historical.",
        "",
        "No causal, biological, or architecture-performance claim is made from failed gates. Neither a positive single fold nor a target-label-balanced context is treated as deployable support. This audit does not evaluate formal final-heldout EEG.",
        "",
        "Code and compact artifacts are on `codex/persist-eeg-trial-context-interaction-audit-seed0-v1`; the immutable pushed commit SHA is provided in the delivery response because a Git commit cannot contain its own hash.",
        "",
        "## Final fold table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for f in folds:
        lines.append("| " + " | ".join(str(round(f[c], 4)) if isinstance(f[c], float) else str(f[c]) for c in columns) + " |")
    lines += ["", f"Pooled primary interaction−best-noninteraction BA {fmt(ci['point_BA'])}, 95% CI [{fmt(ci['ci95_low'])}, {fmt(ci['ci95_high'])}]; exact next action `{summary['next_action']}`; formal final-heldout EEG reads 0.", ""]
    with (OUT / "FINAL_REPORT.md").open("x", encoding="utf-8", newline="\n") as f: f.write("\n".join(lines))


if __name__ == "__main__": main()
