"""Post-heldout technical recovery of the locked V1 aggregation.

The only metric computation change relative to run.py is averaging the
within-fold top-three correction-energy shares, since Protected ranks differ
between folds. This phase is explicitly exploratory under the V1 rule for
post-heldout changes. Model predictions and the final evaluation lock remain
immutable.
"""
from run import *


def recovery_lock() -> None:
    primary = check_final_lock()
    path = PROTOCOL / "AGGREGATION_RECOVERY_LOCK.json"
    code = Path(__file__)
    prediction_rows = []
    for task in TASKS:
        for fold in FOLDS:
            directory = cell_dir(task, fold, "heldout")
            rec = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
            actual = sha(directory / "predictions.npz")
            if rec["status"] != "COMPLETE" or actual != rec["predictions_sha256"]:
                raise RuntimeError("locked heldout predictions changed or incomplete")
            prediction_rows.append({"task": task, "fold": fold, "sha256": actual})
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if sha(path) != (PROTOCOL / "AGGREGATION_RECOVERY_LOCK.sha256").read_text().strip():
            raise RuntimeError("recovery lock hash mismatch")
        if recorded["recovery_code_sha256"] != sha(code) or recorded["prediction_cells"] != prediction_rows:
            raise RuntimeError("recovery source or predictions changed")
        return
    value = {"schema": "PC_REFINE_V1_POST_HELDOUT_TECHNICAL_RECOVERY",
             "status": "EXPLORATORY_POST_HELDOUT",
             "reason": "Original aggregate failed on ragged fold-specific Protected rank vectors.",
             "only_metric_formula_change": "top3 correction-energy share is calculated within each fold then averaged across folds",
             "primary_BA_F1_NLL_formulas": "byte-for-byte unchanged from locked run.py aggregate",
             "original_final_eval_lock_sha256": sha(PROTOCOL / "FINAL_EVAL_LOCK.json"),
             "original_code_sha256": primary["code_sha256"],
             "recovery_code_sha256": sha(code), "prediction_cells": prediction_rows}
    json_write(path, value)
    (PROTOCOL / "AGGREGATION_RECOVERY_LOCK.sha256").write_text(sha(path) + "\n", encoding="utf-8")


def aggregate_recovered() -> None:
    check_final_lock()
    subject_rows, summary_rows, contrasts, rescue, mechanism_rows = [], [], [], [], []
    for task in TASKS:
        parts = []
        for fold in FOLDS:
            target = cell_dir(task, fold, "heldout")
            rec = json.loads((target / "COMPLETE.json").read_text(encoding="utf-8"))
            if rec["status"] != "COMPLETE" or sha(target / "predictions.npz") != rec["predictions_sha256"]: raise RuntimeError("heldout cell incomplete")
            parts.append(np.load(target / "predictions.npz", allow_pickle=False))
        sessions = (0,1,2) if task == "WBCIC_MI" else (1,2)
        session_subject = {}
        for session in sessions:
            y = parts[0][f"S{session}_y"]
            subjects = parts[0][f"S{session}_subjects"].astype(str)
            for p in parts[1:]:
                if not np.array_equal(y, p[f"S{session}_y"]) or not np.array_equal(subjects, p[f"S{session}_subjects"].astype(str)):
                    raise RuntimeError("fold trial order mismatch")
            probabilities = {}
            for variant in VARIANTS:
                z = np.stack([p[f"{variant}_S{session}_z"] for p in parts])
                prob = np.exp(z - z.max(axis=2, keepdims=True)); prob /= prob.sum(axis=2, keepdims=True)
                probabilities[variant] = prob.mean(0)
            for sub in sorted(set(subjects), key=lambda t: int(t.replace("sub-", ""))):
                ix = subjects == sub
                session_subject[sub, session] = {}
                for variant in VARIANTS:
                    prob = probabilities[variant][ix]
                    pred = prob.argmax(1)
                    row = {"task": task, "subject": sub, "session": f"S{session}", "variant": variant,
                           "BA": float(balanced_accuracy_score(y[ix], pred)),
                           "macro_F1": float(f1_score(y[ix], pred, average="macro", zero_division=0)),
                           "NLL": float(-np.log(np.clip(prob[np.arange(len(prob)), y[ix]], 1e-12, 1)).mean())}
                    subject_rows.append(row); session_subject[sub, session][variant] = row
                bp = probabilities["BASELINE"][ix].argmax(1)
                pp = probabilities["PROTECTED_PC_REFINE"][ix].argmax(1)
                mechanism = {"task": task, "subject": sub, "session": f"S{session}"}
                for variant in VARIANTS:
                    for key in ("P_logit_rms", "C_logit_rms", "PC_disagreement"):
                        mechanism[f"{variant}_{key}"] = float(np.mean([p[f"{variant}_S{session}_{key}"][ix].mean() for p in parts]))
                for key in ("correction_norm", "relative_correction"):
                    mechanism[key] = float(np.mean([p[f"PROTECTED_PC_REFINE_S{session}_{key}"][ix].mean() for p in parts]))
                # Fold-specific Protected axes can have different ranks and cannot be averaged componentwise.
                shares = []
                for p in parts:
                    energy = p[f"PROTECTED_PC_REFINE_S{session}_delta_sq"][ix].mean(0)
                    shares.append(float(np.sort(energy)[-min(3,len(energy)):].sum() / max(float(energy.sum()), 1e-12)))
                mechanism["top3_correction_energy_share"] = float(np.mean(shares))
                mechanism_rows.append(mechanism)
                rescue.append({**mechanism, "baseline_errors_rescued": int(((bp != y[ix]) & (pp == y[ix])).sum()),
                               "baseline_correct_damaged": int(((bp == y[ix]) & (pp != y[ix])).sum()), "trials": int(ix.sum())})
        primary = 2
        subjects = sorted({s for s,_ in session_subject}, key=lambda t: int(t.replace("sub-", "")))
        if len(subjects) != (10 if task == "WBCIC_MI" else 14): raise RuntimeError("heldout subject count mismatch")
        vec = {}
        for variant in VARIANTS:
            for metric in ("BA", "macro_F1"):
                vec[variant, metric] = np.asarray([session_subject[s, primary][variant][metric] for s in subjects])
            worst = np.asarray([min(session_subject[s, ses][variant]["BA"] for ses in sessions) for s in subjects])
            vec[variant, "worst_session_BA"] = worst
            summary_rows.append({"task": task, "variant": variant, "subjects": len(subjects), "primary_session": f"S{primary}",
                                 "BA": float(vec[variant, "BA"].mean()), "macro_F1": float(vec[variant, "macro_F1"].mean()),
                                 "worst_session_BA": float(worst.mean()),
                                 "NLL": float(np.mean([session_subject[s, primary][variant]["NLL"] for s in subjects]))})
        for comparator in VARIANTS[:-1]:
            for metric in ("BA", "macro_F1", "worst_session_BA"):
                d, lo, hi = paired_ci(vec["PROTECTED_PC_REFINE", metric], vec[comparator, metric], (task, comparator, metric))
                contrasts.append({"task": task, "contrast": f"PROTECTED_PC_REFINE - {comparator}", "metric": metric,
                                  "difference": d, "CI95_low": lo, "CI95_high": hi, "bootstrap_draws": 20000, "unit": "biological subject"})
        for p in parts: p.close()
    csv_write(OUTPUT / "HELDOUT_SUBJECT_RESULTS.csv", subject_rows)
    session_rows = []
    for task in TASKS:
        for session in ((0,1,2) if task == "WBCIC_MI" else (1,2)):
            for variant in VARIANTS:
                group = [r for r in subject_rows if r["task"] == task and r["session"] == f"S{session}" and r["variant"] == variant]
                expected = 10 if task == "WBCIC_MI" else 14
                if len(group) != expected: raise RuntimeError("session summary subject count mismatch")
                session_rows.append({"task": task, "session": f"S{session}", "variant": variant,
                                     "subjects": expected, **{metric: float(np.mean([r[metric] for r in group]))
                                                            for metric in ("BA", "macro_F1", "NLL")}})
    csv_write(OUTPUT / "HELDOUT_SESSION_SUMMARY.csv", session_rows)
    csv_write(OUTPUT / "HELDOUT_MODEL_TASK_SUMMARY.csv", summary_rows)
    csv_write(OUTPUT / "PAIRED_HELDOUT_CONTRASTS.csv", contrasts)
    csv_write(OUTPUT / "HELDOUT_RESCUE_HARM.csv", rescue)
    csv_write(OUTPUT / "HELDOUT_MECHANISM_AUDIT.csv", mechanism_rows)
    summary = {(r["task"],r["variant"]):r for r in summary_rows}
    lines = ["# PC-Refine EEGNet V1 final report", "", "Seed 0; five-fold probability mean; formal final-heldout subjects; primary future session.", "",
             "| Task | Baseline BA | Protected-PC BA | Δ BA vs Baseline |",
             "| --- | ---: | ---: | ---: |"]
    for task in TASKS:
        b,v = [summary[task, name]["BA"] for name in VARIANTS]
        lines.append(f"| {task} | {b:.4f} | {v:.4f} | {v-b:+.4f} |")
    lines += ["", "## Paired heldout differences", "",
              "Biological-subject bootstrap, 20,000 draws. Units are absolute metric points.", "",
              "| Task | Metric | Difference | 95% CI |", "| --- | --- | ---: | ---: |"]
    for row in contrasts:
        lines.append(f"| {row['task']} | {row['metric']} | {row['difference']:+.4f} | [{row['CI95_low']:+.4f}, {row['CI95_high']:+.4f}] |")
    lines += ["", "## Physical-session subject-equal BA", "",
              "OpenBMI physical S1/S2; WBCIC physical S0/S1/S2 corresponds to paper S1/S2/S3.", "",
              "| Task | Session | Baseline BA | Protected-PC BA |", "| --- | --- | ---: | ---: |"]
    by_session = {(r["task"], r["session"], r["variant"]): r for r in session_rows}
    for task in TASKS:
        for session in ((0,1,2) if task == "WBCIC_MI" else (1,2)):
            label = f"S{session}"
            b = by_session[task, label, "BASELINE"]["BA"]
            p = by_session[task, label, "PROTECTED_PC_REFINE"]["BA"]
            lines.append(f"| {task} | {label} | {b:.4f} | {p:.4f} |")
    gains = [summary[task, "PROTECTED_PC_REFINE"]["BA"] - summary[task, "BASELINE"]["BA"] for task in TASKS]
    lines += ["", "## Interpretation", "", f"Protected-PC has a positive heldout BA difference on {sum(x > 0 for x in gains)}/4 tasks.",
              "The user narrowed this run to the modified model and its matched seed-0 baseline. This comparison cannot isolate a mechanism-specific gain from a generic adapter effect.",
              "Random-PC is a separate exploratory follow-up only for tasks with a positive primary heldout BA difference."]
    (OUTPUT / "FINAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")



def main() -> None:
    recovery_lock()
    aggregate_recovered()
    report = OUTPUT / "FINAL_REPORT.md"
    content = report.read_text(encoding="utf-8")
    marker = "# Post-heldout technical recovery"
    if not content.startswith(marker):
        prefix = (marker + "\n\nThe locked V1 aggregation failed because fold-specific Protected ranks differ. "
                  "This deterministic recovery changes only the post-hoc top-three correction-energy summary. "
                  "The primary BA/F1/NLL calculations, locked predictions, and checkpoints are unchanged. "
                  "Per the protocol, results produced by this post-heldout code repair are labeled exploratory; "
                  "they do not overwrite a confirmatory V1 result.\n\n")
        report.write_text(prefix + content, encoding="utf-8")
    json_write(OUTPUT / "AGGREGATION_RECOVERY_OUTPUT_HASHES.json",
               {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name not in
                ("AGGREGATION_RECOVERY_OUTPUT_HASHES.json",)})


if __name__ == "__main__": main()
