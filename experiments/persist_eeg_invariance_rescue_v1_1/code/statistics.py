from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EXPERIMENT_ROOT, FIGURES, OUTPUTS, load_config, stable_seed, write_csv, write_json


def _boot(frame: pd.DataFrame, column: str, draws: int = 10000, seed: int = 0) -> dict[str, Any]:
    data = frame[["fold", "seed", "subject_id", column]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[np.isfinite(data[column])]
    if not len(data):
        return {"mean": None, "median": None, "ci95": [None, None], "n_subjects": 0, "draws": draws}
    # Keep the frozen fold -> seed -> subject hierarchy, but materialize each
    # run as a NumPy vector once.  The previous implementation performed two
    # pandas boolean filters inside every bootstrap draw and made finalization
    # needlessly take many minutes without changing the statistic.
    run_values: dict[tuple[Any, Any], np.ndarray] = {
        (fold, run_seed): group[column].to_numpy(dtype=float)
        for (fold, run_seed), group in data.groupby(["fold", "seed"], sort=True)
    }
    fold_runs: dict[Any, list[np.ndarray]] = {}
    for (fold, _), values in run_values.items():
        fold_runs.setdefault(fold, []).append(values)
    rng = np.random.default_rng(seed)
    folds = sorted(fold_runs)
    vals = np.empty(int(draws), dtype=float)
    for d in range(int(draws)):
        fpick = rng.choice(folds, len(folds), replace=True)
        means = []
        for fold in fpick:
            runs = fold_runs[fold]
            spick = rng.integers(0, len(runs), size=len(runs))
            run_means = []
            for run_index in spick:
                values = runs[int(run_index)]
                picks = rng.integers(0, len(values), size=len(values))
                run_means.append(float(values[picks].mean()))
            means.append(float(np.mean(run_means)))
        vals[d] = float(np.mean(means))
    raw = data[column].to_numpy(float)
    subject_means = data.groupby("subject_id")[column].mean()
    return {"mean": float(raw.mean()), "median": float(np.median(raw)), "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))], "n_subjects": int(data.subject_id.nunique()), "n_subject_units": int(data.groupby(["fold", "seed", "subject_id"]).ngroups), "draws": int(draws), "positive_subject_fraction": float(np.mean(subject_means > 0)), "nonnegative_subject_fraction": float(np.mean(subject_means >= 0)), "worst_subject": float(subject_means.min()), "fold_positivity": int(np.sum(data.groupby("fold")[column].mean() > 0)), "seed_positivity": int(np.sum(data.groupby(["fold", "seed"])[column].mean() > 0))}


def _p_less(values: Sequence[float]) -> float | None:
    arr = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(arr): return None
    rng = np.random.default_rng(stable_seed("sign", len(arr), float(arr.mean()))); signs = rng.choice([-1., 1.], size=(100000, len(arr))); return float((np.sum((signs * arr[None, :]).mean(1) <= arr.mean()) + 1) / 100001)


def _holm(values: Mapping[str, float | None]) -> dict[str, float | None]:
    valid = sorted([(k, v) for k, v in values.items() if v is not None], key=lambda x: x[1]); out = {k: None for k in values}; running = 0.
    for i, (key, value) in enumerate(valid): running = max(running, min(1., value * (len(valid) - i))); out[key] = running
    return out


def determine_eligibility(identity: pd.DataFrame, task: pd.DataFrame, functional: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, Any]:
    cfg = load_config(); families = ["A_SUBJECT_GRL_EEGNET", "B_EEG_DG", "C_SCLDGN"]; result = {}
    for family in families:
        idf = identity[identity.family == family]; tf = task[task.family == family]; sf = functional[functional.family == family]; af = assignments[assignments.family == family]; valid_runs = af[af.measurement_valid.astype(bool)] if len(af) else af; id_summary = _boot(idf.rename(columns={"subject_id": "subject_id"}), "delta_ID", int(cfg["bootstrap_draws"]), stable_seed("i1", family)); ba_summary = _boot(tf, "delta_BA_INV", int(cfg["bootstrap_draws"]), stable_seed("i3", family)); spl_summary = _boot(sf, "SPL", int(cfg["bootstrap_draws"]), stable_seed("i2", family)); lp_summary = _boot(sf, "L_P", int(cfg["bootstrap_draws"]), stable_seed("lp", family)); ln_summary = _boot(sf, "L_N", int(cfg["bootstrap_draws"]), stable_seed("ln", family)); i1 = bool(id_summary["mean"] is not None and id_summary["mean"] < 0); i1_cert = bool(id_summary["ci95"][1] is not None and id_summary["ci95"][1] < 0); i2 = bool(len(valid_runs) >= int(cfg.get("protected_assignment_min_runs", 4)) and spl_summary["mean"] is not None and spl_summary["mean"] > 0 and lp_summary["mean"] > 0); i2_cert = bool(i2 and spl_summary["ci95"][0] is not None and spl_summary["ci95"][0] > 0 and lp_summary["ci95"][0] is not None and lp_summary["ci95"][0] > 0); i3 = bool(ba_summary["mean"] is not None and ba_summary["mean"] < 0); i3_cert = bool(ba_summary["ci95"][1] is not None and ba_summary["ci95"][1] < 0); qvar = float(sf["q_variance_train"].dropna().mean()) if "q_variance_train" in sf.columns and sf["q_variance_train"].notna().any() else np.nan; measurement_invalid = bool(len(valid_runs) == 0 or sf.empty or (np.isfinite(qvar) and qvar <= float(cfg["functional"]["q_variance_floor"])))
        if measurement_invalid: status = "MEASUREMENT_INVALID"
        elif not i1: status = "NO_MEASURABLE_INVARIANCE_EFFECT"
        elif not i2: status = "INVARIANCE_WITHOUT_SELECTIVE_PROTECTED_LOSS"
        elif not i3: status = "SELECTIVE_PROTECTED_LOSS_NO_TASK_HARM"
        else: status = "ELIGIBLE_PROTECTED_LOSS"
        # Rescue is a prespecified secondary analysis for either an eligible
        # protected-loss family (STATUS_D) or a STATUS_C family.  The latter
        # is explicitly diagnostic only: it does not turn task preservation
        # into evidence of task-harm rescue.  This field is written here,
        # rather than inferred later by rescue.py, so the gate is auditable in
        # the frozen eligibility artifact and cannot silently suppress a
        # permitted STATUS_C run.
        rescue_allowed = bool(
            (not measurement_invalid)
            and i1
            and i2
            and (i3 or status == "SELECTIVE_PROTECTED_LOSS_NO_TASK_HARM")
        )
        raw_p = {"I1": _p_less(idf["delta_ID"].to_numpy(float)) if len(idf) else None, "I2_LP": _p_less(-sf["L_P"].to_numpy(float)) if len(sf) else None, "I2_SPL": _p_less(-sf["SPL"].to_numpy(float)) if len(sf) else None, "I3": _p_less(tf["delta_BA_INV"].to_numpy(float)) if len(tf) else None}
        result[family] = {"runs": int(len(idf)), "valid_assignment_runs": int(len(valid_runs)), "I1": i1, "I1_certified": i1_cert, "I2": i2, "I2_certified": i2_cert, "I3": i3, "I3_certified": i3_cert, "measurement_invalid": measurement_invalid, "status": status, "eligible": bool(status == "ELIGIBLE_PROTECTED_LOSS"), "rescue_allowed": rescue_allowed, "identity": id_summary, "task_harm": ba_summary, "protected_loss": lp_summary, "matched_nonprotected_loss": ln_summary, "selective_loss": spl_summary, "p_values": {"raw": raw_p}, "outer_test_used": False, "outer_membership_enumerated": False}
    for key in ("I1", "I2_LP", "I2_SPL", "I3"):
        corrected = _holm({family: result[family]["p_values"]["raw"][key] for family in families})
        for family in families:
            result[family]["p_values"].setdefault("holm", {})[key] = corrected[family]
    payload = {"families": result, "outer_test_used": False, "outer_membership_enumerated": False}; write_json(OUTPUTS / "ELIGIBILITY.json", payload); return payload


def _make_figures(identity: pd.DataFrame, functional: pd.DataFrame, rescue: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    retention = functional
    retention_path = OUTPUTS / "FUNCTIONAL_RETENTION.csv"
    if retention_path.exists():
        retention = pd.read_csv(retention_path)
    # 1: identity/task train-side selection and final outcome.
    fig, ax = plt.subplots(figsize=(7, 4)); ax.axhline(0, color="k", lw=.8); final = identity.groupby("family").mean(numeric_only=True); ax.scatter(final.get("T_anchor_ID", pd.Series()), final.get("invariant_ID", pd.Series()), s=70); ax.set_xlabel("T_anchor identity BA"); ax.set_ylabel("Invariant identity BA"); ax.set_title("Final cross-session identity audit"); fig.tight_layout(); fig.savefig(FIGURES / "FIGURE1_GRL_SELECTION_FINAL_IDENTITY.png", dpi=160); plt.close(fig)
    # 2: replica-normalized retention.
    fig, ax = plt.subplots(figsize=(7, 4));
    if len(retention) and {"family", "role", "FR"}.issubset(retention.columns):
        p = retention.groupby(["family", "role"]).FR.mean().unstack(); p.plot.bar(ax=ax); ax.set_ylabel("FR (1-MSE)"); ax.set_title("Functional protected retention");
    fig.tight_layout(); fig.savefig(FIGURES / "FIGURE2_REPLICA_NORMALIZED_RETENTION.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4));
    if len(functional):
        for family, group in functional.groupby("family"): ax.scatter(group.SPL, np.arange(len(group))[:len(group)], label=family, alpha=.6)
    ax.axvline(0, color="k", lw=.8); ax.set_xlabel("SPL = L_P - L_N"); ax.set_title("Selective Protected Loss"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "FIGURE3_SELECTIVE_PROTECTED_LOSS.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4));
    if len(identity): ax.scatter(identity.delta_ID, identity.delta_BA_INV, alpha=.6); ax.axvline(0, color="k", lw=.8); ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Δ identity"); ax.set_ylabel("Δ balanced accuracy"); ax.set_title("Identity/task trade-off"); fig.tight_layout(); fig.savefig(FIGURES / "FIGURE4_IDENTITY_BA_PARETO.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4));
    if len(rescue):
        pivot = rescue.pivot_table(index=["family","fold","seed","subject_id"], columns="rescue_method", values="balanced_accuracy", aggfunc="mean");
        if "R4_PERSIST_PROTECTED_RESIDUAL" in pivot and "R3_GENERIC_PERSISTENT_RESIDUAL" in pivot: d = pivot["R4_PERSIST_PROTECTED_RESIDUAL"] - pivot["R3_GENERIC_PERSISTENT_RESIDUAL"]; ax.bar(np.arange(len(d)), d.values)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("PERSIST - generic BA"); ax.set_title("Selective restoration diagnostic"); fig.tight_layout(); fig.savefig(FIGURES / "FIGURE5_PERSIST_GENERIC_DELTA.png", dpi=160); plt.close(fig)


def finalize() -> dict[str, Any]:
    cfg = load_config()
    # IDENTITY_AUDIT is intentionally a run-level table (one row per
    # family/fold/seed).  The primary I1 inference is subject-level, so use
    # the paired subject audit for eligibility/bootstrap while retaining the
    # run-level table for reporting and figures.
    identity = pd.read_csv(OUTPUTS / "IDENTITY_AUDIT.csv")
    identity_subject = pd.read_csv(OUTPUTS / "SUBJECT_LEVEL_AUDIT.csv") if (OUTPUTS / "SUBJECT_LEVEL_AUDIT.csv").exists() else identity
    task = pd.read_csv(OUTPUTS / "TASK_HARM.csv")
    functional = pd.read_csv(OUTPUTS / "SELECTIVE_PROTECTED_LOSS.csv") if (OUTPUTS / "SELECTIVE_PROTECTED_LOSS.csv").exists() else pd.DataFrame()
    assignments = pd.read_csv(OUTPUTS / "PROTECTED_ASSIGNMENT.csv")
    eligibility = determine_eligibility(identity_subject, task, functional, assignments)
    rescue = pd.read_csv(OUTPUTS / "RESCUE_RESULTS.csv") if (OUTPUTS / "RESCUE_RESULTS.csv").exists() else pd.DataFrame()
    rescue_subject = pd.read_csv(OUTPUTS / "RESCUE_SUBJECT_RESULTS.csv") if (OUTPUTS / "RESCUE_SUBJECT_RESULTS.csv").exists() else pd.DataFrame()
    rescue_rows = []
    if len(rescue_subject):
        for family, group in rescue_subject.groupby("family"):
            for method, values in group.groupby("rescue_method").balanced_accuracy:
                boot = _boot(group[group.rescue_method == method].rename(columns={"subject_id": "subject_id"}), "balanced_accuracy", int(cfg["bootstrap_draws"]), stable_seed("rescue", family, method)); rescue_rows.append({"family": family, "rescue_method": method, **boot, "outer_test_used": False})
    rescue_stats = pd.DataFrame(rescue_rows); write_csv(OUTPUTS / "RESCUE_STATISTICS.csv", rescue_stats)
    allowed = [f for f, x in eligibility["families"].items() if x["I1"] and x["I2"] and (x["I3"] or x["status"] == "SELECTIVE_PROTECTED_LOSS_NO_TASK_HARM")]
    supported = []
    for family in allowed:
        if len(rescue_subject):
            pivot = rescue_subject[rescue_subject.family == family].pivot_table(index=["fold","seed","subject_id"], columns="rescue_method", values="balanced_accuracy", aggfunc="mean")
            if "R4_PERSIST_PROTECTED_RESIDUAL" in pivot and "R3_GENERIC_PERSISTENT_RESIDUAL" in pivot:
                d = pd.DataFrame({"fold": pivot.index.get_level_values(0), "seed": pivot.index.get_level_values(1), "subject_id": pivot.index.get_level_values(2), "delta": pivot["R4_PERSIST_PROTECTED_RESIDUAL"] - pivot["R3_GENERIC_PERSISTENT_RESIDUAL"]}); b = _boot(d, "delta", int(cfg["bootstrap_draws"]), stable_seed("rescue-persist", family));
                if b["ci95"][0] is not None and b["ci95"][0] > 0: supported.append(family)
    if not allowed: terminal = "V1_1_MEASUREMENT_INVALID" if any(x["measurement_invalid"] for x in eligibility["families"].values()) else ("V1_1_NO_MEASURABLE_INVARIANCE_EFFECT" if all(not x["I1"] for x in eligibility["families"].values()) else "V1_1_INVARIANCE_WITHOUT_SELECTIVE_PROTECTED_LOSS")
    elif not supported: terminal = "V1_1_ELIGIBLE_BUT_PERSIST_RESCUE_NOT_SUPPORTED" if all(eligibility["families"][f]["I3"] for f in allowed) else "V1_1_SELECTIVE_PROTECTED_LOSS_NO_TASK_HARM"
    elif len(supported) == 1: terminal = "V1_1_PERSIST_RESCUE_SINGLE_FAMILY_ONLY"
    else: terminal = "V1_1_PERSIST_RESCUE_CROSS_FAMILY_SUPPORTED"
    _make_figures(identity, functional, rescue_subject); ready = bool(any(x["I1"] and x["I2"] for x in eligibility["families"].values()) and (supported or any(x["status"] == "SELECTIVE_PROTECTED_LOSS_NO_TASK_HARM" for x in eligibility["families"].values())))
    decision = {"terminal_state": terminal, "families": eligibility["families"], "eligible_families": allowed, "rescue_supported_families": supported, "READY_TO_DESIGN_EXPERIMENT_2": ready, "outer_test_used": False, "outer_membership_enumerated": False}; write_json(OUTPUTS / "FINAL_DECISION.json", decision); write_csv(OUTPUTS / "SUBJECT_LEVEL_RESULTS.csv", identity_subject)
    report = ["# Scientific report", "", "## Direct answers", "", "V1 old PRS was a cross-model latent-coordinate reconstruction and therefore confounded by rotation/remapping non-identifiability. V1.1 uses an independently trained task-only replica and frozen teacher task-evidence targets.", "", f"Terminal state: `{terminal}`.", "", "Family summaries:"]
    for family, row in eligibility["families"].items(): report.append(f"- {family}: I1={row['I1']} (mean ΔID={row['identity']['mean']}, CI={row['identity']['ci95']}), I2={row['I2']} (mean L_P={row['protected_loss']['mean']}, mean L_N={row['matched_nonprotected_loss']['mean']}, mean SPL={row['selective_loss']['mean']}, CI={row['selective_loss']['ci95']}), I3={row['I3']} (mean ΔBA={row['task_harm']['mean']}, CI={row['task_harm']['ci95']}), status={row['status']}.")
    report += ["", f"Rescue-supported families: {supported or 'none'}.", "", "All reported outputs set outer_test_used=false and outer_membership_enumerated=false.", "", "The result is exploratory because V1.1 was redesigned after observing V1."]
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8"); return decision
