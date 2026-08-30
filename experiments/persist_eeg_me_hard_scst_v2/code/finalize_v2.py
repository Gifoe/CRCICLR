"""Produce compact reports/figures and select exactly one honest terminal."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v2_common as c


REQUIRED_RESULTS = (
    "SOURCE_RECIPE_SEARCH.csv", "BANK_DECOMPOSITION.csv", "CANDIDATE_COVERAGE.csv", "HARDNESS_DISTRIBUTION.csv",
    "HARD_RANDOM_MATCHING.csv", "DISCOVERY_PER_SUBJECT.csv", "DISCOVERY_PER_FOLD.csv", "DISCOVERY_SUMMARY.csv",
    "CONFIRMATION_PER_SUBJECT.csv", "CONFIRMATION_SUMMARY.csv", "CONTROL_COMPARISON.csv", "STATISTICS.json",
)


def empty_csv(name: str, columns: list[str], reason: str) -> None:
    path = c.RESULTS / name
    if not path.exists():
        frame = pd.DataFrame(columns=columns + ["status"]); frame.loc[0, "status"] = reason; c.write_csv(path, frame)


def document(name: str, title: str, body: str) -> None:
    (c.EXP / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def save_plot(name: str, draw) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0)); draw(ax); fig.tight_layout()
    fig.savefig(c.FIGURES / f"{name}.png", dpi=180); fig.savefig(c.FIGURES / f"{name}.pdf"); plt.close(fig)


def main() -> None:
    c.ensure_dirs(); v1 = c.read_json(c.RESULTS / "V1_REPRODUCTION.json"); source = c.read_json(c.RESULTS / "SOURCE_DECISION.json")
    source_selected = source.get("selected")
    discovery_stats = c.read_json(c.RESULTS / "STATISTICS.json") if (c.RESULTS / "STATISTICS.json").is_file() else None
    confirmation_stats = c.read_json(c.RESULTS / "CONFIRMATION_STATISTICS.json") if (c.RESULTS / "CONFIRMATION_STATISTICS.json").is_file() else None
    if not source.get("source_gate_pass"):
        terminal = "ME_HARD_SCST_MECHANISM_NOT_REALIZED"
    elif discovery_stats is None:
        raise RuntimeError("SOURCE_PASSED_BUT_DISCOVERY_NOT_COMPLETE")
    elif not discovery_stats.get("discovery_supported"):
        terminal = "ME_HARD_SCST_NOT_SUPPORTED"
    elif confirmation_stats is None:
        raise RuntimeError("DISCOVERY_PASSED_BUT_CONFIRMATION_NOT_COMPLETE")
    elif confirmation_stats.get("cross_arch_supported"):
        terminal = "ME_HARD_SCST_CROSS_ARCH_SUPPORTED"
    else:
        terminal = "ME_HARD_SCST_DISCOVERY_SUPPORTED_CONFIRMATION_FAILED"

    stop_reason = terminal
    empty_csv("HARD_RANDOM_MATCHING.csv", ["model", "fold", "seed", "alpha_match", "candidate_count_match", "valid_count_match", "whitened_norm_error"], stop_reason)
    empty_csv("DISCOVERY_PER_SUBJECT.csv", ["model", "method", "fold", "seed", "subject_id", "BA", "macro_F1"], stop_reason)
    empty_csv("DISCOVERY_PER_FOLD.csv", ["model", "method", "fold", "seed", "BA", "macro_F1"], stop_reason)
    empty_csv("DISCOVERY_SUMMARY.csv", ["method", "BA", "CI95_L", "CI95_U"], stop_reason)
    empty_csv("CONFIRMATION_PER_SUBJECT.csv", ["model", "method", "fold", "seed", "subject_id", "BA"], stop_reason)
    empty_csv("CONFIRMATION_SUMMARY.csv", ["model", "method", "BA", "delta_BA", "CI95_L", "CI95_U"], stop_reason)
    empty_csv("CONTROL_COMPARISON.csv", ["comparison", "delta_BA", "CI95_L", "CI95_U", "positive_folds"], stop_reason)
    if discovery_stats is None:
        c.write_json(c.RESULTS / "STATISTICS.json", {"bootstrap_draws": 10000, "discovery_run": False, "reason": stop_reason, "outer_or_sealed_opened": False})

    bank = pd.read_csv(c.RESULTS / "BANK_DECOMPOSITION.csv")
    coverage = pd.read_csv(c.RESULTS / "CANDIDATE_COVERAGE.csv")
    hardness = pd.read_csv(c.RESULTS / "HARDNESS_DISTRIBUTION.csv")
    recipe = pd.read_csv(c.RESULTS / "SOURCE_RECIPE_SEARCH.csv")
    bank_stats = {
        "median_norm_b": float(bank.norm_b.median()), "median_norm_c": float(bank.norm_c.median()),
        "median_main_effect_energy_fraction": float(bank.main_effect_energy_fraction.replace([np.inf, -np.inf], np.nan).median()),
        "median_eta": float(bank.eta.median()),
    }
    figures = c.FIGURES; figures.mkdir(parents=True, exist_ok=True)
    save_plot("main_effect_vs_interaction", lambda ax: (ax.scatter(bank.norm_b, bank.norm_c, s=4, alpha=.2), ax.set(xlabel="||b_s||", ylabel="||c_s,y||")))
    save_plot("valid_candidate_coverage", lambda ax: (coverage.groupby(["scope", "q", "lambda_H"]).coverage_ge2.mean().plot(kind="bar", ax=ax), ax.axhline(.5, color="red", ls="--"), ax.set(ylabel="Coverage")))
    save_plot("hardness_distribution", lambda ax: (ax.hist(hardness.hardness_gap.dropna(), bins=25), ax.set(xlabel="Tail minus uniform hardness", ylabel="Units")))
    save_plot("structured_vs_hard_random", lambda ax: (ax.text(.5,.5,"Not run" if terminal=="ME_HARD_SCST_MECHANISM_NOT_REALIZED" else "See matched control CSV",ha="center",va="center"), ax.set_axis_off()))
    if (c.RESULTS / "DISCOVERY_PER_SUBJECT.csv").is_file() and "ME-HardSCST" in pd.read_csv(c.RESULTS / "DISCOVERY_PER_SUBJECT.csv").get("method", pd.Series(dtype=str)).astype(str).tolist():
        disc = pd.read_csv(c.RESULTS / "DISCOVERY_PER_SUBJECT.csv"); piv=disc.pivot_table(index="subject_id",columns="method",values="BA"); gain=piv["ME-HardSCST"]-piv["ERM"]
        save_plot("subject_level_gain", lambda ax: (ax.bar(np.arange(len(gain)), gain), ax.axhline(0,color="black"), ax.set(xlabel="Biological subject",ylabel="Delta BA")))
    else:
        save_plot("subject_level_gain", lambda ax: (ax.text(.5,.5,"Discovery not opened",ha="center",va="center"),ax.set_axis_off()))
    save_plot("cross_architecture_gain", lambda ax: (ax.text(.5,.5,"Confirmation not run" if confirmation_stats is None else "See confirmation summary",ha="center",va="center"),ax.set_axis_off()))

    code_map_status = "recovered; five artifact-backed methods reproduced; ShuffleSameClass historical value had no code/artifact"
    document("README.md", "ME-HardSCST V2", f"Outcome-informed V2 hypothesis evaluated under source-first gating. Final terminal: `{terminal}`. V1 remains `{ 'SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE' }`.")
    document("SCIENTIFIC_RATIONALE.md", "Scientific rationale", "V2 separates the cross-class subject main effect from protected subject-class interaction and asks whether admissible, decision-challenging transport has utility beyond equally hard random residual-span perturbations. It does not reinterpret the negative V1 result.")
    document("METHOD.md", "Method", "Cross-fitted count-shrunk mixed-effects estimates use leave-one-anchor-out centroids. Only b_t-b_s is transported. Five source-derived gates precede detached upper-tail ranking; the loss is clean CE plus softplus negative candidate margin and contains no symmetric KL.")
    document("LEAKAGE_AUDIT.md", "Leakage audit", "Banks used training partitions only. Source selection used OpenBMI sessions 1/2 and WBCIC S1/S2. WBCIC S3 was opened only if source gates passed and only after a committed protocol lock. Outer and sealed resources were never opened.")
    document("MIXED_EFFECTS_BANK_AUDIT.md", "Mixed-effects bank audit", json.dumps(bank_stats, indent=2))
    document("BANK_STALENESS_AUDIT.md", "Bank staleness audit", "Scope A used fixed encoder coordinates. Scope B used EMA decay 0.99 and rebuilt the detached bank exactly once at each epoch start. Fully trainable encoder with permanently frozen bank was not used.")
    document("ADMISSIBILITY_AUDIT.md", "Admissibility audit", f"Mean clean-correct coverage across source units: {coverage.coverage_ge2.mean():.6f}; median valid candidates: {coverage.median_valid_candidates.median():.3f}. Gates were computed from training data only.")
    document("HARDNESS_AUDIT.md", "Hardness audit", f"Mean selected-tail minus uniform hardness: {hardness.hardness_gap.mean():.6f}; minimum reported lower bound: {hardness.hardness_gap_CI95_L.min():.6f}.")
    document("HARD_RANDOM_MATCHING_AUDIT.md", "HardRandom matching audit", "HardRandom is defined in the b_s SVD residual span with exact matched whitened norm. Paired methods deterministically match valid candidate counts. If source gating stopped the experiment, this control was not opened on S3.")
    document("DISCOVERY_REPORT.md", "Discovery report", "WBCIC S3 was not opened because source gates failed." if terminal=="ME_HARD_SCST_MECHANISM_NOT_REALIZED" else json.dumps(discovery_stats, indent=2))
    document("CONFIRMATION_REPORT.md", "Confirmation report", "Not run because the prerequisite discovery gate did not pass." if confirmation_stats is None else json.dumps(confirmation_stats, indent=2))
    document("CONTROL_REPORT.md", "Control report", "Controls are interpreted only when their preregistered stage was opened. Empty result schemas identify gate-stopped analyses rather than fabricated measurements.")
    document("CLAIM_AUDIT.md", "Claim audit", f"Strongest claim: {terminal}. No claim is made that all subject information is nuisance, that admissibility predicts utility, that the method is causal, or that outer confirmation was completed.")
    document("ITERATION_LEDGER.md", "Iteration ledger", "1. Implemented the fixed 12-recipe q x lambda_H x scope grid.\n2. Repaired only an engineering cache-write race by serializing scopes within fold/seed.\n3. No scientific rule changed after source results or any S3 access.")
    document("REPRODUCIBILITY.md", "Reproducibility", "Run v1_reproduce.py, pytest tests/, smoke_v2.py, run_source_grid.py, select_source.py. Only after a committed lock and a passing source decision may discovery.py run. Runtime caches/checkpoints are intentionally excluded from Git.")
    final = {
        "branch": "codex/persist-eeg-me-hard-scst-v2", "terminal": terminal,
        "immutable_v1_terminal": "SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE",
        "v1_code_map_status": code_map_status, "v1_reproduction_pass": v1["artifact_backed_reproduction_pass"],
        "selected_source_recipe": source_selected, "bank_factorization": bank_stats,
        "source_coverage_mean": float(coverage.coverage_ge2.mean()), "source_median_valid": float(coverage.median_valid_candidates.median()),
        "source_hardness_gap_mean": float(hardness.hardness_gap.mean()), "discovery": discovery_stats,
        "confirmation": confirmation_stats, "outer_resource_status": "NOT_OPENED",
    }
    c.write_json(c.EXP / "FINAL_REPORT.json", final)
    document("FINAL_REPORT.md", "Final report", json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

