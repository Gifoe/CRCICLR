#!/usr/bin/env python3
"""Finalize an early Stage-0 stop without manufacturing later-stage results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs"
decision = json.loads((OUT / "GRADIENT_TRANSFER_DECISION.json").read_text())
summary = pd.read_csv(OUT / "GRADIENT_TRANSFER_SUMMARY.csv").set_index("dataset")

if decision.get("stage1_authorized"):
    raise RuntimeError("This early-stop finalizer is invalid after Stage 1 authorization")

benchmarks = {
    "OpenBMI_MI": (0.7555, "TCFormer"),
    "OpenBMI_ERP": (0.8557, "TCFormer"),
    "OpenBMI_SSVEP": (0.9234, "TCFormer"),
    "WBCIC_MI": (0.7918, "EEGNet"),
}
pd.DataFrame(columns=["task", "seed", "matched_LiteBN_BA", "CM_GA_BA", "delta_pp"]).to_csv(
    OUT / "SEED0_TASK_RESULTS.csv", index=False
)
pd.DataFrame(columns=[
    "task", "fold", "seed", "attempted_updates", "accepted_updates",
    "rejected_updates", "acceptance_rate", "mean_rho", "median_rho",
    "selected_epoch", "epoch0_fallback",
]).to_csv(OUT / "GA_TRAINING_DIAGNOSTICS.csv", index=False)
pd.DataFrame(columns=[
    "task", "fold", "seed", "epoch", "inner_val_BA", "epoch0_BA",
    "paired_bootstrap_ci_lower", "admissible", "selected",
]).to_csv(OUT / "CHECKPOINT_SELECTION_AUDIT.csv", index=False)
(OUT / "SEED0_BENCHMARK_GATE.json").write_text(json.dumps({
    "status": "NOT_RUN_DUE_STAGE0_FAIL",
    "benchmarks": {task: {"threshold": value, "model": model} for task, (value, model) in benchmarks.items()},
    "exposed_benchmark_evaluation_accessed": False,
    "new_sealed_test_accessed": False,
}, indent=2, sort_keys=True) + "\n")

final = {
    **decision,
    "final_model_candidate_status": "FINAL_MODEL_CANDIDATE_FAIL",
    "stage1_run": False,
    "stage2_run": False,
    "benchmark_results_available": False,
}
(OUT / "FINAL_CM_GA_DECISION.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")

questions = [
    ("1. Does the signal transfer for OpenBMI MI?", "No under the locked gate. Primary AUROC and Spearman were positive, but the Spearman advantage over the different-subject control had a confidence interval crossing zero."),
    ("2. Does it transfer for WBCIC MI?", "No under the locked gate. Primary AUROC and Spearman were positive, but the AUROC advantage over control had a confidence interval crossing zero."),
    ("3. Fraction of residual updates rejected?", "Not applicable; Stage 1 was not authorized."),
    ("4. Does WBCIC remain at least as good as matched LiteBN?", "Not evaluated; the exposed benchmark was not opened."),
    ("5. Does OpenBMI MI exceed 75.55%?", "Not evaluated."),
    ("6. Does OpenBMI ERP exceed 85.57%?", "Not evaluated."),
    ("7. Does OpenBMI SSVEP exceed 92.34%?", "Not evaluated."),
    ("8. Does WBCIC MI exceed 79.18%?", "Not evaluated."),
    ("9. Are all thresholds exceeded simultaneously?", "No evidence; Stage 0 stopped the experiment."),
    ("10. Do three-seed means exceed all thresholds?", "Not applicable; Stage 2 did not run."),
    ("11. Epoch-0 fallback frequency?", "Not applicable; no Stage-1 checkpoint selection occurred."),
    ("12. Learned C/M amplitudes?", "No learning occurred. lambda_channel and all three mixer gamma parameters remain exactly zero at the audited identity initialization."),
    ("13. Enough evidence to freeze LiteBN-CM-GA?", "No."),
]
lines = [
    "# LiteBN-CM-GA final report", "",
    f"Terminal: `{decision['terminal']}`.", "",
    "The single candidate failed its mandatory Stage-0 transfer gate. Per protocol, no residual training, benchmark evaluation, or multiseed run was performed.", "",
    "| Dataset | AUROC | 95% CI | Spearman | 95% CI | control AUROC | AUROC advantage 95% CI | pass |",
    "|---|---:|---|---:|---|---:|---|---|",
]
for dataset, row in summary.iterrows():
    lines.append(
        f"| {dataset} | {row.auroc:.4f} | [{row.auroc_ci95_lower:.4f}, {row.auroc_ci95_upper:.4f}] | "
        f"{row.spearman:.4f} | [{row.spearman_ci95_lower:.4f}, {row.spearman_ci95_upper:.4f}] | "
        f"{row.control_auroc:.4f} | [{row.auroc_advantage_ci95_lower:+.4f}, {row.auroc_advantage_ci95_upper:+.4f}] | {bool(row['pass'])} |"
    )
lines += ["", "## Required questions", ""]
for question, answer in questions:
    lines += [f"### {question}", "", answer, ""]
lines += [
    "EXPOSED_BENCHMARK_EVALUATION_ACCESSED = NO", "",
    "NEW_SEALED_TEST_ACCESSED = NO", "",
    "FINAL_MODEL_CANDIDATE_FAIL",
]
(OUT / "FINAL_CM_GA_REPORT.md").write_text("\n".join(lines) + "\n")
print(decision["terminal"])
