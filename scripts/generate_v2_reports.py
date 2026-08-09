#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
REPO = ROOT / "repo"
BASE = ROOT / "outputs/v2_joint_certified"
REPORTS = BASE / "reports"


def metric_table(frame: pd.DataFrame, key: str, metrics: list[str]) -> pd.DataFrame:
    current = frame[frame.metric.isin(metrics)].groupby(["dataset", "alpha", key, "metric"]).value.mean().reset_index()
    return current.pivot(index=["dataset", "alpha", key], columns="metric", values="value").reset_index()


def markdown(frame: pd.DataFrame, decimals: int = 4) -> str:
    return frame.to_markdown(index=False, floatfmt=f".{decimals}f")


def get(summary: pd.DataFrame, dataset: str, alpha: float, metric: str) -> float:
    row = summary[(summary.dataset == dataset) & np.isclose(summary.alpha, alpha) & (summary.metric == metric)]
    return float(row.iloc[0]["mean"])


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    nested = pd.read_csv(BASE / "nested_dev/DEV_RESULTS_SUMMARY.csv")
    nested.loc[nested.metric == "certified_positive_adaptation_rate", "metric"] = "tta_selection_rate"
    baselines = pd.read_csv(BASE / "baselines/EXTERNAL_BASELINE_RESULTS.csv")
    predictors = pd.read_csv(BASE / "predictors/PREDICTOR_RESULTS_ALL.csv")
    actions = pd.read_csv(BASE / "actions/ACTION_SAFE_ORACLE_HEADROOM.csv")
    source = pd.read_csv(BASE / "source_models/SOURCE_HEAD_COMPARISON.csv")
    simulations = pd.read_csv(BASE / "theory/SIMULATION_V2_RESULTS.csv")
    main_metrics = ["marginal_violation", "csr", "full_set_fallback", "average_set_size", "argmax_error",
                    "macro_f1", "selected_vs_no_tta_gain", "tta_selection_rate", "selected_tta_ppv",
                    "safe_oracle_gain_captured", "joint_validity"]
    main = nested[nested.metric.isin(main_metrics)].pivot(index=["dataset", "alpha", "policy"], columns="metric", values="mean").reset_index()
    baseline_table = metric_table(baselines, "policy", ["marginal_violation", "csr", "full_set_fallback", "average_set_size",
                                                           "argmax_error", "macro_f1", "selected_vs_no_tta_gain", "tta_selection_rate",
                                                           "safe_beneficial_selection_precision"])
    benefit = predictors[predictors.target == "benefit"].groupby(["dataset", "model"])[
        ["gain_mae", "sign_balanced_accuracy", "spearman"]].mean().reset_index()
    risk = predictors[predictors.target == "risk"].groupby(["dataset", "alpha", "model"])[
        ["mae", "underestimation_rate"]].mean().reset_index()
    sim_validity = float(simulations.joint_simultaneous_validity.mean())
    hmc_headroom = float(actions[actions.dataset == "hmc"].safe_oracle_gain.mean()) if "safe_oracle_gain" in actions else np.nan
    mi_headroom = float(actions[actions.dataset == "eegmmidb"].safe_oracle_gain.mean()) if "safe_oracle_gain" in actions else np.nan
    lines = ["# HSC-TTA v2 full development report", "", "## Verdict", "",
             "The implementation and nested development protocol are complete, but the empirical main-method claim fails. The simultaneous certificate is conservative and valid on development episodes, yet it selects no TTA in either task and captures none of the available Safe-Oracle gain. This is a **NO-GO for an ICLR main-method submission in its current form**.", "",
             "## Proposed-policy main table", "", markdown(main), "",
             "`selected_tta_ppv` is undefined when no TTA is selected. Full-set fallback is reported as fallback, never counted as a nontrivial certificate.", "",
             "## Independent policy baselines", "", markdown(baseline_table), "",
             "The agreement policy is a custom U-only agreement heuristic with policy CRC; it is not represented as official TTALine. Tent/EATA are architecture-incompatible because the selected CBraMod heads use LayerNorm rather than adaptable BatchNorm; no fake official implementation is reported.", "",
             "## Predictor audit", "", "### Benefit", "", markdown(benefit), "", "### Risk", "", markdown(risk), "",
             "## Direct answers", "",
             "1. **Are the repaired source models qualified?** Yes for development use: HMC temporal attention and EEGMMIDB official all-patch downstream heads beat weak/majority behavior across five seeds and EEGMMIDB predicts all four classes. This qualification does not imply the selector succeeds.",
             f"2. **How often are actions truly better than No-TTA?** The action audit records substantial raw beneficial cases (T3A roughly 35–39% depending on task/alpha), while robust-residual benefit is much rarer (roughly 7–13%). Exact per-action values are in `ACTION_WIN_RATE.csv`.",
             f"3. **How large is Safe-Oracle headroom?** Mean development Safe-Oracle gain is approximately {hmc_headroom:.4f} for HMC and {mi_headroom:.4f} for EEGMMIDB (dataset aggregation over stored action-audit rows). It exists but is small.",
             "4. **Does the benefit predictor beat simple surrogates?** Not reliably. ElasticNet sometimes improves sign discrimination, but constant-zero often has equal or lower gain MAE. The positive-gain lower bound therefore remains non-positive for every selected candidate.",
             "5. **Is the risk predictor accurate enough?** It has usable ranking/MAE for conservative bounds, but calibration inflation is large. Risk CSR is nonzero, particularly at alpha=0.20, while many subjects still require the sentinel full set.",
             f"6. **Does the joint certificate reach nominal validity?** Yes at the predeclared 0.90 level in nested development: EEGMMIDB is 1.000 and HMC is 0.989/0.996 for alpha 0.10/0.20; simulation mean simultaneous validity is {sim_validity:.3f}. This is marginal episode-level evidence, not conditional or per-subject certainty.",
             "7. **Is joint calibration tighter than separate calibration?** No consistent advantage is established. The separate-calibration ablation is often less conservative; that does not give it the proposed simultaneous post-selection theorem.",
             "8. **Does the proposed policy beat No-TTA+CRC, Best-Fixed+CRC, Entropy Gate+CRC, and agreement+CRC?** No. Its argmax predictions equal No-TTA because no TTA is selected; several heuristic/fixed policies trade validity or utility differently, but the proposed method has no positive utility advantage.",
             "9. **Does it capture Safe-Oracle gain?** No: Safe-Oracle gain captured is 0.000 in all four dataset/alpha blocks.",
             "10. **What fraction selects TTA?** 0.000 in nested development for both tasks and both alphas.",
             "11. **What fraction of selected TTA is truly beneficial?** Undefined because the selected-TTA denominator is zero; it must not be reported as 0% or 100% precision.",
             "12. **Does safety rely on full-set fallback?** Materially yes. EEGMMIDB fallback is high at alpha=0.10 and remains substantial at alpha=0.20; HMC also uses fallback. Exact rates are in the main table.",
             "13. **Is calibration size sufficient?** It is sufficient to obtain conservative marginal validity, not sufficient for useful positive-benefit certification. The m=12/14 folds cannot overcome weak benefit prediction; larger m only addresses quantile granularity.",
             "14. **Which dataset still fails and why?** Both fail the adaptive-utility objective. EEGMMIDB additionally suffers higher error and larger fallback; HMC has real action headroom but the benefit certificate does not identify it.",
             "15. **Are the results sufficient to proceed to a new confirmatory dataset?** No for a confirmatory claim. A new dataset should only be acquired after improving U-only benefit predictability or action reliability on untouched development data and re-freezing the method.", "",
             "## Theoretical scope", "",
             "The theorem controls marginal episode-level risk and non-harm after an arbitrary U-only selector using a simultaneous subject score. It does not guarantee conditional validity for the certified subgroup, deterministic safety for every subject, Macro-F1, a non-full prediction set, or existence of a beneficial TTA action.", "",
             "## Reproducibility and taint", "",
             "All method selection used source-fit and v2 nested-development subjects. Old final outcomes were accessed only by the one-time v1 diagnosis before development and, after `V2_METHOD_FREEZE.json`, by separately labeled exploratory replication. Those replications are not confirmatory and cannot be used to revise v2.", "",
             "A post-freeze evaluator audit corrected risk/set metrics to use each frozen certified index instead of the oracle true critical index. No predictor, action, bound, q, selector, or decision changed; hashes before/after and the unchanged decision hash are recorded in `V2_EVALUATION_CORRECTION.json`."]
    (REPORTS / "V2_FULL_DEVELOPMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readiness = ["# V2 ICLR readiness assessment", "", "## Decision: NO-GO", "",
                 "The current result does not support the main empirical claim. This judgment is driven by the outcome, not implementation effort:", "",
                 "- Safe-Oracle headroom exists, so the action library is not the sole blocker.",
                 "- Nested joint validity is conservative, but TTA selection rate and Safe-Oracle gain captured are both zero.",
                 "- Benefit prediction does not reliably outperform the constant-zero baseline.",
                 "- Aggregate classification utility is exactly No-TTA rather than better than it.",
                 "- Safety materially depends on sentinel full-set fallback.", "",
                 "## What would change the decision", "",
                 "A new development cycle—not the tainted final sets—must demonstrate all of: nonzero positive-benefit certification with subject-level uncertainty, stable selected-TTA PPV, captured Safe-Oracle gain with confidence intervals above zero, nontrivial CSR without dominant full-set fallback, and an advantage over fixed/heuristic policy CRC baselines. Only then should the frozen confirmatory protocol be used."]
    (REPORTS / "V2_ICLR_READINESS_ASSESSMENT.md").write_text("\n".join(readiness) + "\n", encoding="utf-8")

    reproduce = ["# Reproduce HSC-TTA v2", "", "Environment: `/root/miniconda3/envs/hsc_gpu`; project root: `/root/autodl-tmp/hsc_tta_eeg`; repository: `/root/autodl-tmp/hsc_tta_eeg/repo`.", "",
                 "```bash", "cd /root/autodl-tmp/hsc_tta_eeg/repo", "bash scripts/run_v2_full_development.sh --resume --device cuda --batch-size 128", "```", "",
                 "Use `--start-stage`, `--stop-after-stage`, `--datasets`, `--seeds`, or `--dry-run` for bounded reruns. Stages write independent logs, JSON states, and SHA-256 manifests under `outputs/v2_joint_certified`. Do not run stage 18 before stage 17 creates the method freeze.", "",
                 "The repository does not contain EEG data, token embeddings, model checkpoints, or large parquet counterfactuals. Those remain in the server data/output roots."]
    (REPORTS / "REPRODUCE_V2.md").write_text("\n".join(reproduce) + "\n", encoding="utf-8")
    shutil.copy2(REPO / "docs/CONFIRMATORY_DATASET_REQUIREMENTS.md", REPORTS / "CONFIRMATORY_DATASET_REQUIREMENTS.md")
    shutil.copy2(BASE / "certifiability/CERTIFIABILITY_REPORT.md", REPORTS / "CERTIFIABILITY_REPORT.md")

    tracked = [path for path in BASE.rglob("*") if path.is_file() and path.stat().st_size < 10_000_000]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "python": platform.python_version(), "platform": platform.platform(),
                  "development_root": str(BASE),
                  "small_artifact_sha256": {str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked},
                  "taint_policy": "old final results are exploratory only after method freeze"}
    (BASE / "provenance/EXPERIMENT_PROVENANCE_V2.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(REPORTS)


if __name__ == "__main__":
    main()
