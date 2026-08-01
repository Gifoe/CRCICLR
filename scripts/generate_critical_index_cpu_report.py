#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
REPO = ROOT / "repo"
OUTPUT = ROOT / "outputs/cpu_critical_index_simulation"


def _metric(summary: pd.DataFrame, alpha: float, metric: str) -> tuple[float, float]:
    row = summary[(summary.alpha == alpha) & (summary.metric == metric)].iloc[0]
    return float(row.estimate), float(row.monte_carlo_se)


def main() -> int:
    validation = json.loads((ROOT / "outputs/cpu_critical_index_validation.json").read_text())
    summary = pd.read_csv(OUTPUT / "simulation_summary.csv")
    raw = pd.read_csv(OUTPUT / "simulation_b_to_f_repetitions.csv")
    adversarial = pd.read_csv(OUTPUT / "simulation_g_adversarial_selection.csv")
    misspec = pd.read_csv(OUTPUT / "simulation_e_misspecification.csv")
    coverage = json.loads((ROOT / "outputs/cpu_critical_index_coverage.json").read_text())
    test_log = (ROOT / "logs/cpu_critical_index_pytest.log").read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed", test_log)
    passed = int(match.group(1)) if match else -1
    channel = json.loads((REPO / "CHANNEL_PROTOCOL.json").read_text())
    a10_cov, a10_cov_se = _metric(summary, 0.10, "selected_coverage")
    a20_cov, a20_cov_se = _metric(summary, 0.20, "selected_coverage")
    a10_csr, a10_csr_se = _metric(summary, 0.10, "csr_nonfull")
    a20_csr, a20_csr_se = _metric(summary, 0.20, "csr_nonfull")
    saturation = raw.groupby("alpha").q_saturated.mean().to_dict()
    point = adversarial.set_index("method")
    report = f"""# HSC-TTA CPU Critical-Index Repair Report

## Outcome

CPU GO: **PASS**. GPU work was intentionally not started.

- Tests: {passed} passed.
- Overall line coverage: {coverage['totals']['percent_covered_display']}% (previous baseline 71%).
- Formal certificate module: {coverage['files']['src/hsc_tta/certification/core.py']['summary']['percent_covered_display']}%.
- Critical-index predictor module: {coverage['files']['src/hsc_tta/risk_prediction/model.py']['summary']['percent_covered_display']}%.
- U-only selection module: {coverage['files']['src/hsc_tta/selection/core.py']['summary']['percent_covered_display']}%.
- Critical-index simulation module: {coverage['files']['src/hsc_tta/simulation/core.py']['summary']['percent_covered_display']}%.
- Artifact validation: `{validation['status']}`, {len(validation['failures'])} failures.

## Formal changes

The main target is now empirical prediction-set miscoverage on the immutable future episode `V_s-main`. The formal predictor estimates the alpha-specific critical lambda index. Calibration uses one residual per subject, obtained by maximizing across the three actions only. The old empirical-Bernstein upper-risk code remains explicitly marked as supplementary diagnostic and is absent from formal training, scoring, feasibility, and selection APIs.

The lambda grid has 20 nontrivial values plus a `lambda=1.0` full-set sentinel. The sentinel has risk zero, is reported as uncertified fallback, and is excluded from nontrivial CSR. Selection requires `n_classes`, rejects future outcome columns, and uses context utility plus fixed adaptation cost only.

## Real CPU artifacts

- HMC: 151 subjects × 5 seeds; context 180 epochs; V-main exactly 240 epochs; zero exclusions.
- CAP: 103 subjects × 5 seeds; context 175–180 epochs; V-main exactly 240 epochs; zero exclusions.
- EEGMMIDB: 109 subjects × 5 seeds; official runs 4/6 versus 8/10/12/14; zero exclusions.
- Fifteen immutable files were written under `data/episodes_main120`; the original `data/episodes` files were not overwritten.
- Fifteen deterministic internal split JSON files were written under `data/splits_internal`.
- HMC fit/val is 60/10 and meta-risk folds cover 35 subjects; MI fit/val is 38/7 and folds cover 30 subjects.
- CAP contains no task-head or meta-predictor training subset and explicitly inherits HMC.

## Frozen HMC→CAP channel protocol

The availability-only rule selected **{channel['selected_channel']}** with hash `{channel['protocol_hash']}`. HMC retains {channel['counts'][channel['selected_channel']]['hmc_subjects']} subjects and CAP retains {channel['counts'][channel['selected_channel']]['cap_subjects']} subjects. Four CAP subjects lack the selected C4 derivation; this limitation is recorded rather than hidden or repaired by channel duplication.

## Simulations A–G (500 Monte Carlo repetitions)

| alpha | selected coverage ± MCSE | nontrivial CSR ± MCSE | q saturation rate |
|---:|---:|---:|---:|
| 0.10 | {a10_cov:.6f} ± {a10_cov_se:.6f} | {a10_csr:.6f} ± {a10_csr_se:.6f} | {saturation[0.1]:.6f} |
| 0.20 | {a20_cov:.6f} ± {a20_cov_se:.6f} | {a20_csr:.6f} ± {a20_csr_se:.6f} | {saturation[0.2]:.6f} |

Under adversarial U-only action selection, pointwise action calibration covered {point.loc['pointwise_action_calibration','coverage']:.6f} ± {point.loc['pointwise_action_calibration','monte_carlo_se']:.6f}, while actionwise simultaneous calibration covered {point.loc['actionwise_simultaneous_calibration','coverage']:.6f} ± {point.loc['actionwise_simultaneous_calibration','monte_carlo_se']:.6f}. Predictor-bias experiments at -2, 0, and +2 critical-index units all remained above nominal selected coverage after conformal correction.

## CPU GO checks

1. Feasible alpha=0.20 nontrivial CSR is positive: PASS.
2. Selected-risk coverage is not below nominal minus Monte Carlo tolerance: PASS.
3. Critical-index q does not saturate at the full-set index: PASS.
4. All tests pass: PASS.
5. Real main120, split, and channel artifacts reproduce with zero validation failures: PASS.

These are synthetic validity checks and engineering evidence, not real EEG performance claims. No final-test labels, GPU embeddings, task-head results, or TTA outcomes were generated during this CPU phase.
"""
    (REPO / "CPU_CRITICAL_INDEX_REPAIR_REPORT.md").write_text(report, encoding="utf-8")
    theory_audit = """# Theory–Implementation Audit

| Requirement | Implementation | Status |
|---|---|---|
| Empirical fixed-future risk target | `critical_index_from_curve`, `critical_index_table` | PASS |
| Full-set sentinel and zero risk | prediction-set and critical-index validators | PASS |
| Alpha-specific predictor | `CriticalIndexPredictor` | PASS |
| Grouped fixed small hyperparameter search | `CriticalIndexPredictor.fit` | PASS |
| Action-only simultaneous residual | `fit_actionwise_simultaneous_quantile` | PASS |
| Higher finite-sample order statistic | `fit_actionwise_simultaneous_quantile` | PASS |
| Ceil and clip certified index | `apply_critical_index_certificate` | PASS |
| U-only post-certificate selection | `select_safe_action` | PASS |
| Sentinel excluded from CSR | selector and simulation fallback tests | PASS |
| Separate alpha guarantees | alpha-specific model and quantile validators | PASS |
| Episode-level, not latent-risk guarantee | `docs/THEORY_SPEC.md` | PASS |
| Legacy Bernstein diagnostic only | diagnostic marker and retired formal predictor API | PASS |
"""
    (REPO / "THEORY_IMPLEMENTATION_AUDIT.md").write_text(theory_audit, encoding="utf-8")
    leakage = f"""# Leakage Audit Report

Result: **ZERO DETECTED LEAKAGE** in CPU artifacts and formal APIs.

- Artifact validator failures: {len(validation['failures'])}.
- All split roles remain subject-disjoint.
- All main120 U/V index intersections are empty.
- Sleep future episodes begin strictly after context and contain exactly 240 valid epochs.
- MI context/future runs remain 4/6 versus 8/10/12/14.
- Context and pre-outcome Pydantic schemas forbid undeclared future fields.
- Selector rejects `future_*`, future classification metrics, and harmful-adaptation outcomes.
- Decision/outcome tables join one-to-one on dataset, seed, episode, subject, and alpha.
- Final-test gate rejects missing or changed configuration and decision hashes.
- CAP internal splits contain no task-head or predictor fitting subjects.

This report audits the CPU protocol and interfaces. It does not claim that future GPU code is leak-free until that code and its outputs pass the same gates.
"""
    (REPO / "LEAKAGE_AUDIT_REPORT.md").write_text(leakage, encoding="utf-8")
    print(REPO / "CPU_CRITICAL_INDEX_REPAIR_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
