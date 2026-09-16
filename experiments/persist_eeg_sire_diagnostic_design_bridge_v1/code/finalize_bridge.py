#!/usr/bin/env python3
"""Render the frozen P3 bridge interpretation without changing any result."""
from __future__ import annotations

import json
import sys

import pandas as pd

from source_audit import EXP, OLD

sys.path.insert(0, str(OLD / "code"))
import run_pswa_bridge as pswa  # noqa: E402


A = EXP / "outputs/subspace_decomposition"
B = EXP / "outputs/prospective_selection"


def effect_line(frame, task, metric):
    row = frame[(frame.task == task) & (frame.metric == metric)].iloc[0]
    factor = 1 if metric.endswith("_pp") else 100
    return f"{factor*row['mean']:+.2f} [{factor*row.CI95_low:+.2f}, {factor*row.CI95_high:+.2f}] pp"


def main():
    if json.loads((A / "COMPLETION.json").read_text())["status"] != "PASS":
        raise RuntimeError("Part A incomplete")
    if json.loads((B / "COMPLETION.json").read_text())["status"] != "PASS":
        raise RuntimeError("Part B incomplete")
    a = pd.read_csv(A / "TASK_SUMMARY.csv")
    e = pd.read_csv(A / "FULL_VS_B1_PAIRED_EFFECTS.csv")
    checkpoint_audit = pd.read_csv(A / "CHECKPOINT_AUDIT.csv")
    basis_hashes = []
    for row in checkpoint_audit.itertuples(index=False):
        frozen = pswa.load_npz(pswa.session_cache_path(row.task, row.model, int(row.fold), int(row.seed)))
        if frozen["metadata"]["checkpoint_sha256"] != row.checkpoint_sha256:
            raise RuntimeError("Part-A canonical-cache checkpoint hash mismatch")
        basis_hashes.append(frozen["metadata"]["basis_sha256"])
    checkpoint_audit["basis_sha256"] = basis_hashes
    pswa.atomic_csv(A / "CHECKPOINT_AUDIT.csv", checkpoint_audit)
    p = pd.read_csv(B / "POLICY_RESULTS.csv")
    d = pd.read_csv(B / "POLICY_PAIRED_EFFECTS.csv")
    freeze = json.loads((B / "SELECTION_FREEZE.json").read_text())
    w = a[a.task == "WBCIC_MI"].set_index("model")
    full = w.loc["OFFICIAL_FINAL_LITEBN_REFERENCE"]
    b1 = w.loc["B1_SAME_SCALE_63"]
    w_effects = e[e.task == "WBCIC_MI"].set_index("metric")
    protected = w_effects.loc["protected_only_WSBA"]
    complement = w_effects.loc["complement_retained_WSBA"]
    if b1.PSWA_pp <= full.PSWA_pp:
        relation = "B1 did not increase PSWA in WBCIC, contrary to the historical bridge table."
    else:
        relation = ("B1 has higher relative PSWA, but its absolute Protected-only WSBA is lower "
                    "and its random-only WSBA is also lower. The higher PSWA therefore must not be described "
                    "as stronger absolute Protected-only prediction.")
    if complement.CI95_high < 0:
        comp_claim = "The B1−Full complement-retained deficit is negative with a paired CI below zero, consistent with a complement-utility explanation. It is still noncausal."
    else:
        comp_claim = "The B1−Full complement-retained point estimate is negative, but its paired CI includes zero; this decomposition does not attribute the decoder gap to complement utility alone."
    a_report = A / "FINAL_SUBSPACE_DECOMPOSITION_REPORT.md"
    a_lines = a_report.read_text(encoding="utf-8").split("\n## Paired biological-subject effects")[0]
    a_lines += "\n## Paired biological-subject effects\n\n| Task | Metric | B1−Full [95% CI], pp |\n|---|---|---:|\n"
    for task in ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"):
        for metric in ("protected_only_WSBA", "complement_retained_WSBA", "intact_probe_WSBA", "full_model_WSBA", "PEEH_pp", "PSWA_pp"):
            a_lines += f"| {task} | {metric} | {effect_line(e, task, metric)} |\n"
    a_lines += f"\n## WBCIC interpretation\n\n{relation} {comp_claim}\n"
    a_report.write_text(a_lines, encoding="utf-8")
    lines = ["# SIRE-EEG P3 diagnostic-design bridge", "",
             "## Provenance and limits", "",
             "No neural network was retrained. Full/B0 is the server's historical frozen Full checkpoint, **not** a training-protocol-matched B0. Historical code evaluated Full on the same outer-development subjects. Thus Part B is a pre-specified development replay, not a strictly prospective independent confirmation.",
             "Part A reused the exact prior seed0 Full/B1 checkpoints, Protected union and 100 equal-rank controls. The omitted train-fitted canonical basis arrays were deterministically reconstructed and checked against the frozen session-coordinate cache. Table-12 PEEH, PSWA and full-model WSBA were replayed before new summaries.",
             "The final-heldout 14 subjects are absent from Part B. The 40 outer-development subjects were partitioned 8 per fold and opened only after SELECTOR_SPEC and SELECTION_FREEZE were written and hashed in this new run.",
             "", "## Part A: WBCIC decomposition", "",
             "All WSBA values are percentages; PEEH/PSWA are percentage-point contrasts.", "",
             "| Model | Protected-only | Complement-retained | Intact probe | Random-only | Random complement | PEEH | PSWA | Full decoder |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
             f"| Historical Full | {100*full.protected_only_WSBA:.2f} | {100*full.complement_retained_WSBA:.2f} | {100*full.intact_probe_WSBA:.2f} | {100*full.random_only_WSBA:.2f} | {100*full.random_complement_WSBA:.2f} | {full.PEEH_pp:.2f} | {full.PSWA_pp:.2f} | {100*full.full_model_WSBA:.2f} |",
             f"| B1 SameScale63 | {100*b1.protected_only_WSBA:.2f} | {100*b1.complement_retained_WSBA:.2f} | {100*b1.intact_probe_WSBA:.2f} | {100*b1.random_only_WSBA:.2f} | {100*b1.random_complement_WSBA:.2f} | {b1.PEEH_pp:.2f} | {b1.PSWA_pp:.2f} | {100*b1.full_model_WSBA:.2f} |",
             "", "Paired B1−Full effects (20,000 biological-subject bootstrap draws):", "",
             "| Metric | Mean [95% CI], pp |", "|---|---:|"]
    for metric in ("protected_only_WSBA", "complement_retained_WSBA", "intact_probe_WSBA", "full_model_WSBA", "PEEH_pp", "PSWA_pp"):
        lines.append(f"| {metric} | {effect_line(e, 'WBCIC_MI', metric)} |")
    lines += ["", relation, comp_claim,
              "The two probe utilities are not additive, and E_P(h) retains active non-P coordinates plus residual rather than a strict raw orthogonal complement.",
              "All four tasks, including contrary directions, are in `subspace_decomposition/TASK_SUMMARY.csv` and paired effects in `FULL_VS_B1_PAIRED_EFFECTS.csv`.",
              "", "## Part B: frozen development decisions and outer-development replay", "",
              "| Fold | Diagnostic choice | Validation-BA choice | Development reason |",
              "|---:|---|---|---|"]
    for r in freeze["choices"]:
        lines.append(f"| {r['fold']} | {r['diagnostic_choice']} | {r['validation_BA_choice']} | {r['reason']} |")
    lines += ["", "| Policy | Future BA | Macro-F1 | WSBA |", "|---|---:|---:|---:|"]
    for r in p.itertuples(index=False):
        lines.append(f"| {r.policy} | {100*r.future_BA:.2f}% | {100*r.future_macro_F1:.2f}% | {100*r.WSBA:.2f}% |")
    lines += ["", "Diagnostic-guided paired effects versus implementable comparators (percentage points; conditional subject bootstrap):", "",
              "| Comparator | Δ future BA [95% CI] | Δ Macro-F1 [95% CI] | Δ WSBA [95% CI] |",
              "|---|---:|---:|---:|"]
    for comparator in ("Random expectation", "Validation BA", "Fixed Full"):
        rows = d[d.comparison == f"Diagnostic-guided minus {comparator}"].set_index("metric")
        def fmt(metric):
            x = rows.loc[metric]
            return f"{x.delta_mean_pp:+.2f} [{x.CI95_low_pp:+.2f}, {x.CI95_high_pp:+.2f}]"
        lines.append(f"| {comparator} | {fmt('future_BA')} | {fmt('future_macro_F1')} | {fmt('WSBA')} |")
    same = all(r["diagnostic_choice"] == r["validation_BA_choice"] for r in freeze["choices"])
    random_ba = d[(d.comparison == "Diagnostic-guided minus Random expectation") & (d.metric == "future_BA")].iloc[0]
    if random_ba.CI95_low_pp > 0:
        rand_claim = "The frozen diagnostic policy exceeds the exact uniform-random expectation in future BA under the conditional subject bootstrap."
    else:
        rand_claim = "The frozen diagnostic policy does not establish a positive future-BA advantage over uniform random selection: its paired CI includes or falls below zero."
    lines += ["", "## Claim assessment", "",
              "- Supported: relative Protected diagnostics and full-decoder quality are distinct; the WBCIC decomposition rules out equating higher PSWA with higher absolute Protected-only utility.",
              "- Not established: the complement-retained difference alone explains the full-decoder gap; no causal mediation is inferred.",
              f"- Operational replay: {rand_claim}",
              "- " + ("Diagnostic and validation-BA selectors made identical fold choices; no independent decision increment was demonstrated." if same else "Diagnostic and validation-BA selectors differ in at least one fold; inspect the frozen fold table and paired outer effects without retuning."),
              "- Not established: strictly prospective architectural superiority, because the historical Full arm is non-matched and its outer-development outcomes were previously exposed.",
              "", "## Manuscript-ready interpretation", "",
              "In frozen SIRE-EEG ablations, relative Protected-coordinate advantages did not monotonically track full-decoder robustness. A seed0 decomposition showed that Protected-only and P-removed representation probes can move separately, but did not identify a causal decomposition of decoder performance. We therefore used positive PEEH and PSWA as development-side admissibility constraints and chose among admissible architectures by complement-retained probe robustness. After freezing this rule and five fold-level decisions, we evaluated the choices on the corresponding outer-development subjects. Because the historical Full arm was not protocol-matched and had prior outer-development exposure, these outcomes are a development replay; independent P4 confirmation with the unchanged selector remains necessary.",
              "", "## Artifact index", "",
              "- `protocol/SOURCE_AUDIT.md`, `SOURCE_MANIFEST.json`, `PROSPECTIVE_AUDIT.json`, `SELECTOR_SPEC.json`",
              "- `outputs/subspace_decomposition/`: checkpoint, cell, subject, task, paired-effect, replay and final reports",
              "- `outputs/prospective_selection/`: development diagnostics, frozen choices, outer subject results, policy summaries, paired CIs and fold sensitivity",
              "- `runtime/`: cached embeddings and run logs (server only; not committed)"]
    (EXP / "outputs/FINAL_BRIDGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("BRIDGE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
