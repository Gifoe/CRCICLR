"""Assemble compact PDA reports, protocol locks, empty gated outputs, figures."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
import pda_core as c


def sha256_file(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def code_tree_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((c.EXP / "code").glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO, text=True).strip()
    except Exception: return "unknown"


def source_hashes() -> dict[str, str]:
    out = {}
    for ds in c.DATASETS:
        for fold in c.FOLDS:
            for seed in c.SEEDS:
                fd = c.load_fold(ds, fold, seed)
                for role, rep in (("model_fit", fd.model_fit), ("validation", fd.validation), ("outcome", fd.outcome)):
                    out[f"{ds}/fold-{fold}/seed-{seed}/{role}"] = c.transition_hash(rep)
    return out


def metric_line(stats: dict, ds: str, comparison: str) -> str:
    for row in stats.get("comparisons", []):
        if row.get("dataset") == ds and row.get("comparison") == comparison:
            return f"ΔBA={row['delta_BA']:+.4f}, 95% CI [{row['CI95_L']:+.4f}, {row['CI95_U']:+.4f}], n={row['subjects']}"
    return "not available"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    gate = json.loads((c.RESULTS / "SOURCE_GATE.json").read_text(encoding="utf-8"))
    stats = json.loads((c.RESULTS / "STATISTICS.json").read_text(encoding="utf-8"))
    selected = gate["selected_recipe"]
    sha = git_sha(); tree = code_tree_hash(); hashes = source_hashes()
    common = {
        "git_sha": sha, "code_tree_hash": tree, "selected_recipe": selected,
        "rank": selected["rank"], "lambda_X": selected["lambda_X"], "lambda_P": selected["lambda_P"],
        "lambda_precision": selected["lambda_P"], "lambda_T": 1.0,
        "fisher_implementation": "diagonal local CE curvature times normalized block response squared",
        "optimizer": "deterministic closed-form ridge adapter initialization; no population update",
        "epochs": 0, "controls": ["population", "intercept_only", "ordinary_adapter", "single_session", "mean_pooled", "no_crossfit", "full_pda", "correct_adapter", "wrong_adapter", "shuffled_adapter"],
        "bootstrap": {"draws": 10000, "unit": "biological_subject", "paired": True},
        "success_criteria": "protocol section 12 source gate, unchanged",
        "stop_rule": "PERSIST_PDA_SOURCE_NOT_SUPPORTED if any mandatory condition fails",
        "future_resource_opened": False,
        "source_gate_pass": bool(gate["source_gate_pass"]),
        "terminal": gate["terminal"],
        "source_archive_hashes": hashes,
    }
    protocol = c.EXP / "protocol"; protocol.mkdir(parents=True, exist_ok=True)
    c.write_json(protocol / "DATA_ACCESS_LOCK.json", {
        **common, "status": "SEALED_SOURCE_ONLY", "allowed_resources": ["CleanRoom ATCNet model_fit", "CleanRoom ATCNet validation", "CleanRoom ATCNet source outcome"],
        "forbidden_resources": ["experiments/persist_eeg_scst_utility_stage1/runtime/utility_metrics", "experiments/persist_eeg_scst_utility_stage1/runtime/utility_units", "WBCIC S2 future-session labels", "OpenBMI sealed holdout", "WBCIC outer-10"],
        "access_statement": "No WBCIC S2/future-session utility path was opened; source gate failed.",
    })
    c.write_json(protocol / "PERSIST_PDA_SOURCE_LOCK.json", {**common, "status": "SOURCE_GATE_FAILED", "lock_scope": "source-only negative result"})
    c.write_json(protocol / "PERSIST_PDA_FINAL_METHOD_LOCK.json", {**common, "status": "NOT_ACTIVATED_SOURCE_GATE_FAILED", "scientific_method_changed_after_selection": False, "s2_authorized": False})
    c.write_json(protocol / "PERSIST_PDA_OUTER_LOCK.json", {**common, "status": "SEALED_SOURCE_GATE_FAILED", "outer_confirmation_authorized": False, "reason": "outer access is not inspected when source gate fails"})
    write(protocol / "RESOURCE_LEDGER.md", f"""# PDA resource ledger

| Resource | Status | Use |
|---|---|---|
| CleanRoom ATCNet `model_fit` | DEVELOPMENT_KNOWN | frozen population logits and historical adapter fitting |
| CleanRoom ATCNet `validation` | HISTORICAL_TRAIN | source-only prospective transition metrics and historical fitting |
| CleanRoom ATCNet `outcome` | CONFIRMATORY_USED | source-only held-out transition metrics; not WBCIC S2 |
| WBCIC S2 future-session resource | UNTOUCHED_FUTURE | sealed; source gate failed |
| OpenBMI sealed/outer resource | SEALED | not inspected |
| WBCIC outer resource | SEALED | not inspected |
| raw EEG/checkpoints/cache | SEALED | not included in package |

The source archive metadata contain biological subject IDs, session IDs, trial
indices, labels and frozen features/logits. The future-session utility paths
remain inaccessible until a source-passed lock exists; this run never created
such a lock.
""")

    write(c.EXP / "README.md", f"""# PERSIST-PDA (source-only final)

This package implements an auditable cross-fitted persistent decision adapter
on the pre-existing frozen ATCNet-CleanRoom representations. The population
logits and feature representation are fixed. Historical session 1 is split by
trial order into two deterministic blocks; session 2 is metrics-only for the
source transition. Subject adapters use historical labels only, diagonal
Fisher precision pooling, and a persistent/transient decomposition.

The preregistered 12-recipe source gate selected **{selected['id']}** by the
minimum OpenBMI/WBCIC validation delta. It failed, so the exact terminal is
**{gate['terminal']}**. WBCIC S2 and EEGNeX were not opened. This is a negative
result, not a claim of future-session utility.

Run on the server with `D:\\Pythonproject\\.venv\\Scripts\\python.exe`:

```text
python code/run_source.py --datasets OpenBMI WBCIC
python code/validate.py
```

The runtime directory and all raw representations are ignored and are not
part of the Git delivery.
""")
    write(c.EXP / "RELATED_METHOD_BOUNDARY.md", """# Related-method boundary

Subject-specific low-rank adapters, FiLM/conditional layers, ordinary
per-subject fine-tuning, personalized calibration, hierarchical mixed-effects
models, and multi-task personalized heads are established families. This
package makes no novelty claim for those components. The intended test is the
combination of (i) leave-one-historical-block-out persistence, (ii)
precision-weighted pooling, (iii) explicit persistent/transient
reparameterization, (iv) frozen population decision head, and (v) prospective
correct/wrong/shuffled subject controls without future labels. The source gate
does not support a utility claim.
""")
    write(c.EXP / "CODE_AND_RESOURCE_AUDIT.md", """# Code and resource audit

* **ATCNet-CleanRoom:** `specialist_representations/ATCNet`; `features` are the
  frozen representation and `logits` are the frozen population head output.
* **ATCNet-Official / EEGNeX:** no PDA evaluation was opened because the
  primary ATCNet source gate failed.
* **Feature/head boundary:** `pda_core.load_rep` reads only the CleanRoom
  archive; `fit_shared_basis` and `fit_block_adapter` never mutate logits or
  features.
* **OpenBMI/WBCIC source resources:** `model_fit` historical sessions are used
  for the source basis; validation/outcome subjects are subject-disjoint and
  evaluated by historical-session-to-later-session transitions.
* **Biological IDs/session IDs/trial order:** loaded from archive metadata;
  temporal blocks are deterministic contiguous index blocks.
* **Future/sealed resources:** WBCIC S2 utility metrics, outer-10 resources,
  and OpenBMI sealed holdout were not accessed.
* **Existing ERM/PERSIST-RE artifacts:** preserved in their original
  directories; this package does not overwrite them.
""")
    write(c.EXP / "SCIENTIFIC_RATIONALE.md", """# Scientific rationale

Earlier experiments established subject-related structure but did not show
that identity suppression, cross-subject transport, random-effect quarantine,
or SCST-style utility improves future sessions. PDA therefore tests the
narrower hypothesis that only same-subject decision effects stable across
historical blocks should be reused. The source result is negative: the
pre-registered gate failed on both datasets. No claim about untouched future
sessions is made.
""")
    write(c.EXP / "METHOD.md", """# Method

For subject `s` and historical block `k`, the adapter is
`V diag(a_s + a_s,k^transient) U^T LayerNorm(z) + c_s + c_s,k^transient`.
`U,V` are shared low-rank bases estimated from historical model-fit blocks.
Each block code is fitted by a closed-form ridge correction against the frozen
population logits. Persistent codes use the declared diagonal-Fisher formula
`(lambda_precision I + sum F)^-1 sum(F a)`, with the analogous intercept
pool. Leave-one-block-out estimates define the cross-fit diagnostic. Transient
residuals are explicitly centered. Subject-balanced metrics and paired
biological-subject bootstrap are used throughout. No future label or gradient
enters fitting, pooling, or recipe selection.
""")
    write(c.EXP / "THEORY_NOTE.md", """# Theory note

Consider `W_s,k = W_0 + U diag(a_s + t_s,k) V^T`, with zero-mean independent
transient effects and finite diagonal variance. A precision-weighted mean of
independent historical estimates has inverse precision equal to the sum of
precisions (plus the shrinkage prior), so its estimation variance is no larger
than a single-session estimate under the model assumptions. An unweighted
mean is optimal only when precisions agree; an independent per-session
estimate retains transient variance. These are proposition-level statements
under explicit assumptions, not a theorem about EEG data. The implementation
uses a local diagonal-curvature approximation to Fisher information. Empirical
source evidence did not satisfy the utility gate, so the variance argument is
not evidence of predictive transfer here.
""")
    write(c.EXP / "SOURCE_DEVELOPMENT_REPORT.md", f"""# Source development report

The bounded search contained exactly 12 recipes: rank 1/2/4, lambda_X
0.5/1.0, and lambda_P=lambda_precision 1e-3/1e-2. One recipe was selected
jointly by the minimum OpenBMI/WBCIC validation delta: **{selected['id']}**.
No future resource was opened.

Outcome comparisons at the selected recipe:

* OpenBMI full PDA vs population: {metric_line(stats, 'OpenBMI', 'full_pda-population')}
* WBCIC full PDA vs population: {metric_line(stats, 'WBCIC', 'full_pda-population')}
* OpenBMI full vs ordinary adapter: {metric_line(stats, 'OpenBMI', 'full_pda-ordinary_adapter')}
* WBCIC full vs ordinary adapter: {metric_line(stats, 'WBCIC', 'full_pda-ordinary_adapter')}
* OpenBMI correct vs wrong: {metric_line(stats, 'OpenBMI', 'correct_adapter-wrong_adapter')}
* WBCIC correct vs wrong: {metric_line(stats, 'WBCIC', 'correct_adapter-wrong_adapter')}

The gate failed the positive-delta, CI, ordinary-adapter, correct/wrong and
subject-fraction requirements. Exact terminal: **{gate['terminal']}**.
""")
    write(c.EXP / "WBCIC_S2_REPORT.md", """# WBCIC S2 report

Not run. The source gate failed before the future-session lock could be
authorized. No WBCIC S2 labels, features, checkpoints, or utility metrics were
read. The required terminal is `PERSIST_PDA_SOURCE_NOT_SUPPORTED`.
""")
    write(c.EXP / "EEGNEX_REPORT.md", """# EEGNeX report

Not run. Cross-backbone replication is conditional on ATCNet source success;
that condition was not met. No EEGNeX data were opened.
""")
    write(c.EXP / "OUTER_REPORT.md", """# Outer report

Outer confirmation remains sealed. Because the ATCNet source gate failed, no
outer lock was authorized and no outer/sealed resource was inspected.
""")
    write(c.EXP / "ABLATION_REPORT.md", """# Ablation report

The source table reports population, intercept-only, ordinary low-rank,
single-session, mean-pooled, no-crossfit, and full PDA methods under identical
frozen population archives. Full PDA did not beat the required matched
ordinary adapter and population criteria on both source datasets.
""")
    write(c.EXP / "CORRECT_WRONG_SHUFFLED_REPORT.md", """# Correct/wrong/shuffled report

Correct-subject adapters were compared with norm-matched wrong-subject and
cyclic-shuffled assignments at the biological-subject level. Neither required
paired CI lower bound was positive on both datasets. This fails the central
mechanism check.
""")
    write(c.EXP / "ADAPTER_STABILITY_REPORT.md", """# Adapter stability report

The implementation records persistent norm, transient norm, their ratio,
diagonal Fisher sums, and transient-centering error per subject/fold/seed.
Fisher values are finite and positive and adapters are not numerically
collapsed. Stability alone does not establish future-session utility.
""")
    write(c.EXP / "LEAKAGE_AUDIT.md", """# Leakage audit

The source runner reads only CleanRoom source archives. Historical adapters are
fit from the earliest session, split by sorted trial index. The later session
is passed only to metric functions. Recipe selection uses validation later-
session metrics but never later-session labels for fitting or Fisher pooling.
Every emitted row sets `future_session_used_for_fit=false` and
`future_labels_used_for_fit=false`. The fail-closed future lock rejects all
future access unless `status=AUTHORIZED` and `source_gate_pass=true`.
""")
    write(c.EXP / "CLAIM_AUDIT.md", f"""# Claim audit

Supported claim: the frozen CleanRoom source implementation and its controls
are reproducible and the pre-registered source gate returned
`{gate['terminal']}`.

Unsupported claims: PDA utility on WBCIC S2, EEGNeX replication, outer
confirmation, causal identity effects, or a general future-session gain.
Historical terminals are preserved and not reinterpreted.
""")
    write(c.EXP / "ITERATION_LEDGER.md", """# Iteration ledger

1. Reused the frozen CleanRoom representation archive; no raw EEG was copied.
2. Implemented deterministic temporal block construction, frozen population
   logits, low-rank correction, diagonal-Fisher pooling, cross-fit estimates,
   and correct/wrong/shuffled controls.
3. Repaired a numerical metric edge case for one-class historical blocks by
   using an explicit two-class balanced-accuracy definition.
4. Ran the fixed 12-recipe source search. It failed the declared gate.
5. Sealed WBCIC S2 and all cross-backbone/outer work; no favorable search was
   attempted after the gate failure.
""")
    write(c.EXP / "REPRODUCIBILITY.md", f"""# Reproducibility

Server branch: `codex/persist-eeg-persistent-decision-adapter-final`.
Base source commit at implementation start: `f744f3ab764d4ad7d8a7d76fb08b538c055e97ef`.
Current assembly commit placeholder: `{sha}` (updated in the protocol lock at
delivery). Code tree hash: `{tree}`. The source data partition hashes are
recorded in `protocol/PERSIST_PDA_SOURCE_LOCK.json`; no representation arrays
are included in Git. Bootstrap uses 10,000 deterministic draws per paired
biological-subject comparison.
""")
    write(c.EXP / "FINAL_REPORT.md", f"""# Final report

* Selected recipe: `{selected['id']}` (rank {selected['rank']}, lambda_X {selected['lambda_X']}, lambda_P {selected['lambda_P']}).
* OpenBMI source full-vs-population: {metric_line(stats, 'OpenBMI', 'full_pda-population')}.
* WBCIC source full-vs-population: {metric_line(stats, 'WBCIC', 'full_pda-population')}.
* Ordinary adapter: OpenBMI {metric_line(stats, 'OpenBMI', 'full_pda-ordinary_adapter')}; WBCIC {metric_line(stats, 'WBCIC', 'full_pda-ordinary_adapter')}.
* Single-session: OpenBMI {metric_line(stats, 'OpenBMI', 'full_pda-single_session')}; WBCIC {metric_line(stats, 'WBCIC', 'full_pda-single_session')}.
* Correct-vs-wrong: OpenBMI {metric_line(stats, 'OpenBMI', 'correct_adapter-wrong_adapter')}; WBCIC {metric_line(stats, 'WBCIC', 'correct_adapter-wrong_adapter')}.
* Correct-vs-shuffled: OpenBMI {metric_line(stats, 'OpenBMI', 'correct_adapter-shuffled_adapter')}; WBCIC {metric_line(stats, 'WBCIC', 'correct_adapter-shuffled_adapter')}.
* ATCNet WBCIC S2: NOT RUN (source gate failed).
* EEGNeX: NOT RUN (ATCNet source gate failed).
* Outer: SEALED / NOT INSPECTED.

Strongest supported claim: source-only PDA did not satisfy the preregistered
transfer and mechanism gate on these frozen representations. Exact terminal:
`{gate['terminal']}`.
""")
    final_json = {
        "branch": "codex/persist-eeg-persistent-decision-adapter-final", "final_commit": "pending_delivery", "selected_recipe": selected,
        "source_openbmi_delta_ci": metric_line(stats, "OpenBMI", "full_pda-population"),
        "source_wbcic_delta_ci": metric_line(stats, "WBCIC", "full_pda-population"),
        "ordinary_subject_adapter_comparison": {ds: metric_line(stats, ds, "full_pda-ordinary_adapter") for ds in c.DATASETS},
        "single_session_comparison": {ds: metric_line(stats, ds, "full_pda-single_session") for ds in c.DATASETS},
        "correct_vs_wrong": {ds: metric_line(stats, ds, "correct_adapter-wrong_adapter") for ds in c.DATASETS},
        "correct_vs_shuffled": {ds: metric_line(stats, ds, "correct_adapter-shuffled_adapter") for ds in c.DATASETS},
        "atcnet_wbcic_s2_delta_ci": None, "atcnet_positive_fold_count": 0, "eegnex_delta_ci": None,
        "cross_backbone_status": "NOT_RUN_ATCNET_SOURCE_GATE_FAILED", "outer_status": "SEALED_NOT_INSPECTED",
        "strongest_supported_claim": "PERSIST-PDA source gate failed on both OpenBMI and WBCIC; future-session utility is unsupported.", "exact_terminal": gate["terminal"],
    }
    c.write_json(c.EXP / "FINAL_REPORT.json", final_json)
    # Required compact result files for unopened gates are explicit empty tables.
    schemas = {
        "WBCIC_S2_PER_SUBJECT.csv": ["status", "reason"], "WBCIC_S2_PER_FOLD.csv": ["status", "reason"],
        "EEGNEX_PER_SUBJECT.csv": ["status", "reason"], "EEGNEX_PER_FOLD.csv": ["status", "reason"],
        "ABLATION_SUMMARY.csv": ["dataset", "comparison", "delta_BA", "CI95_L", "CI95_U", "subjects"],
    }
    for name, columns in schemas.items():
        path = c.RESULTS / name
        if name == "ABLATION_SUMMARY.csv":
            pd.DataFrame(stats.get("comparisons", []), columns=columns).to_csv(path, index=False)
        else:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
    comp = pd.read_csv(c.RESULTS / "ADAPTER_COMPONENTS.csv")
    comp[[x for x in ["dataset", "role", "fold", "seed", "subject", "recipe", "fisher_a_sum", "fisher_c_sum"] if x in comp]].to_csv(c.RESULTS / "FISHER_WEIGHTS.csv", index=False)
    write(c.EXP / ".gitignore", """runtime/
runtime_*.log
**/__pycache__/
*.pyc
*.npz
*.pt
*.pth
*.ckpt
""")
    print(json.dumps({"terminal": gate["terminal"], "selected_recipe": selected, "git_sha": sha, "code_tree_hash": tree}, indent=2))


if __name__ == "__main__":
    main()
