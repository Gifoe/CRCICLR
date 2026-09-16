#!/usr/bin/env python3
"""Part B outer-development inference; refuses to run without frozen selector."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from source_audit import EXP, OLD, sha, write
from prepare_and_freeze_selection import CANDIDATES, OUT, TASK, infer, load_model, manifest

sys.path.insert(0, str(OLD / "code"))
import run_peeh_bridge as peeh  # noqa: E402
import run_pswa_bridge as pswa  # noqa: E402

METRICS = ("future_BA", "future_macro_F1", "WSBA")
POLICIES = ("Diagnostic-guided", "Random expectation", "Validation BA", "Fixed Full", "Oracle")


def check_freeze():
    path = OUT / "SELECTION_FREEZE.json"
    if not path.is_file():
        raise RuntimeError("selector not frozen: outer-development inference forbidden")
    freeze = json.loads(path.read_text())
    if freeze["status"] != "FROZEN_BEFORE_OUTER_EVALUATION" or freeze["outer_development_labels_read"]:
        raise RuntimeError("invalid selection freeze")
    for path_string, digest in freeze["selection_input_hashes"].items():
        if sha(Path(path_string)) != digest:
            raise RuntimeError(f"selection input changed after freeze: {path_string}")
    choices = freeze["choices"]
    if len(choices) != 5 or {int(x["fold"]) for x in choices} != set(range(5)):
        raise RuntimeError("incomplete frozen choices")
    return freeze


def infer_outer(runtime, bundle, fold, seed, variant, entry, mean, std, device):
    path = Path(entry["checkpoint_path"])
    if sha(path) != entry["checkpoint_sha256"]:
        raise RuntimeError("checkpoint drift after selector freeze")
    model = load_model(runtime, variant, path, seed, device)
    ids = np.arange(len(bundle.rows), dtype=np.int64)
    _, y, logits = infer(model, bundle, ids, mean, std, device)
    subjects = np.asarray([str(row.subject) for row in bundle.rows])
    sessions = np.asarray([int(row.session) for row in bundle.rows])
    rows = []
    for subject in entry["subject_split"]["outer_dev_subjects"]:
        for session in (1, 2):
            mask = (subjects == subject) & (sessions == session)
            if not mask.any():
                raise RuntimeError(f"missing outer-development subject {subject}/S{session}")
            metric = runtime.base.classification_metrics(y[mask], logits[mask])
            rows.append({"task": TASK, "fold": fold, "seed": seed, "variant": variant,
                         "subject_id": subject, "session": f"S{session}", "trials": int(mask.sum()),
                         "BA": metric["BA"], "macro_F1": metric["macro_F1"],
                         "checkpoint_sha256": entry["checkpoint_sha256"],
                         "normalizer_sha256": entry["normalizer_sha256"]})
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def bootstrap(values, *key):
    arr = np.asarray(values, float)
    rng = np.random.default_rng(peeh.stable_seed("P3-design-outer-bootstrap", *key))
    draws = arr[rng.integers(0, len(arr), (20_000, len(arr)))].mean(1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def policy_results(raw, freeze):
    # Seed-average within the same biological subject and session, before WSBA min.
    session = raw.groupby(["fold", "variant", "subject_id", "session"], as_index=False)[["BA", "macro_F1"]].mean()
    subj = []
    for (fold, variant, subject), g in session.groupby(["fold", "variant", "subject_id"]):
        if set(g.session) != {"S1", "S2"}:
            raise RuntimeError("outer session pair missing")
        s2 = g[g.session == "S2"].iloc[0]
        subj.append({"fold": fold, "variant": variant, "subject_id": subject,
                     "future_BA": s2.BA, "future_macro_F1": s2.macro_F1, "WSBA": g.BA.min()})
    candidate = pd.DataFrame(subj)
    if len(candidate) != 160 or candidate.subject_id.nunique() != 40:
        raise RuntimeError("expected 4 candidate outcomes for 40 unique outer-development subjects")
    fold_choices = {int(x["fold"]): x for x in freeze["choices"]}
    oracle = {}
    for fold, group in candidate.groupby("fold"):
        means = group.groupby("variant").future_BA.mean().to_dict()
        oracle[int(fold)] = sorted(CANDIDATES, key=lambda v: (-means[v], CANDIDATES.index(v)))[0]
    policies = []
    for (fold, subject), group in candidate.groupby(["fold", "subject_id"]):
        by_variant = group.set_index("variant")
        if set(by_variant.index) != set(CANDIDATES):
            raise RuntimeError("candidate outcome missing")
        choices = {"Diagnostic-guided": fold_choices[fold]["diagnostic_choice"],
                   "Validation BA": fold_choices[fold]["validation_BA_choice"],
                   "Fixed Full": "B0_FULL_EXISTING", "Oracle": oracle[fold]}
        for policy in POLICIES:
            if policy == "Random expectation":
                row = {m: float(by_variant[m].mean()) for m in METRICS}
                chosen = "uniform exact expectation"
            else:
                chosen = choices[policy]
                row = {m: float(by_variant.loc[chosen, m]) for m in METRICS}
            policies.append({"fold": fold, "subject_id": subject, "policy": policy,
                             "chosen_variant": chosen, **row})
    return candidate, pd.DataFrame(policies), oracle


def finalize(raw, freeze):
    candidate, subject, oracle = policy_results(raw, freeze)
    pswa.atomic_csv(OUT / "CANDIDATE_SUBJECT_RESULTS.csv", candidate)
    pswa.atomic_csv(OUT / "POLICY_SUBJECT_RESULTS.csv", subject)
    results, paired, sensitivity = [], [], []
    for policy in POLICIES:
        frame = subject[subject.policy == policy].sort_values("subject_id")
        chosen = [x["diagnostic_choice"] if policy == "Diagnostic-guided" else
                  x["validation_BA_choice"] if policy == "Validation BA" else
                  "B0_FULL_EXISTING" if policy == "Fixed Full" else
                  oracle[int(x["fold"])] if policy == "Oracle" else "uniform"
                  for x in freeze["choices"]]
        row = {"policy": policy, "n_biological_subjects": len(frame),
               "chosen_architectures_f0_to_f4": "/".join(chosen),
               "oracle_is_unavailable_prospectively": policy == "Oracle"}
        for metric in METRICS:
            vals = frame[metric].to_numpy(float)
            # Shared bootstrap indices make numerically identical policies
            # (Validation BA, Fixed Full and this run's Oracle) report identical CIs.
            lo, hi = bootstrap(vals, "policy-common", metric)
            row.update({metric: vals.mean(), f"{metric}_CI95_low": lo, f"{metric}_CI95_high": hi})
        results.append(row)
    primary = subject[subject.policy == "Diagnostic-guided"].set_index("subject_id")
    for comparator in ("Random expectation", "Validation BA", "Fixed Full"):
        control = subject[subject.policy == comparator].set_index("subject_id")
        if set(primary.index) != set(control.index):
            raise RuntimeError("paired subjects mismatch")
        for metric in METRICS:
            differences = (primary[metric] - control[metric]).sort_index()
            low, high = bootstrap(differences.to_numpy(float), "paired-common", metric)
            paired.append({"comparison": f"Diagnostic-guided minus {comparator}", "metric": metric,
                           "delta_mean_pp": 100 * differences.mean(), "CI95_low_pp": 100 * low,
                           "CI95_high_pp": 100 * high, "biological_subjects": len(differences),
                           "bootstrap_draws": 20_000})
            fold_delta = pd.DataFrame({"fold": primary.loc[differences.index, "fold"].to_numpy(),
                                       "delta": differences.to_numpy()}).groupby("fold").delta.mean().to_numpy(float)
            flo, fhi = bootstrap(fold_delta, "fold-cluster-common", metric)
            sensitivity.append({"comparison": f"Diagnostic-guided minus {comparator}", "metric": metric,
                                "folds": 5, "fold_cluster_delta_pp": 100 * fold_delta.mean(),
                                "fold_cluster_CI95_low_pp": 100 * flo, "fold_cluster_CI95_high_pp": 100 * fhi,
                                "warning": "only five fold clusters; descriptive sensitivity"})
    pswa.atomic_csv(OUT / "POLICY_RESULTS.csv", results)
    pswa.atomic_csv(OUT / "POLICY_PAIRED_EFFECTS.csv", paired)
    pswa.atomic_csv(OUT / "POLICY_FOLD_SENSITIVITY.csv", sensitivity)
    fold_lines = []
    for choice in freeze["choices"]:
        f = int(choice["fold"])
        d = candidate[(candidate.fold == f) & (candidate.variant == choice["diagnostic_choice"])]
        fold_lines.append(f"| {f} | {choice['diagnostic_choice']} | {choice['validation_BA_choice']} | {choice['reason']} | {100*d.future_BA.mean():.2f} | {100*d.WSBA.mean():.2f} |")
    lines = ["# SIRE-EEG diagnostic-guided architecture selection: development replay", "",
             "This is not a strictly prospective independent confirmation. At the user's direction, B0 is the historical frozen Full checkpoint, which the existing multiseed protocol explicitly marks non-matched. Previous outcome exposure cannot be excluded.",
             "The selector and five fold-level choices were frozen and SHA-verified before this program loaded outer-development data. No neural retraining or final-heldout use occurred.",
             "The 40 outer-development subjects were disjoint across folds. The reported subject bootstrap is conditional on five frozen fold-level choices; a five-fold cluster sensitivity is also supplied.",
             "Random is the exact uniform-policy expectation. Oracle selects by fold-mean outer future BA and is a BA upper bound only; its attached F1/WSBA are not necessarily metric-wise upper bounds. Oracle is not deployable.", "",
             "| Policy | Chosen architectures (fold0–4) | Future BA | Future Macro-F1 | WS-BA |",
             "|---|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['policy']} | {r['chosen_architectures_f0_to_f4']} | {100*r['future_BA']:.2f} | {100*r['future_macro_F1']:.2f} | {100*r['WSBA']:.2f} |")
    lines += ["", "## Fold-level decisions", "",
              "| Fold | Diagnostic choice | Validation-BA choice | Development reason | Outer BA | Outer WS-BA |",
              "|---:|---|---|---|---:|---:|", *fold_lines, "",
              "Paired biological-subject 20,000-draw CIs: `POLICY_PAIRED_EFFECTS.csv`. Fold-cluster sensitivity: `POLICY_FOLD_SENSITIVITY.csv`.",
              "Protected PEEH/PSWA were admissibility constraints; complement-retained probe utility chose among admissible candidates. These diagnostics are not a scalar quality ranking and the outcome does not establish a causal mechanism.",
              "A future P4 independent dataset with this unmodified selector is required for genuinely prospective confirmation."]
    pswa.atomic_text(OUT / "FINAL_PROSPECTIVE_SELECTION_REPORT.md", "\n".join(lines))
    write(OUT / "COMPLETION.json", {"status": "PASS", "candidate_seed_cells": 60,
          "outer_development_biological_subjects": 40, "neural_training_runs": 0,
          "selection_freeze_sha256": sha(OUT / "SELECTION_FREEZE.json"),
          "historical_B0_protocol_matched": False, "classification": "pre-specified development replay",
          "random_policy": "exact uniform expectation", "bootstrap_draws": 20_000})
    print("PART_B_COMPLETE outer_subjects=40", flush=True)


def main():
    freeze = check_freeze()
    entries = manifest()
    runtime = peeh.load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = OUT / "OUTER_SUBJECT_RESULTS.csv"
    rows = pd.read_csv(path).to_dict("records") if path.is_file() else []
    for row in rows:
        row["subject_id"] = str(row["subject_id"])
    done = {(int(x["fold"]), int(x["seed"]), x["variant"]) for x in rows}
    for fold in range(5):
        outer = entries[(fold, 0, CANDIDATES[0])]["subject_split"]["outer_dev_subjects"]
        bundle = runtime.base.build_bundle(TASK, outer)
        mean, std, _ = runtime.base.load_tensor_pair(peeh.normalizer_path(TASK, fold))
        for variant in CANDIDATES:
            for seed in range(3):
                if (fold, seed, variant) in done:
                    continue
                entry = entries[(fold, seed, variant)]
                current = infer_outer(runtime, bundle, fold, seed, variant, entry, mean, std, device)
                rows.extend(current)
                pswa.atomic_csv(path, rows)
                done.add((fold, seed, variant))
                print(f"PART_B_OUTER f{fold} s{seed} {variant}", flush=True)
        del bundle
        gc.collect()
    if len(done) != 60 or len(rows) != 960:
        raise RuntimeError(f"outer inference incomplete: cells={len(done)} rows={len(rows)}")
    finalize(pd.DataFrame(rows), freeze)


if __name__ == "__main__":
    main()
