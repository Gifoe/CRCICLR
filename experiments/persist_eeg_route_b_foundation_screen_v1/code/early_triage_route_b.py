"""Early two-fold triage for the Route-B foundation screen.

This runner is intentionally separate from the production five-fold screen.
It reuses only the existing selection/checkpoint caches, evaluates WBCIC first,
then OpenBMI folds 0--1, and writes an explicit early-triage report.  The
amendment lock is written before any held-out metrics are read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(sys.argv[1]).resolve()
BASE_ROOT = Path(sys.argv[2]).resolve()
DEVICE = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cuda:0")
CODE = Path(__file__).with_name("run_foundation_screen.py")
SCHEMA = "PERSIST_EEG_ROUTE_B_EARLY_TRIAGE_V1"
AMENDMENT = {
    "schema": SCHEMA,
    "purpose": "Pause remaining OpenBMI after the currently running fold; complete WBCIC outer folds 0-1 for B0-B4; compare first two folds per dataset before deciding whether to continue the five-fold screen.",
    "datasets": ["OpenBMI", "WBCIC"],
    "outer_folds": [0, 1],
    "methods": [
        "B0_SUBJECT_BALANCED_ERM",
        "B1_SUBJECT_GROUPDRO",
        "B2_SUBJECT_EPISODIC_MLDG",
        "B3_SUBJECT_GRADIENT_STAT_DG",
        "B4_SUBJECT_STYLE_EXTRAPOLATION",
    ],
    "decision_rule": {
        "dataset_mean_delta_BA_pp_min": 0.5,
        "positive_dataset_fold_cells_min": 3,
        "cells_total": 4,
        "continue_full_five_fold_only_if_both_conditions_hold": True,
    },
    "protocol": {
        "seed": 0,
        "inner_K": 5,
        "source_only": True,
        "outcome_labels_read_before_lock": False,
        "OpenBMI_sealed_holdout_opened": False,
        "WBCIC_outer_10_opened": False,
    },
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def sha256_obj(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_route_module():
    spec = importlib.util.spec_from_file_location("route_b_foundation_screen", CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CODE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    # Lock the amendment before any held-out evaluation is performed.
    amendment_payload = dict(AMENDMENT)
    amendment_payload["amendment_sha256"] = sha256_obj(AMENDMENT)
    amendment_payload["outcome_labels_read"] = False
    (ROOT / "EARLY_TRIAGE_PROTOCOL_AMENDMENT.json").write_text(
        json.dumps(amendment_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rb = load_route_module()
    # WBCIC is intentionally first.  run_outer uses this module-global order
    # and max_outer=2, while preserving the frozen five-fold split definition.
    rb.DATASETS = ("WBCIC", "OpenBMI")
    rb.seed_everything(rb.SEED)
    ap, geo, _ = rb.import_audited(BASE_ROOT)
    run, _aux = rb.run_outer(ROOT, BASE_ROOT, ap, geo, DEVICE, max_outer=2)

    sp = pd.DataFrame(run["subject"])
    if sp.empty:
        raise RuntimeError("triage produced no held-out subject rows")
    base = sp[sp.method == "B0_SUBJECT_BALANCED_ERM"].set_index(["dataset", "outer_fold", "subject"])
    rows = []
    for method in rb.METHODS[1:]:
        for dataset in ("OpenBMI", "WBCIC"):
            for fold in (0, 1):
                g = sp[(sp.dataset == dataset) & (sp.outer_fold == fold)]
                bm = float(g[g.method == "B0_SUBJECT_BALANCED_ERM"].BA.mean())
                mm = float(g[g.method == method].BA.mean())
                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "outer_fold": fold,
                    "B0_BA_pp": bm * 100.0,
                    "method_BA_pp": mm * 100.0,
                    "delta_BA_pp": (mm - bm) * 100.0,
                    "positive": bool((mm - bm) >= 0.0),
                })
    cell = pd.DataFrame(rows)
    cell.to_csv(ROOT / "EARLY_TRIAGE_FOLD_CELLS.csv", index=False)
    summary_rows = []
    for method, g in cell.groupby("method", sort=False):
        ds = g.groupby("dataset").delta_BA_pp.mean()
        signs = g.sort_values(["dataset", "outer_fold"]).delta_BA_pp.to_numpy(float)
        positive_cells = int(np.sum(signs >= 0.0))
        both_dataset_means = bool(all(float(ds.get(d, float("-inf"))) >= 0.5 for d in ("OpenBMI", "WBCIC")))
        gate = bool(both_dataset_means and positive_cells >= 3)
        summary_rows.append({
            "method": method,
            "OpenBMI_mean_delta_BA_pp": float(ds.get("OpenBMI", np.nan)),
            "WBCIC_mean_delta_BA_pp": float(ds.get("WBCIC", np.nan)),
            "positive_dataset_fold_cells": positive_cells,
            "dataset_fold_cells": 4,
            "fold_signs": ";".join(f"{v:+.3f}" for v in signs),
            "direction_consistent": bool(positive_cells >= 3),
            "provisional_gate": gate,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "EARLY_TRIAGE_SUMMARY.csv", index=False)
    passed = bool(summary.provisional_gate.any())
    terminal = "CONTINUE_FULL_5_FOLD_SCREEN" if passed else "STOP_FULL_SCREEN_NO_PROVISIONAL_POSITIVE_TRAINING_PRINCIPLE"
    lines = [
        "# Early triage report",
        "",
        f"Terminal: `{terminal}`",
        "",
        "This report uses only OpenBMI and WBCIC outer folds 0-1, with B0 as the frozen subject-balanced ERM baseline.",
        "A method can continue only when both dataset mean ΔBA are at least +0.5 pp and at least 3 of 4 dataset-fold cells are nonnegative.",
        "",
        "| Method | OpenBMI mean ΔBA (pp) | WBCIC mean ΔBA (pp) | Positive cells / 4 | Fold signs (OpenBMI0,1; WBCIC0,1) | Provisional gate |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['method']} | {r['OpenBMI_mean_delta_BA_pp']:.3f} | {r['WBCIC_mean_delta_BA_pp']:.3f} | {r['positive_dataset_fold_cells']}/4 | {r['fold_signs']} | {'PASS' if r['provisional_gate'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"Protocol amendment SHA-256: `{amendment_payload['amendment_sha256']}`",
        "No OpenBMI sealed holdout or WBCIC outer-10 data were opened.",
    ]
    (ROOT / "EARLY_TRIAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "EARLY_TRIAGE_DECISION.json").write_text(
        json.dumps({"schema": SCHEMA, "terminal": terminal, "provisional_positive": passed, "amendment_sha256": amendment_payload["amendment_sha256"], "outcome_labels_read": True, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
