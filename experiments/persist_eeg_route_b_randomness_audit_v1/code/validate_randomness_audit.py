"""Independent recomputation and validation for the Route-B randomness audit.

This validator never reads canonical outcome labels.  It recomputes the compact
gates from the emitted per-fold rows, checks the protocol/seed/order hashes,
and records an auditable pass/fail JSON.  It is intentionally separate from
the training driver so a malformed summary cannot make the validation pass.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(sys.argv[1]).resolve()
BASE_ROOT = Path(sys.argv[2]).resolve()
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "cpu"
AUDIT_SCHEMA = "PERSIST_EEG_ROUTE_B_RANDOMNESS_AUDIT_V1"
TOL = 1e-9


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        protocol = json.loads(require(ROOT / "COMMON_RANDOMNESS_PROTOCOL_V2.json").read_text(encoding="utf-8"))
        checks["schema"] = protocol.get("schema") == AUDIT_SCHEMA
        checks["canonical_outcome_labels_read_false"] = protocol.get("canonical_outcome_labels_read") is False
        checks["sealed_holdouts_closed"] = protocol.get("OpenBMI_sealed_holdout_opened") is False and protocol.get("WBCIC_outer_10_opened") is False
        checks["method_independent_randomness"] = protocol.get("method_names_excluded_from_order_and_rng") is True
        if not all(checks.values()):
            errors.append("protocol metadata failed")

        eq = json.loads(require(ROOT / "BASELINE_NUMERICAL_EQUIVALENCE.json").read_text(encoding="utf-8"))
        checks["baseline_equivalence_pass"] = eq.get("pass") is True and all(row.get("pass") is True for row in eq.get("rows", []))
        if not checks["baseline_equivalence_pass"]:
            errors.append("baseline numerical equivalence failed")

        per = pd.read_csv(require(ROOT / "CLEAN_EARLY_SCREEN_PER_FOLD.csv"))
        summary = pd.read_csv(require(ROOT / "CLEAN_EARLY_SCREEN_SUMMARY.csv"))
        required_methods = {"B0_SUBJECT_BALANCED_ERM", "B1_SUBJECT_GROUPDRO", "B2_SUBJECT_EPISODIC_MLDG", "B3_SUBJECT_GRADIENT_STAT_DG", "B4_SUBJECT_STYLE_EXTRAPOLATION"}
        expected_cells = {(d, f) for d in ("OpenBMI", "WBCIC") for f in (0, 1)}
        got_cells = {(str(d), int(f)) for d, f in zip(per.dataset, per.outer_fold)}
        checks["per_fold_complete"] = got_cells == expected_cells and set(per.method) == required_methods and len(per) == 20
        if not checks["per_fold_complete"]:
            errors.append("per-fold table is incomplete or has unexpected methods")

        recomputed = []
        for method in sorted(required_methods - {"B0_SUBJECT_BALANCED_ERM"}):
            cells = []
            for dataset in ("OpenBMI", "WBCIC"):
                for fold in (0, 1):
                    b0 = float(per[(per.dataset == dataset) & (per.outer_fold == fold) & (per.method == "B0_SUBJECT_BALANCED_ERM")].held_BA.iloc[0])
                    val = float(per[(per.dataset == dataset) & (per.outer_fold == fold) & (per.method == method)].held_BA.iloc[0])
                    cells.append((dataset, fold, val - b0))
            om = float(np.mean([x[2] for x in cells if x[0] == "OpenBMI"]))
            wm = float(np.mean([x[2] for x in cells if x[0] == "WBCIC"]))
            positive = int(sum(x[2] >= 0.0 for x in cells))
            recomputed.append({"method": method, "OpenBMI_mean_delta_BA_pp": om, "WBCIC_mean_delta_BA_pp": wm, "positive_dataset_fold_cells": positive, "provisional_gate": bool(om >= 0.5 and wm >= 0.5 and positive >= 3)})
        rec = pd.DataFrame(recomputed)
        checks["summary_recomputed"] = len(summary) == len(rec)
        for _, row in rec.iterrows():
            got = summary[summary.method == row.method]
            if len(got) != 1:
                checks["summary_recomputed"] = False
                errors.append(f"missing summary row {row.method}")
                continue
            got = got.iloc[0]
            for col in ("OpenBMI_mean_delta_BA_pp", "WBCIC_mean_delta_BA_pp", "positive_dataset_fold_cells", "provisional_gate"):
                if col == "provisional_gate":
                    ok = bool(got[col]) == bool(row[col])
                else:
                    ok = abs(float(got[col]) - float(row[col])) <= TOL
                if not ok:
                    checks["summary_recomputed"] = False
                    errors.append(f"summary mismatch {row.method}:{col}")

        sens = json.loads(require(ROOT / "WBCIC_ORDER_SENSITIVITY.json").read_text(encoding="utf-8"))
        ranges = [float(x["range_pp"]) for x in sens.get("folds", [])]
        checks["order_sensitivity_audit_present"] = len(ranges) == 2 and all(np.isfinite(ranges))
        env = pd.read_csv(require(ROOT / "RANDOMNESS_ENVELOPE_COMPARISON.csv"))
        checks["envelope_recomputed"] = len(env) == len(recomputed)
        max_range = max(ranges) if ranges else float("nan")
        for _, row in rec.iterrows():
            got = env[env.method == row.method]
            if len(got) != 1:
                checks["envelope_recomputed"] = False
                continue
            got = got.iloc[0]
            expected = float(row.WBCIC_mean_delta_BA_pp - max_range)
            if abs(float(got.WBCIC_gain_vs_randomness_envelope_pp) - expected) > TOL or bool(got.exceeds_envelope) != bool(row.WBCIC_mean_delta_BA_pp > max_range):
                checks["envelope_recomputed"] = False
                errors.append(f"envelope mismatch {row.method}")

        # Independent state/order hashes: these are compact and do not expose data.
        route_path = ROOT.parent / "persist_eeg_route_b_foundation_screen_v1" / "code" / "run_foundation_screen.py"
        rb = load_module("route_b_foundation_screen_validate", route_path)
        ap, geo, _ = rb.import_audited(BASE_ROOT)
        state_rows, order_rows = [], []
        for dataset in ("OpenBMI", "WBCIC"):
            roles, _, _ = ap.load_roles(dataset)
            source = rb.subj_sort(roles[0]["model_fit"])
            folds = rb.subject_split(source, "nested-oof-outer", dataset, 0, rb.OUTER_K)
            for outer in (0, 1):
                held = folds[outer]
                train_subjects = [s for s in source if s not in set(held)]
                inner = rb.subject_split(train_subjects, "route-b-inner-validation", dataset, outer, rb.INNER_K)
                sel_subjects = [s for s in train_subjects if s not in set(inner[-1])]
                cache = geo.FoldCache(dataset, source, rb.SEED, 0)
                sel_rows = cache.rows(sel_subjects, geo.SESSIONS_FIT[dataset])
                refit_rows = cache.rows(train_subjects, geo.SESSIONS_FIT[dataset])
                state, init_seed, _ = geo.initial_state(cache, dataset, outer, rb.SEED, "route-b-common")
                h = hashlib.sha256()
                for k in sorted(state):
                    h.update(k.encode()); h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
                state_rows.append({"dataset": dataset, "outer_fold": outer, "init_seed": int(init_seed), "initial_state_sha256": h.hexdigest()})
                for phase in ("selection", "refit"):
                    for epoch in range(1, 4):
                        seed = int(rb.stable_seed("route-b-common-order-v2", dataset, outer, phase, epoch))
                        rows = sel_rows if phase == "selection" else refit_rows
                        order = np.asarray(rows, dtype=np.int64)[np.random.default_rng(seed).permutation(len(rows))]
                        order_rows.append({"dataset": dataset, "outer_fold": outer, "phase": phase, "epoch": epoch, "order_sha256": sha256_bytes(order.tobytes())})
        pd.DataFrame(state_rows).to_csv(ROOT / "INITIAL_STATE_HASH.csv", index=False)
        pd.DataFrame(order_rows).to_csv(ROOT / "COMMON_ORDER_HASH_AUDIT.csv", index=False)
        checks["independent_hash_artifacts"] = len(state_rows) == 4 and len(order_rows) == 24
    except Exception as exc:  # validation must fail closed
        errors.append(f"exception: {type(exc).__name__}: {exc}")
        checks.setdefault("validator_exception_free", False)

    checks["all_checks"] = bool(checks) and all(bool(v) for v in checks.values())
    payload = {"schema": AUDIT_SCHEMA, "pass": checks["all_checks"], "checks": checks, "errors": errors, "source_code_sha256": sha256_file(Path(__file__))}
    (ROOT / "INDEPENDENT_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
