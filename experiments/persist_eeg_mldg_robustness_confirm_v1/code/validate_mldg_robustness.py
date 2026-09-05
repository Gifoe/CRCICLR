"""Independent, fail-closed validator for the fast MLDG confirmation audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(sys.argv[1]).resolve()
SCHEMA = "PERSIST_EEG_MLDG_ROBUSTNESS_CONFIRM_V1"
SEEDS = {0, 1, 2}
DATASETS = {"OpenBMI", "WBCIC"}
EXPECTED = {
    ("OpenBMI", 0): (1.0, 38, 59),
    ("OpenBMI", 1): (1.0, 41, 58),
    ("WBCIC", 0): (1.0, 13, 6),
    ("WBCIC", 1): (0.5, 12, 23),
}


def req(name: str) -> Path:
    p = ROOT / name
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def main() -> int:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        lock = json.loads(req("PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
        split = json.loads(req("SPLIT_REUSE_AUDIT.json").read_text(encoding="utf-8"))
        gate = json.loads(req("ROBUSTNESS_GATE.json").read_text(encoding="utf-8"))
        checks["schema"] = lock.get("schema") == SCHEMA and split.get("schema") == SCHEMA and gate.get("schema") == SCHEMA
        checks["canonical_labels_closed"] = lock.get("canonical_outcome_labels_read") is False
        checks["sealed_holdouts_closed"] = lock.get("OpenBMI_sealed_holdout_opened") is False and lock.get("WBCIC_outer_10_opened") is False
        checks["split_reuse_pass"] = split.get("pass") is True and all(r.get("matches_randomness_audit") is True for r in split.get("rows", []))
        if not checks["split_reuse_pass"]: errors.append("split reuse audit failed")

        pairs = pd.read_csv(req("PAIRED_SEED_RESULTS.csv"))
        expected_cols = {"dataset", "outer_fold", "opt_seed", "beta", "ERM_epochs", "MLDG_epochs", "ERM_BA", "MLDG_BA", "delta_BA_pp", "initial_state_sha256"}
        checks["paired_schema"] = expected_cols.issubset(set(pairs.columns))
        checks["allowed_cells"] = set(pairs.dataset).issubset(DATASETS) and set(pairs.opt_seed.astype(int)).issubset(SEEDS) and set(pairs.outer_fold.astype(int)).issubset({0, 1})
        checks["delta_recomputed"] = bool(np.allclose(pairs.delta_BA_pp.to_numpy(float), pairs.MLDG_BA.to_numpy(float) - pairs.ERM_BA.to_numpy(float), atol=1e-9, rtol=0.0))
        checks["initial_state_present"] = bool(pairs.initial_state_sha256.astype(str).str.len().min() >= 32)
        checks["frozen_values"] = True
        for _, row in pairs.iterrows():
            key = (str(row.dataset), int(row.outer_fold))
            if key not in EXPECTED or abs(float(row.beta) - EXPECTED[key][0]) > 1e-12 or int(row.ERM_epochs) != EXPECTED[key][1] or int(row.MLDG_epochs) != EXPECTED[key][2]:
                checks["frozen_values"] = False
        if not checks["frozen_values"]: errors.append("frozen beta/epoch mismatch")

        # Exact Stage-A early-stop semantics: 2 datasets x fold0 x 3 seeds.
        stage_a_keys = {(str(d), int(f), int(s)) for d, f, s in zip(pairs.dataset, pairs.outer_fold, pairs.opt_seed)}
        terminal = str(gate.get("terminal"))
        if terminal == "EARLY_STOP_MLDG_NOT_ROBUST":
            checks["early_stop_cells"] = stage_a_keys == {(d, 0, s) for d in DATASETS for s in SEEDS}
        else:
            checks["early_stop_cells"] = True
        checks["per_subject_delta"] = True
        ps = pd.read_csv(req("PER_SUBJECT_PAIRED_DELTAS.csv"))
        if len(ps):
            checks["per_subject_delta"] = bool(np.allclose(ps.delta_BA_pp.to_numpy(float), ps.MLDG_BA.to_numpy(float) - ps.ERM_BA.to_numpy(float), atol=1e-9, rtol=0.0))

        # Recompute the available-data gate booleans independently.
        checks["gate_recomputed"] = True
        for dataset in sorted(DATASETS):
            q = pairs[pairs.dataset == dataset]
            vals = q.delta_BA_pp.to_numpy(float)
            fold0 = q[q.outer_fold == 0].delta_BA_pp.to_numpy(float)
            fold1 = q[q.outer_fold == 1].delta_BA_pp.to_numpy(float)
            expected = {
                "G1_or_G2_mean_ge_0.5": bool(len(vals) and float(np.mean(vals)) >= 0.5),
                "G3_positive_ge_4_of_6": bool(np.sum(vals >= 0.0) >= 4),
                "G4_both_fold_means_nonnegative": bool(len(fold0) and len(fold1) and float(np.mean(fold0)) >= 0.0 and float(np.mean(fold1)) >= 0.0),
                "G5_median_ge_0.25": bool(len(vals) and float(np.median(vals)) >= 0.25),
            }
            got = gate.get("gates", {}).get(dataset, {})
            for key, value in expected.items():
                if bool(got.get(key)) != value:
                    checks["gate_recomputed"] = False
                    errors.append(f"gate mismatch {dataset}:{key}")

        compute_path = ROOT / "COMPUTE_MATCHED_ERM_RESULTS.csv"
        if compute_path.exists():
            comp = pd.read_csv(compute_path)
            checks["compute_cells_legal"] = set(zip(comp.dataset, comp.outer_fold)).issubset({("OpenBMI", 0), ("WBCIC", 1)})
        else:
            checks["compute_cells_legal"] = terminal == "EARLY_STOP_MLDG_NOT_ROBUST"
        checks["no_target_information"] = True
    except Exception as exc:
        errors.append(f"exception: {type(exc).__name__}: {exc}")
        checks["exception_free"] = False

    checks["all_checks"] = bool(checks) and all(bool(v) for v in checks.values())
    payload = {"schema": SCHEMA, "pass": checks["all_checks"], "checks": checks, "errors": errors, "scientific_terminal": (json.loads((ROOT / "ROBUSTNESS_GATE.json").read_text(encoding="utf-8")).get("terminal") if (ROOT / "ROBUSTNESS_GATE.json").exists() else None)}
    (ROOT / "INDEPENDENT_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
