from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import DIAGNOSTICS, PROTOCOL, V7_SEED, ensure_directories, write_csv, write_json
from utility.controller import CONTROLLERS, GENERIC_COLUMNS, PERSIST_COLUMNS, crossfit_predict


def run() -> None:
    ensure_directories()
    generator = np.random.default_rng(V7_SEED)
    rows = []
    semantics = {
        "PROTECTED": 0.0,
        "ADAPTABLE": 0.025,
        "HARMFUL": -0.025,
    }
    for subject in range(240):
        latent = generator.normal()
        for component, mean in semantics.items():
            row = {
                "benchmark": "SYNTHETIC_POSITIVE_CONTROL",
                "outer_fold": subject % 5,
                "subject_id": f"synthetic-{subject:03d}",
                "component_id": component,
                "family": "synthetic",
                "future_ce_gain": mean + 0.004 * latent + generator.normal(0.0, 0.002),
                "future_ba_gain": mean,
                "P_persistence": {"PROTECTED": 0.95, "ADAPTABLE": 0.70, "HARMFUL": 0.20}[component],
                "U_signed_utility_prior": mean,
                "D_decision_dependence": {"PROTECTED": 0.80, "ADAPTABLE": 0.35, "HARMFUL": 0.60}[component],
                "G_task_overlap": {"PROTECTED": 0.90, "ADAPTABLE": 0.75, "HARMFUL": -0.70}[component],
                "R_history_transfer": mean + generator.normal(0.0, 0.003),
            }
            for column in GENERIC_COLUMNS:
                row[column] = float(generator.normal())
            rows.append(row)
    frame = pd.DataFrame(rows)
    predicted, metrics = crossfit_predict(frame, True, CONTROLLERS[0], V7_SEED)
    predicted["synthetic_semantic"] = predicted.component_id
    predicted["policy_action"] = np.where(predicted.predicted_utility > 0.005, "ADAPT", "PRESERVE")
    adaptable = predicted.component_id.eq("ADAPTABLE")
    protected = predicted.component_id.eq("PROTECTED")
    harmful = predicted.component_id.eq("HARMFUL")
    summary = {
        "adaptable_sensitivity": float(predicted.loc[adaptable, "policy_action"].eq("ADAPT").mean()),
        "protected_specificity": float(predicted.loc[protected, "policy_action"].eq("PRESERVE").mean()),
        "harmful_rejection_rate": float(predicted.loc[harmful, "policy_action"].eq("PRESERVE").mean()),
        "utility_predictability": metrics,
        "synthetic_only": True,
        "real_EEG_evidence": False,
        "OUTER_TEST_USED": False,
    }
    write_csv(DIAGNOSTICS / "V7_POSITIVE_CONTROL.csv", predicted)
    write_json(DIAGNOSTICS / "V7_POSITIVE_CONTROL.json", summary)
    write_json(PROTOCOL / "V7_POSITIVE_CONTROL_AUDIT.json", {
        "purpose": "mechanism wiring test only",
        "protected": "near-zero coefficient / PRESERVE",
        "adaptable": "positive coefficient / ADAPT",
        "harmful": "rejected / PRESERVE",
        "synthetic_not_real_evidence": True,
        "OUTER_TEST_USED": False,
    })
    print(summary, flush=True)


if __name__ == "__main__":
    run()
