from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import p4_persist_ct as base

ROOT = base.ROOT
OUT = ROOT / "outputs" / "persist_eeg_p4_ct_v2"
rows = []
for fold in base.FOLDS:
    for seed in base.SEEDS:
        p = OUT / "development" / "CT_V1" / f"fold-{fold}" / f"seed-{seed}" / "DEVELOPMENT_RESULT.json"
        if not p.exists():
            raise FileNotFoundError(str(p))
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"fold": fold, "seed": seed, "control_macro_BA": r["control"]["macro_BA"], "ct_macro_BA": r["ct"]["macro_BA"], "delta_macro_BA": r["delta_macro_BA"], "control_mi_BA": r["control"]["task_BA"]["mi"], "ct_mi_BA": r["ct"]["task_BA"]["mi"], "control_erp_BA": r["control"]["task_BA"]["erp"], "ct_erp_BA": r["ct"]["task_BA"]["erp"], "control_ssvep_BA": r["control"]["task_BA"]["ssvep"], "ct_ssvep_BA": r["ct"]["task_BA"]["ssvep"], "outer_test_used": False})
df = pd.DataFrame(rows)
df.to_csv(OUT / "P4_CT_DEVELOPMENT_SUMMARY_CT_V1.csv", index=False)
mean_delta = float(df.delta_macro_BA.mean())
positive = int((df.delta_macro_BA > 0).sum())
task_delta = {t: float((df[f"ct_{t}_BA"] - df[f"control_{t}_BA"]).mean()) for t in ("mi", "erp", "ssvep")}
report = {"status": "P4_CT_NOT_SUPPORTED", "version": "CT_V1", "mean_delta_macro_BA": mean_delta, "positive_runs": positive, "task_mean_delta_BA": task_delta, "gates": {"clean_macro_ge_0p003": mean_delta >= 0.003, "at_least_4_of_6_positive": positive >= 4, "task_ge_0p005": any(v >= 0.005 for v in task_delta.values()), "protection_gate": False, "counterfactual_robustness_gate": False}, "reason": "CT-V0 and CT-V1 failed the prospective clean development gate; no method lock", "outer_test_used": False}
(OUT / "P4_CT_DEVELOPMENT_REPORT_CT_V1.json").write_text(json.dumps(base.clean(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
log_path = OUT / "P4_CT_ADAPTATION_LOG.json"
log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
v0_path = OUT / "P4_CT_DEVELOPMENT_SUMMARY_CT_V0.csv"
v0 = pd.read_csv(v0_path) if v0_path.exists() else pd.DataFrame()
log["development_versions"] = [
    {"version": "CT_V0", "failure": "near-zero clean development gain", "evidence": {"mean_delta_macro_BA": float(v0.delta_macro_BA.mean()) if len(v0) else None, "positive_runs": int((v0.delta_macro_BA > 0).sum()) if len(v0) else None}, "modification": "frozen historical encoder; complete clean + soft worst-case CE + KL consistency", "why_it_addresses_failure": "first complete CT objective", "data_used": ["TRAIN", "VALIDATION"], "outer_test_used": False},
    {"version": "CT_V1", "failure": "CT-V1 clean development gate not met", "evidence": {"mean_delta_macro_BA": mean_delta, "positive_runs": positive, "task_mean_delta_BA": task_delta}, "modification": "K=4; 1.5x empirical nuisance transport amplitude; lambda_CT=0.60; lambda_cons=0.05", "why_it_addresses_failure": "increased counterfactual coverage and worst-case pressure", "data_used": ["TRAIN", "VALIDATION"], "outer_test_used": False},
]
log["final_decision"] = "P4_CT_NOT_SUPPORTED"
log_path.write_text(json.dumps(base.clean(log), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "P4_CT_LOCK_REFUSED.json").write_text(json.dumps({"status": "P4_CT_LOCK_REFUSED", "reason": "CT-V0 and CT-V1 failed clean development gate", "outer_test_used": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "P4_CT_FINAL_REPORT.json").write_text(json.dumps(base.clean({"status": "P4_CT_NOT_SUPPORTED", "audit_v2": "P4_CT_AUDIT_V2_PASS", "development": report, "outer_test_used": False}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "P4_CT_FINAL_REPORT.md").write_text(f"# PERSIST-CT\n\nAudit V2: `P4_CT_AUDIT_V2_PASS`\n\nFinal decision: `P4_CT_NOT_SUPPORTED`\n\nCT-V1 mean macro BA delta versus continued-training control: `{mean_delta:.6f}`. Positive runs: `{positive}/6`.\n\nNo method lock and no outer-test access.\n", encoding="utf-8")
print(json.dumps(base.clean(report), ensure_ascii=False, indent=2))
