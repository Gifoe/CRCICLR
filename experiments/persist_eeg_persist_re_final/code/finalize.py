"""Materialize compact locks and the final source-gated report."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

import persist_re_core as c


def row(frame, dataset, comparison):
    value = frame[(frame.dataset == dataset) & (frame.comparison == comparison)]
    return value.iloc[0].to_dict() if len(value) else {"delta_BA": None, "CI95_L": None, "CI95_U": None}


def main() -> None:
    gate = json.loads((c.RESULTS / "SOURCE_GATE.json").read_text(encoding="utf-8"))
    ablation = pd.read_csv(c.RESULTS / "ABLATION_SUMMARY.csv")
    selected = {d: pd.read_csv(c.RESULTS / f"SOURCE_RECIPE_SELECTION_{d}.csv").iloc[0].to_dict() for d in c.DATASETS}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO, text=True).strip()
    lock_common = {"method": "PERSIST-RE", "git_head_at_lock": head, "outer_resources_touched": False, "sealed_resources_touched": False, "future_utility_accessed": False, "source_gate": gate}
    c.write_json(c.EXP / "protocol" / "DATA_ACCESS_LOCK.json", {**lock_common, "status": "SOURCE_ONLY_COMPLETE", "authorized_resources": ["OpenBMI_development", "WBCIC_S1_development"], "forbidden_resources": ["WBCIC_outer_10", "OpenBMI_sealed_holdout", "WBCIC_session2_utility"]})
    c.write_json(c.EXP / "protocol" / "PERSIST_RE_SOURCE_LOCK.json", {**lock_common, "status": "SOURCE_GATE_FAILED", "selected_recipes": selected, "search_grid": {"rank": [1, 2, 4], "lambda_R": [1e-3, 1e-2], "lambda_P": [0.5, 1.0]}, "controls": list(c.METHODS)})
    c.write_json(c.EXP / "protocol" / "PERSIST_RE_FINAL_METHOD_LOCK.json", {**lock_common, "status": "NOT_AUTHORIZED_SOURCE_GATE_FAILED", "confirmation_architectures_opened": []})
    c.write_json(c.EXP / "protocol" / "OUTER_CONFIRMATION_PROTOCOL.json", {"status": "NOT_AUTHORIZED", "reason": "source gate failed", "outer_resources_touched": False})
    final = {
        "method": "PERSIST-RE", "terminal": "PERSIST_RE_SOURCE_NOT_SUPPORTED", "branch": "codex/persist-eeg-persist-re-final", "git_head_at_generation": head,
        "selected_recipes": selected,
        "source": {"OpenBMI": row(ablation, "OpenBMI", "PERSIST-RE-ERM"), "WBCIC": row(ablation, "WBCIC", "PERSIST-RE-ERM")},
        "comparisons": {d: {k: row(ablation, d, f"{k}-ERM") for k in ("GroupDRO", "ProspectiveOnly", "RandomEffectOnly") } for d in c.DATASETS},
        "source_gate": gate,
        "identity_probe": "source-only mechanism artifact; see results/IDENTITY_PROBE.csv",
        "decision_heterogeneity": "source-only mechanism artifact; see results/DECISION_HETEROGENEITY.csv",
        "ATCNet-Official": "exploratory_only_not_authorized",
        "EEGNeX": "exploratory_only_not_authorized",
        "cross_architecture_status": "NOT_AUTHORIZED_SOURCE_GATE_FAILED",
        "outer_status": "UNTOUCHED",
        "sealed_status": "UNTOUCHED",
        "strongest_supported_claim": "The implementation enforces decision-level random-effect quarantine and population-only inference; no utility improvement is supported by the CleanRoom source gate.",
        "strongest_unsupported_claim": "PERSIST-RE is not supported as a future-session utility method on these source resources.",
    }
    c.write_json(c.EXP / "FINAL_REPORT.json", final)
    lines = ["# Final report", "", "## Terminal", "", "`PERSIST_RE_SOURCE_NOT_SUPPORTED`", "", "The CleanRoom source gate failed before any confirmation architecture was authorized.", "", "## Source results", ""]
    for d in c.DATASETS:
        r = final["source"][d]; lines.append(f"- {d}: delta={r.get('delta_BA')}, paired CI=[{r.get('CI95_L')}, {r.get('CI95_U')}].")
    lines += ["", "## Controls", "", "GroupDRO, ProspectiveOnly, and RandomEffectOnly comparisons are in `results/ABLATION_SUMMARY.csv`; no post-hoc rule was changed.", "", "## Resource boundary", "", "WBCIC outer subjects, WBCIC session-2 utility, and the OpenBMI sealed holdout were untouched.  Exploratory additional-backbone files, if present, are not confirmation evidence.", "", "## Claim boundary", "", final["strongest_supported_claim"]]
    (c.EXP / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__": main()

