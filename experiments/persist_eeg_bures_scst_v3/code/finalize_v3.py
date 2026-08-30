"""Finalize compact V3 artifacts and select the preregistered terminal."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as c


DOCS = ("README.md", "V2_FORENSIC_AUDIT.md", "CODE_MAP.md", "SCIENTIFIC_RATIONALE.md", "METHOD.md", "CROSS_FIT_AUDIT.md", "COVARIANCE_AUDIT.md", "BURES_MAP_AUDIT.md", "ANCHOR_EXCLUSION_AUDIT.md", "TARGET_AFFINITY_AUDIT.md", "RANDOM_AFFINE_AUDIT.md", "SOURCE_DEVELOPMENT_REPORT.md", "ATCNET_OFFICIAL_REPORT.md", "EEGNEX_CONFIRMATION_REPORT.md", "CONTROL_REPORT.md", "CLAIM_AUDIT.md", "ITERATION_LEDGER.md", "REPRODUCIBILITY.md", "FINAL_REPORT.md", "FINAL_REPORT.json")
FIGURES = ("v2_self_neighbor_audit", "mean_vs_second_order_transport", "target_affinity", "displacement_vs_margin", "method_comparison", "subject_level_gain", "cross_architecture_gain")


def _doc(name: str, title: str, body: str) -> None:
    (c.EXP / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _plot(name: str, draw) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0)); draw(ax); fig.tight_layout(); fig.savefig(c.FIGURES / f"{name}.png", dpi=180); fig.savefig(c.FIGURES / f"{name}.pdf"); plt.close(fig)


def _text_plot(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True); ax.set_axis_off()


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.is_file():
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame(columns=columns)


def main() -> None:
    c.ensure_dirs()
    gate = c.read_json(c.RESULTS / "SOURCE_GATE.json") if (c.RESULTS / "SOURCE_GATE.json").is_file() else {"source_gate_pass": False, "terminal_if_stop": "BURES_SCST_IMPLEMENTATION_INVALID", "selected": None, "transport": {}}
    forensic = c.read_json(c.RESULTS / "V2_FORENSIC_SUMMARY.json") if (c.RESULTS / "V2_FORENSIC_SUMMARY.json").is_file() else {}
    stopped = str(gate.get("terminal_if_stop", "")) if not gate.get("source_gate_pass", False) else ""
    source_cols = ["dataset", "method", "q", "lambda_T", "fold", "seed", "subject_id", "BA", "macro_F1"]
    for name, columns in {
        "SOURCE_PER_SUBJECT.csv": source_cols, "SOURCE_PER_FOLD.csv": ["dataset", "method", "q", "lambda_T", "fold", "seed", "BA", "macro_F1"],
        "SOURCE_RECIPE_SEARCH.csv": ["dataset", "method", "q", "lambda_T", "delta_BA"], "METHOD_SUMMARY.csv": ["dataset", "method", "q", "lambda_T", "fold", "seed", "BA"],
        "CONTROL_COMPARISON.csv": ["dataset", "q", "lambda_T", "comparison", "delta_BA", "positive_folds"],
        "BURES_STATISTICS.csv": ["dataset", "method", "q", "lambda_T"], "CANDIDATE_VALIDITY.csv": ["dataset", "method", "q", "lambda_T"], "TARGET_AFFINITY.csv": ["dataset", "method", "q", "lambda_T"],
    }.items():
        path = c.RESULTS / name
        if not path.is_file():
            frame = pd.DataFrame(columns=columns + ["status"]); frame.loc[0, "status"] = stopped or "NOT_AVAILABLE"; c.write_csv(path, frame)
    source = _read(c.RESULTS / "SOURCE_PER_SUBJECT.csv", source_cols)
    recipe = _read(c.RESULTS / "SOURCE_RECIPE_SEARCH.csv", [])
    matching = _read(c.RESULTS / "RANDOM_AFFINE_MATCHING.csv", [])
    official = _read(c.RESULTS / "ATCNET_OFFICIAL_PER_FOLD.csv", [])
    eegnex = _read(c.RESULTS / "EEGNEX_PER_FOLD.csv", [])
    if not gate.get("source_gate_pass", False):
        terminal = str(gate.get("terminal_if_stop", "BURES_SCST_SOURCE_GATE_FAILED"))
    elif (c.RESULTS / "CONFIRMATION_STATUS.json").is_file():
        terminal = "BURES_SCST_IMPLEMENTATION_INVALID"
    else:
        terminal = "BURES_SCST_IMPLEMENTATION_INVALID"
    allowed = {"BURES_SCST_CROSS_ARCH_SUPPORTED", "BURES_SCST_DISCOVERY_ONLY_GENERALITY_NOT_SUPPORTED", "BURES_SCST_NOT_SUPPORTED_ON_OFFICIAL_ATCNET", "BURES_SCST_SOURCE_GATE_FAILED", "BURES_SCST_TRANSPORT_NOT_REALIZED", "BURES_SCST_IMPLEMENTATION_INVALID"}
    if terminal not in allowed: terminal = "BURES_SCST_IMPLEMENTATION_INVALID"
    selected = gate.get("selected")
    selected_row = None
    if selected and len(recipe):
        pick = recipe[(np.isclose(recipe.q, float(selected["q"]))) & (np.isclose(recipe.lambda_T, float(selected["lambda_T"]))) & (recipe.method == "Bures-HardSCST")]
        if len(pick): selected_row = pick.groupby("dataset").delta_BA.mean().to_dict()
    transport = gate.get("transport", {})
    paired = matching[matching.matched_pair.astype(bool)] if "matched_pair" in matching else matching
    matching_audit = {"rows": int(len(matching)), "matched_pairs": int(len(paired)), "empty": bool(len(paired) == 0), "mean_euclidean_norm_mismatch": float(paired.euclidean_norm_mismatch.abs().mean()) if "euclidean_norm_mismatch" in paired and len(paired) else None, "mean_whitened_norm_mismatch": float(paired.whitened_norm_mismatch.abs().mean()) if "whitened_norm_mismatch" in paired and len(paired) else None, "alpha_mismatch_max": float(paired.alpha_mismatch.dropna().abs().max()) if "alpha_mismatch" in paired and len(paired) and paired.alpha_mismatch.notna().any() else None, "per_anchor_count_match": bool((matching.structured_matched_count == matching.random_matched_count).all()) if "structured_matched_count" in matching and len(matching) else False}
    final = {
        "schema": "BURES_SCST_V3_FINAL_REPORT_V1", "branch": "codex/persist-eeg-bures-scst-v3", "terminal": terminal,
        "immutable_v1_terminal": "SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE", "immutable_v2_terminal": "ME_HARD_SCST_NOT_SUPPORTED",
        "v2_forensic": forensic, "anchor_self_neighbor_rate": forensic.get("nearest_self_rate_mean"), "selected_source_recipe": selected,
        "source_development": {"OpenBMI": transport.get("OpenBMI"), "WBCIC_S1_to_S2": transport.get("WBCIC"), "selected_delta_by_dataset": selected_row},
        "target_affinity": transport, "random_affine_matching": matching_audit,
        "atcnet_official": {"ERM_BA": None, "Bures_SCST_BA": None, "delta_BA": None, "paired_CI": None, "positive_folds": None, "status": "NOT_RUN_SOURCE_GATE" if not gate.get("source_gate_pass", False) else "NOT_AVAILABLE"},
        "eegnex": {"ERM_BA": None, "Bures_SCST_BA": None, "delta_BA": None, "paired_CI": None, "positive_folds": None, "status": "NOT_RUN_SOURCE_GATE" if not gate.get("source_gate_pass", False) else "NOT_AVAILABLE"},
        "outer_resource_status": "NOT_OPENED", "s3_opened": False, "strongest_supported_claim": "The V3 source gate did not authorize model-specific S3 confirmation; no utility generalization claim is supported." if not gate.get("source_gate_pass", False) else "No model-specific confirmation claim is supported.",
    }
    # Compact evidence plots; every stopped analysis is rendered explicitly as
    # not run rather than as a fabricated zero measurement.
    f = _read(c.RESULTS / "V2_FORENSIC_METRICS.csv", [])
    if len(f) and "nearest_self_rate" in f:
        _plot("v2_self_neighbor_audit", lambda ax: (ax.bar(["nearest", "top-3", "top-5"], [f.nearest_self_rate.mean(), f.top3_self_rate.mean(), f.top5_self_rate.mean()]), ax.set_ylim(0, 1), ax.set_ylabel("fraction including anchor")))
    else: _plot("v2_self_neighbor_audit", lambda ax: _text_plot(ax, "V2 forensic cache unavailable"))
    gs = _read(c.RESULTS / "GEOMETRY_PER_SUBJECT.csv", [])
    if len(gs) and "target_distance_improvement" in gs:
        _plot("mean_vs_second_order_transport", lambda ax: ax.bar(["distance", "Gaussian NLL"], [gs.target_distance_improvement.mean(), gs.target_nll_improvement.mean()]))
        _plot("target_affinity", lambda ax: ax.bar(["distance", "NLL"], [gs.target_distance_improvement.mean(), gs.target_nll_improvement.mean()]))
    else:
        _plot("mean_vs_second_order_transport", lambda ax: _text_plot(ax, "V3 source geometry not available")); _plot("target_affinity", lambda ax: _text_plot(ax, "V3 source geometry not available"))
    _plot("displacement_vs_margin", lambda ax: _text_plot(ax, "See compact candidate-validity CSV" if len(gs) else "V3 source geometry not available"))
    if len(recipe) and "delta_BA" in recipe and recipe.delta_BA.notna().any():
        value = recipe.groupby("method").delta_BA.mean().dropna().sort_values(); _plot("method_comparison", lambda ax: (value.plot(kind="bar", ax=ax), ax.axhline(0, color="black"), ax.set_ylabel("BA versus ERM"))) if len(value) else _plot("method_comparison", lambda ax: _text_plot(ax, "Source recipe search not opened"))
    else: _plot("method_comparison", lambda ax: _text_plot(ax, "Source recipe search not opened"))
    _plot("subject_level_gain", lambda ax: _text_plot(ax, "Source gate stopped before subject utility" if not gate.get("source_gate_pass", False) else "See source per-subject CSV"))
    _plot("cross_architecture_gain", lambda ax: _text_plot(ax, "ATCNet-Official/EEGNeX confirmation not run"))
    docs = {
        "README.md": ("Bures-SCST V3", f"Final terminal: `{terminal}`. V1 `{final['immutable_v1_terminal']}` and V2 `{final['immutable_v2_terminal']}` are immutable historical results."),
        "V2_FORENSIC_AUDIT.md": ("V2 forensic audit", f"Artifact-backed audit summary:\n\n```json\n{json.dumps(forensic, indent=2)}\n```\n\nA nearest-neighbor self-inclusion rate of {forensic.get('nearest_self_rate_mean')} is a validity limitation of V2. V2 S3 was not rerun."),
        "CODE_MAP.md": ("Code map", "`common.py` provides deterministic paths/bootstrap; `bures.py` provides cross-fitted covariance/Bures geometry; `source_v3.py` runs source methods; `source_gate.py` applies the preregistered gate; `freeze_protocol.py` writes locks; `official_confirmation.py` fails closed for unavailable architecture caches; finalization and validation are compact."),
        "SCIENTIFIC_RATIONALE.md": ("Scientific rationale", "V3 tests class-centered second-order subject style with a Bures map. It does not reinterpret V1/V2 negatives and does not claim nuisance invariance."),
        "METHOD.md": ("Method", "Global class centroids, equal class weighting, cross-fitted subject means/covariances, deterministic d/(n+d) shrinkage, eigenvalue floor, anchor-excluded k=5 validity gates, alpha {0.25,0.50,0.75,1.00}, detached hardness, and a 3+15 epoch EMA schedule."),
        "CROSS_FIT_AUDIT.md": ("Cross-fit audit", "Each anchor uses the opposite stable-hash half of its subject×class cell; fallback is pooled class-balanced statistics, never the anchor row."),
        "COVARIANCE_AUDIT.md": ("Covariance audit", "Covariances are symmetrized, count-shrunk toward pooled within-class covariance, and eigenvalue-floored before square roots/inverses."),
        "BURES_MAP_AUDIT.md": ("Bures map audit", "The symmetric eigendecomposition implements C_s^-1/2 (C_s^1/2 C_t C_s^1/2)^1/2 C_s^-1/2 with symmetrization after products."),
        "ANCHOR_EXCLUSION_AUDIT.md": ("Anchor exclusion audit", "All V3 local support queries exclude the original row id and exact duplicate rows. V2's self-inclusion is quantified in V2_FORENSIC_METRICS.csv."),
        "TARGET_AFFINITY_AUDIT.md": ("Target affinity audit", "Accepted candidates require both target same-class five-nearest distance and target Gaussian NLL to decrease; source-level CIs are reported in SOURCE_GATE.json."),
        "RANDOM_AFFINE_AUDIT.md": ("Random affine audit", f"Matching audit: {json.dumps(matching_audit, sort_keys=True)}. The pipeline fails closed if no matched rows or tolerances fail."),
        "SOURCE_DEVELOPMENT_REPORT.md": ("Source development report", f"Source gate terminal: `{gate.get('terminal_if_stop')}`. Selected recipe: `{json.dumps(selected)}`. Development uses OpenBMI sessions 1→2 and WBCIC S1→S2 only."),
        "ATCNET_OFFICIAL_REPORT.md": ("ATCNet-Official report", "Not run because the V3 source gate did not authorize S3." if not gate.get("source_gate_pass", False) else "Model-specific confirmation cache was unavailable; no result is reported."),
        "EEGNEX_CONFIRMATION_REPORT.md": ("EEGNeX confirmation report", "Not run because the V3 source gate did not authorize S3." if not gate.get("source_gate_pass", False) else "Model-specific confirmation cache was unavailable; no result is reported."),
        "CONTROL_REPORT.md": ("Control report", "ERM, Mixup, V2-ME-HardSCST, Manifold-Mixup, Bures-Uniform, Bures-HardRandom, and Bures-HardSCST share the subject-balanced sampler and optimization budget. Stopped controls are explicit not-run schemas."),
        "CLAIM_AUDIT.md": ("Claim audit", f"Strongest supported claim: {final['strongest_supported_claim']} Final terminal: `{terminal}`."),
        "ITERATION_LEDGER.md": ("Iteration ledger", "Implemented the preregistered Bures V3 geometry, repaired source aggregation baseline handling and anchor-excluded matching, then stopped/continued solely according to the fixed source gate."),
        "REPRODUCIBILITY.md": ("Reproducibility", "Run forensic_v2.py, source_v3.py --all --aggregate, source_gate.py, freeze_protocol.py, official_confirmation.py, finalize_v3.py, and validate_v3.py with the pinned environment. Runtime/cache/checkpoint/raw EEG artifacts are excluded."),
    }
    for name, (title, body) in docs.items(): _doc(name, title, body)
    c.write_json(c.EXP / "FINAL_REPORT.json", final); _doc("FINAL_REPORT.md", "Final report", json.dumps(final, indent=2))
    if not (c.RESULTS / "STATISTICS.json").is_file():
        c.write_json(c.RESULTS / "STATISTICS.json", {"source_units": 0, "source_grid_complete": False, "outer_or_sealed_opened": False, "terminal": terminal})
    print(json.dumps(final, indent=2))


if __name__ == "__main__": main()
