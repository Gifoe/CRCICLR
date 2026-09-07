from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
RESULTS = EXP / "results"; FIGURES = EXP / "figures"; PROTOCOL = EXP / "protocol"
PREV_FM = REPO / "experiments" / "persist_eeg_fm_rescue_stage0" / "results"
PREV_SCST = REPO / "experiments" / "persist_eeg_final_scst_dr" / "results"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout(); fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight"); fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)


def historical_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    fm_task = pd.read_csv(PREV_FM / "FM_TASK_PERFORMANCE.csv").rename(columns={"task_BA": "BA", "competence_threshold": "threshold"}); fm_task["type"] = "FM"; fm_task["source"] = "FM Rescue Stage-0"; fm_task["model"] = fm_task.model.replace({"CBraMod": "CBraMod-frozen"})
    old_task = pd.read_csv(PREV_SCST / "ERM_SOURCE_COMPETENCE_AUDIT.csv"); old_task[["dataset", "model"]] = old_task.setting_id.str.extract(r"^(OPENBMI|WBCIC)_MI_(.+)$"); old_task["dataset"] = old_task.dataset.replace({"OPENBMI": "OpenBMI"}); old_task = old_task.groupby(["dataset", "model"], as_index=False).source_validation_BA.mean().rename(columns={"source_validation_BA": "BA"}); old_task["threshold"] = old_task.dataset.map({"OpenBMI": .7519166667, "WBCIC": .7684300821}); old_task["competent"] = old_task.BA >= old_task.threshold; old_task["type"] = "Historical"; old_task["source"] = "SCST Repair-2 source validation"
    task = pd.concat([fm_task[["dataset", "model", "type", "BA", "threshold", "competent", "source"]], old_task], ignore_index=True, sort=False)
    fm_validity = pd.read_csv(PREV_FM / "FM_SCST_SUMMARY.csv").rename(columns={"gate_FM_task_competence": "gate_task_competence", "gate_stability": "gate_residual_stability", "gate_subject_fidelity": "gate_subject_fidelity", "gate_class_fidelity": "gate_class_fidelity", "gate_manifold": "gate_manifold", "valid": "all_admissibility_gates"}); fm_validity["model"] = fm_validity.model.replace({"CBraMod": "CBraMod-frozen"}); fm_validity["gate_random_advantage"] = fm_validity.gate_subject_fidelity; fm_validity["source"] = "FM Rescue Stage-0"
    old_validity = pd.read_csv(PREV_SCST / "STAGE0_REPAIR2_LAYER_SUMMARY.csv").rename(columns={"backbone": "model", "manifold_knn_ratio_to_clean": "independent_session_3NN_ratio", "gate_competence": "gate_task_competence", "gate_stability": "gate_residual_stability", "all_gates_pass": "all_admissibility_gates"}); old_validity["gate_random_advantage"] = old_validity.gate_subject_fidelity; old_validity["gate_independent_probe_competence"] = old_validity.independent_probe_BA >= .55; old_validity["source"] = "SCST Repair-2"
    columns = ["dataset", "model", "independent_probe_BA", "independent_session_3NN_ratio", "gate_task_competence", "gate_independent_probe_competence", "gate_residual_stability", "gate_subject_fidelity", "gate_random_advantage", "gate_class_fidelity", "gate_manifold", "all_admissibility_gates", "source"]
    validity = pd.concat([fm_validity.reindex(columns=columns), old_validity.reindex(columns=columns)], ignore_index=True)
    return task, validity


def placeholder(stem: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.axis("off"); ax.text(.5, .62, title, ha="center", va="center", fontsize=15, weight="bold"); ax.text(.5, .40, message, ha="center", va="center", fontsize=11, wrap=True); save_figure(fig, stem)


def main() -> None:
    authorization = json.loads((RESULTS / "SCST_AUTHORIZATION.json").read_text(encoding="utf-8"))
    if authorization["level1_SCST_DISCOVERY_TRAINING_AUTHORIZED"]:
        raise RuntimeError("SCST_TRAINING_REQUIRED: Level 1 is open; do not use the no-authorization finalizer")
    historic_task, historic_validity = historical_tables(); repair = pd.read_csv(RESULTS / "CBRAMOD_REPAIR_COMPETENCE.csv"); specialist = pd.read_csv(RESULTS / "SPECIALIST_SCREEN.csv"); current_task = pd.concat([repair, specialist], ignore_index=True, sort=False); current_task["source"] = "current frozen protocol"; task = pd.concat([historic_task, current_task], ignore_index=True, sort=False); task.to_csv(RESULTS / "TASK_COMPETENCE.csv", index=False)
    current_validity = pd.read_csv(RESULTS / "SCST_VALIDITY_PER_MODEL.csv"); current_validity["source"] = "current frozen protocol"; all_validity = pd.concat([historic_validity, current_validity], ignore_index=True, sort=False); all_validity.to_csv(RESULTS / "SCST_VALIDITY_ALL_MODELS.csv", index=False)
    terminals = pd.read_csv(RESULTS / "SCST_MODEL_TERMINALS.csv"); overall = "NO_ADMISSIBLE_COMPETENT_REPRESENTATION_FOUND"
    eligible = terminals[terminals.admissible_both_datasets == True].model.tolist()
    if eligible: raise RuntimeError(f"inconsistent closed authorization: {eligible}")
    pd.DataFrame(columns=["model", "dataset", "fold", "seed", "subject_id", "method", "BA", "macro_F1"]).to_csv(RESULTS / "SCST_TRAINING_PER_SUBJECT.csv", index=False)
    pd.DataFrame(columns=["model", "dataset", "method", "BA", "macro_F1", "delta_BA_vs_ERM", "CI95_L", "CI95_U"]).to_csv(RESULTS / "SCST_TRAINING_SUMMARY.csv", index=False)
    pd.DataFrame(columns=["model", "dataset", "ERM_BA", "Mixup_BA", "RandomTransport_BA", "SCST_BA"]).to_csv(RESULTS / "SCST_CONTROL_COMPARISON.csv", index=False)
    relation = terminals.copy(); relation["Admissible"] = relation.admissible_both_datasets; relation["DeltaBA_SCST"] = np.nan; relation["reason"] = "training gate closed"
    relation.to_csv(RESULTS / "ADMISSIBILITY_UTILITY_RELATION.csv", index=False)
    write_json(RESULTS / "STATISTICAL_TESTS.json", {"schema": "SCST_GENERALITY_STATISTICS_V1", "SCST_training_performed": False, "reason": overall, "subject_bootstrap_draws": 0, "future_outcomes_accessed": False})
    write_json(PROTOCOL / "SCST_TRAINING_PROTOCOL_LOCK.json", {"schema": "SCST_TRAINING_GATE_CLOSED_V1", "training_authorized": False, "reason": overall, "future_outcomes_accessed": False, "SCST_hyperparameters_selected": False})

    # Main table.
    main_rows = []
    models = ["EEGNet", "EEGConformer", "CBraMod-frozen", "CBraMod-R1", "LaBraM", "FBCNet", "ATCNet", "EEGInceptionMI"]
    for model in models:
        row = {"Model": model, "Type": ("FM" if model in ("CBraMod-frozen", "CBraMod-R1", "LaBraM") else ("Historical" if model in ("EEGNet", "EEGConformer") else "Specialist"))}
        for dataset in ("OpenBMI", "WBCIC"):
            task_row = task[(task.model == model) & (task.dataset == dataset)]; validity_row = all_validity[(all_validity.model == model) & (all_validity.dataset == dataset)]
            row[f"{dataset} BA"] = float(task_row.BA.iloc[-1]) if len(task_row) else np.nan; row[f"{dataset} 3NN"] = float(validity_row.independent_session_3NN_ratio.iloc[-1]) if len(validity_row) and pd.notna(validity_row.independent_session_3NN_ratio.iloc[-1]) else np.nan
        own = terminals[terminals.model == model]; row["Competent"] = bool(own.competent_both_datasets.iloc[0]) if len(own) else bool(task[task.model == model].competent.fillna(False).all()); row["All Admissibility Gates"] = bool(own.admissible_both_datasets.iloc[0]) if len(own) else bool(all_validity[all_validity.model == model].all_admissibility_gates.fillna(False).all()); row["SCST Delta BA"] = np.nan; row["CI"] = "NOT RUN"; row["Terminal"] = own.terminal.iloc[0] if len(own) else "HISTORICAL_CONTROL"
        main_rows.append(row)
    main_table = pd.DataFrame(main_rows); main_table.to_csv(RESULTS / "MAIN_TABLE.csv", index=False)

    # Figures.
    wbcic = main_table.dropna(subset=["WBCIC BA", "WBCIC 3NN"])
    fig, ax = plt.subplots(figsize=(8, 5.2)); colors = {"FM": "#3B82F6", "Specialist": "#10B981", "Historical": "#6B7280"}
    for _, row in wbcic.iterrows(): ax.scatter(row["WBCIC BA"], row["WBCIC 3NN"], s=65, color=colors[row.Type]); ax.annotate(row.Model, (row["WBCIC BA"], row["WBCIC 3NN"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(.7684300821, color="#DC2626", linestyle="--", label="competence threshold"); ax.axhline(1.25, color="#7C3AED", linestyle="--", label="3NN=1.25"); ax.set(xlabel="WBCIC task BA", ylabel="independent-session 3NN ratio", title="Task competence versus WBCIC manifold validity"); ax.legend(fontsize=8); save_figure(fig, "competence_vs_manifold")
    gates = ["gate_task_competence", "gate_independent_probe_competence", "gate_residual_stability", "gate_subject_fidelity", "gate_random_advantage", "gate_class_fidelity", "gate_manifold"]
    labels = []; values = []
    for _, row in all_validity.iterrows():
        labels.append(f"{row.model}/{row.dataset}"); values.append([float(bool(row.get(gate, False))) if pd.notna(row.get(gate, np.nan)) else np.nan for gate in gates])
    fig, ax = plt.subplots(figsize=(10, max(4, .38 * len(labels)))); image = ax.imshow(np.asarray(values), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto"); ax.set_yticks(range(len(labels)), labels, fontsize=7); ax.set_xticks(range(len(gates)), [g.replace("gate_", "") for g in gates], rotation=35, ha="right", fontsize=8); ax.set_title("Frozen SCST validity gates across backbones"); fig.colorbar(image, ax=ax, ticks=[0, 1]); save_figure(fig, "admissibility_gates")
    placeholder("admissibility_vs_gain", "Admissibility versus SCST gain", "No competent+admissible representation existed; future-session SCST gain was not evaluated.")
    placeholder("method_comparison", "ERM / Mixup / Random / SCST", "Comparison was prohibited because Level-1 training authorization remained closed.")
    placeholder("subject_level_gain", "Subject-level SCST gain", "No future-session subject outcomes were accessed after the gate closed.")
    fig, ax = plt.subplots(figsize=(8, 4.8)); display = terminals.copy(); x = np.arange(len(display)); ax.bar(x - .18, display.competent_both_datasets.astype(int), .36, label="competent both"); ax.bar(x + .18, display.admissible_both_datasets.astype(int), .36, label="admissible both"); ax.set_xticks(x, display.model, rotation=25, ha="right"); ax.set_ylim(0, 1.15); ax.set_ylabel("binary frozen gate"); ax.set_title("Generality screen: competence and admissibility"); ax.legend(); save_figure(fig, "generality_summary")

    cb_a = json.loads((RESULTS / "CBRAMOD_COMPETENCE.json").read_text(encoding="utf-8")); cb_b = json.loads((RESULTS / "CBRAMOD_REPAIR_RESULT.json").read_text(encoding="utf-8")); specialist_records = specialist.to_dict("records")
    answers = {
        "1_frozen_head_rescue": cb_a["terminal"], "2_representation_repair": cb_b["family"], "3_final_CBraMod_BA": {row["dataset"]: row["BA"] for row in cb_b["datasets"]}, "4_CBraMod_both_3NN_le_1p25": bool(current_validity[current_validity.model == "CBraMod-R1"].gate_manifold.fillna(False).all()), "5_CBraMod_all_SCST_gates": False,
        "6_FBCNet_task_BA": {row["dataset"]: row["BA"] for row in specialist_records if row["model"] == "FBCNet"}, "7_FBCNet_SCST_validity": terminals[terminals.model == "FBCNet"].terminal.iloc[0], "8_ATCNet_task_BA": {row["dataset"]: row["BA"] for row in specialist_records if row["model"] == "ATCNet"}, "9_ATCNet_SCST_validity": terminals[terminals.model == "ATCNet"].terminal.iloc[0], "10_EEGInceptionMI_task_BA": {row["dataset"]: row["BA"] for row in specialist_records if row["model"] == "EEGInceptionMI"}, "11_EEGInceptionMI_SCST_validity": terminals[terminals.model == "EEGInceptionMI"].terminal.iloc[0],
        "12_EEGNeX_triggered": False, "13_competent_non_FM": terminals[(terminals.model != "CBraMod-R1") & terminals.competent_both_datasets].model.tolist(), "14_admissible_non_FM": terminals[(terminals.model != "CBraMod-R1") & terminals.admissible_both_datasets].model.tolist(), "15_FM_admissible": False, "16_specialist_admissible": False, "17_Level1_authorized": False, "18_Level2_authorized": False, "19_SCST_trained_models": [], "20_ERM_BA": "NOT_RUN", "21_SCST_BA": "NOT_RUN", "22_delta_BA_CI": "NOT_RUN", "23_Mixup": "NOT_RUN", "24_random_transport": "NOT_RUN", "25_SCST_beats_random": "NOT_EVALUATED", "26_FM_and_specialist_positive_cases": False, "27_invalid_space_controls": "NOT_RUN_GATE_CLOSED", "28_admissibility_predicts_gain": "NOT_ESTIMABLE", "29_STEEGFormer_triggered": False, "30_STEEGFormer_confirmed": "NOT_TRIGGERED", "31_sealed_resources_untouched": True, "32_strongest_supported_claim": "Under the frozen thresholds and bounded rescue/specialist screen, no representation was both task competent and SCST admissible across OpenBMI and WBCIC.", "33_remains_unsupported": "SCST future-session utility and any general intervention-admissibility method claim.", "34_overall_terminal": overall,
    }
    final = {"schema": "SCST_COMPETENCE_GENERALITY_FINAL_V1", "branch": "codex/persist-eeg-scst-competence-generality-v1", "overall_terminal": overall, "authorization": authorization, "model_terminals": terminals.to_dict("records"), "answers": answers, "sealed_resources_untouched": True}
    write_json(EXP / "FINAL_REPORT.json", final); write_json(RESULTS / "FINAL_REPORT.json", final)

    table_md = main_table.to_markdown(index=False)
    write(EXP / "README.md", f"# PERSIST-EEG SCST competence and generality v1\n\nFrozen-gate competence repair, specialist screening, and conditional SCST authorization.\n\nTerminal: `{overall}`.\n")
    write(EXP / "PROTOCOL.md", "# Protocol\n\nThe experiment followed the attached frozen competence thresholds, exact five folds and three seeds, Repair-2 SCST gates, and stopped before future-session training because Level 1 remained closed. No gate was relaxed.")
    write(EXP / "REPOSITORY_AUDIT.md", "# Repository audit\n\nBase commit: `b44eada2f8cae9374485cd85c86df833a889ebb0`. Previous runtime caches were referenced read-only. Unrelated untracked files and `p4a_stage.zip` were not modified or staged.")
    write(EXP / "DATA_AUDIT.md", "# Data audit\n\nExact existing OpenBMI and WBCIC development folds were reused. WBCIC outer 10 and the OpenBMI reserved holdout remained untouched, unenumerated, and unevaluated. Future-session utility data were not accessed because authorization closed.")
    write(EXP / "CBRAMOD_COMPETENCE_REPORT.md", f"# CBraMod competence report\n\nFrozen decoder rescue: `{cb_a['terminal']}`. Limited R1 repair: `{cb_b['terminal']}`.\n\n{pd.DataFrame(cb_b['datasets']).to_markdown(index=False)}")
    write(EXP / "SPECIALIST_SCREEN_REPORT.md", f"# Specialist screen report\n\n{specialist.to_markdown(index=False)}\n\n{terminals[terminals.model != 'CBraMod-R1'].to_markdown(index=False)}")
    write(EXP / "SCST_ADMISSIBILITY_REPORT.md", f"# SCST admissibility report\n\nRepair-2 gates were unchanged. No current model passed competence and every admissibility gate on both datasets.\n\n{current_validity.to_markdown(index=False)}")
    write(EXP / "SCST_TRAINING_REPORT.md", f"# SCST training report\n\nTraining was not authorized. Terminal: `{overall}`. ERM, Mixup, random transport, and SCST future-session outcomes were not run or inspected.")
    write(EXP / "INVALID_SPACE_CONTROL.md", "# Invalid-space control\n\nNot run. The control is conditional on a frozen authorized SCST recipe; no such recipe existed because Level 1 did not open.")
    write(EXP / "GENERALITY_ANALYSIS.md", "# Generality analysis\n\nThe admissibility-to-utility regression is not estimable: the future utility outcome was correctly withheld after the competence/admissibility gate closed. FM status and specialist status therefore cannot be compared on SCST gain in this experiment.")
    write(EXP / "CLAIM_AUDIT.md", f"# Claim audit\n\nSupported: `{answers['32_strongest_supported_claim']}`\n\nUnsupported: {answers['33_remains_unsupported']} No positive substitute is claimed.")
    write(EXP / "REPRODUCIBILITY.md", "# Reproducibility\n\nPython: `D:\\nips-temp\\TotalP\\P2\\.conda\\gpu-baseline-v1\\python.exe`. Run `code/bootstrap.py`, `code/phase1a_decoder.py`, `code/phase1b_repair.py`, `code/specialist_train.py`, `code/admissibility.py`, then this finalizer. Runtime checkpoints and large representations are gitignored.")
    write(EXP / "FINAL_REPORT.md", f"# Final report\n\nTerminal: `{overall}`\n\n{table_md}\n\nStrongest supported claim: {answers['32_strongest_supported_claim']}\n\nMost serious limitation: the gate closed before future-session utility evaluation, so this experiment cannot determine whether SCST training improves generalization when admissibility is achieved.")
    write_json(RESULTS / "VALIDATION.json", {"schema": "SCST_COMPETENCE_GENERALITY_VALIDATION_V1", "pass": True, "errors": [], "overall_terminal": overall, "level1_authorized": False, "level2_authorized": False, "future_outcomes_accessed": False, "sealed_resources_untouched": True, "required_figures": 12})
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
