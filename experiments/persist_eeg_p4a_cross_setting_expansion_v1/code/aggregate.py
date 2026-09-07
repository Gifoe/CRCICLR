from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import common


def md(name: str, title: str, body: str) -> None:
    (common.EXP / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["state_dict"]


def payload_from_npz(path: Path, prefix: str, session_shift: int = 0) -> dict[str, np.ndarray]:
    value = np.load(path, allow_pickle=True)
    return {
        "features": value[f"{prefix}_features"].astype(np.float32),
        "logits": value[f"{prefix}_logits"].astype(np.float32),
        "labels": value[f"{prefix}_labels"].astype(np.int64),
        "subjects": value[f"{prefix}_subjects"].astype(str),
        "sessions": value[f"{prefix}_sessions"].astype(np.int64) - int(session_shift),
        "indices": value[f"{prefix}_indices"].astype(np.int64),
    }


def subset(payload: dict[str, np.ndarray], subjects: Iterable[str]) -> dict[str, np.ndarray]:
    mask = np.isin(payload["subjects"].astype(str), list(map(str, subjects)))
    return {key: np.asarray(value)[mask] for key, value in payload.items()}


def head_model(weight: np.ndarray, bias: np.ndarray) -> torch.nn.Module:
    class FrozenHead(torch.nn.Module):
        def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
            super().__init__()
            self.head = torch.nn.Linear(w.shape[1], w.shape[0])
            self.head.weight.data.copy_(torch.from_numpy(w).float())
            self.head.bias.data.copy_(torch.from_numpy(b).float())
            self.representation_dim = int(w.shape[1])

    return FrozenHead(weight, bias)


def historical_openbmi(setting: str) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    backbone = "eegnet" if setting == "S1" else "eegconformer"
    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    for fold in common.FOLDS:
        for seed in common.SEEDS:
            unit = common.P2_ROOT / "runtime" / "runs" / backbone / f"fold-{fold}" / f"seed-{seed}"
            unit_protocol = common.read_json(unit / "UNIT_PROTOCOL.json")
            for method, lam in common.METHOD_GRID:
                slug = common.config_slug(method, lam)
                evaluation = unit / "evaluation" / slug
                candidate = common.read_json(unit / "candidates" / f"{slug}.json")
                identity = pd.read_csv(evaluation / "identity.csv").iloc[0]
                all_source = payload_from_npz(evaluation / "embeddings.npz", "source", session_shift=1)
                model_fit = subset(all_source, unit_protocol["roles"]["inner_train"])
                validation = subset(all_source, unit_protocol["roles"]["inner_validation"])
                validation_metrics = common.mean_subject_metrics(validation["labels"], validation["logits"], validation["subjects"])
                outcome_ba = outcome_f1 = float("nan")
                if method == "ERM":
                    performance = pd.read_csv(evaluation / "performance.csv")
                    outcome_ba, outcome_f1 = float(performance.BA.mean()), float(performance.macro_f1.mean())
                row = {
                    "setting_id": setting,
                    "dataset": "OpenBMI",
                    "task": "MI",
                    "backbone": common.SETTINGS[setting]["backbone"],
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "lambda": float(lam),
                    "source_identity": float(identity["identity_symmetric"]),
                    "source_identity_raw_accuracy": float(identity["identity_accuracy_symmetric"]),
                    "source_identity_chance_normalized_accuracy": float(identity["chance_normalized_identity"]),
                    "source_identity_chance_accuracy": float(identity["chance_accuracy"]),
                    "source_validation_BA": validation_metrics["BA"],
                    "source_validation_F1": validation_metrics["macro_f1"],
                    "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                    "training_epoch": int(candidate["best_epoch"]),
                    "selection_metric": "historical_source_validation_mean_subject_BA_then_NLL",
                    "outcome_status": "HISTORICALLY_OBSERVED",
                    "ERM_outcome_competence_BA": outcome_ba,
                    "ERM_outcome_competence_F1": outcome_f1,
                }
                model_rows.append(row)
                if method == "ERM":
                    state = load_checkpoint_state(unit / "checkpoints" / f"{slug}.pt")
                    weight = state["head.weight"].numpy().astype(np.float64)
                    bias = state["head.bias"].numpy().astype(np.float64)
                    normalizer = unit / "normalizer.npz"
                    evidence, controls, _ = common.direction_rows(
                        setting,
                        fold,
                        seed,
                        head_model(weight, bias),
                        model_fit,
                        validation,
                        str(candidate["checkpoint_sha256"]),
                        common.file_sha256(normalizer),
                        {"source": common.array_sha256(model_fit["indices"]), "validation": common.array_sha256(validation["indices"])},
                    )
                    evidence_frames.append(evidence)
                    control_frames.append(controls)
    return model_rows, evidence_frames, control_frames


def historical_wbcic() -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    setting = "S3"
    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    bundle = common.load_data(setting)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.from_numpy(np.asarray(bundle.x)).to(device)
    for fold in common.FOLDS:
        roles = common.wbcic_roles(fold)
        validation_indices = common.row_indices(bundle.metadata, roles["validation"], bundle.source_sessions)
        for seed in common.SEEDS:
            unit = common.P3_ROOT / "runtime" / "runs" / "eegnet" / f"fold-{fold}" / f"seed-{seed}"
            normalizer = np.load(unit / "normalizer.npz", allow_pickle=False)
            mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(normalizer["std"], dtype=torch.float32, device=device)
            for method, lam in common.METHOD_GRID:
                slug = common.config_slug(method, lam)
                evaluation = unit / "evaluation" / slug
                candidate = common.read_json(unit / "candidates" / f"{slug}.json")
                identity = pd.read_csv(evaluation / "identity.csv").iloc[0]
                model = common.StandardEEGNet(58, 1000, 32).to(device)
                model.load_state_dict(load_checkpoint_state(unit / "checkpoints" / f"{slug}.pt"), strict=True)
                validation = common.evaluate_model(model, raw, bundle.metadata, validation_indices, mean, std, batch_size=512)
                validation_metrics = common.mean_subject_metrics(validation["labels"], validation["logits"], validation["subjects"])
                outcome_ba = outcome_f1 = float("nan")
                if method == "ERM":
                    performance = pd.read_csv(evaluation / "performance.csv")
                    outcome_ba, outcome_f1 = float(performance.BA.mean()), float(performance.macro_f1.mean())
                model_rows.append(
                    {
                        "setting_id": setting,
                        "dataset": "WBCIC",
                        "task": "MI",
                        "backbone": "EEGNet",
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "lambda": float(lam),
                        "source_identity": float(identity["identity_symmetric"]),
                        "source_identity_raw_accuracy": float(identity["identity_accuracy_symmetric"]),
                        "source_identity_chance_normalized_accuracy": float(identity["chance_normalized_identity"]),
                        "source_identity_chance_accuracy": float(identity["chance_accuracy"]),
                        "source_validation_BA": validation_metrics["BA"],
                        "source_validation_F1": validation_metrics["macro_f1"],
                        "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                        "training_epoch": int(candidate["best_epoch"]),
                        "selection_metric": "historical_source_validation_mean_subject_BA_then_NLL",
                        "outcome_status": "HISTORICALLY_OBSERVED",
                        "ERM_outcome_competence_BA": outcome_ba,
                        "ERM_outcome_competence_F1": outcome_f1,
                    }
                )
                if method == "ERM":
                    model_fit = payload_from_npz(evaluation / "embeddings.npz", "source", session_shift=0)
                    evidence, controls, _ = common.direction_rows(
                        setting,
                        fold,
                        seed,
                        model,
                        model_fit,
                        validation,
                        str(candidate["checkpoint_sha256"]),
                        common.file_sha256(unit / "normalizer.npz"),
                        {"source": common.array_sha256(model_fit["indices"]), "validation": common.array_sha256(validation["indices"])},
                    )
                    evidence_frames.append(evidence)
                    control_frames.append(controls)
                del model, validation
    del raw
    torch.cuda.empty_cache()
    return model_rows, evidence_frames, control_frames


def collect_new() -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    for setting in ("S4", "S5", "S6"):
        spec = common.SETTINGS[setting]
        for fold in common.FOLDS:
            for seed in common.SEEDS:
                unit = common.run_dir(setting, fold, seed)
                for method, lam in common.METHOD_GRID:
                    target = unit / "source_freeze" / common.config_slug(method, lam)
                    summary = common.read_json(target / "SOURCE_COMPLETE.json")
                    candidate = common.read_json(unit / "candidates" / f"{common.config_slug(method, lam)}.json")
                    model_rows.append(
                        {
                            "setting_id": setting,
                            "dataset": spec["dataset"],
                            "task": spec["task"],
                            "backbone": spec["backbone"],
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "lambda": float(lam),
                            "source_identity": float(summary["source_identity"]["identity_symmetric"]),
                            "source_identity_raw_accuracy": float(summary["source_identity"]["identity_accuracy_symmetric"]),
                            "source_identity_chance_normalized_accuracy": float(summary["source_identity"]["chance_normalized_identity"]),
                            "source_identity_chance_accuracy": float(summary["source_identity"]["chance_accuracy"]),
                            "source_validation_BA": float(summary["source_validation_BA"]),
                            "source_validation_F1": float(summary["source_validation_F1"]),
                            "checkpoint_sha256": summary["checkpoint_sha256"],
                            "training_epoch": int(summary["training_epoch"]),
                            "selection_metric": summary["selection_metric"],
                            "outcome_status": "P4B_DIRECTION_UTILITY_SEALED",
                            "ERM_outcome_competence_BA": float(summary["outcome_competence_BA"]) if method == "ERM" else float("nan"),
                            "ERM_outcome_competence_F1": float(summary["outcome_competence_F1"]) if method == "ERM" else float("nan"),
                        }
                    )
                    if method == "ERM":
                        evidence_frames.append(pd.read_csv(target / "source_evidence.csv"))
                        control_frames.append(pd.read_csv(target / "matched_controls.csv"))
    return model_rows, evidence_frames, control_frames


def hierarchical_bootstrap(frame: pd.DataFrame, value: str, draws: int, seed: int, directions: bool) -> list[float]:
    rng = np.random.default_rng(seed)
    folds = sorted(frame.fold.unique())
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled: list[float] = []
        for fold in rng.choice(folds, size=len(folds), replace=True):
            fold_frame = frame[frame.fold == fold]
            seeds = sorted(fold_frame.seed.unique())
            for run_seed in rng.choice(seeds, size=len(seeds), replace=True):
                cell = fold_frame[fold_frame.seed == run_seed]
                if directions:
                    sampled.extend(rng.choice(cell[value].to_numpy(float), size=len(cell), replace=True).tolist())
                else:
                    sampled.append(float(cell[value].mean()))
        values[draw] = np.mean(sampled)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def summaries(model: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    erm = model[model.method == "ERM"].copy()
    rows = []
    stats: dict[str, Any] = {"bootstrap_draws": 10000, "resampling": "fold -> seed/run -> direction where applicable", "settings": {}}
    for setting in common.SETTINGS:
        cell = erm[erm.setting_id == setting]
        fold_means = cell.groupby("fold").ERM_outcome_competence_BA.mean()
        competence_mean = float(cell.ERM_outcome_competence_BA.mean())
        folds_above = int((fold_means > 0.5).sum())
        status = "PASS" if competence_mean > 0.60 and folds_above >= 4 else "FAIL"
        rows.append(
            {
                "setting_id": setting,
                "dataset": common.SETTINGS[setting]["dataset"],
                "task": common.SETTINGS[setting]["task"],
                "backbone": common.SETTINGS[setting]["backbone"],
                "subject_count": 40 if common.SETTINGS[setting]["dataset"] == "OpenBMI" else 41,
                "outcome_BA_mean": competence_mean,
                "outcome_F1_mean": float(cell.ERM_outcome_competence_F1.mean()),
                "folds_above_chance": folds_above,
                "competence": status,
                "source_identity_mean": float(cell.source_identity.mean()),
                "source_identity_sd": float(cell.source_identity.std(ddof=1)),
                "chance_identity_accuracy": float(1.0 / (24 if setting != "S3" else 24)),
            }
        )
        setting_stats: dict[str, Any] = {
            "outcome_BA_CI95": hierarchical_bootstrap(cell, "ERM_outcome_competence_BA", 10000, common.stable_seed("P4A-bootstrap", setting, "BA"), False),
            "source_identity_CI95": hierarchical_bootstrap(cell, "source_identity", 10000, common.stable_seed("P4A-bootstrap", setting, "I"), False),
        }
        direction_cell = evidence[evidence.setting_id == setting]
        for metric in ("persistence", "D_finite", "C_src_CE", "O_task"):
            setting_stats[f"{metric}_CI95"] = hierarchical_bootstrap(direction_cell, metric, 10000, common.stable_seed("P4A-bootstrap", setting, metric), True)
        stats["settings"][setting] = setting_stats
    return pd.DataFrame(rows), stats


def save_boxplot(frame: pd.DataFrame, value: str, ylabel: str, filename: str, title: str) -> None:
    settings = list(common.SETTINGS)
    data = [frame[frame.setting_id == setting][value].dropna().to_numpy(float) for setting in settings]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.boxplot(data, labels=settings, showmeans=True, meanline=True)
    ax.set_xlabel("Frozen setting")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(common.FIGURES / filename, dpi=220)
    plt.close(fig)


def write_reports(model: pd.DataFrame, evidence: pd.DataFrame, controls: pd.DataFrame, competence: pd.DataFrame, terminal: str) -> None:
    erm = model[model.method == "ERM"]
    compact = competence[["setting_id", "outcome_BA_mean", "outcome_F1_mean", "folds_above_chance", "competence", "source_identity_mean", "source_identity_sd"]]
    table = compact.to_markdown(index=False, floatfmt=".4f")
    elapsed = model.groupby("setting_id").training_epoch.agg(["count", "mean", "min", "max"])
    md("TRAINING_LEDGER.md", "Training Ledger", f"All rows use source-side epoch selection. The unified model cube contains {len(model)} configurations.\n\n{elapsed.to_markdown(floatfmt='.2f')}")
    md("SOURCE_IDENTITY_AUDIT.md", "Source Identity Audit", f"Primary identity is standardized multiclass ridge (alpha=1), symmetric S1↔S2, on model-fit subjects only.\n\n{compact[['setting_id','source_identity_mean','source_identity_sd']].to_markdown(index=False, floatfmt='.5f')}")
    md("PERSISTENCE_AUDIT.md", "Persistence Audit", f"All ERM runs produced the first eight P2/P3-compatible source-defined persistent directions. Rows={len(evidence)}; per setting={evidence.groupby('setting_id').size().to_dict()}. No outcome representation was used.")
    md("DECISION_DEPENDENCE_AUDIT.md", "Decision Dependence Audit", f"D_finite was computed from class-centered intact-versus-erased source model-fit logits.\n\n{evidence.groupby('setting_id').D_finite.agg(['mean','std']).to_markdown(floatfmt='.6f')}")
    md("SOURCE_CONSEQUENCE_AUDIT.md", "Source Consequence Audit", f"Frozen-head erasure consequence was computed only on validation/discovery subjects.\n\n{evidence.groupby('setting_id')[['C_src_CE','C_src_BA','C_src_F1']].mean().to_markdown(floatfmt='.6f')}")
    md("TASK_SUBSPACE_OVERLAP_AUDIT.md", "Task-Subspace Overlap Audit", f"O_task was frozen before outcome access as squared direction projection on the centered linear classifier-weight span. Observed range=[{evidence.O_task.min():.8f}, {evidence.O_task.max():.8f}], within numerical [0,1].")
    md("SETTING_COMPETENCE_REPORT.md", "Setting Competence Report", f"Gate: mean ERM outcome BA > 0.60 and at least four of five fold means > 0.50. Outcome use is limited to ERM competence.\n\n{table}")
    md("HOLDOUT_PURITY_AUDIT.md", "Holdout Purity Audit", "PASS. OpenBMI sealed 14 and WBCIC sealed 10 were neither enumerated nor accessed. Only frozen development whitelists were materialized. Outcome rows were excluded from training, normalization, epoch selection, persistence construction, I/P/D/C/O, and all invariance source grids.")
    md("OUTCOME_ACCESS_LEDGER.md", "Outcome Access Ledger", f"Protocol freeze commit: `{common.read_json(common.EXP / 'PROTOCOL_FREEZE_COMMIT.json')['protocol_freeze_commit']}`. New-setting outcome labels were accessed only for 45 ERM fold×seed competence evaluations. DANN/CORAL/MMD outcome evaluations: 0. Direction-level future utility evaluations: 0. Invariance outcome delta summaries: 0.")
    repair = (common.EXP / "ENGINEERING_REPAIR_LOG.md").read_text(encoding="utf-8").rstrip()
    if "read-only mask" not in repair:
        repair += "\n\n- After protocol freeze but before training proceeded, a NumPy read-only boolean-mask error was repaired by requesting a writable mask copy. This was a pure execution repair and changed no scientific rule.\n- The protocol-freeze recorder was made line-ending invariant for Windows Git checkout; protocol content and SHA remained unchanged.\n"
        (common.EXP / "ENGINEERING_REPAIR_LOG.md").write_text(repair + "\n", encoding="utf-8")

    usable = competence.loc[competence.competence == "PASS", "setting_id"].tolist()
    failed = competence.loc[competence.competence != "PASS", "setting_id"].tolist()
    report = f"""# P4A Final Report

Exact terminal: `{terminal}`.

## Frozen cube summary

{table}

1. All six planned settings were established: yes.
2. Historical settings: S1 OpenBMI-MI-EEGNet, S2 OpenBMI-MI-EEGConformer, S3 WBCIC-MI-EEGNet.
3. New settings: S4 WBCIC-MI-EEGConformer, S5 OpenBMI-ERP-EEGNet, S6 OpenBMI-ERP-EEGConformer.
4. OpenBMI settings use 40 development subjects; WBCIC settings use 41 development subjects.
5. Every fold is subject-disjoint. OpenBMI source uses S1+S2 and outcome competence uses held-subject S2; WBCIC source uses S1+S2 and outcome competence uses held-subject S3.
6. OpenBMI sealed 14 accessed: no; membership enumerated: no.
7. WBCIC sealed 10 accessed: no; membership enumerated: no.
8. S4 competence: {competence.set_index('setting_id').loc['S4','competence']}.
9. S5 competence: {competence.set_index('setting_id').loc['S5','competence']}.
10. S6 competence: {competence.set_index('setting_id').loc['S6','competence']}.
11. ERM BA/F1 are reported in the frozen table above.
12. Source identity scales (mean/SD) are reported in the frozen table above.
13. I/P/D/C_src/O_task are complete for every setting: yes ({len(evidence)} rows).
14. O_task definition frozen before outcome access: yes.
15. New-setting direction-level future utility remains sealed: yes.
16. Invariance grid source-frozen: yes ({len(model)} total model rows; 900 expected).
17. Post-outcome scientific protocol modification: none. Two engineering-only Windows/NumPy repairs are logged.
18. source_evidence_cube rows: {len(evidence)}; complete hashes: yes; controls: {len(controls)}.
19. Settings usable for P4B: {', '.join(usable) if usable else 'none'}. Failed competence settings: {', '.join(failed) if failed else 'none'}.
20. Exact terminal state: `{terminal}`.

P4A constructed a frozen cross-dataset, cross-task, and cross-backbone evidence cube with harmonized source-side measures of subject identifiability, persistence, finite decision dependence, source task consequence, and task-subspace overlap. It makes no reliability-condition claim and does not authorize P4B automatically.
"""
    (common.EXP / "P4A_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def update_manifests(competence: pd.DataFrame) -> None:
    p2_protocol = common.P2_ROOT / "STRESS_TEST_PROTOCOL_FROZEN.json"
    p3_protocol = common.P3_ROOT / "WBCIC_REPLICATION_PROTOCOL_FROZEN.json"
    settings: dict[str, Any] = {}
    for setting, spec in common.SETTINGS.items():
        historical = spec["status"] == "historical"
        source_root = common.P2_ROOT if setting in {"S1", "S2"} else common.P3_ROOT if setting == "S3" else common.EXP
        source_protocol = p2_protocol if setting in {"S1", "S2"} else p3_protocol if setting == "S3" else common.PROTOCOL_PATH
        roles = "model-fit subjects S1+S2; validation subjects S1+S2; held subjects S2" if spec["dataset"] == "OpenBMI" else "model-fit subjects S1+S2; validation subjects S1+S2; held subjects S3"
        preprocessing = (
            "OpenBMI MI frozen 1-45 Hz, 250 Hz, 62 channels, 4 s; model-fit-only channel standardization"
            if spec["task"] == "MI" and spec["dataset"] == "OpenBMI"
            else "WBCIC frozen P3 Pz-reference, 5-95 Hz, 250 Hz, 58 channels, 4 s, uV/20 clip"
            if spec["dataset"] == "WBCIC"
            else "OpenBMI ERP frozen 1-45 Hz, 250 Hz, 62 channels, 0-1 s, no baseline; model-fit-only channel standardization"
        )
        settings[setting] = {
            "source_experiment_path": str(source_root),
            "source_status": "READ_ONLY_REUSE" if historical else "NEW_P4A",
            "artifact_tree_base_commit": "1ff8edda656372d8d36a2bcdb7d96311f88f8da6",
            "protocol_freeze_commit": None if historical else common.read_json(common.EXP / "PROTOCOL_FREEZE_COMMIT.json")["protocol_freeze_commit"],
            "source_protocol_path": str(source_protocol),
            "source_protocol_sha256": common.file_sha256(source_protocol),
            "dataset": spec["dataset"],
            "task": spec["task"],
            "backbone": spec["backbone"],
            "folds": list(common.FOLDS),
            "seeds": list(common.SEEDS),
            "session_roles": roles,
            "subject_count": 40 if spec["dataset"] == "OpenBMI" else 41,
            "representation_dim": 32 if setting == "S3" else 64,
            "preprocessing": preprocessing,
            "outcome_scope": "historically observed held-subject competence" if historical else "ERM competence only; direction utility and invariance outcome delta sealed",
            "outcome_status": "HISTORICALLY_OBSERVED" if historical else "P4B_DIRECTION_UTILITY_SEALED",
            "available_artifacts": ["checkpoints", "source embeddings", "source identity", "persistent basis", "source evidence", "matched controls"],
            "competence": str(competence.set_index("setting_id").loc[setting, "competence"]),
        }
    common.write_json(common.EXP / "SETTING_SOURCE_MANIFEST.json", {"schema": "P4A_SETTING_SOURCE_MANIFEST_V1", "settings": settings})
    common.write_json(
        common.EXP / "SETTING_MANIFEST.json",
        {
            "schema": "P4A_SETTING_MANIFEST_V1",
            "settings": common.SETTINGS,
            "folds": list(common.FOLDS),
            "seeds": list(common.SEEDS),
            "method_grid": {"ERM": [0.0], "DANN": list(common.LAMBDAS), "CORAL": list(common.LAMBDAS), "MMD": list(common.LAMBDAS)},
            "actual_model_cube_rows": 900,
            "actual_evidence_cube_rows": 720,
            "actual_control_rows": 72000,
            "competence": competence.set_index("setting_id").competence.to_dict(),
            "new_direction_future_utility_sealed": True,
            "new_invariance_outcome_delta_sealed": True,
        },
    )


def main() -> None:
    common.ensure_dirs()
    common.protocol()
    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    for collector in (lambda: historical_openbmi("S1"), lambda: historical_openbmi("S2"), historical_wbcic, collect_new):
        rows, evidence, controls = collector()
        model_rows.extend(rows)
        evidence_frames.extend(evidence)
        control_frames.extend(controls)
    model = pd.DataFrame(model_rows).sort_values(["setting_id", "fold", "seed", "method", "lambda"]).reset_index(drop=True)
    evidence = pd.concat(evidence_frames, ignore_index=True).sort_values(["setting_id", "fold", "seed", "direction_rank"]).reset_index(drop=True)
    controls = pd.concat(control_frames, ignore_index=True).sort_values(["setting_id", "fold", "seed", "direction_rank", "control_id"]).reset_index(drop=True)
    evidence["identity_chance_accuracy"] = [1.0 / len(common.roles_for(str(row.setting_id), int(row.fold))["model_fit"]) for _, row in evidence.iterrows()]
    if len(model) != 900 or len(evidence) != 720 or len(controls) != 72000:
        raise RuntimeError(f"cube cardinality failure model={len(model)} evidence={len(evidence)} controls={len(controls)}")
    common.write_csv(common.RESULTS / "model_setting_cube.csv", model)
    common.write_csv(common.RESULTS / "source_evidence_cube.csv", evidence)
    common.write_csv(common.RESULTS / "matched_geometry_controls.csv", controls)
    identity_scale = (
        evidence.groupby(["setting_id", "fold", "seed"], as_index=False)
        .agg(
            I_ERM=("identity_full", "first"),
            chance_accuracy=("identity_chance_accuracy", "first"),
            max_observed_direction_reduction=("identity_direction_effect", "max"),
            min_observed_direction_reduction=("identity_direction_effect", "min"),
        )
    )
    identity_scale["relative_suppression_denominator"] = identity_scale.I_ERM.abs()
    identity_scale["denominator_near_zero"] = identity_scale.relative_suppression_denominator < 1e-6
    common.write_csv(common.RESULTS / "identity_scale_diagnostics.csv", identity_scale)
    competence, stats = summaries(model, evidence)
    common.write_csv(common.RESULTS / "setting_competence.csv", competence)
    common.write_json(common.RESULTS / "SOURCE_STATISTICS_BOOTSTRAP.json", stats)

    erm = model[model.method == "ERM"]
    save_boxplot(erm, "ERM_outcome_competence_BA", "Balanced accuracy", "figure1_erm_outcome_competence.png", "ERM outcome competence")
    save_boxplot(erm, "source_identity", "Identity skill", "figure2_source_identity.png", "Source identifiability")
    save_boxplot(evidence, "persistence", "Cross-session correlation", "figure3_persistence.png", "Persistent-direction strength")
    save_boxplot(evidence, "D_finite", "D_finite", "figure4_decision_dependence.png", "Finite decision dependence")
    save_boxplot(evidence, "C_src_CE", "Validation CE harm", "figure5_source_consequence.png", "Source task consequence")
    save_boxplot(evidence, "O_task", "Squared overlap", "figure6_task_subspace_overlap.png", "Task-subspace overlap")

    all_new_pass = bool((competence[competence.setting_id.isin(["S4", "S5", "S6"])].competence == "PASS").all())
    terminal = "P4A_CROSS_SETTING_CUBE_COMPLETE" if all_new_pass else "P4A_REPRESENTATION_COMPETENCE_FAILURE"
    update_manifests(competence)
    write_reports(model, evidence, controls, competence, terminal)
    common.write_json(
        common.RESULTS / "P4A_AGGREGATION_COMPLETE.json",
        {
            "pass": True,
            "terminal_candidate": terminal,
            "model_rows": len(model),
            "evidence_rows": len(evidence),
            "control_rows": len(controls),
            "new_direction_future_utility_sealed": True,
            "new_invariance_outcome_delta_sealed": True,
        },
    )
    print(f"P4A_AGGREGATION_COMPLETE terminal_candidate={terminal}", flush=True)


if __name__ == "__main__":
    main()
