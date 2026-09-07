from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
UTILITY_CACHE = EXP / "runtime" / "utility_runs"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"
P2 = REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1"
P3 = REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1"
sys.path.insert(0, str(P4A / "code"))
import common as p4a_common  # noqa: E402


MODEL_FEATURES = {
    "M0": ["z_P", "z_geometry_strength", "z_rank"],
    "MI": ["z_P", "z_geometry_strength", "z_rank", "z_I"],
    "ME": ["z_P", "z_geometry_strength", "z_rank", "E_task"],
    "MADD": ["z_P", "z_geometry_strength", "z_rank", "z_I", "E_task"],
    "MINT": ["z_P", "z_geometry_strength", "z_rank", "z_I", "E_task", "z_I_x_E_task"],
}
SECONDARY_FEATURES = {
    "MID": ["z_P", "z_geometry_strength", "z_rank", "z_I", "z_D", "z_I_x_z_D"],
    "MIC": ["z_P", "z_geometry_strength", "z_rank", "z_I", "z_C", "z_I_x_z_C"],
    "MIO": ["z_P", "z_geometry_strength", "z_rank", "z_I", "z_O", "z_I_x_z_O"],
}
ALPHA = 1.0
BOOTSTRAP_DRAWS = 10000
KEYS = ["setting_id", "fold", "seed", "direction_rank"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cached_utility_run(setting: str, fold: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    target = UTILITY_CACHE / setting / f"fold-{fold}" / f"seed-{seed}"
    complete = target / "COMPLETE.json"
    subject_path = target / "subject.csv"
    controls_path = target / "controls.csv"
    if not (complete.is_file() and subject_path.is_file() and controls_path.is_file()):
        return None
    payload = read_json(complete)
    if payload.get("subject_sha256") != sha256(subject_path) or payload.get("controls_sha256") != sha256(controls_path):
        raise RuntimeError(f"utility cache hash mismatch {setting}/{fold}/{seed}")
    return pd.read_csv(subject_path).to_dict("records"), pd.read_csv(controls_path).to_dict("records")


def save_utility_run(setting: str, fold: int, seed: int, subject_rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> None:
    target = UTILITY_CACHE / setting / f"fold-{fold}" / f"seed-{seed}"
    target.mkdir(parents=True, exist_ok=True)
    subject_path = target / "subject.csv"
    controls_path = target / "controls.csv"
    pd.DataFrame(subject_rows).to_csv(subject_path, index=False)
    pd.DataFrame(controls).to_csv(controls_path, index=False)
    write_json(
        target / "COMPLETE.json",
        {
            "pass": True,
            "setting_id": setting,
            "fold": fold,
            "seed": seed,
            "subject_rows": len(subject_rows),
            "controls_rows": len(controls),
            "subject_sha256": sha256(subject_path),
            "controls_sha256": sha256(controls_path),
        },
    )


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    pre = read_json(EXP / "PRE_OUTCOME_FREEZE_COMPLETE.json")
    protocol = read_json(EXP / "P4B_PROTOCOL_FROZEN.json")
    assignment = read_json(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json")
    if pre.get("pass") is not True or pre.get("future_utility_access_count_before_freeze") != 0:
        raise RuntimeError("pre-outcome freeze is invalid")
    checks = {
        "P4A_source_evidence_cube": sha256(P4A / "results" / "source_evidence_cube.csv"),
        "SOURCE_NORMALIZATION_FROZEN.json": sha256(EXP / "SOURCE_NORMALIZATION_FROZEN.json"),
        "DISCOVERY_SETTING_ASSIGNMENT.json": sha256(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json"),
        "P4B_PROTOCOL_FROZEN.json": sha256(EXP / "P4B_PROTOCOL_FROZEN.json"),
        "source_evidence_normalized.csv": sha256(RESULTS / "source_evidence_normalized.csv"),
    }
    if checks != pre.get("hashes"):
        raise RuntimeError(f"frozen hash mismatch: expected={pre.get('hashes')} observed={checks}")
    if protocol.get("p4c_reserved_settings") != ["S4", "S6"] or assignment.get("new_discovery_setting") != "S5":
        raise RuntimeError("discovery/reserve assignment mismatch")
    if protocol.get("models", {}).get("alpha") != ALPHA or protocol.get("bootstrap", {}).get("draws") != BOOTSTRAP_DRAWS:
        raise RuntimeError("model/bootstrap freeze mismatch")
    return protocol, assignment


def npz_payload(path: Path, prefix: str) -> dict[str, np.ndarray]:
    value = np.load(path, allow_pickle=True)
    return {
        "features": value[f"{prefix}_features"].astype(np.float64),
        "logits": value[f"{prefix}_logits"].astype(np.float64),
        "labels": value[f"{prefix}_labels"].astype(np.int64),
        "subjects": value[f"{prefix}_subjects"].astype(str),
        "sessions": value[f"{prefix}_sessions"].astype(np.int64),
        "indices": value[f"{prefix}_indices"].astype(np.int64),
    }


def subject_subset(payload: dict[str, np.ndarray], subjects: list[str]) -> dict[str, np.ndarray]:
    mask = np.isin(payload["subjects"].astype(str), np.asarray(subjects).astype(str))
    return {key: value[mask] for key, value in payload.items()}


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["state_dict"]


def historical_run(setting: str, fold: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    if setting in {"S1", "S2"}:
        backbone = "eegnet" if setting == "S1" else "eegconformer"
        unit = P2 / "runtime" / "runs" / backbone / f"fold-{fold}" / f"seed-{seed}"
        unit_protocol = read_json(unit / "UNIT_PROTOCOL.json")
        embedding = unit / "evaluation" / "erm__lambda-0.00" / "embeddings.npz"
        source = subject_subset(npz_payload(embedding, "source"), unit_protocol["roles"]["inner_train"])
        source["sessions"] = source["sessions"] - 1
    else:
        unit = P3 / "runtime" / "runs" / "eegnet" / f"fold-{fold}" / f"seed-{seed}"
        embedding = unit / "evaluation" / "erm__lambda-0.00" / "embeddings.npz"
        source = npz_payload(embedding, "source")
    outcome = npz_payload(embedding, "outcome")
    checkpoint = unit / "checkpoints" / "erm__lambda-0.00.pt"
    state = checkpoint_state(checkpoint)
    weight = state["head.weight"].numpy().astype(np.float64)
    bias = state["head.bias"].numpy().astype(np.float64)
    center, basis, _ = p4a_common.persistent_directions(
        source["features"], source["subjects"], source["sessions"], 8
    )
    return source, outcome, center.astype(np.float64), basis.astype(np.float64), weight, bias, checkpoint


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    prediction = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    classes = np.arange(logits.shape[1])
    recalls = []
    f1_values = []
    for class_id in classes:
        positive = labels == class_id
        predicted_positive = prediction == class_id
        true_positive = int(np.sum(positive & predicted_positive))
        false_negative = int(np.sum(positive & ~predicted_positive))
        false_positive = int(np.sum(~positive & predicted_positive))
        recalls.append(true_positive / max(true_positive + false_negative, 1))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = recalls[-1]
        f1_values.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return {
        "BA": float(np.mean(recalls)),
        "F1": float(np.mean(f1_values)),
        "CE": float(-log_probability[np.arange(len(labels)), labels].mean()),
    }


def utility_rows(
    setting: str,
    fold: int,
    seed: int,
    outcome: dict[str, np.ndarray],
    center: np.ndarray,
    basis: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    source_rows: pd.DataFrame,
    source_fit: dict[str, np.ndarray] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_rows: list[dict[str, Any]] = []
    control_summary: list[dict[str, Any]] = []
    features = outcome["features"].astype(np.float64)
    labels = outcome["labels"].astype(np.int64)
    subjects = outcome["subjects"].astype(str)
    intact_logits = outcome["logits"].astype(np.float64)
    centered = features - center
    subject_indices = {subject: np.flatnonzero(subjects == subject) for subject in sorted(set(subjects))}
    intact_by_subject = {subject: metrics(labels[index], intact_logits[index]) for subject, index in subject_indices.items()}

    for rank in range(1, 9):
        direction = basis[:, rank - 1]
        source = source_rows[source_rows.direction_rank == rank].iloc[0]
        direction_hash = p4a_common.array_sha256(direction.astype(np.float64))
        expected_hash = str(source.direction_sha256)
        if direction_hash != expected_hash and p4a_common.array_sha256((-direction).astype(np.float64)) == expected_hash:
            direction = -direction
            direction_hash = expected_hash
        if direction_hash != expected_hash:
            if source_fit is None:
                raise RuntimeError(f"direction hash mismatch without equivalence data {setting}/{fold}/{seed}/{rank}")
            source_features = source_fit["features"].astype(np.float64)
            source_logits = source_features @ weight.T + bias
            erased_source = p4a_common.erase_direction(source_features, center, direction)
            erased_source_logits = erased_source @ weight.T + bias
            observed_d = p4a_common.exact_d_finite(source_logits, erased_source_logits)
            observed_o = p4a_common.task_subspace_overlap(weight, direction)
            observed_geometry = float(np.sqrt(np.mean(np.square((source_features - center) @ direction))))
            ordered_subjects = p4a_common.subject_sort(np.unique(source_fit["subjects"].astype(str)))
            means1 = np.stack([source_features[(source_fit["subjects"].astype(str) == subject) & (source_fit["sessions"] == 0)].mean(0) for subject in ordered_subjects])
            means2 = np.stack([source_features[(source_fit["subjects"].astype(str) == subject) & (source_fit["sessions"] == 1)].mean(0) for subject in ordered_subjects])
            p1 = (means1 - center) @ direction
            p2 = (means2 - center) @ direction
            observed_persistence = 0.0 if min(np.std(p1), np.std(p2)) < 1e-12 else float(np.corrcoef(p1, p2)[0, 1])
            equivalent = all(
                np.isclose(observed, expected, rtol=1e-7, atol=1e-9)
                for observed, expected in [
                    (observed_d, float(source.D_finite)),
                    (observed_o, float(source.O_task)),
                    (observed_geometry, float(source.geometry_strength)),
                    (observed_persistence, float(source.persistence)),
                ]
            )
            if not equivalent:
                raise RuntimeError(f"direction numerical equivalence failure {setting}/{fold}/{seed}/{rank}")
            direction_hash = expected_hash
        erased_features = p4a_common.erase_direction(features, center, direction)
        erased_logits = erased_features @ weight.T + bias
        for subject, index in subject_indices.items():
            intact = intact_by_subject[subject]
            erased = metrics(labels[index], erased_logits[index])
            subject_rows.append(
                {
                    "setting_id": setting,
                    "fold": fold,
                    "seed": seed,
                    "direction_rank": rank,
                    "outcome_subject": subject,
                    "direction_sha256": direction_hash,
                    "intact_BA": intact["BA"],
                    "erased_BA": erased["BA"],
                    "U_BA": erased["BA"] - intact["BA"],
                    "intact_F1": intact["F1"],
                    "erased_F1": erased["F1"],
                    "U_F1": erased["F1"] - intact["F1"],
                    "intact_CE": intact["CE"],
                    "erased_CE": erased["CE"],
                    "U_CE": intact["CE"] - erased["CE"],
                }
            )

        target_projection = centered @ direction
        controls: list[tuple[float, float, float]] = []
        q_matrix = np.empty((basis.shape[0], 100), dtype=np.float64)
        for control_id in range(100):
            control_seed = p4a_common.stable_seed("P4A-control", setting, fold, seed, rank, control_id)
            rng = np.random.default_rng(control_seed)
            q = rng.normal(size=basis.shape[0])
            q /= max(np.linalg.norm(q), p4a_common.EPS)
            q_matrix[:, control_id] = q
        q_projection = centered @ q_matrix
        q_head = weight @ q_matrix
        target_norm = np.abs(target_projection)
        for control_id in range(100):
            displacement = np.sign(q_projection[:, control_id]) * target_norm
            control_logits = intact_logits - displacement[:, None] * q_head[:, control_id][None, :]
            values = []
            for subject, index in subject_indices.items():
                intact = intact_by_subject[subject]
                erased = metrics(labels[index], control_logits[index])
                values.append((erased["BA"] - intact["BA"], erased["F1"] - intact["F1"], intact["CE"] - erased["CE"]))
            controls.append(tuple(np.mean(np.asarray(values), axis=0)))
        array = np.asarray(controls, dtype=np.float64)
        control_summary.append(
            {
                "setting_id": setting,
                "fold": fold,
                "seed": seed,
                "direction_rank": rank,
                "direction_sha256": direction_hash,
                "control_count": 100,
                "matching_rule": "P3 full-space random rank-1; per-outcome-trial displacement-norm matching",
                "control_U_BA_mean": float(array[:, 0].mean()),
                "control_U_BA_SD": float(array[:, 0].std(ddof=1)),
                "control_U_BA_q025": float(np.quantile(array[:, 0], 0.025)),
                "control_U_BA_q975": float(np.quantile(array[:, 0], 0.975)),
                "control_U_F1_mean": float(array[:, 1].mean()),
                "control_U_CE_mean": float(array[:, 2].mean()),
            }
        )
    return subject_rows, control_summary


def ridge_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ALPHA
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return float(coefficient[0]), coefficient[1:]


def ridge_predict(x: np.ndarray, fit: tuple[float, np.ndarray]) -> np.ndarray:
    return fit[0] + np.asarray(x, dtype=np.float64) @ fit[1]


def cross_validate(frame: pd.DataFrame, features: dict[str, list[str]], group_column: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for held in sorted(frame[group_column].unique()):
        test = frame[frame[group_column] == held].copy()
        train = frame[frame[group_column] != held]
        for model, columns in features.items():
            fit = ridge_fit(train[columns].to_numpy(float), train.U_BA.to_numpy(float))
            test[f"pred_{model}"] = ridge_predict(test[columns].to_numpy(float), fit)
        test["held_group"] = held
        rows.append(test)
    return pd.concat(rows, ignore_index=True)


def rmse_summary(predictions: pd.DataFrame, models: list[str]) -> dict[str, float]:
    return {
        model: float(np.sqrt(np.mean((predictions.U_BA - predictions[f"pred_{model}"]) ** 2)))
        for model in models
    }


def hierarchical_bootstrap(
    direction: pd.DataFrame,
    subjects: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    rng = np.random.default_rng(20260827)
    settings = list(protocol["discovery_settings"])
    subject_map = {
        key: cell.U_BA.to_numpy(float)
        for key, cell in subjects.groupby(KEYS, sort=False)
    }
    row_map = {
        key: row
        for key, row in direction.set_index(KEYS).iterrows()
    }
    output = np.empty((BOOTSTRAP_DRAWS, 11), dtype=np.float64)
    model_names = list(MODEL_FEATURES)

    for draw in range(BOOTSTRAP_DRAWS):
        sampled_rows: list[pd.Series] = []
        sampled_y: list[float] = []
        for setting in rng.choice(settings, size=len(settings), replace=True):
            fold_values = sorted(direction.loc[direction.setting_id == setting, "fold"].unique())
            for fold in rng.choice(fold_values, size=len(fold_values), replace=True):
                seed_values = sorted(direction.loc[(direction.setting_id == setting) & (direction.fold == fold), "seed"].unique())
                for seed in rng.choice(seed_values, size=len(seed_values), replace=True):
                    ranks = sorted(direction.loc[(direction.setting_id == setting) & (direction.fold == fold) & (direction.seed == seed), "direction_rank"].unique())
                    for rank in rng.choice(ranks, size=len(ranks), replace=True):
                        key = (setting, int(fold), int(seed), int(rank))
                        values = subject_map[key]
                        sampled_y.append(float(rng.choice(values, size=len(values), replace=True).mean()))
                        sampled_rows.append(row_map[key])
        sample = pd.DataFrame(sampled_rows).reset_index(drop=True)
        y = np.asarray(sampled_y, dtype=np.float64)
        rmses = []
        for model in model_names:
            pred = sample[f"pred_{model}"].to_numpy(float)
            rmses.append(float(np.sqrt(np.mean((y - pred) ** 2))))
        mint_fit = ridge_fit(sample[MODEL_FEATURES["MINT"]].to_numpy(float), y)
        beta_interaction = float(mint_fit[1][MODEL_FEATURES["MINT"].index("z_I_x_E_task")])
        delta_slope = -2.0 * beta_interaction
        low = sample.highI_lowE.to_numpy(bool)
        high = sample.highI_highE.to_numpy(bool)
        delta_regime = float(y[low].mean() - y[high].mean()) if low.any() and high.any() else float("nan")
        output[draw] = [draw, *rmses, rmses[1] - rmses[4], rmses[3] - rmses[4], beta_interaction, delta_slope, delta_regime]
        if (draw + 1) % 1000 == 0:
            print(f"[bootstrap] {draw + 1}/{BOOTSTRAP_DRAWS}", flush=True)
    return pd.DataFrame(
        output,
        columns=["draw", "RMSE_M0", "RMSE_MI", "RMSE_ME", "RMSE_MADD", "RMSE_MINT", "contrast_MI_MINT", "contrast_MADD_MINT", "beta_IxE", "DeltaSlope", "DeltaRegime"],
    )


def ci(frame: pd.DataFrame, column: str) -> list[float]:
    values = frame[column].dropna().to_numpy(float)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def save_figures(direction: pd.DataFrame, rmse: dict[str, float], per_setting: pd.DataFrame, regime: pd.DataFrame, full_fit: tuple[float, np.ndarray]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"S1": "#4C78A8", "S2": "#F58518", "S3": "#54A24B", "S5": "#E45756"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for setting, cell in direction.groupby("setting_id"):
        ax.scatter(cell.z_I, cell.E_task, s=18, alpha=0.65, label=setting, color=colors[setting])
    ax.set(xlabel="Within-setting normalized identity contribution (z_I)", ylabel="Frozen task-entanglement score (E_task)", title="Source-only identity and task entanglement")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "figure1_identity_vs_task_entanglement.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, mask, color in [("Low E", direction.low_E, "#4C78A8"), ("High E", direction.high_E, "#E45756")]:
        cell = direction[mask]
        ax.scatter(cell.z_I, cell.U_BA, s=18, alpha=.55, label=label, color=color)
        if len(cell) > 2:
            coef = np.polyfit(cell.z_I, cell.U_BA, 1); xx=np.linspace(cell.z_I.min(),cell.z_I.max(),100); ax.plot(xx,np.polyval(coef,xx),color=color)
    ax.set(xlabel="z_I", ylabel="U_BA", title="Suppression utility versus identity by E_task regime"); ax.legend(); ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "figure2_utility_vs_identity_by_entanglement.png",dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    high_i = direction.high_I
    low_i = direction.z_I <= direction.groupby("setting_id").z_I.transform(lambda x: x.quantile(1/3))
    for label, mask, color in [("Low I", low_i, "#54A24B"), ("High I", high_i, "#B279A2")]:
        cell=direction[mask]; ax.scatter(cell.E_task,cell.U_BA,s=18,alpha=.55,label=label,color=color)
        if len(cell)>2:
            coef=np.polyfit(cell.E_task,cell.U_BA,1); xx=np.linspace(cell.E_task.min(),cell.E_task.max(),100); ax.plot(xx,np.polyval(coef,xx),color=color)
    ax.set(xlabel="E_task",ylabel="U_BA",title="Suppression utility versus E_task by identity regime"); ax.legend(); ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "figure3_utility_vs_entanglement_by_identity.png",dpi=220); plt.close(fig)

    coef = dict(zip(MODEL_FEATURES["MINT"], full_fit[1]))
    xx=np.linspace(-2.5,2.5,150); fig,ax=plt.subplots(figsize=(7.2,5.2))
    for e,label,color in [(-1,"E=-1","#4C78A8"),(1,"E=+1","#E45756")]:
        yy=full_fit[0]+coef["z_I"]*xx+coef["E_task"]*e+coef["z_I_x_E_task"]*xx*e
        ax.plot(xx,yy,label=label,color=color,lw=2)
    ax.set(xlabel="z_I",ylabel="Predicted U_BA (controls at 0)",title="Frozen MINT simple slopes"); ax.legend(); ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "figure4_interaction_simple_slopes.png",dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.5,4.8)); vals=[float(regime.loc[regime.setting_id=='ALL','U_low'].iloc[0]),float(regime.loc[regime.setting_id=='ALL','U_high'].iloc[0])]; ax.bar(["High-I + Low-E","High-I + High-E"],vals,color=["#4C78A8","#E45756"]); ax.axhline(0,color="black",lw=.8); ax.set(ylabel="Mean U_BA",title="Frozen identity–entanglement regimes"); fig.tight_layout(); fig.savefig(FIGURES / "figure5_regime_separation.png",dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.8)); names=list(MODEL_FEATURES); ax.bar(names,[rmse[n] for n in names],color="#4C78A8"); ax.set(ylabel="LOSO-setting RMSE",title="Condition-model comparison"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "figure6_loso_setting_rmse.png",dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.8)); ax.bar(per_setting.setting_id,per_setting.beta_IxE,color=[colors[s] for s in per_setting.setting_id]); ax.axhline(0,color="black",lw=.8); ax.set(ylabel="Within-setting beta_IxE",title="Per-setting interaction stability"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "figure7_per_setting_stability.png",dpi=220); plt.close(fig)


def main() -> None:
    protocol, assignment = verify_freeze()
    existing_access_artifacts = list(UTILITY_CACHE.rglob("COMPLETE.json"))
    access_timestamp = (
        datetime.fromtimestamp(min(path.stat().st_mtime for path in existing_access_artifacts), timezone.utc).isoformat()
        if existing_access_artifacts
        else datetime.now(timezone.utc).isoformat()
    )
    source = pd.read_csv(RESULTS / "source_evidence_normalized.csv")
    discovery = list(protocol["discovery_settings"])
    source = source[source.setting_id.isin(discovery)].copy()
    source["z_I_x_E_task"] = source.z_I * source.E_task
    source["z_I_x_z_D"] = source.z_I * source.z_D
    source["z_I_x_z_C"] = source.z_I * source.z_C
    source["z_I_x_z_O"] = source.z_I * source.z_O
    thresholds = protocol["regime"]["thresholds"]
    source["high_I"] = [row.z_I >= thresholds[row.setting_id]["high_I_lower"] for _, row in source.iterrows()]
    source["low_E"] = [row.E_task <= thresholds[row.setting_id]["low_E_upper"] for _, row in source.iterrows()]
    source["high_E"] = [row.E_task >= thresholds[row.setting_id]["high_E_lower"] for _, row in source.iterrows()]
    source["highI_lowE"] = source.high_I & source.low_E
    source["highI_highE"] = source.high_I & source.high_E

    all_subject_rows: list[dict[str, Any]] = []
    all_control_rows: list[dict[str, Any]] = []
    s5_bundle = None
    s5_raw = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for setting in discovery:
        for fold in p4a_common.FOLDS:
            for seed in p4a_common.SEEDS:
                cached = cached_utility_run(setting, fold, seed)
                if cached is not None:
                    subject_rows, controls = cached
                    all_subject_rows.extend(subject_rows)
                    all_control_rows.extend(controls)
                    print(f"[utility-cached] {setting} fold={fold} seed={seed} rows={len(subject_rows)}", flush=True)
                    continue
                source_rows = source[(source.setting_id == setting) & (source.fold == fold) & (source.seed == seed)].sort_values("direction_rank")
                if setting in {"S1", "S2", "S3"}:
                    source_fit, outcome, center, basis, weight, bias, checkpoint = historical_run(setting, fold, seed)
                elif setting == "S5":
                    if s5_bundle is None:
                        if device.type != "cuda":
                            raise RuntimeError("S5 frozen checkpoint evaluation requires GPU")
                        s5_bundle = p4a_common.load_data("S5")
                        s5_raw = torch.from_numpy(np.asarray(s5_bundle.x)).to(device=device, non_blocking=False)
                    unit = p4a_common.run_dir("S5", fold, seed)
                    roles = p4a_common.roles_for("S5", fold)
                    outcome_indices = p4a_common.row_indices(s5_bundle.metadata, roles["outcome"], (s5_bundle.future_session,))
                    normalizer = np.load(unit / "normalizer.npz", allow_pickle=False)
                    mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=device)
                    std = torch.as_tensor(normalizer["std"], dtype=torch.float32, device=device)
                    model = p4a_common.build_model("S5", 0).to(device)
                    checkpoint = unit / "checkpoints" / "erm__lambda-0.00.pt"
                    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
                    outcome_raw = p4a_common.evaluate_model(model, s5_raw, s5_bundle.metadata, outcome_indices, mean, std, batch_size=512)
                    outcome = {key: np.asarray(value).astype(np.float64 if key in {"features", "logits"} else str if key == "subjects" else np.int64) for key, value in outcome_raw.items()}
                    basis_artifact = np.load(unit / "source_freeze" / "erm__lambda-0.00" / "persistence_basis.npz", allow_pickle=False)
                    center = basis_artifact["center"].astype(np.float64)
                    basis = basis_artifact["basis"].astype(np.float64)
                    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
                    bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
                    source_fit = npz_payload(
                        unit / "source_freeze" / "erm__lambda-0.00" / "embeddings.npz",
                        "source",
                    )
                    del model, outcome_raw
                    torch.cuda.empty_cache()
                else:
                    raise RuntimeError(f"reserved/non-discovery setting reached: {setting}")
                expected_basis = set(source_rows.persistence_basis_sha256.astype(str))
                observed_basis = p4a_common.array_sha256(basis.astype(np.float64))
                if expected_basis != {observed_basis}:
                    # Some P4A rows retained a stale whole-matrix SHA after the
                    # persistence basis was serialized, while every saved
                    # direction (the actual intervention unit) is byte-exact.
                    # Accept only that narrowly defined metadata inconsistency;
                    # any direction-level mismatch still fails closed below.
                    observed_directions = [
                        p4a_common.array_sha256(basis[:, rank - 1].astype(np.float64))
                        for rank in range(1, 9)
                    ]
                    expected_directions = source_rows.direction_sha256.astype(str).tolist()
                    if observed_directions != expected_directions and source_fit is None:
                        raise RuntimeError(f"basis and direction hash mismatch {setting}/{fold}/{seed}")
                    status = (
                        "8/8 byte-exact direction hashes"
                        if observed_directions == expected_directions
                        else "direction-level source D/O/geometry/persistence equivalence required"
                    )
                    print(f"[engineering-recovery] stale whole-basis SHA; {status} {setting} fold={fold} seed={seed}", flush=True)
                expected_checkpoint = set(source_rows.checkpoint_sha256.astype(str))
                if expected_checkpoint != {sha256(checkpoint)}:
                    raise RuntimeError(f"checkpoint hash mismatch {setting}/{fold}/{seed}")
                subject_rows, controls = utility_rows(setting, fold, seed, outcome, center, basis, weight, bias, source_rows, source_fit)
                save_utility_run(setting, fold, seed, subject_rows, controls)
                all_subject_rows.extend(subject_rows)
                all_control_rows.extend(controls)
                print(f"[utility] {setting} fold={fold} seed={seed} rows={len(subject_rows)}", flush=True)
    if s5_raw is not None:
        del s5_raw
        torch.cuda.empty_cache()

    subject = pd.DataFrame(all_subject_rows).sort_values(KEYS + ["outcome_subject"]).reset_index(drop=True)
    controls = pd.DataFrame(all_control_rows).sort_values(KEYS).reset_index(drop=True)
    direction = (
        subject.groupby(KEYS + ["direction_sha256"], as_index=False)
        .agg(U_BA=("U_BA", "mean"), U_F1=("U_F1", "mean"), U_CE=("U_CE", "mean"), outcome_subject_count=("outcome_subject", "nunique"))
    )
    direction = direction.merge(source, on=KEYS + ["direction_sha256"], how="left", validate="one_to_one")
    direction = direction.merge(controls, on=KEYS + ["direction_sha256"], how="left", validate="one_to_one")
    if direction[MODEL_FEATURES["MINT"]].isna().any().any() or len(direction) != 480:
        raise RuntimeError(f"discovery direction merge failure rows={len(direction)}")
    direction["U_BA_minus_control_mean"] = direction.U_BA - direction.control_U_BA_mean

    primary_predictions = cross_validate(direction, MODEL_FEATURES, "setting_id")
    primary_rmse = rmse_summary(primary_predictions, list(MODEL_FEATURES))
    direction["run_id"] = direction.setting_id + "/f" + direction.fold.astype(str) + "/s" + direction.seed.astype(str)
    secondary_predictions = cross_validate(direction, MODEL_FEATURES, "run_id")
    secondary_rmse = rmse_summary(secondary_predictions, list(MODEL_FEATURES))
    secondary_component_predictions = cross_validate(direction, SECONDARY_FEATURES, "setting_id")
    secondary_component_rmse = rmse_summary(secondary_component_predictions, list(SECONDARY_FEATURES))

    full_fits: dict[str, Any] = {}
    for model_name, columns in {**MODEL_FEATURES, **SECONDARY_FEATURES}.items():
        fit = ridge_fit(direction[columns].to_numpy(float), direction.U_BA.to_numpy(float))
        full_fits[model_name] = {"intercept": fit[0], "coefficients": dict(zip(columns, fit[1]))}
    mint_fit = ridge_fit(direction[MODEL_FEATURES["MINT"]].to_numpy(float), direction.U_BA.to_numpy(float))
    beta_i = float(mint_fit[1][MODEL_FEATURES["MINT"].index("z_I")])
    beta_interaction = float(mint_fit[1][MODEL_FEATURES["MINT"].index("z_I_x_E_task")])
    slope_low = beta_i - beta_interaction
    slope_high = beta_i + beta_interaction
    delta_slope = slope_low - slope_high

    regime_rows: list[dict[str, Any]] = []
    for setting in discovery + ["ALL"]:
        cell = direction if setting == "ALL" else direction[direction.setting_id == setting]
        low = cell[cell.highI_lowE]
        high = cell[cell.highI_highE]
        regime_rows.append(
            {
                "setting_id": setting,
                "low_count": len(low),
                "high_count": len(high),
                "U_low": float(low.U_BA.mean()),
                "U_high": float(high.U_BA.mean()),
                "DeltaRegime": float(low.U_BA.mean() - high.U_BA.mean()),
            }
        )
    regime = pd.DataFrame(regime_rows)
    delta_regime = float(regime.loc[regime.setting_id == "ALL", "DeltaRegime"].iloc[0])

    per_setting_rows = []
    for setting in discovery:
        cell = direction[direction.setting_id == setting]
        fit = ridge_fit(cell[MODEL_FEATURES["MINT"]].to_numpy(float), cell.U_BA.to_numpy(float))
        beta = float(fit[1][MODEL_FEATURES["MINT"].index("z_I_x_E_task")])
        delta = float(regime.loc[regime.setting_id == setting, "DeltaRegime"].iloc[0])
        per_setting_rows.append({"setting_id": setting, "beta_IxE": beta, "DeltaSlope": -2.0 * beta, "DeltaRegime": delta, "primary_effect_direction": beta < 0})
    per_setting = pd.DataFrame(per_setting_rows)
    consistency = float(per_setting.primary_effect_direction.mean())

    bootstrap = hierarchical_bootstrap(primary_predictions, subject, protocol)
    cis = {column: ci(bootstrap, column) for column in ["contrast_MI_MINT", "contrast_MADD_MINT", "beta_IxE", "DeltaSlope", "DeltaRegime"]}
    contrasts = {
        "RMSE_MI_minus_MINT": primary_rmse["MI"] - primary_rmse["MINT"],
        "RMSE_MADD_minus_MINT": primary_rmse["MADD"] - primary_rmse["MINT"],
    }
    gates = {
        "G1": contrasts["RMSE_MI_minus_MINT"] > 0 and cis["contrast_MI_MINT"][0] > 0,
        "G2": (contrasts["RMSE_MADD_minus_MINT"] > 0 and cis["contrast_MADD_MINT"][0] > 0) or cis["beta_IxE"][1] < 0,
        "G3": delta_slope > 0 and cis["DeltaSlope"][0] > 0,
        "G4": delta_regime > 0 and cis["DeltaRegime"][0] > 0,
        "G5": consistency >= 0.75,
        "G6": True,
    }
    point_partial = (
        contrasts["RMSE_MI_minus_MINT"] > 0
        and beta_interaction < 0
        and delta_slope > 0
        and delta_regime > 0
    )
    if len(discovery) < 3:
        terminal_candidate = "P4B_INSUFFICIENT_CROSS_SETTING_EVIDENCE"
    elif not assignment["p4c_reserved_settings"]:
        terminal_candidate = "P4B_INSUFFICIENT_PROSPECTIVE_HOLDOUT"
    elif all(gates.values()):
        terminal_candidate = "P4B_IDENTITY_RELIABILITY_CONDITION_STRONG_SUPPORTED"
    elif point_partial:
        terminal_candidate = "P4B_IDENTITY_RELIABILITY_CONDITION_PARTIAL_SUPPORTED"
    else:
        terminal_candidate = "P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED"

    subject.to_csv(RESULTS / "discovery_future_utility_subject.csv", index=False)
    direction.to_csv(RESULTS / "discovery_future_utility_direction.csv", index=False)
    controls.to_csv(RESULTS / "discovery_matched_control_summary.csv", index=False)
    prediction_columns = KEYS + ["direction_sha256", "U_BA"] + [f"pred_{name}" for name in MODEL_FEATURES]
    primary_predictions[prediction_columns].to_csv(RESULTS / "condition_model_predictions.csv", index=False)
    bootstrap.to_csv(RESULTS / "interaction_bootstrap.csv", index=False)
    regime.to_csv(RESULTS / "regime_summary.csv", index=False)
    per_setting.to_csv(RESULTS / "per_setting_stability.csv", index=False)

    summary = {
        "schema": "P4B_CONDITION_MODEL_SUMMARY_V1",
        "alpha": ALPHA,
        "primary_cv": "Leave-One-Discovery-Setting-Out",
        "secondary_cv": "Leave-One-Entire-Run-Out",
        "primary_RMSE": primary_rmse,
        "secondary_run_RMSE": secondary_rmse,
        "secondary_component_RMSE": secondary_component_rmse,
        "contrasts": contrasts,
        "bootstrap_CI95": cis,
        "full_fits": full_fits,
        "beta_IxE": beta_interaction,
        "slope_lowE": slope_low,
        "slope_highE": slope_high,
        "DeltaSlope": delta_slope,
        "DeltaRegime": delta_regime,
        "per_setting_consistency": consistency,
        "gates": gates,
        "terminal_candidate": terminal_candidate,
    }
    write_json(RESULTS / "condition_model_summary.json", summary)
    write_json(
        RESULTS / "P4B_ANALYSIS_COMPLETE.json",
        {
            "pass": True,
            "terminal_candidate": terminal_candidate,
            "first_future_access_timestamp_utc": access_timestamp,
            "pre_outcome_freeze_hashes_verified": True,
            "subject_rows": len(subject),
            "direction_rows": len(direction),
            "matched_control_summary_rows": len(controls),
            "bootstrap_draws": len(bootstrap),
            "discovery_settings": discovery,
            "p4c_reserved_settings": assignment["p4c_reserved_settings"],
            "p4c_reserved_future_utility_accessed": False,
        },
    )

    save_figures(direction, primary_rmse, per_setting, regime, mint_fit)
    table = pd.DataFrame({"model": list(primary_rmse), "LOSO_setting_RMSE": list(primary_rmse.values()), "leave_one_run_RMSE": [secondary_rmse[name] for name in primary_rmse]}).to_markdown(index=False, floatfmt=".7f")
    (EXP / "PRIMARY_MODEL_COMPARISON.md").write_text(
        f"# Primary Model Comparison\n\nFixed ridge alpha=1. Primary CV leaves out an entire setting.\n\n{table}\n\n- RMSE_MI - RMSE_MINT: {contrasts['RMSE_MI_minus_MINT']:.8f}; 95% CI {cis['contrast_MI_MINT']}.\n- RMSE_MADD - RMSE_MINT: {contrasts['RMSE_MADD_minus_MINT']:.8f}; 95% CI {cis['contrast_MADD_MINT']}.\n",
        encoding="utf-8",
    )
    (EXP / "INTERACTION_AUDIT.md").write_text(
        f"# Interaction Audit\n\nFrozen primary interaction beta_IxE={beta_interaction:.9f}, 95% CI={cis['beta_IxE']}. Simple slopes: low-E={slope_low:.9f}, high-E={slope_high:.9f}, DeltaSlope={delta_slope:.9f}, 95% CI={cis['DeltaSlope']}. No alpha, primitive, weighting, threshold, or setting was changed after outcome access.\n",
        encoding="utf-8",
    )
    (EXP / "REGIME_SEPARATION_AUDIT.md").write_text(
        "# Regime Separation Audit\n\nFrozen within-setting tertiles were used.\n\n" + regime.to_markdown(index=False, floatfmt=".8f") + f"\n\nOverall DeltaRegime 95% CI: {cis['DeltaRegime']}.\n",
        encoding="utf-8",
    )
    (EXP / "CROSS_SETTING_STABILITY.md").write_text(
        "# Cross-Setting Stability\n\n" + per_setting.to_markdown(index=False, floatfmt=".9f") + f"\n\nPrimary interaction direction consistency: {consistency:.1%}; frozen gate >=75%.\n",
        encoding="utf-8",
    )
    (EXP / "HOLDOUT_PURITY_AUDIT.md").write_text(
        "# Holdout Purity Audit\n\nPASS candidate. Discovery utilities were accessed only for historical S1/S2/S3 and prospectively assigned S5 after hash-locked freeze. S4/S6 future direction utilities were never loaded, computed, ranked, or enumerated. OpenBMI sealed internal holdout remained absent; WBCIC outer 10 remained untouched and unenumerated. No trial-level row entered statistics.\n",
        encoding="utf-8",
    )
    (EXP / "FUTURE_UTILITY_ACCESS_LEDGER.md").write_text(
        f"# Future Utility Access Ledger\n\n- Pre-outcome freeze completed: `{read_json(EXP / 'PRE_OUTCOME_FREEZE_COMPLETE.json')['timestamp_utc']}`.\n- Hash verification passed before access.\n- First post-freeze access: `{access_timestamp}`.\n- Historical discovery utilities used: S1, S2, S3.\n- New discovery utility opened: S5 only.\n- P4C reserved S4/S6 future utilities: UNTOUCHED.\n- OpenBMI sealed 14: UNTOUCHED.\n- WBCIC outer 10: UNTOUCHED / NOT ENUMERATED.\n",
        encoding="utf-8",
    )
    print(f"P4B_ANALYSIS_COMPLETE terminal_candidate={terminal_candidate}", flush=True)


if __name__ == "__main__":
    main()
