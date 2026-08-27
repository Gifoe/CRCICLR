from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
RUN_CACHE = RUNTIME / "future_utility_runs"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"
P4B = REPO / "experiments" / "persist_eeg_p4b_identity_reliability_discovery_v1"
KEYS = ["setting_id", "fold", "seed", "direction_rank"]

sys.path.insert(0, str(HERE))
from p4c_safety_common import control_metrics, metrics, now_utc, read_json, sha256, stable_seed, write_json  # noqa: E402

sys.path.insert(0, str(P4A / "code"))
import common as p4a_common  # noqa: E402


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    pre = read_json(EXP / "P4C_SAFETY_PREOUTCOME_FREEZE.json")
    protocol = read_json(EXP / "P4C_SAFETY_PROTOCOL_FROZEN.json")
    if pre.get("pass") is not True or pre.get("future_outcome_access_count_before_freeze") != 0:
        raise RuntimeError("invalid P4C-Safety pre-outcome freeze")
    paths = {
        "P4C_SAFETY_INPUT_AUDIT.json": EXP / "P4C_SAFETY_INPUT_AUDIT.json",
        "P4B_FINAL_VALIDATION.json": P4B / "results" / "P4B_FINAL_VALIDATION.json",
        "P4B_PROTOCOL_FROZEN.json": P4B / "P4B_PROTOCOL_FROZEN.json",
        "SOURCE_NORMALIZATION_FROZEN.json": P4B / "SOURCE_NORMALIZATION_FROZEN.json",
        "source_evidence_normalized.csv": P4B / "results" / "source_evidence_normalized.csv",
        "P4C_SAFETY_PROTOCOL_FROZEN.json": EXP / "P4C_SAFETY_PROTOCOL_FROZEN.json",
        "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv": RESULTS / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv",
        "p4c_safety_source_preflight_runs.csv": RESULTS / "p4c_safety_source_preflight_runs.csv",
        "discovery_regime_reproduction.csv": RESULTS / "discovery_regime_reproduction.csv",
        "regime_coverage.csv": RESULTS / "regime_coverage.csv",
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    if observed != pre.get("hashes"):
        raise RuntimeError(f"pre-outcome freeze hash mismatch: {observed}")
    assignments = pd.read_csv(RESULTS / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv")
    if len(assignments) != 240 or set(assignments.setting_id) != {"S4", "S6"}:
        raise RuntimeError("assignment cardinality or setting mismatch")
    if protocol.get("matched_random", {}).get("count") != 100:
        raise RuntimeError("matched random count changed")
    return pre, protocol, assignments


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def cache_paths(setting: str, fold: int, seed: int) -> tuple[Path, Path, Path]:
    target = RUN_CACHE / setting / f"fold-{fold}" / f"seed-{seed}"
    return target / "subject.csv", target / "direction.csv", target / "COMPLETE.json"


def load_cache(setting: str, fold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    subject_path, direction_path, complete_path = cache_paths(setting, fold, seed)
    if not (subject_path.is_file() and direction_path.is_file() and complete_path.is_file()):
        return None
    payload = read_json(complete_path)
    if payload.get("subject_sha256") != sha256(subject_path) or payload.get("direction_sha256") != sha256(direction_path):
        raise RuntimeError(f"cache hash mismatch {setting}/{fold}/{seed}")
    return pd.read_csv(subject_path), pd.read_csv(direction_path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_cache(setting: str, fold: int, seed: int, subject: pd.DataFrame, direction: pd.DataFrame) -> None:
    subject_path, direction_path, complete_path = cache_paths(setting, fold, seed)
    atomic_csv(subject, subject_path)
    atomic_csv(direction, direction_path)
    write_json(complete_path, {
        "pass": True,
        "setting_id": setting,
        "fold": fold,
        "seed": seed,
        "subject_rows": len(subject),
        "direction_rows": len(direction),
        "subject_sha256": sha256(subject_path),
        "direction_sha256": sha256(direction_path),
        "completed_at_utc": now_utc(),
    })


def evaluate_run(setting: str, fold: int, seed: int, bundle: Any, raw: torch.Tensor, device: torch.device, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = p4a_common.run_dir(setting, fold, seed)
    frozen = assignments[(assignments.setting_id == setting) & (assignments.fold == fold) & (assignments.seed == seed)].sort_values("direction_rank")
    if len(frozen) != 8:
        raise RuntimeError(f"frozen assignment cardinality {setting}/{fold}/{seed}")
    roles = p4a_common.roles_for(setting, fold)
    outcome_indices = p4a_common.row_indices(bundle.metadata, roles["outcome"], (bundle.future_session,))
    if len(outcome_indices) == 0:
        raise RuntimeError(f"empty reserved outcome {setting}/{fold}/{seed}")
    normalizer_path = run / "normalizer.npz"
    checkpoint = run / "checkpoints" / "erm__lambda-0.00.pt"
    normalizer = np.load(normalizer_path, allow_pickle=False)
    mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalizer["std"], dtype=torch.float32, device=device)
    model = p4a_common.build_model(setting, 0).to(device)
    state = checkpoint_state(checkpoint)
    model.load_state_dict(state, strict=True)
    outcome = p4a_common.evaluate_model(model, raw, bundle.metadata, outcome_indices, mean, std, batch_size=512)
    features = outcome["features"].astype(np.float64)
    intact_logits = outcome["logits"].astype(np.float64)
    labels = outcome["labels"].astype(np.int64)
    subjects = outcome["subjects"].astype(str)
    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
    del model, state
    torch.cuda.empty_cache()
    basis_artifact = np.load(run / "source_freeze" / "erm__lambda-0.00" / "persistence_basis.npz", allow_pickle=False)
    center = basis_artifact["center"].astype(np.float64)
    basis = basis_artifact["basis"].astype(np.float64)
    centered = features - center
    if set(frozen.checkpoint_sha256.astype(str)) != {sha256(checkpoint)} or set(frozen.normalizer_sha256.astype(str)) != {sha256(normalizer_path)}:
        raise RuntimeError(f"checkpoint/normalizer changed after freeze {setting}/{fold}/{seed}")
    subject_indices = {name: np.flatnonzero(subjects == name) for name in p4a_common.subject_sort(np.unique(subjects))}
    intact_by_subject = {name: metrics(labels[index], intact_logits[index]) for name, index in subject_indices.items()}
    subject_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for rank in range(1, 9):
        row = frozen[frozen.direction_rank == rank].iloc[0]
        direction = basis[:, rank - 1]
        direction_hash = p4a_common.array_sha256(direction)
        if direction_hash != str(row.direction_sha256):
            raise RuntimeError(f"direction changed after freeze {setting}/{fold}/{seed}/{rank}")
        target_projection = centered @ direction
        erased_logits = p4a_common.erase_direction(features, center, direction) @ weight.T + bias
        q_matrix = np.empty((basis.shape[0], 100), dtype=np.float64)
        for control_id in range(100):
            rng = np.random.default_rng(stable_seed("P4A-control", setting, fold, seed, rank, control_id))
            q = rng.normal(size=basis.shape[0])
            q /= max(np.linalg.norm(q), p4a_common.EPS)
            q_matrix[:, control_id] = q
        q_projection = centered @ q_matrix
        q_head = weight @ q_matrix
        displacement = np.sign(q_projection) * np.abs(target_projection)[:, None]
        control_logits = intact_logits[:, None, :] - displacement[:, :, None] * q_head.T[None, :, :]
        control_subject_values: list[np.ndarray] = []
        rank_rows: list[dict[str, Any]] = []
        for subject, index in subject_indices.items():
            intact = intact_by_subject[subject]
            erased = metrics(labels[index], erased_logits[index])
            controls = control_metrics(labels[index], control_logits[index])
            control_u = np.column_stack([controls["BA"] - intact["BA"], controls["F1"] - intact["F1"], intact["CE"] - controls["CE"]])
            control_subject_values.append(control_u)
            payload = {
                "setting_id": setting,
                "dataset": str(row.dataset),
                "task": str(row.task),
                "backbone": str(row.backbone),
                "fold": fold,
                "seed": seed,
                "direction_rank": rank,
                "outcome_subject": subject,
                "regime_label": str(row.regime_label),
                "highest_identity": bool(row.highest_identity),
                "direction_sha256": direction_hash,
                "checkpoint_sha256": str(row.checkpoint_sha256),
                "BA_intact": intact["BA"],
                "BA_erased": erased["BA"],
                "U_BA": erased["BA"] - intact["BA"],
                "F1_intact": intact["F1"],
                "F1_erased": erased["F1"],
                "U_F1": erased["F1"] - intact["F1"],
                "CE_intact": intact["CE"],
                "CE_erased": erased["CE"],
                "U_CE": intact["CE"] - erased["CE"],
                "control_U_BA_mean": float(control_u[:, 0].mean()),
                "control_U_F1_mean": float(control_u[:, 1].mean()),
                "control_U_CE_mean": float(control_u[:, 2].mean()),
                "SpecificU_BA": float(erased["BA"] - intact["BA"] - control_u[:, 0].mean()),
            }
            rank_rows.append(payload)
            subject_rows.append(payload)
        subject_control = np.stack(control_subject_values, axis=0)
        control_by_draw = subject_control.mean(axis=0)
        rank_frame = pd.DataFrame(rank_rows)
        direction_rows.append({
            "setting_id": setting,
            "dataset": str(row.dataset),
            "task": str(row.task),
            "backbone": str(row.backbone),
            "fold": fold,
            "seed": seed,
            "direction_rank": rank,
            "regime_label": str(row.regime_label),
            "highest_identity": bool(row.highest_identity),
            "direction_sha256": direction_hash,
            "checkpoint_sha256": str(row.checkpoint_sha256),
            "outcome_subject_count": rank_frame.outcome_subject.nunique(),
            "U_BA": rank_frame.U_BA.mean(),
            "U_F1": rank_frame.U_F1.mean(),
            "U_CE": rank_frame.U_CE.mean(),
            "control_count": 100,
            "matching_rule": "P4A full-space Gaussian; per-trial displacement-norm matching",
            "control_U_BA_mean": float(control_by_draw[:, 0].mean()),
            "control_U_BA_SD": float(control_by_draw[:, 0].std(ddof=1)),
            "control_U_BA_q025": float(np.quantile(control_by_draw[:, 0], .025)),
            "control_U_BA_q975": float(np.quantile(control_by_draw[:, 0], .975)),
            "control_U_F1_mean": float(control_by_draw[:, 1].mean()),
            "control_U_CE_mean": float(control_by_draw[:, 2].mean()),
            "SpecificU_BA": float(rank_frame.SpecificU_BA.mean()),
        })
        del control_logits, q_matrix, q_projection, q_head, displacement
    return pd.DataFrame(subject_rows), pd.DataFrame(direction_rows)


def main() -> None:
    pre, protocol, assignments = verify_freeze()
    marker = RUNTIME / "OUTCOME_ACCESS_STARTED.json"
    if marker.is_file():
        access = read_json(marker)
    else:
        access = {
            "schema": "P4C_SAFETY_OUTCOME_ACCESS_STARTED_V1",
            "first_outcome_access_timestamp_utc": now_utc(),
            "preoutcome_freeze_timestamp_utc": pre["timestamp_utc"],
            "preoutcome_freeze_sha256": sha256(EXP / "P4C_SAFETY_PREOUTCOME_FREEZE.json"),
            "reserved_settings_opened": protocol["reserved_settings"],
            "post_outcome_scientific_modification": False,
        }
        write_json(marker, access)
    if access["first_outcome_access_timestamp_utc"] <= pre["timestamp_utc"]:
        raise RuntimeError("outcome access predates freeze")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P4C-Safety frozen representation evaluation requires server GPU")
    all_subject: list[pd.DataFrame] = []
    all_direction: list[pd.DataFrame] = []
    for setting in protocol["reserved_settings"]:
        bundle = p4a_common.load_data(setting)
        raw = torch.from_numpy(np.asarray(bundle.x)).to(device=device, non_blocking=False)
        print(f"[{setting}] loaded {tuple(raw.shape)} on {torch.cuda.get_device_name(0)}", flush=True)
        for fold in p4a_common.FOLDS:
            for seed in p4a_common.SEEDS:
                cached = load_cache(setting, fold, seed)
                if cached is None:
                    subject, direction = evaluate_run(setting, fold, seed, bundle, raw, device, assignments)
                    save_cache(setting, fold, seed, subject, direction)
                    print(f"[future] {setting} fold={fold} seed={seed} rows={len(subject)}", flush=True)
                else:
                    subject, direction = cached
                    print(f"[future-cached] {setting} fold={fold} seed={seed}", flush=True)
                all_subject.append(subject)
                all_direction.append(direction)
        del raw, bundle
        torch.cuda.empty_cache()
    subject = pd.concat(all_subject, ignore_index=True).sort_values(KEYS + ["outcome_subject"]).reset_index(drop=True)
    direction = pd.concat(all_direction, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if len(direction) != 240 or subject.duplicated(KEYS + ["outcome_subject"]).any() or any("trial" in name.lower() for name in subject.columns):
        raise RuntimeError("subject-first cardinality/schema failure")
    atomic_csv(subject, RESULTS / "p4c_safety_future_utility_subject.csv")
    atomic_csv(direction, RESULTS / "p4c_safety_future_utility_direction.csv")
    write_json(RESULTS / "P4C_SAFETY_OUTCOME_EVALUATION_COMPLETE.json", {
        "pass": True,
        "completed_at_utc": now_utc(),
        "first_outcome_access_timestamp_utc": access["first_outcome_access_timestamp_utc"],
        "preoutcome_freeze_sha256": access["preoutcome_freeze_sha256"],
        "reserved_settings": protocol["reserved_settings"],
        "subject_rows": len(subject),
        "direction_rows": len(direction),
        "subject_sha256": sha256(RESULTS / "p4c_safety_future_utility_subject.csv"),
        "direction_sha256": sha256(RESULTS / "p4c_safety_future_utility_direction.csv"),
        "matched_control_count": 100,
        "post_outcome_scientific_modification": False,
    })
    (EXP / "P4C_SAFETY_OUTCOME_ACCESS_LEDGER.md").write_text(
        "# P4C-Safety Outcome Access Ledger\n\n"
        f"- Pre-outcome freeze: `{pre['timestamp_utc']}`.\n"
        f"- First S4/S6 outcome access: `{access['first_outcome_access_timestamp_utc']}`.\n"
        "- Both reserved settings were evaluated under the frozen source-only regime.\n"
        "- No retraining, MINT refit, threshold change, setting change or direction reselection occurred.\n"
        "- OpenBMI sealed internal holdout: UNTOUCHED.\n"
        "- WBCIC outer 10: UNTOUCHED / NOT ENUMERATED.\n",
        encoding="utf-8",
    )
    print(f"P4C_SAFETY_OUTCOMES_COMPLETE subject_rows={len(subject)} direction_rows={len(direction)}", flush=True)


if __name__ == "__main__":
    main()
