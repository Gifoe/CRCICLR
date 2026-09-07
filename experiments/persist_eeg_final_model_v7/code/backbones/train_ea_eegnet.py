"""Train a history-Euclidean-aligned wide EEGNet on authorized folds.

The alignment matrix for every subject is estimated from legal history
sessions only.  Population training may use future sessions from non-outcome
subjects; outcome future data and labels are scoring-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, V7_SEED, ensure_directories, logit, sigmoid, stable_seed, v6_outputs, write_csv, write_json
from protocol.datasets import load_fold


class HistoryEAEEGNet(nn.Module):
    def __init__(self, channels: int, alignments: np.ndarray, use_ea: bool):
        super().__init__()
        self.use_ea = bool(use_ea)
        self.register_buffer("alignment", torch.as_tensor(alignments, dtype=torch.float32), persistent=True)
        self.temporal = nn.Conv2d(1, 16, (1, 32), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.spatial = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.5)
        self.depth = nn.Conv2d(32, 32, (1, 8), padding="same", groups=32, bias=False)
        self.point = nn.Conv2d(32, 32, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.5)
        self.embedding = nn.Sequential(nn.Linear(32 * 15, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def front_end(self, x: torch.Tensor, subject_index: torch.Tensor) -> torch.Tensor:
        value = x[..., ::2]
        value = value - value.mean(dim=-1, keepdim=True)
        if self.use_ea:
            value = torch.bmm(self.alignment[subject_index], value)
        energy = torch.sqrt(torch.clamp(torch.mean(torch.square(value), dim=(-2, -1), keepdim=True), min=1e-10))
        return value / energy

    def forward_features(self, x: torch.Tensor, subject_index: torch.Tensor) -> torch.Tensor:
        value = self.front_end(x, subject_index).unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.depth(value)
        value = self.point(value)
        value = self.drop2(self.pool2(F.elu(self.bn3(value))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor, subject_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x, subject_index))


def _paths(benchmark: str) -> tuple[Path, Path, Path]:
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    return (
        CACHE / f"{prefix}_RAW_EPOCHS_FLOAT16.npy",
        CACHE / f"{prefix}_RAW_METADATA.parquet",
        CACHE / f"{prefix}_HISTORY_EA_MATRICES.npy",
    )


def _batch(raw, indices, metadata: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(raw, torch.Tensor):
        tensor_index = torch.as_tensor(indices, dtype=torch.long, device=device)
        x = raw[tensor_index].float()
        cpu_index = tensor_index.detach().cpu().numpy()
    else:
        values = np.asarray(raw[indices], dtype=np.float32)
        x = torch.from_numpy(values).to(device)
        cpu_index = np.asarray(indices, dtype=int)
    y = torch.as_tensor(metadata.label.to_numpy(int)[cpu_index], dtype=torch.long, device=device)
    subject = torch.as_tensor(metadata.subject_index.to_numpy(int)[cpu_index], dtype=torch.long, device=device)
    return x, y, subject


def _train(
    model: HistoryEAEEGNet,
    raw,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(getattr(model, "learning_rate", 1e-3)), weight_decay=5e-4)
    batch_size = int(getattr(model, "batch_size", 512))
    label_smoothing = float(getattr(model, "label_smoothing", 0.0))
    alignment_weight = float(getattr(model, "session_alignment_weight", 0.0))
    generator = torch.Generator(device=device).manual_seed(seed)
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    history = []
    for epoch in range(int(epochs)):
        model.train()
        permutation = train_tensor[torch.randperm(len(train_tensor), generator=generator, device=device)]
        total = 0.0
        count = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start:start + batch_size]
            x, y, subject = _batch(raw, index, metadata, device)
            cpu_index = index.detach().cpu().numpy() if isinstance(index, torch.Tensor) else np.asarray(index, dtype=int)
            sessions = torch.as_tensor(metadata.session_id.to_numpy(int)[cpu_index], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if alignment_weight > 0.0:
                    feature = model.forward_features(x, subject)
                    logits = model.head(feature)
                else:
                    feature = None
                    logits = model(x, subject)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                if feature is not None:
                    normalized = F.normalize(feature.float(), dim=1)
                    alignment_terms = []
                    for label_value in torch.unique(y):
                        class_mask = y == label_value
                        class_center = F.normalize(normalized[class_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                        for session_value in torch.unique(sessions[class_mask]):
                            cell = class_mask & (sessions == session_value)
                            if int(cell.sum()) >= 2:
                                session_center = F.normalize(normalized[cell].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                                alignment_terms.append(1.0 - torch.sum(session_center * class_center))
                    if alignment_terms:
                        loss = loss + alignment_weight * torch.stack(alignment_terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
            count += len(index)
        history.append({"epoch": epoch + 1, "train_loss": total / max(count, 1)})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[EA EEGNet] epoch={epoch + 1}/{epochs} loss={history[-1]['train_loss']:.4f}", flush=True)
    return history


def _extract(
    model: HistoryEAEEGNet,
    raw,
    metadata: pd.DataFrame,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((len(metadata), 64), dtype=np.float32)
    logits = np.empty(len(metadata), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        batch_size = int(getattr(model, "batch_size", 512))
        for start in range(0, len(metadata), batch_size):
            index = np.arange(start, min(start + batch_size, len(metadata)))
            x, _, subject = _batch(raw, index, metadata, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                feature = model.forward_features(x, subject)
                output = model.head(feature)
            features[index] = feature.float().cpu().numpy()
            logits[index] = (output[:, 1] - output[:, 0]).float().cpu().numpy()
    return features, logits


def _subject_head(history_features: np.ndarray, history_labels: np.ndarray, future_features: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(history_features)
    model = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=2_000, random_state=V7_SEED)
    model.fit(scaler.transform(history_features), history_labels)
    return np.asarray(model.decision_function(scaler.transform(future_features)), dtype=float)


def _calibration(history_logit: np.ndarray, history_labels: np.ndarray, future_logit: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2_000, random_state=V7_SEED)
    model.fit(history_logit[:, None], history_labels)
    return np.asarray(model.decision_function(future_logit[:, None]), dtype=float)


def _strong_anchor(benchmark: str) -> tuple[pd.DataFrame, str]:
    if benchmark == "openbmi":
        path = v6_outputs() / "diagnostics" / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv"
        method = "MI_SPECIFIC_BACKBONE_ADAPTED"
    else:
        path = v6_outputs() / "diagnostics" / "WBCIC_FUTURE_SESSION_POPULATION_PREDICTIONS.csv"
        method = "V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED"
    frame = pd.read_csv(path)
    return frame.loc[frame.method_id.astype(str).eq(method)].copy(), method


def _frame(data, subject: str, metadata: pd.DataFrame, future_mask: np.ndarray, probability: np.ndarray, method_id: str, history_used: bool) -> pd.DataFrame:
    part = metadata.loc[future_mask].copy()
    return pd.DataFrame({
        "benchmark": data.benchmark,
        "method_id": method_id,
        "trial_uid": part.trial_uid.astype(str).to_numpy(),
        "subject_id": str(subject),
        "outer_fold": data.fold,
        "label": part.label.to_numpy(int),
        "probability": probability,
        "prediction": probability >= 0.5,
        "target_history_labels_used": history_used,
        "target_future_labels_used_for_fit": False,
        "exploratory": True,
        "OUTER_TEST_USED": False,
    })


def _leaderboard(predictions: pd.DataFrame, reference_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows = []
    for (method, subject), group in predictions.groupby(["method_id", "subject_id"]):
        subject_rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "method_id": str(method),
            "subject_id": str(subject),
            "outer_fold": int(group.outer_fold.iloc[0]),
            "BA": float(balanced_accuracy_score(group.label, group.prediction)),
            "NLL": float(log_loss(group.label, np.clip(group.probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
        })
    subjects = pd.DataFrame(subject_rows)
    reference = subjects.loc[subjects.method_id.eq(reference_method)].set_index("subject_id").BA
    rows = []
    for method, group in subjects.groupby("method_id"):
        delta = group.BA.to_numpy(float) - reference.loc[group.subject_id].to_numpy(float)
        rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "method_id": method,
            "subjects": len(group),
            "mean_subject_BA": float(group.BA.mean()),
            "mean_subject_NLL": float(group.NLL.mean()),
            "reference_method_id": reference_method,
            "Delta_BA": float(delta.mean()),
            "positive_subject_fraction": float(np.mean(delta > 0.0)),
            "nonnegative_subject_fraction": float(np.mean(delta >= 0.0)),
            "worst_subject_delta": float(delta.min()),
            "target_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        })
    return pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False), subjects


def run(benchmark: str, variant: str, epochs: int, folds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    ensure_directories()
    raw_path, metadata_path, alignment_path = _paths(benchmark)
    raw_disk = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    alignments = np.load(alignment_path, allow_pickle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    raw = torch.from_numpy(np.asarray(raw_disk)).to(device) if device.type == "cuda" else raw_disk
    use_ea = variant == "ea"
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    method_prefix = "HISTORY_EA_EEGNET" if use_ea else "CAPACITY_MATCHED_EEGNET"
    anchor, anchor_method = _strong_anchor(benchmark)
    anchor_index = anchor.set_index("trial_uid")
    predictions = [anchor]
    training_rows = []
    for fold in folds:
        data = load_fold(benchmark, fold)
        nonoutcome = set(data.nonoutcome_subjects)
        train_indices = np.flatnonzero(metadata.subject_id.astype(str).isin(nonoutcome).to_numpy())
        model = HistoryEAEEGNet(int(raw.shape[1]), alignments, use_ea).to(device)
        history = _train(model, raw, metadata, train_indices, epochs, device, stable_seed(V7_SEED, benchmark, variant, fold))
        features, logits = _extract(model, raw, metadata, device)
        cache_prefix = CACHE / f"{prefix}_{method_prefix}_FOLD_{fold}"
        np.save(cache_prefix.with_name(cache_prefix.name + "_FEATURES.npy"), features, allow_pickle=False)
        np.save(cache_prefix.with_name(cache_prefix.name + "_LOGITS.npy"), logits, allow_pickle=False)
        fold_metadata = metadata.copy()
        fold_metadata["outer_fold"] = fold
        fold_metadata.to_parquet(cache_prefix.with_name(cache_prefix.name + "_METADATA.parquet"), index=False)
        checkpoint = cache_prefix.with_suffix(".pt")
        torch.save({
            "model": model.state_dict(),
            "benchmark": benchmark,
            "variant": variant,
            "epochs": epochs,
            "nonoutcome_subjects": sorted(nonoutcome),
            "outcome_subjects_hashed_only": True,
            "target_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        }, checkpoint)
        for subject in data.outcome_subjects:
            subject_mask = metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
            history_mask = subject_mask & metadata.session_id.astype(int).isin(data.history_sessions).to_numpy()
            future_mask = subject_mask & metadata.session_id.astype(int).eq(data.future_session).to_numpy()
            history_y = metadata.loc[history_mask, "label"].to_numpy(int)
            frozen_logit = logits[future_mask].astype(float)
            head_logit = _subject_head(features[history_mask], history_y, features[future_mask])
            calibrated_logit = _calibration(logits[history_mask].astype(float), history_y, frozen_logit)
            values = {
                f"{method_prefix}_FROZEN": frozen_logit,
                f"{method_prefix}_FIXED_HEAD": 0.5 * (frozen_logit + head_logit),
                f"{method_prefix}_FIXED_CALIBRATION": 0.5 * (frozen_logit + calibrated_logit),
            }
            anchor_probability = anchor_index.loc[metadata.loc[future_mask, "trial_uid"].astype(str), "probability"].to_numpy(float)
            anchor_logit = logit(anchor_probability)
            for method_id, value in list(values.items()):
                values[f"ANCHOR_BLEND__{method_id}"] = 0.5 * (anchor_logit + value)
            for method_id, value in values.items():
                predictions.append(_frame(data, subject, metadata, future_mask, sigmoid(value), method_id, "FIXED_HEAD" in method_id or "CALIBRATION" in method_id))
        training_rows.append({
            "benchmark": data.benchmark,
            "outer_fold": fold,
            "variant": variant,
            "epochs": epochs,
            "last_train_loss": history[-1]["train_loss"],
            "history": json.dumps(history),
            "population_fit_subjects": len(nonoutcome),
            "outcome_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        })
        print(f"[{benchmark} {variant}] fold={fold} complete loss={history[-1]['train_loss']:.4f}", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    if folds != (0, 1, 2, 3, 4):
        allowed = set().union(*(set(load_fold(benchmark, fold).outcome_subjects) for fold in folds))
        prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    leaderboard, subjects = _leaderboard(prediction_frame, anchor_method)
    tag = f"{prefix}_{method_prefix}"
    write_csv(LEADERBOARD / f"{tag}.csv", leaderboard)
    write_csv(DIAGNOSTICS / f"{tag}_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / f"{tag}_SUBJECT_RESULTS.csv", subjects)
    write_csv(DIAGNOSTICS / f"{tag}_TRAINING.csv", pd.DataFrame(training_rows))
    write_json(PROTOCOL / f"{tag}_AUDIT.json", {
        "history_alignment": "subject covariance from legal history sessions only" if use_ea else "identity capacity-matched control",
        "population_fit": "all non-outcome subject sessions",
        "outcome_adaptation": "fixed S1 or S1/S2 history-only head/calibration",
        "outcome_future_labels_used_for_fit_or_selection": False,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    print(leaderboard.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--variant", choices=("ea", "identity"), default="ea")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (30 if args.benchmark == "openbmi" else 20)
    run(args.benchmark, args.variant, epochs, (args.fold,) if args.fold is not None else (0, 1, 2, 3, 4))


if __name__ == "__main__":
    main()
