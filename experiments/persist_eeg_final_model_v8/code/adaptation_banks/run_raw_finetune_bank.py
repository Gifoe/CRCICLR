"""Raw-encoder legal-history adaptation bank and transformation headroom audit.

This family reuses V6 source-fold population checkpoints and exposes several
predeclared, discovery-query-screened adaptation rules as residual actions on
the locked V7 generic anchor.  It diagnoses whether deeper encoder movement,
rather than feature-head adaptation, creates deployment-level headroom.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_banks.run_query_bank import _parts, _upsert, Episode
from common import CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED, ensure_directories, logit, stable_seed, stage0_root, v6_outputs, v7_outputs, write_csv, write_json
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


class EEGNet(nn.Module):
    def __init__(self, channels: int, f1: int = 16, f2: int = 32, dropout: float = 0.5, wbcic_style: bool = False):
        super().__init__()
        self.wbcic_style = wbcic_style
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, 2 * f1, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * f1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        depth_name = "separable_depth" if wbcic_style else "depth"
        point_name = "separable_point" if wbcic_style else "point"
        setattr(self, depth_name, nn.Conv2d(2 * f1, 2 * f1, (1, 16), padding="same", groups=2 * f1, bias=False))
        setattr(self, point_name, nn.Conv2d(2 * f1, f2, 1, bias=False))
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        if wbcic_style:
            self.embedding = nn.Linear(f2 * 31, 64)
            self.embedding_norm = nn.LayerNorm(64)
        else:
            self.embedding = nn.Sequential(nn.Linear(f2 * 31, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = self.bn1(self.temporal(x.unsqueeze(1)))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        depth = self.separable_depth if self.wbcic_style else self.depth
        point = self.separable_point if self.wbcic_style else self.point
        value = self.drop2(self.pool2(F.elu(self.bn3(point(depth(value))))))
        if self.wbcic_style:
            return self.embedding_norm(F.elu(self.embedding(value.flatten(1))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class ShallowConvNet(nn.Module):
    def __init__(self, channels: int, filters: int = 40, dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, filters, (1, 25), bias=False)
        self.spatial = nn.Conv2d(filters, filters, (channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(filters * 61, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = self.bn(self.spatial(self.temporal(x.unsqueeze(1))))
        value = torch.log(torch.clamp(self.pool(torch.square(value)), min=1e-6))
        return self.embedding(self.dropout(value).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def _source(benchmark: str, fold: int, channels: int) -> tuple[nn.Module, dict, dict]:
    if benchmark == "wbcic":
        path = v6_outputs() / "cache" / f"WBCIC_LARGE_EEGNET_FOLD_{fold}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = EEGNet(channels, 16, 32, 0.5, True)
    else:
        path = v6_outputs() / "cache" / f"OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_{fold}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        configuration = payload["configuration"]
        if configuration["architecture"] == "eegnet":
            model = EEGNet(channels, int(configuration["f1"]), int(configuration["f2"]), float(configuration["dropout"]), False)
        else:
            model = ShallowConvNet(channels, int(configuration["filters"]), float(configuration["dropout"]))
    model.load_state_dict(payload["model"], strict=True)
    return model, payload, {"source_checkpoint": str(path), "source_payload": {k: v for k, v in payload.items() if k != "model"}}


def _normalizer(benchmark: str, fold: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    if benchmark == "wbcic":
        return np.zeros((channels, 1), dtype=np.float32), np.ones((channels, 1), dtype=np.float32)
    best = (54, 54, 25, 47, 44)[fold]
    path = stage0_root() / "outputs" / "persist_eeg_p2p3" / "backbone" / "checkpoints" / "eegnet" / f"fold-{fold}" / "seed-0" / "trajectory" / f"epoch-{best:03d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(payload["channel_mean"], dtype=np.float32)[:, None], np.asarray(payload["channel_std"], dtype=np.float32)[:, None]


def _configs(benchmark: str) -> list[dict]:
    if benchmark == "wbcic":
        return [
            {"id": "ADABN", "strategy": "adabn", "lr": 0.0, "epochs": 1},
            {"id": "HEAD", "strategy": "head", "lr": 3e-4, "epochs": 24},
            {"id": "TAIL", "strategy": "tail", "lr": 1e-4, "epochs": 8},
            {"id": "FULL", "strategy": "full", "lr": 1e-4, "epochs": 8},
        ]
    return [
        {"id": "ADABN", "strategy": "adabn", "lr": 0.0, "epochs": 1},
        {"id": "HEAD_SHORT", "strategy": "head", "lr": 1e-3, "epochs": 10},
        {"id": "HEAD_LONG", "strategy": "head", "lr": 3e-4, "epochs": 30},
        {"id": "TAIL_SLOW", "strategy": "tail", "lr": 1e-4, "epochs": 10},
        {"id": "TAIL_FAST", "strategy": "tail", "lr": 3e-4, "epochs": 10},
        {"id": "FULL_SLOW", "strategy": "full", "lr": 1e-5, "epochs": 10},
        {"id": "FULL_LONG", "strategy": "full", "lr": 3e-5, "epochs": 30},
    ]


def _set_trainable(model: nn.Module, strategy: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        keep = (
            strategy == "full"
            or (strategy == "head" and name.startswith("head"))
            or (strategy == "tail" and (name.startswith("embedding") or name.startswith("head")))
        )
        if keep:
            parameter.requires_grad_(True)
            selected.append(parameter)
    return selected


def _predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            xb = torch.as_tensor(x[start:start + 128], dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(xb)
            result.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
    return np.concatenate(result)


def _adapt(model: nn.Module, history_x: np.ndarray, history_y: np.ndarray, configuration: dict, device: torch.device, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    adapted = copy.deepcopy(model).to(device)
    strategy = configuration["strategy"]
    if strategy == "adabn":
        adapted.train()
        with torch.inference_mode():
            for start in range(0, len(history_x), 64):
                adapted(torch.as_tensor(history_x[start:start + 64], dtype=torch.float32, device=device))
        return adapted
    parameters = _set_trainable(adapted, strategy)
    optimizer = torch.optim.AdamW(parameters, lr=float(configuration["lr"]), weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    y_tensor = torch.as_tensor(history_y, dtype=torch.long)
    for _ in range(int(configuration["epochs"])):
        adapted.train()
        permutation = torch.randperm(len(history_y), generator=generator)
        for start in range(0, len(history_y), 64):
            index = permutation[start:start + 64]
            xb = torch.as_tensor(history_x[index], dtype=torch.float32, device=device)
            yb = y_tensor[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(adapted(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
    return adapted


def _subject_raw(raw: np.ndarray, metadata: pd.DataFrame, subject: str, history_sessions: tuple[int, ...], future_session: int, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
    history = mask & metadata.session_id.astype(int).isin(history_sessions).to_numpy()
    future = mask & metadata.session_id.astype(int).eq(future_session).to_numpy()
    hi = metadata.loc[history, "source_index"].to_numpy(int)
    fi = metadata.loc[future, "source_index"].to_numpy(int)
    hx = (np.asarray(raw[hi], dtype=np.float32) - mean[None]) / np.maximum(std[None], 1e-8)
    fx = (np.asarray(raw[fi], dtype=np.float32) - mean[None]) / np.maximum(std[None], 1e-8)
    return hx, metadata.loc[history, "label"].to_numpy(int), fx, metadata.loc[future, "label"].to_numpy(int), metadata.loc[future, "trial_uid"].astype(str).to_numpy()


def run(benchmark: str, folds: tuple[int, ...]) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    benchmark_name = "OpenBMI_MI_S1_to_S2" if benchmark == "openbmi" else "WBCIC_S1S2_to_S3_authorized_development"
    family_slug = "RAW_ENCODER_FINETUNE_BANK"
    family_id = f"{benchmark_name}__{family_slug}"
    baseline, baseline_source_method = baseline_predictions(benchmark)
    protocol = load_feature_fold(benchmark, 0, "CONFORMER_NORM").protocol
    baseline = baseline.loc[baseline.subject_id.astype(str).isin(protocol.search_subjects)].copy()
    baseline["method_id"] = "B_STRONG_MATCHED_V7"
    baseline["family_id"] = family_id
    baseline["source_fold"] = baseline.outer_fold.astype(int)
    baseline["benchmark"] = benchmark_name
    baseline["internal_holdout_used"] = False
    predictions = [baseline]
    configs = _configs(benchmark)
    primary = [f"{family_slug}__{value['id']}_RESIDUAL50" for value in configs]
    audits = []
    raw = np.load(v7_outputs() / "cache" / f"{prefix}_RAW_EPOCHS_FLOAT16.npy", mmap_mode="r", allow_pickle=False)
    for fold in folds:
        data = load_feature_fold(benchmark, fold, "CONFORMER_NORM")
        assert_search_only(list(data.search_outcome_subjects), benchmark)
        if not data.search_outcome_subjects:
            continue
        source_model, source_payload, source_audit = _source(benchmark, fold, int(raw.shape[1]))
        source_model = source_model.to(device)
        mean, std = _normalizer(benchmark, fold, int(raw.shape[1]))
        for subject in data.search_outcome_subjects:
            hx, hy, fx, fy, uid = _subject_raw(raw, data.metadata, subject, data.protocol.history_sessions, data.protocol.future_session, mean, std)
            frozen = _predict(source_model, fx, device)
            locked = logit(baseline.set_index("trial_uid").loc[uid, "probability"].to_numpy(float))
            shell = Episode(str(subject), fold, np.empty(0), np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0), np.empty((len(fy), 0)), frozen, fy.astype(np.float32), uid)
            for configuration, method in zip(configs, primary):
                adapted_model = _adapt(source_model, hx, hy, configuration, device, stable_seed(V8_SEED, benchmark, fold, subject, configuration["id"]))
                adapted = _predict(adapted_model, fx, device)
                predictions.extend(_parts([shell], adapted, family_id, f"{family_slug}__{configuration['id']}_STANDALONE"))
                predictions.extend(_parts([shell], locked + 0.5 * (adapted - frozen), family_id, method))
            print(f"[{benchmark} {family_slug}] fold={fold} subject={subject} configs={len(configs)}", flush=True)
        audits.append({
            "benchmark": benchmark_name, "family_id": family_id, "source_fold": fold,
            "source": source_audit, "configs": configs, "search_outcome_subjects": len(data.search_outcome_subjects),
            "source_model_preexisted_V8_split": True, "internal_holdout_used": False, "OUTER_TEST_USED": False,
        })
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(set(load_feature_fold(benchmark, fold, "CONFORMER_NORM").search_outcome_subjects) for fold in folds))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "folds": list(folds), "experts": len(configs), "baseline_source_method": baseline_source_method,
        "training_objective": "V6 discovery-query-screened raw encoder rules; V8 uses fixed residual actions",
        "deployment_transform": "locked strong anchor plus half raw adaptation residual",
        "diagnostic_status": "structural raw-adaptation headroom audit, not a newly meta-learned direction bank",
    })
    tag = f"{prefix}_{family_slug}"
    write_csv(DIAGNOSTICS / f"{tag}_SEARCH_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / f"{tag}_SUBJECT_RESULTS.csv", report["subjects"])
    write_json(DIAGNOSTICS / f"{tag}_TRAINING_AUDIT.json", audits)
    write_json(HEADROOM / f"{tag}_HEADROOM.json", summary)
    write_csv(HEADROOM / f"{tag}_SUBJECT_ORACLE.csv", report["oracle"])
    write_csv(HEADROOM / f"{tag}_EXPERT_COMPETENCE.csv", report["competence"])
    write_csv(HEADROOM / f"{tag}_EXPERT_DIVERSITY.csv", report["diversity"])
    write_csv(HEADROOM / f"{tag}_ORACLE_BY_FOLD.csv", report["folds"])
    _upsert(HEADROOM / "HEADROOM_FAMILY_TABLE.csv", pd.DataFrame([summary]), ["benchmark", "family_id"])
    _upsert(HEADROOM / "EXPERT_COMPETENCE.csv", report["competence"], ["benchmark", "family_id", "method_id"])
    _upsert(HEADROOM / "EXPERT_DIVERSITY.csv", report["diversity"], ["benchmark", "family_id", "expert_left", "expert_right"])
    _upsert(HEADROOM / "SUBJECT_ORACLE.csv", report["oracle"], ["benchmark", "family_id", "subject_id"])
    _upsert(HEADROOM / "ORACLE_BY_FOLD.csv", report["folds"], ["benchmark", "family_id", "source_fold"])
    write_json(PROTOCOL / f"{tag}_LEGALITY.json", {
        "population_checkpoint": "locked pre-V8 V6 source-fold checkpoint",
        "target_adaptation": "V8_SEARCH outcome legal history only", "future_labels": "scoring only",
        "internal_holdout_used": False, "WBCIC_outer_split_opened": False, "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural diagnostic: test whether normalization, head, tail, or full raw-encoder movement supplies headroom absent from feature adapters.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, folds)


if __name__ == "__main__":
    main()
