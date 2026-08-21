"""Shrinkage-SPD history whitening with a query-trained spatial metric bank."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_banks.run_query_bank import Episode, _parts, _upsert
from common import CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED, ensure_directories, logit, stable_seed, v7_outputs, write_csv, write_json
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


@dataclass
class SPDEpisode:
    shell: Episode
    history_cov: np.ndarray
    history_y: np.ndarray
    history_session: np.ndarray
    query_cov: np.ndarray
    query_base: np.ndarray


class SPDTransportBank(nn.Module):
    def __init__(self, dimension: int, experts: int, filters: int, max_sessions: int):
        super().__init__()
        self.experts = experts
        self.filters = filters
        generator = torch.Generator().manual_seed(81337)
        rows = []
        for _ in range(experts):
            random = torch.randn(dimension, filters, generator=generator)
            q, _ = torch.linalg.qr(random, mode="reduced")
            rows.append(q.T)
        self.spatial = nn.Parameter(torch.stack(rows))
        self.session_logits = nn.Parameter(torch.zeros(experts, max_sessions))
        if max_sessions > 1:
            with torch.no_grad():
                for expert in range(experts):
                    self.session_logits[expert] = torch.linspace(-1.0, 1.0, max_sessions) * expert / max(experts - 1, 1)
        self.gain_raw = nn.Parameter(torch.linspace(-1.0, 1.0, experts))
        self.bias = nn.Parameter(torch.zeros(experts))
        self.base_scale = nn.Parameter(torch.ones(experts))

    def _features(self, covariance: torch.Tensor) -> torch.Tensor:
        spatial = F.normalize(self.spatial, dim=2)
        power = torch.einsum("kpd,ndm,kpm->nkp", spatial, covariance, spatial)
        return torch.log(torch.clamp(power, min=1e-6))

    def episode_logits(self, episode: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._features(episode["history_cov"])
        query = self._features(episode["query_cov"])
        sessions = torch.unique(episode["history_session"], sorted=True)
        session_weight = torch.softmax(self.session_logits[:, :len(sessions)], dim=1)
        prototypes = []
        for label in (0.0, 1.0):
            cells = []
            for session in sessions:
                mask = (episode["history_session"] == session) & (episode["history_y"] == label)
                cells.append(history[mask].mean(dim=0))
            cells = torch.stack(cells, dim=1)
            prototypes.append(torch.sum(cells * session_weight[:, :, None], dim=1))
        distance0 = torch.mean(torch.square(query - prototypes[0][None]), dim=2)
        distance1 = torch.mean(torch.square(query - prototypes[1][None]), dim=2)
        score = distance0 - distance1
        gain = F.softplus(self.gain_raw) + 0.05
        logits = self.base_scale[None] * episode["query_base"][:, None] + gain[None] * score + self.bias[None]
        return logits, score


def _build_covariance_cache(benchmark: str, device: torch.device) -> tuple[np.ndarray, pd.DataFrame, dict]:
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    path = CACHE / f"{prefix}_V8_SEARCH_SPECTRAL_COV_FLOAT16.npy"
    metadata_path = CACHE / f"{prefix}_V8_SEARCH_SPECTRAL_COV_METADATA.parquet"
    if path.is_file() and metadata_path.is_file():
        covariance = np.load(path, mmap_mode="r", allow_pickle=False)
        metadata = pd.read_parquet(metadata_path)
        if len(covariance) != len(metadata) or metadata.subject_id.nunique() != (40 if benchmark == "openbmi" else 31):
            raise RuntimeError("Malformed V8 search spectral covariance cache")
        return covariance, metadata, {"reused": True, "rows": len(metadata), "path": str(path)}
    canonical = pd.read_parquet(CACHE / f"{prefix}_SEARCH_ROWS_FOLD_0.parquet").copy()
    canonical = canonical.sort_values("source_index").reset_index(drop=True)
    canonical.insert(0, "cov_index", np.arange(len(canonical), dtype=np.int64))
    raw = np.load(v7_outputs() / "cache" / f"{prefix}_RAW_EPOCHS_FLOAT16.npy", mmap_mode="r", allow_pickle=False)
    channels = int(raw.shape[1])
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(canonical), channels, channels))
    frequencies = torch.fft.rfftfreq(int(raw.shape[2]), d=1.0 / 250.0).to(device)
    band = (frequencies >= 8.0) & (frequencies <= 30.0)
    identity = torch.eye(channels, dtype=torch.float32, device=device)
    source_indices = canonical.source_index.to_numpy(int)
    for start in range(0, len(canonical), 32):
        index = source_indices[start:start + 32]
        x = torch.as_tensor(np.asarray(raw[index], dtype=np.float32), dtype=torch.float32, device=device)
        x = x - x.mean(dim=2, keepdim=True)
        spectrum = torch.fft.rfft(x, dim=2)[:, :, band]
        covariance = (
            torch.einsum("ncf,ndf->ncd", spectrum.real, spectrum.real)
            + torch.einsum("ncf,ndf->ncd", spectrum.imag, spectrum.imag)
        ) / int(band.sum())
        trace = torch.diagonal(covariance, dim1=1, dim2=2).sum(dim=1)
        covariance = covariance / torch.clamp(trace[:, None, None], min=1e-12)
        covariance = 0.95 * covariance + 0.05 * identity[None] / channels
        output[start:start + len(index)] = covariance.cpu().numpy().astype(np.float16)
        if start == 0 or (start // 32) % 100 == 0:
            print(f"[{benchmark} SPD cache] {min(start + len(index), len(canonical))}/{len(canonical)}", flush=True)
    output.flush()
    del output
    canonical.to_parquet(metadata_path, index=False)
    return np.load(path, mmap_mode="r", allow_pickle=False), canonical, {
        "reused": False, "rows": len(canonical), "channels": channels,
        "sampling_rate_hz": 250, "band_hz": [8.0, 30.0], "shrinkage": 0.05,
        "internal_holdout_rows": 0, "path": str(path), "OUTER_TEST_USED": False,
    }


def _project_covariances(covariance: np.ndarray, indices: np.ndarray, projection: np.ndarray, device: torch.device) -> np.ndarray:
    result = np.empty((len(indices), projection.shape[1], projection.shape[1]), dtype=np.float32)
    p = torch.as_tensor(projection, dtype=torch.float32, device=device)
    for start in range(0, len(indices), 256):
        values = torch.as_tensor(np.asarray(covariance[indices[start:start + 256]], dtype=np.float32), device=device)
        projected = p.T[None] @ values @ p[None]
        result[start:start + len(values)] = projected.cpu().numpy()
    return result


def _inverse_root(matrix: np.ndarray) -> np.ndarray:
    value, vector = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = max(float(np.mean(value)) * 1e-4, 1e-8)
    return ((vector * (1.0 / np.sqrt(np.maximum(value, floor)))[None, :]) @ vector.T).astype(np.float32)


def _aligned_episode(
    data,
    subject: str,
    covariances: np.ndarray,
    cov_metadata: pd.DataFrame,
    projection: np.ndarray,
    device: torch.device,
) -> SPDEpisode:
    rows = cov_metadata.loc[cov_metadata.subject_id.astype(str).eq(str(subject))].copy()
    history = rows.session_id.astype(int).isin(data.protocol.history_sessions).to_numpy()
    future = rows.session_id.astype(int).eq(data.protocol.future_session).to_numpy()
    hcov = _project_covariances(covariances, rows.loc[history, "cov_index"].to_numpy(int), projection, device)
    qcov = _project_covariances(covariances, rows.loc[future, "cov_index"].to_numpy(int), projection, device)
    reference = hcov.mean(axis=0)
    whitening = _inverse_root(reference)
    hcov = np.einsum("ab,nbc,cd->nad", whitening, hcov, whitening, optimize=True)
    qcov = np.einsum("ab,nbc,cd->nad", whitening, qcov, whitening, optimize=True)
    uids = rows.loc[future, "trial_uid"].astype(str).to_numpy()
    feature_rows = data.metadata.set_index("trial_uid").loc[uids]
    source_index = feature_rows.source_index.to_numpy(int)
    query_base = np.asarray(data.logits[source_index], dtype=np.float32)
    y = rows.loc[future, "label"].to_numpy(np.float32)
    shell = Episode(
        str(subject), data.source_fold, np.empty(0), np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0),
        np.empty((len(y), 0)), query_base, y, uids,
    )
    return SPDEpisode(
        shell=shell,
        history_cov=hcov.astype(np.float32),
        history_y=rows.loc[history, "label"].to_numpy(np.float32),
        history_session=rows.loc[history, "session_id"].to_numpy(np.int64),
        query_cov=qcov.astype(np.float32),
        query_base=query_base,
    )


def _tensor_episode(value: SPDEpisode, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history_cov": torch.as_tensor(value.history_cov, dtype=torch.float32, device=device),
        "history_y": torch.as_tensor(value.history_y, dtype=torch.float32, device=device),
        "history_session": torch.as_tensor(value.history_session, dtype=torch.long, device=device),
        "query_cov": torch.as_tensor(value.query_cov, dtype=torch.float32, device=device),
        "query_base": torch.as_tensor(value.query_base, dtype=torch.float32, device=device),
        "query_y": torch.as_tensor(value.shell.query_y, dtype=torch.float32, device=device),
    }


def _balanced_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = []
    for expert in range(logits.shape[1]):
        terms = []
        for label in (0.0, 1.0):
            mask = labels == label
            terms.append(F.binary_cross_entropy_with_logits(logits[mask, expert], labels[mask]))
        result.append(0.5 * (terms[0] + terms[1]))
    return torch.stack(result)


def _train(train: list[SPDEpisode], target: list[SPDEpisode], experts: int, filters: int, epochs: int, tau: float, lambda_mean: float, seed: int, device: torch.device):
    train_tensors = [_tensor_episode(item, device) for item in train]
    target_tensors = [_tensor_episode(item, device) for item in target]
    max_sessions = max(len(np.unique(item.history_session)) for item in train)
    torch.manual_seed(seed)
    model = SPDTransportBank(train[0].history_cov.shape[1], experts, filters, max_sessions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    log_k = float(np.log(experts))
    best_value = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses, base_losses = [], []
        for episode in train_tensors:
            logits, _ = model.episode_logits(episode)
            losses.append(_balanced_losses(logits, episode["query_y"]))
            base_losses.append(_balanced_losses(episode["query_base"][:, None].expand(-1, experts), episode["query_y"]))
        losses = torch.stack(losses)
        base_loss = torch.stack(base_losses).mean()
        coverage = (-tau * torch.logsumexp(-losses / tau, dim=1) + tau * log_k).mean()
        mean_loss = losses.mean()
        assignment = torch.softmax(-losses / tau, dim=1)
        balance = experts * torch.mean(torch.square(assignment.mean(dim=0) - 1.0 / experts))
        spatial = F.normalize(model.spatial, dim=2)
        row_gram = spatial @ spatial.transpose(1, 2)
        orthogonality = torch.mean(torch.square(row_gram - torch.eye(filters, device=device)[None]))
        flat = F.normalize(spatial.flatten(1), dim=1)
        redundancy = torch.mean(torch.square(flat @ flat.T - torch.eye(experts, device=device)))
        competence = torch.relu(losses.mean(dim=0) - base_loss - 0.04).mean()
        objective = coverage + lambda_mean * mean_loss + 0.10 * balance + 0.03 * competence + 0.005 * orthogonality + 0.004 * redundancy
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(objective.detach())
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
        if epoch in {0, 49, 99, 199, 399, epochs - 1}:
            history.append({
                "epoch": epoch + 1, "objective": value, "coverage_loss": float(coverage.detach()),
                "mean_loss": float(mean_loss.detach()), "assignment": assignment.mean(dim=0).detach().cpu().numpy().tolist(),
                "gain": (F.softplus(model.gain_raw) + 0.05).detach().cpu().numpy().tolist(),
            })
    model.load_state_dict(best_state)
    model.eval()
    predictions, bases = [], []
    with torch.no_grad():
        for episode in target_tensors:
            logits, _ = model.episode_logits(episode)
            predictions.append(logits.cpu().numpy())
            bases.append(episode["query_base"].cpu().numpy())
    audit = {
        "best_objective": best_value, "history": history,
        "gain": (F.softplus(model.gain_raw) + 0.05).detach().cpu().numpy().tolist(),
        "session_weight": torch.softmax(model.session_logits, dim=1).detach().cpu().numpy().tolist(),
    }
    return np.concatenate(predictions), np.concatenate(bases), audit, model


def run(benchmark: str, experts: int, filters: int, reduced_dimension: int, epochs: int, folds: tuple[int, ...], tau: float, lambda_mean: float) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    benchmark_name = "OpenBMI_MI_S1_to_S2" if benchmark == "openbmi" else "WBCIC_S1S2_to_S3_authorized_development"
    family_slug = f"SPD_TRANSPORT_K{experts}_P{filters}_D{reduced_dimension}"
    family_id = f"{benchmark_name}__{family_slug}"
    covariance, cov_metadata, cache_audit = _build_covariance_cache(benchmark, device)
    baseline, baseline_source_method = baseline_predictions(benchmark)
    protocol = load_feature_fold(benchmark, 0, "MI_SPECIFIC").protocol
    baseline = baseline.loc[baseline.subject_id.astype(str).isin(protocol.search_subjects)].copy()
    baseline["method_id"] = "B_STRONG_MATCHED_V7"
    baseline["family_id"] = family_id
    baseline["source_fold"] = baseline.outer_fold.astype(int)
    baseline["benchmark"] = benchmark_name
    baseline["internal_holdout_used"] = False
    predictions = [baseline]
    primary = [f"{family_slug}__E{k}_RESIDUAL50" for k in range(experts)]
    audits = []
    for fold in folds:
        data = load_feature_fold(benchmark, fold, "MI_SPECIFIC")
        assert_search_only(list(data.meta_subjects) + list(data.search_outcome_subjects), benchmark)
        if not data.search_outcome_subjects:
            continue
        meta_mask = cov_metadata.subject_id.astype(str).isin(data.meta_subjects).to_numpy()
        pooled = np.asarray(covariance[cov_metadata.loc[meta_mask, "cov_index"].to_numpy(int)], dtype=np.float32).mean(axis=0)
        eigenvalue, eigenvector = np.linalg.eigh(0.5 * (pooled + pooled.T))
        projection = eigenvector[:, np.argsort(eigenvalue)[-reduced_dimension:]]
        meta = [_aligned_episode(data, subject, covariance, cov_metadata, projection, device) for subject in data.meta_subjects]
        outcome = [_aligned_episode(data, subject, covariance, cov_metadata, projection, device) for subject in data.search_outcome_subjects]
        adapted, cached_base, audit, model = _train(meta, outcome, experts, filters, epochs, tau, lambda_mean, stable_seed(V8_SEED, benchmark, family_slug, fold), device)
        uids = np.concatenate([item.shell.query_uid for item in outcome])
        locked = logit(baseline.set_index("trial_uid").loc[uids, "probability"].to_numpy(float))
        shells = [item.shell for item in outcome]
        for expert in range(experts):
            residual = adapted[:, expert] - cached_base
            predictions.extend(_parts(shells, adapted[:, expert], family_id, f"{family_slug}__E{expert}_STANDALONE"))
            predictions.extend(_parts(shells, locked + 0.5 * residual, family_id, primary[expert]))
        torch.save({
            "model": model.state_dict(), "projection": projection, "source_fold": fold,
            "meta_subjects": list(data.meta_subjects), "search_outcome_subjects": list(data.search_outcome_subjects),
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        }, CACHE / f"{prefix}_{family_slug}_FOLD_{fold}.pt")
        audits.append({
            "benchmark": benchmark_name, "family_id": family_id, "source_fold": fold,
            "meta_subjects": len(meta), "search_outcome_subjects": len(outcome), **audit,
            "internal_holdout_used": False, "OUTER_TEST_USED": False,
        })
        print(f"[{benchmark} {family_slug}] fold={fold} objective={audit['best_objective']:.5f}", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(set(load_feature_fold(benchmark, fold, "MI_SPECIFIC").search_outcome_subjects) for fold in folds))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "experts": experts, "spatial_filters": filters, "reduced_dimension": reduced_dimension,
        "epochs": epochs, "tau": tau, "lambda_mean": lambda_mean, "folds": list(folds),
        "baseline_source_method": baseline_source_method,
        "training_objective": "history-whitened shrinkage SPD class prototypes with query-trained spatial metric coverage",
        "deployment_transform": "locked strong anchor plus half learned SPD residual",
        "covariance_cache": cache_audit,
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
        "covariance_rows": "V8_SEARCH only", "frequency_band_hz": [8.0, 30.0], "shrinkage": 0.05,
        "spatial_reduction_fit": "source-fold non-outcome V8_SEARCH subjects only",
        "target_whitening": "legal target history only", "target_future_batch_statistics_used": False,
        "search_outcome_future_labels_used_for_fit_or_selection": False, "internal_holdout_used": False,
        "WBCIC_outer_split_opened": False, "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural hypothesis: shrinkage spectral SPD geometry, history-only whitening, and learned spatial metrics provide complementary cross-session structure absent from deep embeddings.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--filters", type=int, default=8)
    parser.add_argument("--reduced-dimension", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--lambda-mean", type=float, default=0.35)
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, args.experts, args.filters, args.reduced_dimension, args.epochs, folds, args.tau, args.lambda_mean)


if __name__ == "__main__":
    main()
