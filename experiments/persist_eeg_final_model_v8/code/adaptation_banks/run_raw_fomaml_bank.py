"""First-order raw-signal meta-learning bank for legal future-session adaptation.

The population initialization is first trained only on source-fold V8_SEARCH
meta subjects.  Each FOMAML expert then performs a legal-history inner update
and receives its outer gradient exclusively from the later session of those
meta subjects.  Search outcome future labels are used only by the Phase-A
subject-oracle diagnostic.  Internal holdout and WBCIC outer rows are absent.
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
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_banks.run_query_bank import Episode, _parts, _upsert
from backbones.run_multiscale_bank import MultiScaleTCNBank, _subject_balanced_ce
from common import (
    CACHE, DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, V8_SEED,
    ensure_directories, logit, stable_seed, v7_outputs, write_csv, write_json,
)
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


EXPERT_CONFIGS = (
    {"id": "HEAD4", "strategy": "head", "inner_steps": 4, "inner_lr": 3e-3},
    {"id": "TAIL3", "strategy": "tail", "inner_steps": 3, "inner_lr": 3e-4},
    {"id": "FULL1", "strategy": "full", "inner_steps": 1, "inner_lr": 1e-4},
)


def _indices(metadata: pd.DataFrame, subject: str, history_sessions: tuple[int, ...], future_session: int) -> tuple[np.ndarray, np.ndarray]:
    subject_mask = metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
    history = subject_mask & metadata.session_id.astype(int).isin(history_sessions).to_numpy()
    future = subject_mask & metadata.session_id.astype(int).eq(future_session).to_numpy()
    return (
        metadata.loc[history, "local_index"].to_numpy(dtype=int, copy=True),
        metadata.loc[future, "local_index"].to_numpy(dtype=int, copy=True),
    )


def _balanced_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = []
    for label in (0, 1):
        mask = labels == label
        if bool(mask.any()):
            values.append(F.cross_entropy(logits[mask].float(), labels[mask]))
    return torch.stack(values).mean() if values else F.cross_entropy(logits.float(), labels)


def _population_logits(model: MultiScaleTCNBank, x: torch.Tensor) -> torch.Tensor:
    return model(x)[1]


def _trainable(model: MultiScaleTCNBank, strategy: str) -> list[torch.nn.Parameter]:
    selected = []
    for name, parameter in model.named_parameters():
        keep = (
            strategy == "full"
            or (strategy == "head" and name.startswith("population_head"))
            or (
                strategy == "tail"
                and (name.startswith("embedding") or name.startswith("population_head"))
            )
        )
        if keep:
            selected.append(parameter)
    if not selected:
        raise RuntimeError(f"No inner parameters selected for {strategy}")
    return selected


def _inner_adapt(
    model: MultiScaleTCNBank,
    x: torch.Tensor,
    y: torch.Tensor,
    configuration: dict,
    seed: int,
    augment: bool,
) -> list[float]:
    optimizer = torch.optim.SGD(
        _trainable(model, str(configuration["strategy"])),
        lr=float(configuration["inner_lr"]),
        momentum=0.0,
    )
    generator = torch.Generator(device=x.device).manual_seed(seed)
    history = []
    for _ in range(int(configuration["inner_steps"])):
        model.train()
        value = x
        if augment:
            shift = int(torch.randint(-8, 9, (1,), generator=generator, device=x.device))
            value = torch.roll(value, shift, dims=2)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            loss = _balanced_loss(_population_logits(model, value), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable(model, str(configuration["strategy"])), 5.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _fomaml(
    initial: MultiScaleTCNBank,
    raw_search: torch.Tensor,
    metadata: pd.DataFrame,
    subjects: tuple[str, ...],
    history_sessions: tuple[int, ...],
    future_session: int,
    configuration: dict,
    meta_epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[MultiScaleTCNBank, list[dict]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=2e-4)
    rng = np.random.default_rng(seed)
    histories = []
    for meta_epoch in range(meta_epochs):
        order = list(subjects)
        rng.shuffle(order)
        query_losses = []
        optimizer.zero_grad(set_to_none=True)
        for subject in order:
            hi, qi = _indices(metadata, subject, history_sessions, future_session)
            history_x = raw_search[torch.as_tensor(hi, dtype=torch.long, device=device)].float()
            query_x = raw_search[torch.as_tensor(qi, dtype=torch.long, device=device)].float()
            history_y = torch.as_tensor(
                metadata.iloc[hi].label.to_numpy(dtype=int, copy=True),
                dtype=torch.long,
                device=device,
            )
            query_y = torch.as_tensor(
                metadata.iloc[qi].label.to_numpy(dtype=int, copy=True),
                dtype=torch.long,
                device=device,
            )
            adapted = copy.deepcopy(model).to(device)
            _inner_adapt(
                adapted, history_x, history_y, configuration,
                stable_seed(seed, meta_epoch, subject, "inner"), True,
            )
            adapted.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                query_loss = _balanced_loss(_population_logits(adapted, query_x), query_y)
            gradients = torch.autograd.grad(
                query_loss, tuple(adapted.parameters()), allow_unused=True,
            )
            scale = 1.0 / max(len(order), 1)
            for parameter, gradient in zip(model.parameters(), gradients):
                if gradient is None:
                    continue
                value = scale * gradient.detach()
                parameter.grad = value.clone() if parameter.grad is None else parameter.grad.add_(value)
            query_losses.append(float(query_loss.detach()))
            del adapted, history_x, query_x
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        row = {
            "meta_epoch": meta_epoch + 1,
            "mean_query_loss": float(np.mean(query_losses)),
            "min_query_loss": float(np.min(query_losses)),
            "max_query_loss": float(np.max(query_losses)),
        }
        histories.append(row)
        print(
            f"[raw FOMAML {configuration['id']}] epoch={meta_epoch + 1}/{meta_epochs} "
            f"query_loss={row['mean_query_loss']:.4f}",
            flush=True,
        )
    return model, histories


def _predict(model: MultiScaleTCNBank, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            value = x[start:start + 256].float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = _population_logits(model, value)
            rows.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
    return np.concatenate(rows)


def run(benchmark: str, meta_epochs: int, folds: tuple[int, ...]) -> dict:
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    benchmark_name = "OpenBMI_MI_S1_to_S2" if benchmark == "openbmi" else "WBCIC_S1S2_to_S3_authorized_development"
    family_slug = "RAW_FOMAML_MULTISCALE_BANK"
    family_id = f"{benchmark_name}__{family_slug}"
    protocol = load_feature_fold(benchmark, 0, "CONFORMER_NORM").protocol
    canonical = pd.read_parquet(CACHE / f"{prefix}_SEARCH_ROWS_FOLD_0.parquet").sort_values("source_index").reset_index(drop=True)
    canonical["local_index"] = np.arange(len(canonical), dtype=int)
    raw_disk = np.load(v7_outputs() / "cache" / f"{prefix}_RAW_EPOCHS_FLOAT16.npy", mmap_mode="r", allow_pickle=False)
    raw_search = torch.as_tensor(
        np.asarray(raw_disk[canonical.source_index.to_numpy(int)], dtype=np.float16),
        device=device,
    )
    baseline, baseline_source_method = baseline_predictions(benchmark)
    baseline = baseline.loc[baseline.subject_id.astype(str).isin(protocol.search_subjects)].copy()
    baseline["method_id"] = "B_STRONG_MATCHED_V7"
    baseline["family_id"] = family_id
    baseline["source_fold"] = baseline.outer_fold.astype(int)
    baseline["benchmark"] = benchmark_name
    baseline["internal_holdout_used"] = False
    predictions = [baseline]
    primary = []
    for configuration in EXPERT_CONFIGS:
        primary.extend((
            f"{family_slug}__{configuration['id']}_BLEND50",
            f"{family_slug}__{configuration['id']}_DELTA50",
        ))
    audits = []
    for fold in folds:
        data = load_feature_fold(benchmark, fold, "CONFORMER_NORM")
        assert_search_only(list(data.meta_subjects) + list(data.search_outcome_subjects), benchmark)
        if not data.search_outcome_subjects:
            continue
        source_path = CACHE / f"{prefix}_MULTISCALE_TCN_LR_STAGED_BANK_K4_R8_FOLD_{fold}.pt"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Run competence-first multiscale fold {fold} before raw FOMAML: {source_path}"
            )
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        initial = MultiScaleTCNBank(int(raw_search.shape[1]), 4, adapter_rank=8).to(device)
        initial.load_state_dict(payload["model"], strict=True)
        trained = []
        fold_audit = []
        for configuration in EXPERT_CONFIGS:
            model, meta_history = _fomaml(
                initial, raw_search, canonical, data.meta_subjects,
                data.protocol.history_sessions, data.protocol.future_session,
                configuration, meta_epochs,
                stable_seed(V8_SEED, benchmark, family_slug, fold, configuration["id"]),
                device,
            )
            trained.append((configuration, model))
            fold_audit.append({
                "configuration": configuration,
                "meta_history": meta_history,
            })
        baseline_index = baseline.set_index("trial_uid")
        for subject in data.search_outcome_subjects:
            hi, qi = _indices(canonical, subject, data.protocol.history_sessions, data.protocol.future_session)
            history_x = raw_search[torch.as_tensor(hi, dtype=torch.long, device=device)]
            query_x = raw_search[torch.as_tensor(qi, dtype=torch.long, device=device)]
            history_y = torch.as_tensor(
                canonical.iloc[hi].label.to_numpy(dtype=int, copy=True),
                dtype=torch.long,
                device=device,
            )
            query_y = canonical.iloc[qi].label.to_numpy(np.float32)
            uid = canonical.iloc[qi].trial_uid.astype(str).to_numpy()
            locked = logit(baseline_index.loc[uid, "probability"].to_numpy(float))
            shell = Episode(str(subject), fold, np.empty(0), np.empty((0, 0)), np.empty(0), np.empty(0), np.empty(0), np.empty((len(qi), 0)), np.empty(len(qi)), query_y, uid)
            for configuration, meta_model in trained:
                frozen = _predict(meta_model, query_x, device)
                adapted = copy.deepcopy(meta_model).to(device)
                _inner_adapt(
                    adapted, history_x.float(), history_y, configuration,
                    stable_seed(V8_SEED, benchmark, fold, subject, configuration["id"], "deploy"),
                    False,
                )
                dynamic = _predict(adapted, query_x, device)
                stem = f"{family_slug}__{configuration['id']}"
                predictions.extend(_parts([shell], dynamic, family_id, f"{stem}_STANDALONE"))
                predictions.extend(_parts([shell], 0.5 * (locked + dynamic), family_id, f"{stem}_BLEND50"))
                predictions.extend(_parts([shell], locked + 0.5 * (dynamic - frozen), family_id, f"{stem}_DELTA50"))
                del adapted
            print(f"[{benchmark} raw FOMAML] fold={fold} subject={subject}", flush=True)
        checkpoint = CACHE / f"{prefix}_{family_slug}_FOLD_{fold}.pt"
        torch.save({
            "models": [model.state_dict() for _, model in trained],
            "configurations": list(EXPERT_CONFIGS),
            "source_fold": fold,
            "meta_subjects": list(data.meta_subjects),
            "search_outcome_subjects": list(data.search_outcome_subjects),
            "internal_holdout_used": False,
            "OUTER_TEST_USED": False,
        }, checkpoint)
        audits.append({
            "benchmark": benchmark_name,
            "source_fold": fold,
            "meta_epochs": meta_epochs,
            "experts": fold_audit,
            "population_initialization": str(source_path),
            "search_outcome_future_labels_used_for_fit": False,
            "internal_holdout_used": False,
            "OUTER_TEST_USED": False,
        })
    prediction_frame = pd.concat(predictions, ignore_index=True)
    allowed = set().union(*(
        set(load_feature_fold(benchmark, fold, "CONFORMER_NORM").search_outcome_subjects)
        for fold in folds
    ))
    prediction_frame = prediction_frame.loc[prediction_frame.subject_id.astype(str).isin(allowed)].copy()
    report = summarize_headroom(prediction_frame, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "folds": list(folds),
        "meta_epochs": meta_epochs,
        "experts": len(EXPERT_CONFIGS),
        "baseline_source_method": baseline_source_method,
        "training_objective": "first-order meta-gradient from future-session balanced query loss after legal-history raw-signal inner adaptation",
        "population_initialization": "competence-first V8_SEARCH-only multiscale temporal-spatial encoder",
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
        "partition": "V8_SEARCH rows materialized before training",
        "inner_loop": "target legal history labels only",
        "outer_meta_loop": "future sessions of source-fold meta subjects only",
        "search_outcome_future_labels_used_for_fit_or_selection": False,
        "internal_holdout_used": False,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural hypothesis: raw-signal FOMAML can learn a population initialization whose legal-history head, tail, or full update transfers to a later session.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--meta-epochs", type=int, default=4)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, args.meta_epochs, folds)


if __name__ == "__main__":
    main()
