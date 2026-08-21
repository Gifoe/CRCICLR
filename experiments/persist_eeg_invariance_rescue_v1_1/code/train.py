from __future__ import annotations

"""V1.1 paired training and representation cache.

All final models are fitted on the 34 legal development-train subjects.  The
inner GRL selector is executed before outcome labels are touched.  T_anchor
and I_invariant share an initial state and loader seed; T_replica is an
independent task-only control.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import CHECKPOINTS, OUTPUTS, SMOKE, balanced_accuracy, ce_loss, ensure_directories, load_config, macro_f1, seed_all, sha256_file, softmax, stable_seed, write_csv, write_json
from data import MIDataset, domain_map, load_development_split, load_manifest, make_loader, normalizer, select_frame
from losses import coral_loss, expert_conditional_alignment_loss, expert_mmd_loss, supervised_contrastive_loss
from models import build_model, grl_lambda, method_family, parameter_count


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def method_key(method_id: str, role: str) -> str:
    return f"{method_id}__{role}"


def _run_root(mode: str, fold: int, seed: int, method_id: str, role: str) -> Path:
    base = SMOKE if mode == "smoke" else OUTPUTS / "runs"
    return base / f"fold-{fold}" / f"seed-{seed}" / method_key(method_id, role)


def checkpoint_path(mode: str, fold: int, seed: int, method_id: str, role: str) -> Path:
    return (_run_root(mode, fold, seed, method_id, role) / "best.pt") if mode == "smoke" else CHECKPOINTS / f"fold-{fold}" / f"{method_key(method_id, role)}.pt"


def representation_path(fold: int, seed: int, method_id: str, role: str) -> Path:
    return OUTPUTS / "cache" / "representations" / f"fold-{fold}" / f"seed-{seed}" / f"{method_key(method_id, role)}.npz"


def _loss(method_id: str, output: Any, labels: torch.Tensor, domains: torch.Tensor, step: int, config: Mapping[str, Any], invariant: bool) -> tuple[torch.Tensor, dict[str, float]]:
    task = F.cross_entropy(output.logits.float(), labels)
    total = task
    parts: dict[str, torch.Tensor] = {"task_ce": task}
    family = method_family(method_id)
    if family == "A" and invariant and grl_lambda(method_id) > 0:
        domain = F.cross_entropy(output.domain_logits.float(), domains)
        total = total + domain
        parts["subject_ce"] = domain
    elif family == "B" and invariant:
        weights = config["eeg_dg_weights"]
        marginal = expert_mmd_loss(output.expert_features, step)
        conditional = expert_conditional_alignment_loss(output.expert_features, labels, step)
        domain = F.cross_entropy(output.domain_logits.float(), domains)
        total = total + float(weights["marginal_mmd"]) * marginal + float(weights["conditional_alignment"]) * conditional + float(weights["domain_classification"]) * domain
        parts.update(marginal_mmd=marginal, conditional_alignment=conditional, domain_ce=domain)
    elif family == "C" and invariant:
        weights = config["scldgn_weights"]
        alignment = coral_loss(output.features.float(), labels, domains, step)
        contrastive = supervised_contrastive_loss(output.projection.float(), labels, step)
        total = total + float(weights["coral"]) * alignment + float(weights["supervised_contrastive"]) * contrastive
        parts.update(coral=alignment, supervised_contrastive=contrastive)
    values = {name: float(value.detach().cpu()) for name, value in parts.items()}
    values["total"] = float(total.detach().cpu())
    return total, values


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    model.eval(); logits, labels, positions = [], [], []
    for signals, target, _, position in loader:
        output = model(signals.to(device, non_blocking=True))
        logits.append(output.logits.float().cpu().numpy()); labels.append(target.numpy()); positions.append(position.numpy())
    if not labels:
        return {"accuracy": None, "balanced_accuracy": None, "macro_f1": None, "cross_entropy": None, "n": 0,
                "labels": np.empty(0, dtype=np.int64), "predictions": np.empty(0, dtype=np.int64), "logits": np.empty((0, 2), dtype=np.float32), "positions": np.empty(0, dtype=np.int64)}
    y = np.concatenate(labels).astype(np.int64); score = np.concatenate(logits).astype(np.float32); pred = score.argmax(axis=1)
    return {"accuracy": float(np.mean(pred == y)), "balanced_accuracy": balanced_accuracy(y, pred), "macro_f1": macro_f1(y, pred), "cross_entropy": ce_loss(y, softmax(score)), "n": int(len(y)), "labels": y, "predictions": pred, "logits": score, "positions": np.concatenate(positions).astype(np.int64)}


def _subject_ba(meta: pd.DataFrame, result: Mapping[str, Any]) -> float | None:
    if not result.get("n", 0): return None
    lookup = meta.set_index("manifest_position"); frame = lookup.loc[result["positions"]].reset_index(drop=True); vals = []
    for _, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        idx = group.index.to_numpy(dtype=np.int64); vals.append(balanced_accuracy(result["labels"][idx], result["predictions"][idx]))
    return float(np.mean(vals)) if vals else None


def _state_payload(model: torch.nn.Module, method_id: str, role: str, fold: int, seed: int, train_subjects: Sequence[str], norm_path: Path, epoch: int, history: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    return {"model": model.state_dict(), "method_id": method_id, "role": role, "fold": int(fold), "seed": int(seed), "epoch": int(epoch), "history": list(history), "parameter_count": parameter_count(model), "train_subjects": list(map(str, train_subjects)), "normalizer_sha256": sha256_file(norm_path), "mode": mode, "outer_test_used": False, "outer_membership_enumerated": False}


def train_model(method_id: str, role: str, fold: int, seed: int, device: torch.device, train_subjects: Sequence[str], eval_subjects: Sequence[str] | None = None, init_state: Mapping[str, torch.Tensor] | None = None, order_seed: int | None = None, mode: str = "full", force: bool = False, epochs: int | None = None) -> dict[str, Any]:
    config = load_config(); ensure_directories(); split = load_development_split(fold)
    train_subjects = tuple(sorted(map(str, train_subjects)))
    if not set(train_subjects).issubset(set(split.original_train_subjects)): raise RuntimeError("training subjects exceed frozen development train set")
    manifest = load_manifest(split); mean, std, norm_path = normalizer(fold, manifest, train_subjects)
    train_frame = select_frame(manifest, train_subjects, config["train_sessions"])
    eval_frame = select_frame(manifest, eval_subjects or (), [2]) if eval_subjects else train_frame.iloc[:0].copy()
    family = method_family(method_id); hyper = config["training"]["family_hyperparameters"][family]
    run_seed = stable_seed("v11-train", mode, fold, seed, method_id, role) if order_seed is None else int(order_seed); seed_all(run_seed)
    # Keep the subject-head width fixed to the frozen 34-subject development
    # universe even for inner folds; otherwise an inner checkpoint cannot be
    # loaded by the common representation extractor.
    model = build_model(method_id, len(split.original_train_subjects), config).to(device)
    if init_state is not None: model.load_state_dict({key: value.detach().clone() for key, value in init_state.items()}, strict=True)
    ckpt = checkpoint_path(mode, fold, seed, method_id, role); done_path = _run_root(mode, fold, seed, method_id, role) / "TRAINING_COMPLETE.json"
    if not force and ckpt.exists() and done_path.exists():
        saved = json.loads(done_path.read_text(encoding="utf-8"))
        if saved.get("status") == "COMPLETE" and saved.get("outer_test_used") is False: return saved
    train_loader = make_loader(MIDataset(train_frame, mean, std, domain_map(train_subjects)), int(hyper["batch_size"]), True, run_seed)
    eval_loader = make_loader(MIDataset(eval_frame, mean, std, None), int(hyper["batch_size"]), False, run_seed + 1) if len(eval_frame) else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hyper["learning_rate"]), weight_decay=float(hyper["weight_decay"]))
    max_epochs = int(epochs if epochs is not None else (config["training"]["smoke_epochs"] if mode == "smoke" else config["training"]["final_epochs"]))
    ckpt.parent.mkdir(parents=True, exist_ok=True); _run_root(mode, fold, seed, method_id, role).mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []; best = -np.inf; started = time.time(); global_step = 0
    for epoch in range(1, max_epochs + 1):
        model.train(); epoch_parts: dict[str, list[float]] = {}
        for signals, labels, domains, _ in train_loader:
            signals, labels, domains = signals.to(device, non_blocking=True), labels.to(device, non_blocking=True), domains.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output = model(signals, grl_strength=grl_lambda(method_id) if role == "I_invariant" else 0.0)
                loss, parts = _loss(method_id, output, labels, domains, global_step, config, role == "I_invariant")
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss {method_id} {role} step {global_step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"])); optimizer.step()
            for key, value in parts.items(): epoch_parts.setdefault(key, []).append(value)
            global_step += 1
        if eval_loader is not None:
            validation = evaluate(model, eval_loader, device); score = _subject_ba(eval_frame, validation)
        else:
            validation = {"balanced_accuracy": None, "cross_entropy": None}; score = None
        row = {"epoch": epoch, **{f"train_{k}": float(np.mean(v)) for k, v in epoch_parts.items()}, "eval_BA": validation["balanced_accuracy"], "eval_subject_BA": score, "eval_CE": validation["cross_entropy"]}; history.append(row)
        if score is not None and score >= best:
            best = float(score); torch.save(_state_payload(model, method_id, role, fold, seed, train_subjects, norm_path, epoch, history, mode), ckpt)
        elif eval_loader is None:
            torch.save(_state_payload(model, method_id, role, fold, seed, train_subjects, norm_path, epoch, history, mode), ckpt)
        print(f"[train] {mode} f{fold}s{seed} {method_id} {role} epoch={epoch}/{max_epochs} BA={score}", flush=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload = {"status": "COMPLETE", "mode": mode, "fold": int(fold), "seed": int(seed), "method_id": method_id, "role": role, "family": family, "best_epoch": int(state["epoch"]), "best_eval_BA": None if not np.isfinite(best) else float(best), "epochs_executed": len(history), "parameter_count": int(state["parameter_count"]), "checkpoint": str(ckpt), "elapsed_seconds": time.time() - started, "train_rows": int(len(train_frame)), "eval_rows": int(len(eval_frame)), "outer_test_used": False, "outer_membership_enumerated": False, "paired_initialization": bool(init_state is not None), "paired_order_seed": int(order_seed) if order_seed is not None else None}
    write_json(done_path, payload); write_csv(_run_root(mode, fold, seed, method_id, role) / "TRAINING_HISTORY.csv", pd.DataFrame(history)); return payload


def load_trained_model(method_id: str, role: str, fold: int, seed: int, device: torch.device, mode: str = "full") -> torch.nn.Module:
    config = load_config(); split = load_development_split(fold); model = build_model(method_id, len(split.original_train_subjects), config)
    payload = torch.load(checkpoint_path(mode, fold, seed, method_id, role), map_location="cpu", weights_only=False)
    if payload["method_id"] != method_id or payload["role"] != role or int(payload["fold"]) != int(fold) or int(payload["seed"]) != int(seed): raise RuntimeError("checkpoint provenance mismatch")
    model.load_state_dict(payload["model"], strict=True); return model.to(device).eval()


@torch.inference_mode()
def extract_one(method_id: str, role: str, fold: int, seed: int, device: torch.device, force: bool = False, mode: str = "full") -> Path:
    path = representation_path(fold, seed, method_id, role); provenance_path = path.with_suffix(".json")
    if path.exists() and provenance_path.exists() and not force: return path
    config = load_config(); split = load_development_split(fold); manifest = load_manifest(split); mean, std, norm_path = normalizer(fold, manifest, split.model_fit_subjects); frame = select_frame(manifest, split.allowed_subjects, [1, 2]); model = load_trained_model(method_id, role, fold, seed, device, mode=mode)
    batch_size = int(config["training"]["family_hyperparameters"][method_family(method_id)]["batch_size"]); loader = make_loader(MIDataset(frame, mean, std, None), batch_size, False, stable_seed("v11-extract", fold, seed, method_id, role)); features, logits, positions = [], [], []
    for signals, _, _, position in loader:
        output = model(signals.to(device, non_blocking=True)); features.append(output.features.float().cpu().numpy()); logits.append(output.logits.float().cpu().numpy()); positions.append(position.numpy())
    h, score, pos = np.concatenate(features).astype(np.float32), np.concatenate(logits).astype(np.float32), np.concatenate(positions).astype(np.int64); order = np.argsort(pos); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(".part.npz")
    np.savez_compressed(temp, positions=pos[order], features=h[order], logits=score[order], outer_test_used=np.asarray(False), outer_membership_enumerated=np.asarray(False)); os.replace(temp, path)
    write_json(provenance_path, {"method_id": method_id, "role": role, "fold": int(fold), "seed": int(seed), "mode": mode, "shape": list(h.shape), "manifest_positions_sha256": __import__("hashlib").sha256(pos[order].tobytes()).hexdigest(), "checkpoint_sha256": sha256_file(checkpoint_path(mode, fold, seed, method_id, role)), "normalizer_sha256": sha256_file(norm_path), "outer_test_used": False, "outer_membership_enumerated": False}); return path


def load_representation(method_id: str, fold: int, seed: int, role: str = "T_anchor") -> dict[str, np.ndarray]:
    value = np.load(representation_path(fold, seed, method_id, role), allow_pickle=False)
    if bool(value["outer_test_used"].item()): raise RuntimeError("outer lock violation")
    return {name: value[name] for name in ("positions", "features", "logits")}


def _make_initial_state(method_id: str, fold: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    config = load_config(); split = load_development_split(fold); seed_all(stable_seed("paired-init", fold, seed, method_id)); model = build_model(method_id, len(split.model_fit_subjects), config).to(device); return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _inner_identity(fold: int, candidate: float, inner: int, fit_subjects: Sequence[str], eval_subjects: Sequence[str], device: torch.device) -> float:
    from spectrum import identity_from_cached_inner
    method = f"A1_SUBJECT_GRL_EEGNET_L{int(round(candidate * 1000)):04d}"; split = load_development_split(fold); config = load_config(); manifest = load_manifest(split); mean, std, _ = normalizer(fold, manifest, fit_subjects); train_frame = select_frame(manifest, fit_subjects, [1, 2]); eval_frame = select_frame(manifest, eval_subjects, [1, 2]); init = _make_initial_state("A0_TASK_ONLY_EEGNET", fold, 9000 + inner, device); order = stable_seed("inner-order", fold, inner)
    train_model("A0_TASK_ONLY_EEGNET", "T_anchor", fold, 9000 + inner, device, fit_subjects, eval_subjects, init_state=init, order_seed=order, mode="smoke", force=True, epochs=int(config["training"]["inner_epochs"])); train_model(method, "I_invariant", fold, 9000 + inner, device, fit_subjects, eval_subjects, init_state=init, order_seed=order, mode="smoke", force=True, epochs=int(config["training"]["inner_epochs"]))
    # Reloading the two small smoke representations is deterministic and keeps
    # the selector independent of any final outcome artefact.
    extract_one("A0_TASK_ONLY_EEGNET", "T_anchor", fold, 9000 + inner, device, force=True, mode="smoke"); extract_one(method, "I_invariant", fold, 9000 + inner, device, force=True, mode="smoke")
    return float(identity_from_cached_inner(fold, 9000 + inner, eval_subjects, "A0_TASK_ONLY_EEGNET", method))


def select_grl(fold: int, device: torch.device, force: bool = False) -> dict[str, Any]:
    config = load_config(); path = OUTPUTS / "HYPERPARAM_SELECTION.csv"; rows = []
    split = load_development_split(fold); subjects = list(split.model_fit_subjects); rng = np.random.default_rng(stable_seed("grl-inner-fold", fold)); rng.shuffle(subjects); folds = np.array_split(np.asarray(subjects), int(config["grl_selection_inner_splits"]))
    for candidate in map(float, config["grl_candidate_grid"]):
        deltas_id, deltas_ba, min_ba = [], [], []; inner_ok = True
        for inner, held in enumerate(folds):
            eval_subjects = [str(x) for x in held]; fit_subjects = [x for x in subjects if x not in set(eval_subjects)]
            if not eval_subjects or len(fit_subjects) < 2: continue
            identity = _inner_identity(fold, candidate, inner, fit_subjects, eval_subjects, device); deltas_id.append(identity)
            # Evaluate BA from the two saved training records; this is held-out
            # inner Session-2 only and cannot see the nine outcome subjects.
            a = json.loads((_run_root("smoke", fold, 9000 + inner, "A0_TASK_ONLY_EEGNET", "T_anchor") / "TRAINING_COMPLETE.json").read_text())
            method = f"A1_SUBJECT_GRL_EEGNET_L{int(round(candidate * 1000)):04d}"; inv = json.loads((_run_root("smoke", fold, 9000 + inner, method, "I_invariant") / "TRAINING_COMPLETE.json").read_text()); deltas_ba.append(float(inv["best_eval_BA"] - a["best_eval_BA"])); min_ba.append(min(float(a["best_eval_BA"]), float(inv["best_eval_BA"])))
        rows.append({"fold": fold, "candidate_lambda": candidate, "mean_delta_ID_inner": float(np.mean(deltas_id)) if deltas_id else np.nan, "mean_delta_BA_inner": float(np.mean(deltas_ba)) if deltas_ba else np.nan, "min_inner_BA": float(np.min(min_ba)) if min_ba else np.nan, "all_inner_competent": bool(min_ba and min(min_ba) >= float(config["grl_selection_competence_floor"])), "inner_splits_observed": len(deltas_id), "outer_test_used": False, "outer_membership_enumerated": False})
    frame = pd.DataFrame(rows); good = frame[(frame.mean_delta_ID_inner <= float(config["grl_selection_identity_threshold"])) & (frame.mean_delta_BA_inner >= float(config["grl_selection_task_delta_floor"])) & frame.all_inner_competent]
    if len(good): selected = float(good.sort_values("candidate_lambda").iloc[0].candidate_lambda); status = "SELECTED_TRAIN_SIDE_GRL"
    else:
        fallback = frame[(frame.mean_delta_BA_inner >= float(config["grl_selection_task_delta_floor"])) & frame.all_inner_competent].sort_values(["mean_delta_ID_inner", "candidate_lambda"]); selected = float(fallback.iloc[0].candidate_lambda) if len(fallback) else float(frame.iloc[0].candidate_lambda); status = "NO_TRAIN_SIDE_INVARIANCE_MANIPULATION" if not bool((frame.mean_delta_ID_inner < 0).any()) else "NO_CANDIDATE_MEETS_FROZEN_GATES"
    frame["selected"] = np.isclose(frame.candidate_lambda.astype(float), selected); frame["selection_status"] = status
    # Preserve every fold's candidate audit; later folds must not erase the
    # earlier train-side selection records.
    old = pd.read_csv(path) if path.exists() and not force else pd.DataFrame()
    if len(old):
        old = old[old.fold != int(fold)]
        frame = pd.concat([old, frame], ignore_index=True, sort=False)
    write_csv(path, frame); result = frame[(frame.fold == int(fold)) & frame.selected].iloc[0].to_dict(); result.update(selected_lambda=selected, selection_status=status); return result


def run_full(device: torch.device, force: bool = False) -> list[dict[str, Any]]:
    config = load_config(); rows, selected_rows = [], []
    for fold in map(int, config["development_folds"]):
        selection = select_grl(fold, device, force=force); selected_rows.append(selection); value = float(selection["selected_lambda"]); a_method = "A0_TASK_ONLY_EEGNET"; i_method = f"A1_SUBJECT_GRL_EEGNET_L{int(round(value * 1000)):04d}"; split = load_development_split(fold)
        for seed in map(int, config["seeds"]):
            for family, task_method, inv_method in [("A_SUBJECT_GRL_EEGNET", a_method, i_method), ("B_EEG_DG", "B0_EEG_DG_TASK_ONLY", "B1_EEG_DG_FULL"), ("C_SCLDGN", "C0_SCLDGN_TASK_ONLY", "C1_SCLDGN_FULL")]:
                init = _make_initial_state(task_method, fold, seed, device); order = stable_seed("final-paired-order", fold, seed, family)
                rows.append(train_model(task_method, "T_anchor", fold, seed, device, split.model_fit_subjects, init_state=init, order_seed=order, mode="full", force=force)); rows.append(train_model(inv_method, "I_invariant", fold, seed, device, split.model_fit_subjects, init_state=init, order_seed=order, mode="full", force=force)); rows.append(train_model(task_method, "T_replica", fold, seed, device, split.model_fit_subjects, order_seed=stable_seed("replica-order", fold, seed, family), mode="full", force=force))
                for method, role in [(task_method, "T_anchor"), (task_method, "T_replica"), (inv_method, "I_invariant")]: extract_one(method, role, fold, seed, device, force=force)
    write_csv(OUTPUTS / "RUN_LEDGER.csv", pd.DataFrame(rows)); return rows


def run_smoke(device: torch.device, force: bool = False) -> list[dict[str, Any]]:
    config = load_config(); split = load_development_split(0); subjects = list(split.model_fit_subjects[:8]); eval_subjects = list(split.model_fit_subjects[8:10]); rows = []
    for method in ("A0_TASK_ONLY_EEGNET", "A1_SUBJECT_GRL_EEGNET_L0100", "B0_EEG_DG_TASK_ONLY", "B1_EEG_DG_FULL", "C0_SCLDGN_TASK_ONLY", "C1_SCLDGN_FULL"):
        role = "I_invariant" if method.startswith("A1") or method.endswith("FULL") else "T_anchor"; init = _make_initial_state("A0_TASK_ONLY_EEGNET" if method.startswith("A1") else method, 0, 777, device) if role == "I_invariant" else None
        rows.append(train_model(method, role, 0, 777, device, subjects, eval_subjects, init_state=init, order_seed=stable_seed("smoke", method), mode="smoke", force=force, epochs=int(config["training"]["smoke_epochs"])))
    write_csv(OUTPUTS / "SMOKE_RESULTS.csv", pd.DataFrame(rows)); return rows
