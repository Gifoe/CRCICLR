from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO = Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK").resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_ema_seed0_v1"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("EMA_RUNTIME", "/root/rivermind-data/litebn_ema_seed0_runtime")).resolve()
sys.path.insert(0, str(REPO / "experiments/persist_eeg_litebn_rgeo_seed0_v1/code"))
import run_rgeo_seed0 as r

SEED = 0
EMA_DECAY = 0.90
MAX_EPOCHS = 60
MIN_SELECTION_EPOCH = 10
PATIENCE = 10
LR = 3e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
TIE_TOL = 1e-12
TASK_ORDER = ["OpenBMI_MI", "WBCIC_MI", "OpenBMI_ERP", "OpenBMI_SSVEP"]


def score(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(1)
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def update_ema(ema: torch.nn.Module, online: torch.nn.Module, initialized: bool) -> bool:
    online_state = online.state_dict()
    with torch.no_grad():
        for key, ema_value in ema.state_dict().items():
            online_value = online_state[key].detach()
            if not initialized:
                ema_value.copy_(online_value)
            elif ema_value.dtype.is_floating_point:
                ema_value.mul_(EMA_DECAY).add_(online_value, alpha=1.0 - EMA_DECAY)
            else:
                ema_value.copy_(online_value)
    return True


def eval_heldout_mi(model: torch.nn.Module, baseline: torch.nn.Module, dataset: str,
                     held_ids: list[str], mean: np.ndarray, std: np.ndarray,
                     device: torch.device) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    ema_values: list[dict[str, float]] = []
    base_values: list[dict[str, float]] = []
    model.eval()
    baseline.eval()
    for subject in held_ids:
        x, y = r.final_assets.load_eval_subject(dataset, subject)
        with torch.no_grad():
            ema_logits = r.infer_raw(model, x, mean, std, device)
            base_logits = r.infer_raw(baseline, x, mean, std, device)
        for method, logits in (("LiteBN_EMA", ema_logits), ("LiteBN_BASELINE", base_logits)):
            metrics = score(y, logits)
            rows.append({"scope": "heldout_internal", "subject_id": str(subject),
                         "method": method, **metrics, "trials": int(len(y))})
            (ema_values if method == "LiteBN_EMA" else base_values).append(metrics)
    def average(values: list[dict[str, float]]) -> dict[str, float]:
        return {key: float(np.mean([row[key] for row in values])) for key in ("BA", "macro_F1", "accuracy")}
    return rows, average(ema_values), average(base_values)


def eval_heldout_task(model: torch.nn.Module, baseline: torch.nn.Module, task: str,
                      held_ids: list[str], mean: np.ndarray, std: np.ndarray,
                      device: torch.device) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    bundle = r.task_core.load_bundle(task, held_ids, sessions=(2,))
    cache = r.Cache("TASK", bundle, mean, std, device)
    ema_rows, _, _ = r.subject_eval(model, cache, bundle, held_ids, (2,))
    base_rows, _, _ = r.subject_eval(baseline, cache, bundle, held_ids, (2,))
    rows: list[dict[str, Any]] = []
    for subject in held_ids:
        for method, source in (("LiteBN_EMA", ema_rows), ("LiteBN_BASELINE", base_rows)):
            metrics = source[str(subject)]
            rows.append({"scope": "heldout_internal", "subject_id": str(subject),
                         "method": method, **metrics})
    def average(source: dict[str, dict[str, float]]) -> dict[str, float]:
        return {key: float(np.mean([source[str(s)][key] for s in held_ids])) for key in ("BA", "macro_F1", "accuracy")}
    return rows, average(ema_rows), average(base_rows)


def train_fold(name: str, kind: str, bundle: Any, fold: dict[str, Any], cache: Any,
               device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    cell = RUNTIME / name.lower() / f"fold{int(fold['fold_id'])}"
    cell.mkdir(parents=True, exist_ok=True)
    latest = cell / "checkpoint_latest.pt"
    selected = cell / "selected_best.pt"
    online = r.build_model(name, bundle.channels).to(device)
    ema = r.build_model(name, bundle.channels).to(device)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    init_hash = r.state_hash(online)
    optimizer = torch.optim.AdamW(online.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_subjects = [str(x) for x in fold["inner_train_subjects"]]
    train_indices = r.indices(bundle, train_subjects, r.source_sessions(name))
    class_weight = None
    weight_info: dict[str, Any] = {"weighted_cross_entropy": False}
    if name == "OpenBMI_ERP":
        class_weight, weight_info = r.task_core.class_weights(bundle, train_subjects)
        if class_weight is not None:
            class_weight = class_weight.to(device)
    if kind == "MI":
        manifest = r.episode_epochs(name, kind, bundle, fold)
        dataset_key = "openbmi" if name == "OpenBMI_MI" else "wbcic"
        manifest_path = r.MI_RUNTIME / f"{dataset_key}_fold{fold['fold_id']}" / "manifest_runtime" / "episode_manifests" / f"{dataset_key}_fold{fold['fold_id']}.json"
        manifest_sha = r.sha256(manifest_path)
    else:
        manifest = None
        manifest_sha = r.sha256(r.FIVEFOLD) + f"_{name}_task_batches"
    start = 1
    history: list[dict[str, Any]] = []
    best_ba = -float("inf")
    best_f1 = -float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    ema_initialized = False
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["init_sha256"] != init_hash or saved["manifest_sha256"] != manifest_sha:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        online.load_state_dict(saved["online_state"])
        ema.load_state_dict(saved["ema_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start = int(saved["epoch"]) + 1
        history = saved["history"]
        best_ba = float(saved["best_ema_val_BA"])
        best_f1 = float(saved["best_ema_val_macro_F1"])
        best_epoch = saved["best_epoch"]
        best_state = saved["best_state"]
        bad_epochs = int(saved.get("bad_epochs", 0))
        ema_initialized = bool(saved.get("ema_initialized", True))
        rng = saved.get("rng")
        if rng is not None:
            torch.set_rng_state(rng["torch"].detach().cpu())
            if "cuda" in rng and device.type == "cuda":
                torch.cuda.set_rng_state_all([x.detach().cpu() for x in rng["cuda"]])
    started = time.perf_counter()
    stopped_early = False
    for epoch in range(start, MAX_EPOCHS + 1):
        online.train()
        losses: list[float] = []
        batches = manifest[epoch - 1] if kind == "MI" else [{"indices": x.tolist()} for x in r.task_batches(train_indices, int(fold["fold_id"]), name, epoch)]
        for episode in batches:
            if kind == "MI":
                indices = np.asarray(episode["support_indices"] + episode["query_indices"], dtype=np.int64)
            else:
                indices = np.asarray(episode["indices"], dtype=np.int64)
            x, y = cache.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = online(x)
                loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite CE in {name} fold {fold['fold_id']}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        ema_initialized = update_ema(ema, online, ema_initialized)
        _, ema_ba, ema_f1, ema_acc = r.subject_eval(ema, cache, bundle, fold["inner_val_subjects"], r.future_session(name))
        _, online_ba, online_f1, _ = r.subject_eval(online, cache, bundle, fold["inner_val_subjects"], r.future_session(name))
        ba_improved = ema_ba > best_ba + TIE_TOL
        f1_tie_improved = abs(ema_ba - best_ba) <= TIE_TOL and ema_f1 > best_f1 + TIE_TOL
        selected_now = epoch >= MIN_SELECTION_EPOCH and (ba_improved or f1_tie_improved)
        if selected_now:
            best_ba = float(ema_ba)
            best_f1 = float(ema_f1)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(ema.state_dict())
            bad_epochs = 0
        elif epoch >= MIN_SELECTION_EPOCH:
            bad_epochs += 1
        row = {"epoch": int(epoch), "train_CE": float(np.mean(losses)),
               "online_inner_val_BA": float(online_ba), "online_inner_val_macro_F1": float(online_f1),
               "EMA_inner_val_BA": float(ema_ba), "EMA_inner_val_macro_F1": float(ema_f1),
               "EMA_inner_val_accuracy": float(ema_acc), "selected": bool(selected_now),
               "bad_epochs": int(bad_epochs), "early_stopped": False}
        history.append(row)
        rng = {"torch": torch.get_rng_state()}
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save({"epoch": epoch, "history": history, "best_ema_val_BA": best_ba,
                    "best_ema_val_macro_F1": best_f1, "best_epoch": best_epoch,
                    "best_state": best_state, "online_state": online.state_dict(),
                    "ema_state": ema.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "rng": rng, "bad_epochs": bad_epochs,
                    "ema_initialized": ema_initialized, "init_sha256": init_hash,
                    "manifest_sha256": manifest_sha}, latest)
        if epoch == 1 or epoch % 5 == 0 or selected_now:
            print(f"[{name} f{fold['fold_id']}] epoch={epoch:02d} EMA_BA={ema_ba:.4f} online_BA={online_ba:.4f} bad={bad_epochs}", flush=True)
        if epoch >= MIN_SELECTION_EPOCH and bad_epochs >= PATIENCE:
            history[-1]["early_stopped"] = True
            stopped_early = True
            print(f"[{name} f{fold['fold_id']}] EARLY_STOP epoch={epoch:02d} best_epoch={best_epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"no eligible EMA checkpoint for {name} fold {fold['fold_id']}")
    ema.load_state_dict(best_state)
    ema.eval()
    torch.save(ema.state_dict(), selected)
    info = {"task": name, "fold": int(fold["fold_id"]), "seed": SEED,
            "selected_epoch": int(best_epoch), "epochs_completed": len(history),
            "early_stopped": bool(stopped_early), "best_EMA_inner_val_BA": float(best_ba),
            "best_EMA_inner_val_macro_F1": float(best_f1), "history": history,
            "checkpoint_path": str(selected), "checkpoint_sha256": r.sha256(selected),
            "init_sha256": init_hash, "manifest_sha256": manifest_sha,
            "runtime_seconds": time.perf_counter() - started, "ema_decay": EMA_DECAY,
            "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRAD_CLIP},
            "selection": {"min_epoch": MIN_SELECTION_EPOCH, "metric": "EMA subject-mean BA; macro-F1 tie-break", "patience": PATIENCE},
            "class_weight_info": weight_info}
    r.write_json(cell / "TRAINING.json", info)
    return ema, info


def append_outputs(fold_rows: list[dict[str, Any]], subject_rows: list[dict[str, Any]], logs: list[dict[str, Any]], summaries: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(OUT / "TASK_RESULTS.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(OUT / "SUBJECT_RESULTS.csv", index=False)
    r.write_json(OUT / "TRAINING_LOGS.json", logs)
    r.write_json(OUT / "FINAL_DECISION.json", decision)
    lines = ["# LiteBN-EMA seed0 report", "", "Internal development diagnostic; current historical heldout is not an untouched final test.", ""]
    for summary in summaries:
        lines += [f"## {summary['task']}", "", f"- Outer LiteBN BA: {summary['outer_baseline_BA']:.6f}; EMA BA: {summary['outer_EMA_BA']:.6f}; delta: {summary['outer_delta_pp']:+.3f} pp",
                  f"- Heldout LiteBN BA: {summary['heldout_baseline_BA']:.6f}; EMA BA: {summary['heldout_EMA_BA']:.6f}; delta: {summary['heldout_delta_pp']:+.3f} pp",
                  f"- Positive outer folds: {summary['positive_outer_folds']}/{summary['folds']}; early-stop epochs: {summary['early_stop_epochs']}",
                  f"- Decision: {summary['decision']}", ""]
    if summaries:
        lines += [f"Outer equal-task mean delta: {np.mean([x['outer_delta_pp'] for x in summaries]):+.3f} pp",
                  f"Heldout equal-task mean delta: {np.mean([x['heldout_delta_pp'] for x in summaries]):+.3f} pp", ""]
    lines += ["FINAL_HELDOUT_ACCESSED = NO", "CURRENT_INTERNAL_HELDOUT_ACCESSED = YES", "FINAL_TEST_USED_FOR_TUNING = NO"]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in (CODE, PROTOCOL, OUT, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)
    search, folds, split_sha = r.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r.write_json(PROTOCOL / "LITEBN_EMA_PROTOCOL.json", {
        "experiment": "LiteBN-EMA", "seed": SEED, "tasks": TASK_ORDER, "folds": 5,
        "architecture": "exact historical CompactLite BN / LiteBN_BASELINE",
        "only_change": "epoch-level EMA training weights", "ema_decay": EMA_DECAY,
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRAD_CLIP,
        "max_epochs": MAX_EPOCHS, "min_selection_epoch": MIN_SELECTION_EPOCH, "patience": PATIENCE,
        "selection": "EMA inner future-session subject-mean BA; macro-F1 tie-break; earliest epoch",
        "split_sha256": split_sha, "historical_baseline_checkpoint_reused": True,
        "current_internal_heldout": True, "final_heldout_accessed": False, "final_test_used_for_tuning": False})
    r.write_json(PROTOCOL / "HOLDOUT_AUDIT.json", {"current_internal_heldout_accessed": True,
        "untouched_final_test_accessed": False, "outer_true_subjects_accessed": False,
        "scope": "historical internal V8 holdout diagnostic"})
    (EXP / "BUG_REPAIR_LEDGER.md").write_text("# BUG_REPAIR_LEDGER\n\nNo scientific protocol repair was applied. This run adds only fixed epoch-level EMA and fixed early stopping to exact LiteBN.\n", encoding="utf-8")
    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    memberships, _ = r.final_assets.holdout_memberships()
    terminal = "IN_PROGRESS"
    for name in TASK_ORDER:
        dataset = "OpenBMI" if name.startswith("OpenBMI") else "WBCIC"
        kind, bundle = r.get_bundle(name, search[dataset])
        held_ids = [str(x) for x in memberships[dataset]]
        task_outer_d: list[float] = []
        task_held_d: list[float] = []
        task_infos: list[dict[str, Any]] = []
        for fold in folds[dataset]:
            mean, std, _ = r.normalizer(kind, bundle, fold["inner_train_subjects"])
            cache = r.Cache(kind, bundle, mean, std, device)
            ema, info = train_fold(name, kind, bundle, fold, cache, device)
            baseline = r.build_model(name, bundle.channels).to(device)
            baseline_path = r.baseline_path(name, int(fold["fold_id"]))
            baseline.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=False), strict=True)
            baseline.eval(); ema.eval()
            ema_rows, ema_outer_ba, ema_outer_f1, ema_outer_acc = r.subject_eval(ema, cache, bundle, fold["outer_dev_subjects"], r.future_session(name))
            base_rows, base_outer_ba, base_outer_f1, base_outer_acc = r.subject_eval(baseline, cache, bundle, fold["outer_dev_subjects"], r.future_session(name))
            for subject in fold["outer_dev_subjects"]:
                for method, source in (("LiteBN_EMA", ema_rows), ("LiteBN_BASELINE", base_rows)):
                    subject_rows.append({"task": name, "fold": int(fold["fold_id"]), "scope": "outer_development", "subject_id": str(subject), "method": method, **source[str(subject)]})
            if kind == "MI":
                held_rows, ema_held, base_held = eval_heldout_mi(ema, baseline, dataset, held_ids, mean, std, device)
            else:
                held_rows, ema_held, base_held = eval_heldout_task(ema, baseline, name.split("_", 1)[1], held_ids, mean, std, device)
            for row in held_rows:
                row.update({"task": name, "fold": int(fold["fold_id"])})
                subject_rows.append(row)
            outer_delta = (ema_outer_ba - base_outer_ba) * 100.0
            held_delta = (ema_held["BA"] - base_held["BA"]) * 100.0
            task_outer_d.append(outer_delta); task_held_d.append(held_delta)
            fold_rows.append({"task": name, "fold": int(fold["fold_id"]), "seed": SEED,
                "selected_epoch": info["selected_epoch"], "epochs_completed": info["epochs_completed"], "early_stopped": info["early_stopped"],
                "best_EMA_inner_val_BA": info["best_EMA_inner_val_BA"], "best_EMA_inner_val_macro_F1": info["best_EMA_inner_val_macro_F1"],
                "historical_LiteBN_outer_BA": base_outer_ba, "EMA_outer_BA": ema_outer_ba, "outer_delta_pp": outer_delta,
                "historical_LiteBN_outer_macro_F1": base_outer_f1, "EMA_outer_macro_F1": ema_outer_f1,
                "historical_LiteBN_heldout_BA": base_held["BA"], "EMA_heldout_BA": ema_held["BA"], "heldout_delta_pp": held_delta,
                "historical_LiteBN_heldout_macro_F1": base_held["macro_F1"], "EMA_heldout_macro_F1": ema_held["macro_F1"],
                "runtime_seconds": info["runtime_seconds"], "checkpoint_sha256": info["checkpoint_sha256"],
                "baseline_checkpoint": str(baseline_path), "baseline_checkpoint_sha256": r.sha256(baseline_path)})
            logs.append(info); task_infos.append(info)
            append_outputs(fold_rows, subject_rows, logs, summaries, {"terminal": "IN_PROGRESS", "tasks_completed": [x["task"] for x in summaries]})
            del baseline, ema, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
        task_outer_mean = float(np.mean(task_outer_d)); task_held_mean = float(np.mean(task_held_d))
        task_decision = "CONTINUE"
        if name == "OpenBMI_MI" and ((task_outer_mean < 0 and task_held_mean < 0) or task_held_mean <= -0.5):
            task_decision = "STOP_MODEL"
        elif name != "OpenBMI_MI" and (task_outer_mean <= -0.5 or task_held_mean <= -0.5):
            task_decision = "STOP_MODEL"
        rows_for_task = [row for row in fold_rows if row["task"] == name]
        summary = {"task": name, "folds": 5,
            "outer_baseline_BA": float(np.mean([x["historical_LiteBN_outer_BA"] for x in rows_for_task])),
            "outer_EMA_BA": float(np.mean([x["EMA_outer_BA"] for x in rows_for_task])), "outer_delta_pp": task_outer_mean,
            "heldout_baseline_BA": float(np.mean([x["historical_LiteBN_heldout_BA"] for x in rows_for_task])),
            "heldout_EMA_BA": float(np.mean([x["EMA_heldout_BA"] for x in rows_for_task])), "heldout_delta_pp": task_held_mean,
            "positive_outer_folds": int(sum(value > 0 for value in task_outer_d)),
            "positive_heldout_folds": int(sum(value > 0 for value in task_held_d)),
            "early_stop_epochs": [x["selected_epoch"] for x in task_infos if x["early_stopped"]], "decision": task_decision}
        summaries.append(summary)
        print("TASK_COMPLETE", json.dumps(summary, sort_keys=True), flush=True)
        append_outputs(fold_rows, subject_rows, logs, summaries, {"terminal": task_decision if task_decision == "STOP_MODEL" else "IN_PROGRESS", "tasks_completed": [x["task"] for x in summaries], "summaries": summaries})
        if task_decision == "STOP_MODEL":
            terminal = "STOP_MODEL"
            break
    if terminal != "STOP_MODEL":
        if len(summaries) == len(TASK_ORDER):
            outer_mean = float(np.mean([x["outer_delta_pp"] for x in summaries]))
            held_mean = float(np.mean([x["heldout_delta_pp"] for x in summaries]))
            terminal = "CONTINUE_TO_MULTI_SEED" if outer_mean > 0 and held_mean > 0 and all(x["outer_delta_pp"] >= 0 and x["heldout_delta_pp"] >= 0 for x in summaries) else "STOP_MODEL"
        else:
            terminal = "STOP_MODEL"
    decision = {"terminal": terminal, "tasks_completed": [x["task"] for x in summaries], "summaries": summaries,
                "final_heldout_accessed": False, "current_internal_heldout_accessed": True, "final_test_used_for_tuning": False}
    append_outputs(fold_rows, subject_rows, logs, summaries, decision)
    r.write_json(OUT / "FINAL_DECISION.json", decision)
    print("LITEBN_EMA_COMPLETE", terminal, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
