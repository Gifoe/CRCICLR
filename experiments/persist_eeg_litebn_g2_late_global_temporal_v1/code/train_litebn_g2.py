#!/usr/bin/env python3
"""Locked LiteBN-G2 seed-0 evaluation: one late 64-d GQA+RoPE block only."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import evaluate_exposed_benchmark as exposed
import sequential_gate as gates
from litebn_g2 import LiteBNG2

REPO = Path(os.environ.get("LITEBN_G2_REPO", "/root/rivermind-data/CRCICLR_G2_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_g2_late_global_temporal_v1"
CODE, OUT, PROTOCOL, RUNTIME = EXP / "code", EXP / "outputs", EXP / "protocol", EXP / "runtime"
BASE_RUNNER = REPO / "experiments/persist_eeg_linux_w0r0_screen_v1/code/run_w0r0_seed0.py"
SEED, EPOCHS, LR_G, LR_L, WEIGHT_DECAY, CLIP = 0, 40, 3e-4, 3e-5, 5e-4, 5.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


os.environ["W0R0_REPO"] = str(REPO)
runner = load_module("litebn_g2_base_runner", BASE_RUNNER)
base, _unused = runner.load_modules(); base.RUNTIME = RUNTIME; base.SEED = SEED


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"); os.replace(temporary, path)


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".part"); frame.to_csv(temporary, index=False); os.replace(temporary, path)


def total_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def group_norm(parameters: list[torch.nn.Parameter], attribute: str | None = None) -> float:
    values = [(parameter.grad if attribute == "grad" else parameter) for parameter in parameters]
    values = [value.detach().float().reshape(-1) for value in values if value is not None]
    return float(torch.linalg.vector_norm(torch.cat(values)).cpu()) if values else 0.0


def update_norm(before: list[torch.Tensor], after: list[torch.nn.Parameter]) -> float:
    return float(torch.linalg.vector_norm(torch.cat([(parameter.detach() - old).float().reshape(-1) for old, parameter in zip(before, after)])).cpu())


def mean_subject_metrics(values: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in values.values()])) for key in ("BA", "macro_F1", "accuracy")}


def build_model(task: str, fold: int, device: torch.device) -> tuple[LiteBNG2, Path]:
    checkpoint = runner.baseline_path(task, fold)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"exact B0 unavailable: {checkpoint}")
    b0 = base.build_model("LiteBN_BASELINE", task); state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    b0.load_state_dict(state, strict=True)
    if tuple(state["depth1.weight"].shape) != (48, 1, 1, 15) or tuple(state["point1.weight"].shape) != (64, 48, 1, 1):
        raise RuntimeError(f"not the historical 64-feature LiteBN: {checkpoint}")
    set_seed(SEED)
    return LiteBNG2(b0).to(device), checkpoint


def exact_identity(model: LiteBNG2, bundle, cache, subjects, mean, std) -> dict[str, Any]:
    labels, b0_logits, g2_logits = [], [], []; future = (int(base.TASKS[bundle.task]["future_session"]),)
    model.eval()
    with torch.inference_mode():
        for subject in base.subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], future)
            for offset in range(0, len(indices), 128):
                value, label = cache.batch(indices[offset:offset + 128], mean, std); b0, _ = model.base(value); g2, _ = model(value)
                labels.append(label.cpu().numpy()); b0_logits.append(b0.float().cpu().numpy()); g2_logits.append(g2.float().cpu().numpy())
    y, b0, g2 = np.concatenate(labels), np.concatenate(b0_logits), np.concatenate(g2_logits)
    b_metrics, g_metrics = base.classification_metrics(y, b0), base.classification_metrics(y, g2); maximum = float(np.max(np.abs(b0 - g2)))
    return {"max_abs_logit_difference": maximum, "exact_predictions": bool(np.array_equal(b0.argmax(1), g2.argmax(1))),
            "BA_equal": b_metrics["BA"] == g_metrics["BA"], "accuracy_equal": b_metrics["accuracy"] == g_metrics["accuracy"],
            "macro_F1_equal": b_metrics["macro_F1"] == g_metrics["macro_F1"],
            "pass": bool(maximum <= 1e-7 and np.array_equal(b0.argmax(1), g2.argmax(1)) and b_metrics == g_metrics)}


def batch_schedule(task: str, fold: dict, bundle) -> dict:
    if base.TASKS[task]["mi_protocol"]:
        episodes, metadata = base.mi_manifest(bundle, fold, task)
        return {**metadata, "episodes": episodes}
    return {"kind": "historical_task_full_permutation_batch64", "manifest_sha256": None}


def selected_attention(model: LiteBNG2, bundle, cache, subjects, mean, std) -> dict[str, float | bool]:
    values: list[tuple[float, float, float, float, float]] = []; future = (int(base.TASKS[bundle.task]["future_session"]),)
    model.eval()
    with torch.inference_mode():
        for subject in base.subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], future)
            for offset in range(0, len(indices), 128):
                value, _ = cache.batch(indices[offset:offset + 128], mean, std); model(value)
                block = model.global_block
                values.append((block.last_attention_entropy, block.last_normalized_attention_entropy, *block.last_entropy_quantiles))
    array = np.asarray(values, dtype=np.float64); normalized = float(array[:, 1].mean())
    return {"selected_mean_attention_entropy": float(array[:, 0].mean()), "selected_normalized_attention_entropy": normalized,
            "selected_entropy_p10": float(array[:, 2].mean()), "selected_entropy_p50": float(array[:, 3].mean()),
            "selected_entropy_p90": float(array[:, 4].mean()), "attention_selective": bool(normalized < .97)}


def train_fold(task: str, fold: dict, device: torch.device) -> tuple[dict, dict, dict, dict, dict, list[dict]]:
    fold_id = int(fold["fold_id"]); allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
    bundle = base.build_bundle(task, allowed)
    if set(bundle.subjects) != set(allowed):
        raise RuntimeError("canonical inner membership mismatch")
    mean, std, norm_meta = base.load_tensor_pair(runner.normalizer_source(task, fold_id)); cache = base.RawGPUCache(bundle, device)
    model, b0_path = build_model(task, fold_id, device)
    g_named, l_named = model.group_g_named_parameters(), model.group_l_named_parameters(); g_params, l_params = [p for _, p in g_named], [p for _, p in l_named]
    breakdown = model.global_block.parameter_breakdown()
    if breakdown["global_temporal_block_total"] != 29314:
        raise RuntimeError(f"unexpected G2 block count: {breakdown}")
    identity = exact_identity(model, bundle, cache, fold["inner_val_subjects"], mean, std)
    identity_row = {"task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id, "seed": SEED, "checkpoint_path": str(b0_path),
                    "checkpoint_sha256": runner.sha256(b0_path), "normalizer_sha256": norm_meta["mean_std_sha256"], **identity}
    if not identity["pass"]:
        raise RuntimeError(f"PROTOCOL_FAIL epoch0 identity: {task}/f{fold_id}")
    frozen_before, bn_before = model.frozen_parameter_sha256(), model.bn_sha256()
    epoch0 = base.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std); epoch0_metric = mean_subject_metrics(epoch0)
    best_epoch, best_ba, best_state = 0, epoch0_metric["BA"], clone_state(model)
    trajectory = [{"task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id, "seed": SEED, "epoch": 0,
                   "train_loss": None, "inner_val_BA": epoch0_metric["BA"], "selected": True, "alpha_attn": 0.0, "alpha_ffn": 0.0,
                   "Q_norm": group_norm([model.global_block.q.weight, model.global_block.q.bias]), "K_norm": group_norm([model.global_block.k.weight, model.global_block.k.bias]),
                   "V_norm": group_norm([model.global_block.v.weight, model.global_block.v.bias]), "W_O_norm": group_norm([model.global_block.output.weight, model.global_block.output.bias]),
                   "FFN_norm": group_norm(list(model.global_block.ffn.parameters())), "group_G_grad_norm": 0.0, "group_L_grad_norm": 0.0,
                   "group_G_update_norm": 0.0, "group_L_update_norm": 0.0, "mean_attention_entropy": None, "normalized_attention_entropy": None,
                   "attention_entropy_p10": None, "attention_entropy_p50": None, "attention_entropy_p90": None}]
    schedule = batch_schedule(task, fold, bundle); weights, _class_meta = base.class_weights(bundle, fold["inner_train_subjects"]); weight = None if weights is None else weights.to(device)
    optimizer = torch.optim.AdamW([{"params": g_params, "lr": LR_G}, {"params": l_params, "lr": LR_L}], weight_decay=WEIGHT_DECAY)
    scaler, amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"), device.type == "cuda"; train_indices = bundle.indices(fold["inner_train_subjects"], base.TASKS[task]["source_sessions"])
    set_seed(SEED + 100_000); started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train(True)
        if any(module.training for module in model.base.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
            raise RuntimeError("BN lock failure")
        batches = schedule["episodes"][epoch - 1] if base.TASKS[task]["mi_protocol"] else base.task_epoch_batches(train_indices, task, fold_id, epoch)
        losses, grad_g, grad_l, update_g, update_l, entropy = [], [], [], [], [], []
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std); optimizer.zero_grad(set_to_none=True)
            before_g, before_l = [parameter.detach().clone() for parameter in g_params], [parameter.detach().clone() for parameter in l_params]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(value); loss = F.cross_entropy(logits, labels, weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite task loss: {task}/f{fold_id}/e{epoch}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_g.append(group_norm(g_params, "grad")); grad_l.append(group_norm(l_params, "grad")); torch.nn.utils.clip_grad_norm_(g_params + l_params, CLIP)
            scaler.step(optimizer); scaler.update(); update_g.append(update_norm(before_g, g_params)); update_l.append(update_norm(before_l, l_params)); losses.append(float(loss.detach().cpu()))
            entropy.append((model.global_block.last_attention_entropy, model.global_block.last_normalized_attention_entropy, *model.global_block.last_entropy_quantiles))
        validation = base.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std); val = mean_subject_metrics(validation); chose = val["BA"] > best_ba + 1e-12
        if chose:
            best_epoch, best_ba, best_state = epoch, val["BA"], clone_state(model)
            for row in trajectory: row["selected"] = False
        measures = np.asarray(entropy, dtype=np.float64).mean(axis=0)
        trajectory.append({"task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id, "seed": SEED, "epoch": epoch, "train_loss": float(np.mean(losses)),
                           "inner_val_BA": val["BA"], "selected": chose, "alpha_attn": float(model.global_block.alpha_attn.detach().cpu()), "alpha_ffn": float(model.global_block.alpha_ffn.detach().cpu()),
                           "Q_norm": group_norm([model.global_block.q.weight, model.global_block.q.bias]), "K_norm": group_norm([model.global_block.k.weight, model.global_block.k.bias]),
                           "V_norm": group_norm([model.global_block.v.weight, model.global_block.v.bias]), "W_O_norm": group_norm([model.global_block.output.weight, model.global_block.output.bias]), "FFN_norm": group_norm(list(model.global_block.ffn.parameters())),
                           "group_G_grad_norm": float(np.mean(grad_g)), "group_L_grad_norm": float(np.mean(grad_l)), "group_G_update_norm": float(np.mean(update_g)), "group_L_update_norm": float(np.mean(update_l)),
                           "mean_attention_entropy": float(measures[0]), "normalized_attention_entropy": float(measures[1]), "attention_entropy_p10": float(measures[2]), "attention_entropy_p50": float(measures[3]), "attention_entropy_p90": float(measures[4])})
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[G2 {task} f{fold_id}] e={epoch:02d} valBA={val['BA']:.5f} best={best_ba:.5f}@{best_epoch}", flush=True)
    model.load_state_dict(best_state, strict=True)
    frozen_after, bn_after = model.frozen_parameter_sha256(), model.bn_sha256()
    frozen_audit = {"task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id, "seed": SEED, "frozen_parameter_sha256_before": frozen_before, "frozen_parameter_sha256_after": frozen_after, "bitwise_unchanged": frozen_before == frozen_after}
    bn_audit = {"task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id, "seed": SEED, "bn_sha256_before": bn_before, "bn_sha256_after": bn_after, "bitwise_unchanged": bn_before == bn_after, "all_bn_eval": not any(module.training for module in model.base.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))}
    if not frozen_audit["bitwise_unchanged"] or not bn_audit["bitwise_unchanged"] or not bn_audit["all_bn_eval"]:
        raise RuntimeError("PROTOCOL_FAIL frozen parameter or BN state changed")
    selectivity = {"task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id, "seed": SEED, "selected_epoch": best_epoch,
                   "alpha_attn_selected": float(model.global_block.alpha_attn.detach().cpu()), "alpha_ffn_selected": float(model.global_block.alpha_ffn.detach().cpu()),
                   **selected_attention(model, bundle, cache, fold["inner_val_subjects"], mean, std)}
    checkpoint = RUNTIME / "checkpoints" / task.lower() / f"fold{fold_id}_seed{SEED}" / "selected_best.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_torch_save(checkpoint, {"state_dict": model.state_dict(), "selected_epoch": best_epoch, "epoch0_inner_val_BA": epoch0_metric["BA"], "selected_inner_val_BA": best_ba,
                                        "selection": "highest subject-equal inner-validation BA, epoch0..40, earliest tie", "matched_B0_path": str(b0_path), "matched_B0_sha256": runner.sha256(b0_path), "normalizer_sha256": norm_meta["mean_std_sha256"], "attention_selectivity": selectivity})
    selection = {"task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id, "seed": SEED, "selected_epoch": best_epoch, "epoch0_fallback": best_epoch == 0,
                 "epoch0_inner_val_BA": epoch0_metric["BA"], "selected_inner_val_BA": best_ba, "checkpoint_path": str(checkpoint), "checkpoint_sha256": runner.sha256(checkpoint), "matched_B0_path": str(b0_path), "matched_B0_sha256": runner.sha256(b0_path), "normalizer_sha256": norm_meta["mean_std_sha256"], "alpha_attn_selected": selectivity["alpha_attn_selected"], "alpha_ffn_selected": selectivity["alpha_ffn_selected"], "elapsed_seconds": time.perf_counter() - started}
    group_audit = {"task": task, "fold": fold_id, "seed": SEED, "global_block_parameters": int(sum(parameter.numel() for parameter in g_params)), "group_L_parameters": int(sum(parameter.numel() for parameter in l_params)), "group_G_lr": LR_G, "group_L_lr": LR_L, "weight_decay": WEIGHT_DECAY, "group_G_names": ";".join(name for name, _ in g_named), "group_L_names": ";".join(name for name, _ in l_named), "global_breakdown": json.dumps(breakdown, sort_keys=True)}
    record = {"checkpoint": checkpoint, "identity": identity_row, "frozen": frozen_audit, "bn": bn_audit, "selection": selection, "groups": group_audit, "selectivity": selectivity, "trajectory": trajectory}
    del model, cache, bundle; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return record, identity_row, frozen_audit, bn_audit, selectivity, trajectory


def init_documents(device: torch.device) -> None:
    for path in (EXP, CODE, OUT, PROTOCOL, RUNTIME): path.mkdir(parents=True, exist_ok=True)
    _search, folds, split_hash = base.load_folds(); model, _checkpoint = build_model("OpenBMI_SSVEP", int(folds["OpenBMI"][0]["fold_id"]), device)
    spec = model.model_spec(); spec.update({"historical_LiteBN_total_parameters": total_parameters(model.base), "LiteBN_G2_total_parameters": total_parameters(model), "exact_total_parameter_count": total_parameters(model), "split_sha256": split_hash, "optimizer_parameter_groups": {"G": {"lr": LR_G, "parameters": "all late Global Temporal Block parameters"}, "L": {"lr": LR_L, "parameters": "backend2/readout parameters only"}}, "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": CLIP, "max_epochs": EPOCHS})
    write_json(OUT / "MODEL_SPEC.json", spec)
    write_json(PROTOCOL / "SOURCE_PROVENANCE.json", {"seed": SEED, "fivefold_split_sha256": split_hash, "historical_litebn_source": str(base.CARRIER_CODE / "run_carrier_screen.py"), "historical_litebn_source_sha256": base.sha256_file(base.CARRIER_CODE / "run_carrier_screen.py"), "baseline": "strict-loaded historical 64-feature LiteBN checkpoint", "baseline_retrained": False, "per_fold_checkpoint_and_normalizer_audit": "outputs/EPOCH0_IDENTITY_AUDIT.csv", "selection_checkpoint_audit": "outputs/CHECKPOINT_SELECTION.csv"})
    write_json(PROTOCOL / "DATA_SCOPE_AUDIT.json", {"NEW_SEALED_TEST_ACCESSED": "NO", "evaluation_status": "EXPOSED_DEVELOPMENT_BENCHMARK", "OpenBMI_V8_INTERNAL": list(exposed.OPENBMI_V8_INTERNAL), "WBCIC_FINAL_TRUE_OUTER": list(exposed.WBCIC_FINAL_TRUE_OUTER_EXPOSED), "checkpoint_selection": "inner validation only", "closed_routes": ["C", "M", "S", "XS", "X", "gradient guards", "routers", "fusion", "ensemble", "DG/meta-learning", "BN adaptation", "statistics residual", "extra transformer layers"]})
    write_json(PROTOCOL / "RUNTIME_AUDIT.json", {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "optimizer": "AdamW", "group_G_lr": LR_G, "group_L_lr": LR_L, "weight_decay": WEIGHT_DECAY, "clip": CLIP, "epochs": EPOCHS})
    (PROTOCOL / "LITEBN_G2_PROTOCOL.md").write_text("# LiteBN-G2 protocol\n\nThis locked experiment inserts one 4Q/2KV, d=64 GQA+RoPE block after historical LiteBN backend block 2 and before Pool8. It has two zero-initialized ReZero scalars. Stem, backend1, and every BatchNorm state are frozen; only the late global block plus declared backend2/readout parameters train. Exposed benchmarks are opened only after each task has five frozen checkpoints.\n", encoding="utf-8")
    (EXP / "README.md").write_text("# LiteBN-G2 late global temporal evaluation\n\nOne fixed late-attention architecture is evaluated sequentially against already-exposed development benchmarks. It does not reuse closed residual, guard, router, fusion, or ensemble routes.\n", encoding="utf-8")
    del model


def persist(identity, frozen, bn, groups, selectivity, trajectory, selections, tasks, decision) -> None:
    write_csv(OUT / "EPOCH0_IDENTITY_AUDIT.csv", identity); write_csv(OUT / "FROZEN_PARAMETER_AUDIT.csv", frozen); write_csv(OUT / "BN_LOCK_AUDIT.csv", bn); write_csv(OUT / "PARAMETER_GROUP_AUDIT.csv", groups); write_csv(OUT / "ATTENTION_SELECTIVITY.csv", selectivity); write_csv(OUT / "TRAINING_TRAJECTORY.csv", trajectory); write_csv(OUT / "CHECKPOINT_SELECTION.csv", selections); write_csv(OUT / "SEED0_TASK_RESULTS.csv", tasks); write_json(OUT / "SEED0_GATE_DECISION.json", decision)


def run_seed0(device: torch.device) -> dict:
    _search, folds, _split = base.load_folds(); identity, frozen, bn, groups, selectivity, trajectory, selections, tasks, outcomes = [], [], [], [], [], [], [], [], {}
    stopped, stop_label = False, None
    for task in gates.TASK_ORDER:
        if stopped:
            benchmark, _model_name = exposed.BENCHMARK[task]; tasks.append({"task": task, "benchmark_best": benchmark, "matched_LiteBN": np.nan, "LiteBN_G2": np.nan, "delta_vs_LiteBN_pp": np.nan, "delta_vs_benchmark_pp": np.nan, "selected_epochs": "", "epoch0_fallback_count": np.nan, "mean_selected_normalized_attention_entropy": np.nan, "entropy_lt_097_folds": np.nan, "PASS": False, "Executed": "NO", "reason": "EARLY_STOP"}); continue
        dataset = base.TASKS[task]["dataset"]; records, checkpoints = [], {}
        for fold in folds[dataset]:
            record, identity_row, frozen_row, bn_row, selectivity_row, history = train_fold(task, fold, device); records.append(record); checkpoints[int(fold["fold_id"])] = record["checkpoint"]
            identity.append(identity_row); frozen.append(frozen_row); bn.append(bn_row); groups.append(record["groups"]); selectivity.append(selectivity_row); trajectory.extend(history); selections.append(record["selection"])
            persist(identity, frozen, bn, groups, selectivity, trajectory, selections, tasks, {"status": "RUNNING", "current_task": task})
            print(f"G2_FOLD_FROZEN {task} fold={fold['fold_id']} selected={record['selection']['selected_epoch']}", flush=True)
        result, _ = exposed.evaluate_task(base=base, runner=runner, task=task, folds=folds[dataset], checkpoints=checkpoints, device=device, output=OUT / f"{task}_EXPOSED_SUBJECT_RESULTS.csv")
        passed, label = gates.gate(task, result); outcomes[task] = {**result, "pass": passed, "gate": label}
        selected_entropy = float(np.mean([record["selectivity"]["selected_normalized_attention_entropy"] for record in records])); selective_folds = int(sum(record["selectivity"]["attention_selective"] for record in records))
        tasks.append({"task": task, "benchmark_best": result["benchmark_best"], "matched_LiteBN": result["matched_LiteBN_BA"], "LiteBN_G2": result["LiteBN_G2_BA"], "delta_vs_LiteBN_pp": result["delta_vs_LiteBN_pp"], "delta_vs_benchmark_pp": result["delta_vs_benchmark_pp"], "selected_epochs": ";".join(str(record["selection"]["selected_epoch"]) for record in records), "epoch0_fallback_count": int(sum(record["selection"]["epoch0_fallback"] for record in records)), "mean_selected_normalized_attention_entropy": selected_entropy, "entropy_lt_097_folds": selective_folds, "PASS": passed, "Executed": "YES", "reason": label})
        stopped, stop_label = (not passed), (label if not passed else None)
        persist(identity, frozen, bn, groups, selectivity, trajectory, selections, tasks, {"status": label, "outcomes": outcomes, "early_stopped": stopped})
        print(f"G2_TASK_COMPLETE {task} gate={label} ba={result['LiteBN_G2_BA']:.6f}", flush=True)
    terminal = stop_label or "SEED0_ALL_TASK_PASS_MULTISEED_PENDING"
    decision = {"terminal": terminal, "seed": SEED, "outcomes": outcomes, "early_stopped": stopped, "NEW_SEALED_TEST_ACCESSED": "NO", "seed1_run": False, "seed2_run": False}
    persist(identity, frozen, bn, groups, selectivity, trajectory, selections, tasks, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed0", action="store_true"); args = parser.parse_args()
    if not args.seed0: raise RuntimeError("only --seed0 is accepted in this locked first stage")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda"); init_documents(device); decision = run_seed0(device); write_json(OUT / "RUN_STATE.json", decision); print(f"LITEBN_G2_SEED0_COMPLETE terminal={decision['terminal']}", flush=True)


if __name__ == "__main__":
    main()
