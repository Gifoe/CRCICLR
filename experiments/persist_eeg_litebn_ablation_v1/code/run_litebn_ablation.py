#!/usr/bin/env python3
"""Seed-0, five-fold independent LiteBN ablation study."""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO = Path(os.environ.get("ABLATION_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_ablation_v1"
OUT, RUN, PROTOCOL = EXP / "outputs", EXP / "runtime", EXP / "protocol"
REMAIN = REPO / "experiments/persist_eeg_litebn_tfformer_remaining_tasks_seed0_v1"
sys.path[:0] = [str(EXP / "code"), str(REMAIN / "code")]
spec = importlib.util.spec_from_file_location("litebn_ablation_base", REMAIN / "code/run_remaining_seed0.py")
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)
from litebn_variants import LiteBNAblation  # noqa: E402

base, baseline_runner = frozen.base, frozen.runner
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
VARIANTS = LiteBNAblation.VARIANTS
ALL_VARIANTS = ("B0_FULL",) + VARIANTS
SEED, EPOCHS, MIN_EPOCH, PATIENCE, BOOTSTRAPS = 0, 60, 10, 8, 20000
OPENBMI_INTERNAL = ["4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54"]
WBCIC_TRUE_OUTER = ["sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51"]
TRUE_OUTER_ROOT = Path("/root/rivermind-data/persist_eeg_cache/wbcic_true_outer_v1/wbcic_epochs")
NORMALIZERS = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime/normalizers")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def set_seed(value: int = SEED) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


class NormalizedCache:
    def __init__(self, raw, mean, std):
        self.raw, self.mean, self.std = raw, mean, std

    def batch(self, indices):
        return self.raw.batch(np.asarray(indices, dtype=np.int64), self.mean, self.std)


def build_variant(task: str, variant: str, device):
    reference = base.build_model("LiteBN_BASELINE", task)
    channels = int(reference.spatial[0].kernel_size[0])
    classes = int(reference.head.out_features)
    del reference
    set_seed()
    return LiteBNAblation(channels, classes, variant).to(device)


def full_checkpoint(task: str, fold: int) -> Path:
    return Path(baseline_runner.baseline_path(task, fold))


def build_full(task: str, fold: int, device):
    model = base.build_model("LiteBN_BASELINE", task)
    path = full_checkpoint(task, fold)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
    return model.to(device), path


def evaluate(model, bundle, cache, subjects):
    model.eval()
    rows = {}
    session = int(base.TASKS[bundle.task]["future_session"])
    with torch.no_grad():
        for subject in subjects:
            indices = bundle.indices([subject], (session,))
            logits = []
            for start in range(0, len(indices), 128):
                x, _ = cache.batch(indices[start:start + 128])
                logits.append(model(x)[0].float().cpu().numpy())
            rows[str(subject)] = base.classification_metrics(bundle.labels(indices), np.concatenate(logits))
            rows[str(subject)]["trials"] = len(indices)
    return float(np.mean([value["BA"] for value in rows.values()])), rows


def train_one(task, fold, variant, bundle, cache, class_weight, device):
    fold_id = int(fold["fold_id"])
    directory = RUN / "checkpoints" / task / variant / f"fold{fold_id}"
    latest, selected = directory / "latest.pt", directory / "selected.pt"
    model = build_variant(task, variant, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    invariant = {
        "model": "LiteBN", "task": task, "variant": variant, "fold": fold_id,
        "seed": SEED, "epochs_cap": EPOCHS, "minimum_checkpoint_epoch": MIN_EPOCH,
        "early_stop_patience": PATIENCE, "optimizer": "AdamW", "lr": 3e-4,
        "weight_decay": 5e-4, "gradient_clip": 5.0,
        "loss": "weighted CE" if task == "OpenBMI_ERP" else "CE",
    }
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["invariant"] != invariant:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start = int(saved["epoch"]) + 1
        history, best, best_epoch, best_state = saved["history"], saved["best"], saved["best_epoch"], saved["best_state"]
    train_indices = bundle.indices(fold["inner_train_subjects"], base.TASKS[task]["source_sessions"])
    episodes = base.mi_manifest(bundle, fold, task)[0] if base.TASKS[task]["mi_protocol"] else None
    weight = None if class_weight is None else class_weight.to(device)
    started, early_stopped, last_epoch = time.perf_counter(), False, start - 1
    for epoch in range(start, EPOCHS + 1):
        last_epoch = epoch
        model.train()
        batches = episodes[epoch - 1] if episodes is not None else base.task_epoch_batches(train_indices, task, fold_id, epoch)
        losses = []
        for indices in batches:
            x, y = cache.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y, weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {task} {variant} fold{fold_id} epoch{epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation_ba, _ = evaluate(model, bundle, cache, fold["inner_val_subjects"])
        chosen = epoch >= MIN_EPOCH and validation_ba > best + 1e-12
        if chosen:
            best, best_epoch, best_state = validation_ba, epoch, state(model)
        history.append({
            "model": "LiteBN", "task": task, "variant": variant, "seed": SEED,
            "fold": fold_id, "epoch": epoch, "train_loss": float(np.mean(losses)),
            "inner_val_BA": validation_ba, "eligible": epoch >= MIN_EPOCH,
            "selected_at_epoch": chosen,
        })
        if epoch == 1 or epoch % 5 == 0 or chosen:
            print(f"LITEABL {task} {variant} f{fold_id} e{epoch:02d} BA={validation_ba:.6f}", flush=True)
        if epoch % 5 == 0:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = latest.with_suffix(".pt.part")
            torch.save({
                "epoch": epoch, "state": state(model), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(), "history": history,
                "best": best, "best_epoch": best_epoch, "best_state": best_state,
                "invariant": invariant,
            }, temporary)
            os.replace(temporary, latest)
        if epoch >= MIN_EPOCH and best_epoch is not None and epoch - best_epoch >= PATIENCE:
            early_stopped = True
            print(f"LITEABL_EARLY_STOP {task} {variant} f{fold_id} epoch={epoch} best={best_epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"no checkpoint selected: {task} {variant} fold{fold_id}")
    model.load_state_dict(best_state)
    replay, _ = evaluate(model, bundle, cache, fold["inner_val_subjects"])
    if abs(replay - best) > 1e-12:
        raise RuntimeError(f"checkpoint replay mismatch: {task} {variant} fold{fold_id}")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_suffix(".pt.part")
    torch.save({"state_dict": state(model), "epoch": best_epoch, "inner_val_BA": best, "invariant": invariant}, temporary)
    os.replace(temporary, selected)
    return {
        "model": "LiteBN", "task": task, "variant": variant, "seed": SEED,
        "fold": fold_id, "selected_epoch": best_epoch, "inner_val_BA": best,
        "epochs_ran": last_epoch, "early_stopped": early_stopped,
        "checkpoint": str(selected), "checkpoint_sha256": sha(selected),
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "history": history,
    }


class FastOuterRawCache:
    def __init__(self, device):
        values, labels = [], []
        for subject in WBCIC_TRUE_OUTER:
            values.append(torch.from_numpy(np.asarray(np.load(TRUE_OUTER_ROOT / subject / "ses-2_epochs.npy", allow_pickle=False), dtype=np.float32)))
            labels.append(torch.from_numpy(np.asarray(np.load(TRUE_OUTER_ROOT / subject / "ses-2_labels.npy", allow_pickle=False), dtype=np.int64)))
        self.x = torch.cat(values).to(device)
        self.y = torch.cat(labels).to(device)
        self.device = device

    def batch(self, indices, mean, std):
        index = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long, device=self.device)
        value = self.x.index_select(0, index)
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=self.device)[None, :, None]
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=self.device)[None, :, None]
        return (value - mean_tensor) / torch.clamp(std_tensor, min=1e-6), self.y.index_select(0, index)


def true_outer_bundle():
    rows = []
    for subject in WBCIC_TRUE_OUTER:
        labels = np.load(TRUE_OUTER_ROOT / subject / "ses-2_labels.npy", allow_pickle=False)
        signal = TRUE_OUTER_ROOT / subject / "ses-2_epochs.npy"
        rows.extend(base.Row(subject, 2, str(signal), index, int(label)) for index, label in enumerate(labels))
    return base.SignalBundle("WBCIC_MI", WBCIC_TRUE_OUTER, rows)


def selected_model(task, fold, variant, device):
    if variant == "B0_FULL":
        return build_full(task, fold, device)
    model = build_variant(task, variant, device)
    path = RUN / f"checkpoints/{task}/{variant}/fold{fold}/selected.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), path


def evaluate_all(device):
    subject_rows = []
    _, fold_map, _ = base.load_folds()
    for task in TASKS:
        subjects = WBCIC_TRUE_OUTER if task == "WBCIC_MI" else OPENBMI_INTERNAL
        for fold in fold_map[base.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            if task == "WBCIC_MI":
                bundle = true_outer_bundle()
                mean, std, metadata = base.load_tensor_pair(NORMALIZERS / f"wbcic_mi_fold{fold_id}.npz")
                raw = FastOuterRawCache(device)
                scope = "already-accessed WBCIC true-outer future-session S2"
            else:
                bundle = base.build_bundle(task, subjects)
                mean, std, metadata = base.load_tensor_pair(baseline_runner.normalizer_source(task, fold_id))
                raw = base.RawGPUCache(bundle, device)
                scope = "OpenBMI V8 internal future-session S2"
            cache = NormalizedCache(raw, mean, std)
            for variant in ALL_VARIANTS:
                model, checkpoint = selected_model(task, fold_id, variant, device)
                _, rows = evaluate(model, bundle, cache, subjects)
                for subject in subjects:
                    subject_rows.append({
                        "model": "LiteBN", "variant": variant, "task": task, "seed": SEED,
                        "fold": fold_id, "subject_id": subject, "BA": rows[subject]["BA"],
                        "macro_F1": rows[subject]["macro_F1"], "accuracy": rows[subject]["accuracy"],
                        "trials": rows[subject]["trials"], "evaluation_scope": scope,
                        "normalizer_sha256": metadata["mean_std_sha256"], "checkpoint_sha256": sha(checkpoint),
                    })
                del model
                torch.cuda.empty_cache()
            atomic_csv(OUT / "LITEBN_ABLATION_SUBJECT_RESULTS.csv", subject_rows)
            print(f"LITEABL_EVAL_DONE {task} f{fold_id}", flush=True)
            del cache, raw, bundle
            gc.collect()
            torch.cuda.empty_cache()
    return subject_rows


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summarize(subject_rows):
    frame = pd.DataFrame(subject_rows)
    biological = frame.groupby(["task", "variant", "subject_id"], as_index=False)[["BA", "macro_F1"]].mean()
    rows = []
    for variant in ALL_VARIANTS:
        row, task_bas, task_deltas = {"model": "LiteBN", "variant": variant}, [], []
        for task in TASKS:
            current = biological[(biological.task == task) & (biological.variant == variant)].set_index("subject_id")
            full = biological[(biological.task == task) & (biological.variant == "B0_FULL")].set_index("subject_id")
            shared = sorted(set(current.index) & set(full.index))
            delta = 100 * (current.loc[shared, "BA"].to_numpy() - full.loc[shared, "BA"].to_numpy())
            mean_delta, low, high = bootstrap(delta)
            key = task.replace("OpenBMI_", "OpenBMI-").replace("WBCIC_", "WBCIC-")
            row[f"{key}_BA"] = float(current.BA.mean())
            row[f"{key}_macro_F1"] = float(current.macro_F1.mean())
            row[f"{key}_delta_BA_pp_vs_FULL"] = mean_delta
            row[f"{key}_delta_CI95_low_pp"] = low
            row[f"{key}_delta_CI95_high_pp"] = high
            task_bas.append(float(current.BA.mean()))
            task_deltas.append(mean_delta)
        row["Mean_BA"] = float(np.mean(task_bas))
        row["Mean_delta_BA_pp_vs_FULL"] = float(np.mean(task_deltas))
        rows.append(row)
    atomic_csv(OUT / "LITEBN_ABLATION_SUMMARY.csv", rows)
    return rows


def report(summary):
    lookup = {row["variant"]: row for row in summary}
    lines = [
        "# Independent LiteBN ablation report", "",
        "Seed 0, five folds. All ablation cells used the exact frozen loaders, train-only normalizers, losses, and subject-equal inner-validation checkpoint metric.", "",
        "User-approved acceleration: maximum 60 epochs, minimum checkpoint epoch 10, and early stopping after 8 epochs without strict eligible inner-validation BA improvement.", "",
        "Execution-order deviation: the user explicitly paused TFFormer after its saved checkpoint state and requested LiteBN next at full GPU allocation. This does not change either independent model protocol.", "",
        "WBCIC deviation: all checkpoints were frozen from source/development data before direct comparison on the already-accessed true-outer future-session S2 ten-subject set. This is an exposed ablation benchmark, not untouched confirmation.", "",
        "| Variant | OpenBMI-MI BA | OpenBMI-ERP BA | OpenBMI-SSVEP BA | WBCIC-MI BA | Mean BA | Mean delta vs Full (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ALL_VARIANTS:
        row = lookup[variant]
        lines.append(f"| {variant} | {row['OpenBMI-MI_BA']:.4f} | {row['OpenBMI-ERP_BA']:.4f} | {row['OpenBMI-SSVEP_BA']:.4f} | {row['WBCIC-MI_BA']:.4f} | {row['Mean_BA']:.4f} | {row['Mean_delta_BA_pp_vs_FULL']:+.3f} |")
    lines += ["", "## Per-task paired subject analysis", ""]
    for variant in VARIANTS:
        row = lookup[variant]
        lines += [f"### {variant}", "", "| Task | Ablation BA | Delta vs Full (pp) | Paired 95% bootstrap CI (pp) |", "|---|---:|---:|---:|"]
        for task in TASKS:
            key = task.replace("OpenBMI_", "OpenBMI-").replace("WBCIC_", "WBCIC-")
            lines.append(f"| {key} | {row[f'{key}_BA']:.4f} | {row[f'{key}_delta_BA_pp_vs_FULL']:+.3f} | [{row[f'{key}_delta_CI95_low_pp']:+.3f}, {row[f'{key}_delta_CI95_high_pp']:+.3f}] |")
        lines.append("")
    lines += ["The results support conclusions only about the multi-scale temporal stem, filter-specific spatial richness, hierarchical temporal backend, and normalization choice.", ""]
    (OUT / "FINAL_LITEBN_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def add_full_rows(records):
    existing = {(row["task"], row["variant"], int(row["fold"])) for row in records}
    for task in TASKS:
        for fold in range(5):
            if (task, "B0_FULL", fold) in existing:
                continue
            checkpoint = full_checkpoint(task, fold)
            records.append({
                "model": "LiteBN", "task": task, "variant": "B0_FULL", "seed": SEED,
                "fold": fold, "selected_epoch": None, "inner_val_BA": None,
                "epochs_ran": 60, "early_stopped": False, "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint), "elapsed_seconds": None,
                "trainable_parameters": sum(p.numel() for p in base.build_model("LiteBN_BASELINE", task).parameters()),
                "history": [], "reused_protocol_identical": True,
            })


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    for directory in (OUT, RUN, PROTOCOL):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_json(PROTOCOL / "ABLATION_PROTOCOL_LOCK.json", {
        "model": "LiteBN", "independent_model_study": True, "seed": SEED,
        "folds": list(range(5)), "tasks": list(TASKS), "variants": list(ALL_VARIANTS),
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 5e-4,
        "gradient_clip": 5.0, "epochs_cap": EPOCHS, "minimum_checkpoint_epoch": MIN_EPOCH,
        "early_stop_patience": PATIENCE, "checkpoint_metric": "subject-equal inner-validation BA",
        "execution_order_user_override": "TFFormer paused with resumable checkpoints; LiteBN runs next at full GPU allocation",
        "OpenBMI_evaluation": "existing V8 internal future-session S2",
        "WBCIC_evaluation": {
            "user_override": True, "scope": "already-accessed true-outer future-session S2 ten subjects",
            "subjects": WBCIC_TRUE_OUTER, "selection_use": False,
            "interpretation": "exposed ablation benchmark; not untouched or one-shot",
        },
    })
    device = torch.device("cuda")
    _, fold_map, _ = base.load_folds()
    log_path = RUN / "TRAINING_LOGS.json"
    records = json.loads(log_path.read_text()) if log_path.is_file() else []
    add_full_rows(records)
    atomic_json(log_path, records)
    done = {(row["task"], row["variant"], int(row["fold"])) for row in records}
    for task in TASKS:
        for fold in fold_map[base.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            pending = [variant for variant in VARIANTS if (task, variant, fold_id) not in done]
            if not pending:
                continue
            bundle = base.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
            mean, std, _ = base.load_tensor_pair(baseline_runner.normalizer_source(task, fold_id))
            raw = base.RawGPUCache(bundle, device)
            cache = NormalizedCache(raw, mean, std)
            weight, _ = base.class_weights(bundle, fold["inner_train_subjects"])
            print(f"LITEABL_CACHE {task} f{fold_id} pending={pending}", flush=True)
            for variant in pending:
                record = train_one(task, fold, variant, bundle, cache, weight, device)
                records.append(record)
                done.add((task, variant, fold_id))
                atomic_json(log_path, records)
                atomic_csv(OUT / "LITEBN_ABLATION_RUNS.csv", [{key: value for key, value in row.items() if key != "history"} for row in records])
                atomic_csv(OUT / "TRAINING_TRAJECTORY.csv", [entry for row in records for entry in row.get("history", [])])
            del cache, raw, bundle
            gc.collect()
            torch.cuda.empty_cache()
    expected = len(TASKS) * len(ALL_VARIANTS) * 5
    if len({(row["task"], row["variant"], int(row["fold"])) for row in records}) != expected:
        raise RuntimeError("not all LiteBN checkpoints are frozen; refusing evaluation")
    subject_rows = evaluate_all(device)
    summary = summarize(subject_rows)
    report(summary)
    atomic_json(OUT / "FINAL_LITEBN_ABLATION_STATUS.json", {
        "status": "COMPLETE", "run_count": expected, "subject_row_count": len(subject_rows),
        "all_checkpoints_frozen_before_WBCIC_true_outer_ablation_comparison": True,
        "WBCIC_true_outer_is_exposed_for_ablation_comparison": True,
    })
    print("LITEABL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
