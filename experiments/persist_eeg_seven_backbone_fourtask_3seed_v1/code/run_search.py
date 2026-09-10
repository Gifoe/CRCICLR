"""One resumable, non-overlapping frozen SEARCH training cell.

This entry point is disabled until BENCHMARK_PROTOCOL_FREEZE.md and its SHA256
sidecar are committed.  It contains no final-held-out loader or path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from backbone_models import MODELS, MODEL_NATIVE_RESAMPLING, admission_metadata, build_model
from benchmark_data import load_search_fold
from tech_official import recipe_config

# Keep CPU-side tensor/OpenMP work below the server's 32 logical CPUs.  This
# is a host scheduling bound only; it does not change numerical recipes.
_cpu_threads = int(os.environ.get("SEVEN_CPU_THREADS", "24"))
if _cpu_threads > 0:
    torch.set_num_threads(_cpu_threads)


EXP = Path(__file__).resolve().parents[1]
PROTOCOL, OUTPUTS = EXP / "protocol", EXP / "outputs"
RUNTIME = Path(os.environ["SEVEN_RUNTIME"]).resolve()
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
CLASSIC = {"EEGNet", "LiteBN", "TCFormer"}
FOUNDATION = {"ST-EEGFormer-small", "LaBraM-base", "CBraMod"}


def _clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(_clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        # Serialize into memory first.  On Windows, PyTorch's direct zip
        # writer can raise ``unexpected pos`` on a large file when a file
        # scanner briefly interferes with the destination handle.  A
        # BytesIO serialization keeps the previous valid checkpoint intact and
        # makes the final filesystem operation a plain sequential write.
        buffer = io.BytesIO()
        torch.save(value, buffer)
        with temporary.open("wb") as handle:
            handle.write(buffer.getbuffer())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        # Windows can surface a transient iostream/badbit when the large
        # foundation checkpoint is written while the drive is near capacity.
        # Preserve the previous valid checkpoint and let the cell resume from
        # it; a failed write must never destroy a valid state.
        try: temporary.unlink()
        except FileNotFoundError: pass
        raise


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _set_seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch"].detach().cpu())
    if torch.cuda.is_available() and "cuda" in value:
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in value["cuda"]])


def _subject_metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    prediction = logits.argmax(1); rows = []
    for subject in sorted(np.unique(subjects.astype(str)), key=lambda value: int(value.replace("sub-", ""))):
        mask = subjects.astype(str) == subject; truth, pred = labels[mask], prediction[mask]
        rows.append({"subject": subject, "BA": float(balanced_accuracy_score(truth, pred)),
                     "macro_F1": float(f1_score(truth, pred, average="macro", zero_division=0)), "trials": int(mask.sum())})
    return {"subject_equal_BA": float(np.mean([row["BA"] for row in rows])),
            "subject_equal_macro_F1": float(np.mean([row["macro_F1"] for row in rows])),
            "trial_accuracy": float(accuracy_score(labels, prediction)), "subjects": rows}


def _resample(model: str, value: np.ndarray) -> np.ndarray:
    spec = MODEL_NATIVE_RESAMPLING[model]
    if spec is None: return np.ascontiguousarray(value, dtype=np.float32)
    result = resample_poly(value.astype(np.float64, copy=False), int(spec["target_hz"]), int(spec["source_hz"]), axis=-1)
    return np.ascontiguousarray(result.astype(np.float32))


def _frozen_recipe(task: str, model: str) -> dict[str, Any]:
    selected = {line.split(",")[0]: line.split(",")[3]
                for line in (PROTOCOL / "TECH_RECIPE_SELECTION.csv").read_text(encoding="utf-8").splitlines()[1:]}
    if model == "TeCh":
        name = selected[task]; config = recipe_config(channels=1, samples=1, classes=2, recipe=name)
        return {"name": name, "epochs": int(config.train_epochs), "min_epoch": 10, "batch_size": int(config.batch_size),
                "lr": float(config.learning_rate), "weight_decay": 0.0, "gradient_clip": 5.0,
                "selection": "frozen inner-validation subject-equal BA"}
    if model in CLASSIC:
        return {"name": "COMMON_SUPERVISED_V1", "epochs": 60, "min_epoch": 10, "batch_size": 128,
                "lr": 3e-4, "weight_decay": 5e-4, "gradient_clip": 5.0, "selection": "frozen inner-validation subject-equal BA"}
    batch = {"ST-EEGFormer-small": 16, "LaBraM-base": 64, "CBraMod": 64}[model]
    return {"name": "PRETRAINED_FINETUNE_V1", "epochs": 60, "min_epoch": 10, "batch_size": batch,
            "lr": 1e-4, "weight_decay": 5e-4, "gradient_clip": 5.0, "selection": "frozen inner-validation subject-equal BA"}


def _batch_order(length: int, *, task: str, model: str, fold: int, seed: int, epoch: int, size: int) -> list[np.ndarray]:
    digest = hashlib.sha256(f"SEVEN-BACKBONE|{task}|{model}|{fold}|{seed}|{epoch}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    values = rng.permutation(length)
    return [values[start:start + size] for start in range(0, len(values), size)]


def _evaluate(model: torch.nn.Module, x: torch.Tensor, labels: np.ndarray, subjects: np.ndarray, batch: int) -> dict[str, Any]:
    parts = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch): parts.append(model(x[start:start + batch]).float().cpu().numpy())
    return _subject_metrics(labels, np.concatenate(parts), subjects)


def _freeze_hash() -> str:
    document, sidecar = PROTOCOL / "BENCHMARK_PROTOCOL_FREEZE.md", PROTOCOL / "BENCHMARK_PROTOCOL_FREEZE.sha256"
    if not document.is_file() or not sidecar.is_file():
        raise RuntimeError("SEARCH is disabled before the committed BENCHMARK_PROTOCOL_FREEZE")
    actual, expected = _sha(document), sidecar.read_text(encoding="utf-8").strip()
    if actual != expected: raise RuntimeError("BENCHMARK_PROTOCOL_FREEZE hash mismatch")
    return actual


def _run_cell(task: str, model_name: str, fold_id: int, seed: int) -> dict[str, Any]:
    freeze_hash = _freeze_hash()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slug = model_name.lower().replace("-", "_")
    cell = RUNTIME / "search_cells" / task.lower() / slug / f"fold{fold_id}_seed{seed}"
    record_path, latest_path, selected_path, lock = cell / "record.json", cell / "latest.pt", cell / "selected.pt", cell / "RUNNING.lock"
    if record_path.is_file() and selected_path.is_file(): return json.loads(record_path.read_text(encoding="utf-8"))
    try: lock.mkdir(parents=True, exist_ok=False)
    except FileExistsError: raise RuntimeError(f"refusing overlapping or stale cell: {lock}")
    started = time.perf_counter()
    try:
        data = load_search_fold(task, fold_id)
        recipe = _frozen_recipe(task, model_name); tech = recipe["name"] if model_name == "TeCh" else None
        _set_seed(seed)
        model = build_model(model_name, dataset=data["dataset"], channels=data["channels"], samples=data["samples"], classes=data["classes"], tech_recipe=tech).to(device)
        initial_hash = _state_sha(model); _set_seed(seed + 100000)
        train_x = torch.from_numpy(_resample(model_name, data["train_x"])).to(device, non_blocking=True)
        val_x = torch.from_numpy(_resample(model_name, data["val_x"])).to(device, non_blocking=True)
        outer_x = torch.from_numpy(_resample(model_name, data["outer_x"])).to(device, non_blocking=True)
        train_y = torch.as_tensor(data["train_y"], dtype=torch.long, device=device)
        weight = None
        if data["weighted_cross_entropy"]:
            counts = np.bincount(data["train_y"], minlength=data["classes"])
            weight = torch.as_tensor(len(data["train_y"]) / (data["classes"] * counts), dtype=torch.float32, device=device)
        invariant = hashlib.sha256(json.dumps({"task": task, "model": model_name, "fold": fold_id, "seed": seed,
            "initial_state": initial_hash, "split": data["split_sha256"], "normalizer": data["normalizer"], "recipe": recipe,
            "freeze": freeze_hash, "adapter": MODEL_NATIVE_RESAMPLING[model_name]}, sort_keys=True).encode()).hexdigest()
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
        start_epoch, history, best, best_epoch, best_state = 1, [], -float("inf"), 0, None
        if latest_path.is_file():
            saved = torch.load(latest_path, map_location=device, weights_only=False)
            if saved.get("invariant") != invariant: raise RuntimeError("cell resume invariant mismatch")
            model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"]); _restore_rng(saved["rng"])
            start_epoch, history, best, best_epoch, best_state = int(saved["epoch"]) + 1, list(saved["history"]), float(saved["best"]), int(saved["best_epoch"]), saved["best_state"]
        for epoch in range(start_epoch, recipe["epochs"] + 1):
            model.train(); losses = []
            for indices in _batch_order(len(train_y), task=task, model=model_name, fold=fold_id, seed=seed, epoch=epoch, size=recipe["batch_size"]):
                idx = torch.as_tensor(indices, dtype=torch.long, device=device); optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(train_x.index_select(0, idx)), train_y.index_select(0, idx), weight=weight)
                if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss in {task}/{model_name}/fold{fold_id}/seed{seed}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"]); optimizer.step(); losses.append(float(loss.detach().cpu()))
            inner = _evaluate(model, val_x, data["val_y"], data["val_subjects"], recipe["batch_size"])
            selected = epoch >= recipe["min_epoch"] and inner["subject_equal_BA"] > best + 1e-12
            if selected: best, best_epoch, best_state = float(inner["subject_equal_BA"]), epoch, copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            history.append({"epoch": epoch, "cross_entropy": float(np.mean(losses)), "inner_validation": inner, "selected": selected})
            if selected or epoch % 5 == 0 or epoch == recipe["epochs"]:
                _torch(latest_path, {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(),
                                     "best": best, "best_epoch": best_epoch, "best_state": best_state, "history": history, "invariant": invariant})
            if epoch == 1 or epoch % 5 == 0 or selected: print(f"[{task} {model_name} f{fold_id} s{seed}] e={epoch} loss={history[-1]['cross_entropy']:.4f} innerBA={inner['subject_equal_BA']:.4f}", flush=True)
        if best_state is None: raise RuntimeError("no eligible selected epoch")
        model.load_state_dict(best_state, strict=True); _torch(selected_path, {"state_dict": model.state_dict(), "invariant": invariant, "selected_epoch": best_epoch})
        outer = _evaluate(model, outer_x, data["outer_y"], data["outer_subjects"], recipe["batch_size"])
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        record = {"schema": "SEVEN_BACKBONE_SEARCH_CELL_V1", "task": task, "dataset": data["dataset"], "model": model_name,
                  "fold": fold_id, "seed": seed, "classes": data["classes"], "normalizer": data["normalizer"], "split_sha256": data["split_sha256"],
                  "protocol_freeze_sha256": freeze_hash, "recipe": recipe, "adapter": MODEL_NATIVE_RESAMPLING[model_name],
                  "initial_state_sha256": initial_hash, "invariant_sha256": invariant, "selected_epoch": best_epoch, "best_inner_validation_BA": best,
                  "outer_development": outer, "history": history, "checkpoint_path": str(selected_path), "checkpoint_sha256": _sha(selected_path),
                  "elapsed_seconds_this_invocation": time.perf_counter() - started, "peak_cuda_bytes": peak, **admission_metadata(model)}
        _json(record_path, record); return record
    finally:
        if lock.exists(): shutil.rmtree(lock)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--task", choices=TASKS, required=True); parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True); parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args(); _run_cell(args.task, args.model, args.fold, args.seed)
    print("SEARCH_CELL_COMPLETE", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
