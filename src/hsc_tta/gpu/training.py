from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from hsc_tta.gpu.embeddings import load_embedding
from hsc_tta.models import TaskHead


GRID = ((3e-4, 1e-4), (3e-4, 1e-3), (1e-3, 1e-4), (1e-3, 1e-3))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _embedding_file(root: Path, dataset: str, subject: str) -> Path:
    return root / "outputs" / "full_experiment" / "embeddings" / dataset / f"{subject.split(':',1)[1]}.h5"


def _load_subjects(root: Path, dataset: str, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, labels, subject_index = [], [], []
    for index, subject in enumerate(subjects):
        data = load_embedding(_embedding_file(root, dataset, subject))
        x, y = np.asarray(data["embedding"], np.float32), np.asarray(data["label"], np.int64)
        embeddings.append(x)
        labels.append(y)
        subject_index.append(np.full(len(y), index, dtype=np.int16))
    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(subject_index)


def _subject_metrics(y: np.ndarray, prediction: np.ndarray, subject_index: np.ndarray,
                     subjects: list[str], n_classes: int) -> pd.DataFrame:
    rows = []
    labels = np.arange(n_classes)
    for index, subject in enumerate(subjects):
        mask = subject_index == index
        rows.append({"subject_id": subject,
                     "macro_f1": f1_score(y[mask], prediction[mask], labels=labels, average="macro", zero_division=0),
                     "balanced_accuracy": balanced_accuracy_score(y[mask], prediction[mask]),
                     "n_windows": int(mask.sum())})
    return pd.DataFrame(rows)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TaskHeadTrainer:
    def __init__(self, root: str | Path, device: str = "cuda", batch_size: int = 1024):
        self.root, self.device, self.batch_size = Path(root), torch.device(device), int(batch_size)

    def train(self, dataset: str, seed: int, *, resume: bool = True) -> dict[str, object]:
        output = self.root / "outputs" / "full_experiment" / "task_heads" / dataset / f"seed_{seed}"
        best_file = output / "task_head_best.pt"
        if resume and best_file.exists():
            payload = torch.load(best_file, map_location="cpu", weights_only=False)
            return {"dataset": dataset, "seed": seed, "status": "resumed", "hash": payload["state_hash"]}
        split = json.loads((self.root / "data" / "splits_internal" / dataset / f"seed_{seed}.json").read_text())
        fit_subjects, val_subjects = split["task_head_fit"], split["task_head_val"]
        x_fit, y_fit, _ = _load_subjects(self.root, dataset, fit_subjects)
        x_val, y_val, val_subject_index = _load_subjects(self.root, dataset, val_subjects)
        n_classes = 5 if dataset == "hmc" else 4
        counts = np.bincount(y_fit, minlength=n_classes).astype(float)
        class_weights = counts.sum() / np.maximum(counts * n_classes, 1.0)
        x_fit_t = torch.from_numpy(x_fit)
        y_fit_t = torch.from_numpy(y_fit)
        x_val_t = torch.from_numpy(x_val).to(self.device)
        training_rows: list[dict[str, object]] = []
        candidates: list[dict[str, object]] = []
        last_state: dict[str, torch.Tensor] | None = None
        for candidate_index, (learning_rate, weight_decay) in enumerate(GRID):
            _seed_everything(seed * 100 + candidate_index)
            model = TaskHead(200, n_classes, 256, 0.2).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device))
            generator = torch.Generator().manual_seed(seed * 100 + candidate_index)
            best_metric, best_balanced, stale = -1.0, -1.0, 0
            best_state = None
            for epoch in range(1, 31):
                model.train()
                order = torch.randperm(len(y_fit_t), generator=generator)
                losses = []
                for start in range(0, len(order), self.batch_size):
                    indices = order[start : start + self.batch_size]
                    xb, yb = x_fit_t[indices].to(self.device), y_fit_t[indices].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite task-head loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.inference_mode():
                    pred = model(x_val_t).argmax(1).cpu().numpy()
                subject_frame = _subject_metrics(y_val, pred, val_subject_index, val_subjects, n_classes)
                macro = float(subject_frame["macro_f1"].mean())
                balanced = float(subject_frame["balanced_accuracy"].mean())
                training_rows.append({"candidate": candidate_index, "epoch": epoch,
                                      "learning_rate": learning_rate, "weight_decay": weight_decay,
                                      "train_loss": float(np.mean(losses)), "subject_macro_f1": macro,
                                      "subject_balanced_accuracy": balanced})
                if (macro, balanced) > (best_metric + 1e-12, best_balanced + 1e-12):
                    best_metric, best_balanced, stale = macro, balanced, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                    if stale >= 5:
                        break
            assert best_state is not None
            last_state = copy.deepcopy(model.state_dict())
            candidates.append({"learning_rate": learning_rate, "weight_decay": weight_decay,
                               "macro_f1": best_metric, "balanced_accuracy": best_balanced,
                               "state_dict": best_state})
        chosen = sorted(candidates, key=lambda row: (-float(row["macro_f1"]), -float(row["balanced_accuracy"]),
                                                     float(row["learning_rate"]), -float(row["weight_decay"])))[0]
        state = chosen.pop("state_dict")
        assert isinstance(state, dict) and last_state is not None
        model = TaskHead(200, n_classes, 256, 0.2).to(self.device)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            pred = model(x_val_t).argmax(1).cpu().numpy()
        validation = _subject_metrics(y_val, pred, val_subject_index, val_subjects, n_classes)
        output.mkdir(parents=True, exist_ok=True)
        state_hash = _state_hash(state)
        torch.save({"state_dict": state, "input_dim": 200, "n_classes": n_classes, "hidden_dim": 256,
                    "dropout": 0.2, "learning_rate": chosen["learning_rate"],
                    "weight_decay": chosen["weight_decay"], "state_hash": state_hash,
                    "fit_subjects": fit_subjects, "val_subjects": val_subjects}, best_file)
        torch.save({"state_dict": last_state, "n_classes": n_classes}, output / "task_head_last.pt")
        pd.DataFrame(training_rows).to_csv(output / "training_log.csv", index=False)
        validation.to_parquet(output / "validation_subject_metrics.parquet", index=False)
        (output / "task_head_config.yaml").write_text(
            f"input_dim: 200\nhidden_dim: 256\ndropout: 0.2\nlearning_rate: {chosen['learning_rate']}\nweight_decay: {chosen['weight_decay']}\n",
            encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps({**chosen, "candidates": [
            {k: v for k, v in row.items() if k != "state_dict"} for row in candidates]}, indent=2), encoding="utf-8")
        (output / "checkpoint_hash.json").write_text(json.dumps({"sha256": state_hash}, indent=2), encoding="utf-8")
        (output / "class_weights.json").write_text(json.dumps({"counts": counts.tolist(),
                                                                 "weights": class_weights.tolist()}, indent=2), encoding="utf-8")
        del x_fit, y_fit, x_val, y_val, x_fit_t, y_fit_t, x_val_t, model
        torch.cuda.empty_cache()
        return {"dataset": dataset, "seed": seed, "status": "complete", "hash": state_hash, **chosen}


def load_task_head(path: str | Path, device: str = "cuda") -> tuple[TaskHead, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TaskHead(int(payload["input_dim"]), int(payload["n_classes"]), int(payload["hidden_dim"]),
                     float(payload["dropout"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
