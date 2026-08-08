from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import EEGWindowDataset
from .models import Stage0Transformer, TorchOperator


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    patience: int = 5
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: Stage0Transformer, loader: DataLoader, operator: TorchOperator, device: torch.device, num_classes: int) -> dict[str, object]:
    model.eval(); predictions, labels, probabilities, indices = [], [], [], []
    op = operator.to(device)
    for signal, target, index in loader:
        logits = model(signal.to(device, non_blocking=True), op)
        probability = logits.softmax(dim=1)
        predictions.extend(probability.argmax(dim=1).cpu().numpy())
        probabilities.extend(probability.cpu().numpy()); labels.extend(target.numpy()); indices.extend(index.numpy())
    y, pred, prob = np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)
    result: dict[str, object] = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "labels": y, "predictions": pred, "probabilities": prob, "indices": np.asarray(indices),
    }
    if num_classes == 2 and len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, prob[:, 1]))
    return result


def train_model(
    method: str, seed: int, train_data: EEGWindowDataset, validation_data: EEGWindowDataset,
    train_operators: list[TorchOperator], validation_operator: TorchOperator,
    num_classes: int, output_checkpoint: str | Path, config: TrainConfig,
    model_kwargs: dict[str, object] | None = None,
) -> tuple[Stage0Transformer, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage0Transformer(method, num_classes, **(model_kwargs or {})).to(device)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, config.batch_size * 2, shuffle=False, num_workers=config.num_workers, pin_memory=device.type == "cuda")
    counts = np.bincount(train_data.label, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1) / num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, stale, best_state, history = -np.inf, 0, None, []
    device_operators = [operator.to(device) for operator in train_operators]
    for epoch in range(config.epochs):
        model.train(); losses = []
        for batch_index, (signal, target, _) in enumerate(train_loader):
            operator = device_operators[(batch_index + epoch) % len(device_operators)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(signal.to(device, non_blocking=True), operator)
                loss = criterion(logits, target.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(model, validation_loader, validation_operator, device, num_classes)
        score = float(validation["balanced_accuracy"])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_ba": score, "learning_rate": float(scheduler.get_last_lr()[0])})
        if score > best_score + 1e-5:
            best_score, stale = score, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    destination = Path(output_checkpoint); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"method": method, "seed": seed, "state_dict": best_state, "history": history, "parameter_count": model.parameter_count}, destination)
    return model, history
