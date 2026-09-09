"""Outcome-blind source, tensor, gradient, and checkpoint admission audit."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from backbone_models import MODELS, MODEL_NATIVE_RESAMPLING, admission_metadata, build_model


EXP = Path(__file__).resolve().parents[1]
PROTOCOL = EXP / "protocol"
TASKS = {
    "OpenBMI_MI": {"dataset": "OpenBMI", "channels": 62, "samples": 1000, "classes": 2},
    "OpenBMI_ERP": {"dataset": "OpenBMI", "channels": 62, "samples": 250, "classes": 2},
    "OpenBMI_SSVEP": {"dataset": "OpenBMI", "channels": 62, "samples": 1000, "classes": 4},
    "WBCIC_MI": {"dataset": "WBCIC", "channels": 58, "samples": 1000, "classes": 2},
}


def stable_seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jsonable(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def adapted_input(model_name: str, raw: torch.Tensor) -> torch.Tensor:
    spec = MODEL_NATIVE_RESAMPLING[model_name]
    if spec is None: return raw
    ratio = (int(spec["target_hz"]), int(spec["source_hz"]))
    # This is the protocol-declared deterministic polyphase adapter, not a
    # learned preprocessing layer.  It is deliberately performed in float64
    # before conversion to the model dtype.
    values = resample_poly(raw.detach().cpu().numpy().astype(np.float64), ratio[0], ratio[1], axis=-1).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(values)).to(raw.device)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    choices = {row.split(",")[0]: row.split(",")[3] for row in (PROTOCOL / "TECH_RECIPE_SELECTION.csv").read_text(encoding="utf-8").splitlines()[1:]}
    rows: list[dict[str, Any]] = []
    for task, meta in TASKS.items():
        for model_name in MODELS:
            stable_seed(913 + len(rows))
            recipe = choices[task] if model_name == "TeCh" else None
            model = build_model(model_name, dataset=meta["dataset"], channels=meta["channels"], samples=meta["samples"], classes=meta["classes"], tech_recipe=recipe).to(device)
            raw = torch.randn((2, meta["channels"], meta["samples"]), dtype=torch.float32, device=device)
            labels = torch.tensor([0, meta["classes"] - 1], dtype=torch.long, device=device)
            value = adapted_input(model_name, raw)
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            model.train(); logits = model(value); loss = F.cross_entropy(logits, labels)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            finite_grads = bool(grads) and all(g is None or bool(torch.isfinite(g).all()) for g in grads)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                first, second = model(value), model(value)
            repeat = float((first - second).abs().max().detach().cpu())
            shape_ok = tuple(first.shape) == (2, meta["classes"])
            finite = bool(torch.isfinite(first).all()) and bool(torch.isfinite(loss))
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            record = {
                "task": task, "model": model_name, "dataset": meta["dataset"], "classes": meta["classes"],
                "raw_input_shape": list(raw.shape), "model_input_shape": list(value.shape), "logits_shape": list(first.shape),
                "loss_finite": finite, "gradient_finite": finite_grads, "deterministic_eval_max_abs_diff": repeat,
                "shape_ok": shape_ok, "peak_cuda_bytes_batch2_train_step": peak,
                "model_native_adapter": MODEL_NATIVE_RESAMPLING[model_name], "tech_recipe": recipe,
                **admission_metadata(model),
            }
            record["pass"] = bool(shape_ok and finite and finite_grads and repeat == 0.0)
            if not record["pass"]: raise RuntimeError(f"backbone admission failed: {record}")
            rows.append(record)
            del raw, value, logits, first, second, loss, optimizer, model
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    payload = {"schema": "SEVEN_BACKBONE_ADMISSION_V1", "outcome_access": "none; synthetic tensors only", "device": str(device),
               "all_pass": len(rows) == len(TASKS) * len(MODELS) and all(row["pass"] for row in rows), "records": rows}
    payload["sha256"] = hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True).encode()).hexdigest()
    atomic_json(PROTOCOL / "BACKBONE_ADMISSION.json", payload)
    print("BACKBONE_ADMISSION_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
