"""Stage-0 v2 repair and re-falsification runner.

This module is intentionally separate from the historical v1 pipeline.  It
freezes the repaired task/operator protocol before fitting, runs Gate 0 first,
and only then opens the Gate A/B comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .basis import CanonicalBasis, normalize_electrode_name, standard_1020_coordinates
from .data import EEGWindowDataset, fixed_subject_split
from .lifting import numerical_audit
from .models import Stage0Transformer, TorchOperator, make_torch_operator
from .operators_v2 import generate_eegmmidb_v2_operators
from .training import TrainConfig, evaluate, seed_everything, train_model


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class SanityCNN(nn.Module):
    """Small EEGNet-like spatial/temporal sanity baseline."""
    def __init__(self, channels: int = 64, classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=25, padding=12, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7, groups=1, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            nn.AvgPool1d(8), nn.Dropout(0.1),
        )
        self.head = nn.Linear(64, classes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(value).mean(dim=-1))


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    pred = logits.argmax(axis=1)
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "accuracy": float((labels == pred).mean()),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
        "predicted_counts": np.bincount(pred, minlength=2).tolist(),
    }
    if len(np.unique(labels)) == 2:
        out["auroc"] = float(roc_auc_score(labels, logits[:, 1]))
    return out


class Stage0V2:
    def __init__(self, repository: Path, config_path: Path):
        self.repository = repository.resolve()
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.project = Path(self.config["project_root"])
        self.output = self.repository / "results/stage0_v2"
        for name in ("plots", "checkpoints", "logs", "operators"):
            (self.output / name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, self.output / "stage0_v2_config.yaml")
        self.names: list[str] = []
        self.coordinates: np.ndarray | None = None
        self.basis: CanonicalBasis | None = None
        self.source_coefficients: np.ndarray | None = None
        self.source_b: np.ndarray | None = None
        self.views = []
        self.split: dict[str, list[str]] = {}

    def prepare(self) -> None:
        paths = sorted(self.project.glob(self.config["dataset"]["eegmmidb"]["processed_glob"]))
        if not paths:
            raise FileNotFoundError("EEGMMIDB processed caches are absent")
        with h5py.File(paths[0]) as handle:
            raw_names = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["channel_names"][:]]
        self.names = [normalize_electrode_name(v) for v in raw_names]
        self.coordinates = np.stack([standard_1020_coordinates(self.names)[name] for name in self.names])
        bcfg = self.config["canonical_basis"]
        self.basis = CanonicalBasis.fixed(bcfg["dimension"], bcfg["bandwidth"], bcfg["grid_size"], bcfg["ridge"])
        psi = self.basis.evaluate(self.coordinates)
        n = len(self.names)
        self.source_coefficients = np.eye(n) - np.ones((n, n)) / n
        self.source_b = self.source_coefficients @ psi
        self.views = generate_eegmmidb_v2_operators(self.names, self.source_b, self.coordinates, self.source_coefficients)
        paths = sorted(str(v) for v in paths)
        scfg = self.config["split"]
        self.split = fixed_subject_split(paths, scfg["seed"], scfg["train_fraction"], scfg["validation_fraction"])

    def _task_audit(self) -> None:
        runs = [int(v) for v in self.config["dataset"]["eegmmidb"]["included_runs"]]
        excluded = [int(v) for v in self.config["dataset"]["eegmmidb"]["excluded_runs"]]
        counts = {"train": {"0": 0, "1": 0}, "validation": {"0": 0, "1": 0}, "test": {"0": 0, "1": 0}}
        windows_by_subject: dict[str, int] = {}
        run_ids_by_split: dict[str, list[int]] = {k: [] for k in self.split}
        hashes: dict[str, list[str]] = {k: [] for k in self.split}
        recording_ids: dict[str, list[str]] = {k: [] for k in self.split}
        path_split = {path: split for split, paths in self.split.items() for path in paths}
        for path, split in path_split.items():
            with h5py.File(path, "r") as handle:
                run_id = np.asarray(handle["run_id"][:], dtype=int)
                labels = np.asarray(handle["label"][:], dtype=int)
                signal = handle["signal"]
                mask = np.isin(run_id, runs) & np.isin(labels, [0, 1])
                windows_by_subject[Path(path).stem] = int(mask.sum())
                for value in np.unique(run_id[mask]):
                    run_ids_by_split[split].append(int(value))
                for label in labels[mask]:
                    counts[split][str(int(label))] += 1
                recs = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["recording_id"][:]]
                recording_ids[split].extend(np.asarray(recs, dtype=object)[mask].tolist())
                for idx in np.flatnonzero(mask):
                    hashes[split].append(_hash_array(np.asarray(signal[idx], dtype=np.float32)))
        hash_sets = {k: set(v) for k, v in hashes.items()}
        cross_split_duplicates = {f"{a}_{b}": len(hash_sets[a] & hash_sets[b]) for a in hash_sets for b in hash_sets if a < b}
        self.task = {
            "dataset": "EEGMMIDB 1.0.0",
            "official_semantics_reference": "https://physionet.org/content/eegmmidb/1.0.0/",
            "task": "binary left-hand versus right-hand motor imagery",
            "included_runs": runs,
            "excluded_runs": excluded,
            "run_semantics": {"4,8,12": "T1/T2 left/right imagery", "6,10,14": "T1/T2 fists/feet imagery; excluded"},
            "label_mapping": {"0": "T1 / left-hand imagery", "1": "T2 / right-hand imagery"},
            "subjects": {k: [Path(v).stem for v in vals] for k, vals in self.split.items()},
            "class_counts": counts,
            "windows_per_subject": windows_by_subject,
            "run_ids_by_split": {k: sorted(set(v)) for k, v in run_ids_by_split.items()},
            "recording_ids_by_split": {k: sorted(set(v)) for k, v in recording_ids.items()},
            "duplicate_window_hashes": {k: len(v) - len(set(v)) for k, v in hashes.items()},
            "cross_split_duplicate_hashes": cross_split_duplicates,
            "leakage_checks": {"subject_disjoint": len(set(self.split["train"]) & set(self.split["test"])) == 0 and len(set(self.split["validation"]) & set(self.split["test"])) == 0, "window_hash_disjoint": all(v == 0 for v in cross_split_duplicates.values())},
        }
        _json(self.output / "task_definition.json", self.task)
        _json(self.output / "subject_splits.json", {"eegmmidb": {k: [Path(v).stem for v in vals] for k, vals in self.split.items()}, "isruc": {"status": "DATA_NOT_LOCALLY_VERIFIED"}})

    def _operator_audit(self) -> None:
        assert self.source_b is not None and self.source_coefficients is not None and self.basis is not None
        rng = np.random.default_rng(20270808)
        x = rng.normal(size=(self.source_b.shape[1], 11))
        y0 = self.source_b @ x
        rows, consistency = [], []
        for view in self.views:
            npz = self.output / "operators" / f"{view.operator_id}.npz"
            np.savez_compressed(npz, A_matrix=view.A, B_matrix=view.B, electrode_coefficients=view.electrode_coefficients)
            row = view.audit_row(self.source_b)
            row.update({"A_hash": row.pop("a_sha256"), "B_hash": row.pop("b_sha256"), "A_matrix": str(npz.relative_to(self.output)) + "::A_matrix", "B_matrix": str(npz.relative_to(self.output)) + "::B_matrix"})
            singular = np.linalg.svd(view.B, compute_uv=False)
            row["effective_rank"] = int(np.sum(singular > (1e-6 * singular[0] if len(singular) else 0.0)))
            row["family"] = view.operator_family
            row["legal"] = bool(row["is_legal"])
            signal_residual = float(np.linalg.norm(view.A @ y0 - view.B @ x) / (np.linalg.norm(view.B @ x) + 1e-15))
            row["signal_consistency_residual"] = signal_residual
            row["positive_terms"] = int(np.sum(view.electrode_coefficients > 1e-8))
            row["negative_terms"] = int(np.sum(view.electrode_coefficients < -1e-8))
            row["total_terms"] = int(np.sum(np.abs(view.electrode_coefficients) > 1e-8))
            row["reference_pool_size"] = int(max(row["negative_terms"], 0))
            rows.append(row)
            consistency.append({"operator_id": view.operator_id, "family": view.operator_family, "split": view.split, "residual": signal_residual, "legal": bool(signal_residual < 1e-10)})
        catalog = pd.DataFrame(rows)
        catalog.to_csv(self.output / "operator_catalog.csv", index=False)
        catalog.to_csv(self.output / "d0_operator_audit.csv", index=False)
        pd.DataFrame(consistency).to_csv(self.output / "d0_signal_operator_consistency.csv", index=False)
        _json(self.output / "matrix_numerical_tests.json", [{**numerical_audit(view.B, self.config["lifting"]["selected_alpha"]), "operator_id": view.operator_id} for view in self.views])
        _json(self.output / "operator_splits.json", {k: [v.operator_id for v in self.views if v.split == k] for k in ("train", "validation", "test")})
        protocol = {
            "schema_version": "moa-eeg-stage0-v2",
            "matched_operator": "source_car64",
            "source_observation": "Y0 = C64 @ Yraw; Yt = A_t @ Y0; B_t = A_t @ C64 @ Psi",
            "train_validation_test_operator_ids": json.loads((self.output / "operator_splits.json").read_text()),
            "gate_a_primary_excludes": ["polarity"],
            "gate_a": {"overall_mean_drop_pp": self.config["gates"]["A_mean_drop_pp"], "minimum_family_mean_drop_pp": self.config["gates"]["A_family_drop_pp"], "minimum_families": 2},
            "gate_b": {"gain_pp": self.config["gates"]["B_gain_pp"], "comparison": "B6 vs strongest of B2-B5, overall and composite-only"},
            "seeds": self.config["training"]["seeds"],
            "isruc": "DATA_NOT_LOCALLY_VERIFIED",
        }
        (self.output / "operator_protocol_v2.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
        (self.output / "PROTOCOL_FREEZE.md").write_text("""# Stage-0 v2 protocol freeze\n\nThis file is written before Gate-0 performance. The binary task is fixed to EEGMMIDB runs 4/8/12 (T1/T2 left/right imagery), with subject-level split seed 2027 and three training seeds. The matched reference is CAR64 (`source_car64`), frozen before any result is read. All B2–B6 methods share the same Stage0Transformer, temporal stem, optimizer, schedule, and seeds. B4 uses only continuous active/reference/count metadata; no family category or signed B vector. B2–B6 have no channel-row positional identity; canonical component identity is reserved for B7/B8. Gate A aggregates all held-out non-polarity operators and requires both an overall mean and at least two family means. Gate B compares B6 against the strongest B2–B5 model overall and on composite-only operators.\n\nNo protocol, operator, threshold, seed, or matched-reference choice may be changed after performance inspection.\n""", encoding="utf-8")
        (self.output / "INVALIDATION_REFERENCE.md").write_text("""# v1 invalidation reference\n\nThe preserved `../stage0_invalid_v1/` directory is not a valid scientific Gate-B conclusion. It is invalidated for task learnability, raw-vs-CAR Y/B inconsistency, single-worst-operator Gate-A triggering, test-only B4 categorical features, and unstable channel-row positional embeddings. v2 addresses each defect before reopening scientific comparisons.\n""", encoding="utf-8")
        # schema-complete outputs exist even if Gate 0 stops the workflow.
        schemas = {
            "training_runs.csv": ["dataset", "method", "seed", "status", "parameter_count", "best_epoch", "best_validation_ba"],
            "matched_unseen_all.csv": ["dataset", "method", "seed", "operator_id", "operator_family", "evaluation", "balanced_accuracy", "macro_f1", "operator_drop", "rei"],
            "matched_unseen_agg.csv": ["dataset", "method", "matched_ba", "unseen_ba", "operator_drop", "macro_f1"],
            "operator_family_results.csv": ["dataset", "method", "operator_family", "balanced_accuracy", "operator_drop"],
            "composite_operator_results.csv": ["dataset", "method", "operator_family", "balanced_accuracy", "operator_drop"],
            "feature_scale_audit.csv": ["method", "operator_id", "split", "raw_mean", "raw_std", "raw_l2", "layernorm_mean", "layernorm_std", "layernorm_l2"],
            "observability_scores.csv": ["operator_id", "O_dim", "O_eff", "O_task", "O_task_soft"],
            "observability_failure_correlation.csv": ["score", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
        }
        for filename, columns in schemas.items():
            if not (self.output / filename).exists():
                pd.DataFrame(columns=columns).to_csv(self.output / filename, index=False)
        _json(self.output / "basis_audit.json", self.basis.audit())

    def audit(self) -> None:
        self.prepare(); self._task_audit(); self._operator_audit()
        _json(self.output / "gate_summary.json", {"status": "STAGE0_V2_REPAIR_REQUIRED", "gates": {"0": "NOT_RUN", "A": "NOT_RUN", "B": "NOT_RUN"}})

    def _torch_ops(self) -> dict[str, TorchOperator]:
        assert self.basis is not None and self.coordinates is not None and self.source_coefficients is not None
        return {v.operator_id: make_torch_operator(v, self.coordinates, self.basis.centers, self.config["lifting"]["selected_alpha"], self.source_coefficients) for v in self.views}

    def _datasets(self) -> dict[str, EEGWindowDataset]:
        split_ids = json.loads((self.output / "subject_splits.json").read_text())["eegmmidb"]
        root = self.project / "data/processed/eegmmidb"
        runs = self.config["dataset"]["eegmmidb"]["included_runs"]
        return {k: EEGWindowDataset([root / f"{sid}.h5" for sid in ids], include_runs=runs) for k, ids in split_ids.items()}

    def _cnn_epoch(self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[float, dict[str, Any]]:
        train = optimizer is not None
        model.train(train); all_y, all_logits, losses = [], [], []
        for signal, target, _ in loader:
            signal, target = signal.to(device), target.to(device)
            with torch.set_grad_enabled(train):
                logits = model(signal)
                loss = nn.functional.cross_entropy(logits, target)
                if train:
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach())); all_y.append(target.cpu().numpy()); all_logits.append(logits.detach().cpu().numpy())
        return float(np.mean(losses)), _metrics(np.concatenate(all_y), np.concatenate(all_logits))

    def _run_cnn(self, data: dict[str, EEGWindowDataset]) -> dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows, histories = [], []
        for seed in self.config["training"]["seeds"]:
            seed_everything(seed); model = SanityCNN(len(self.names), 2).to(device)
            train_loader = DataLoader(data["train"], self.config["training"]["batch_size"], shuffle=True, pin_memory=device.type == "cuda")
            val_loader = DataLoader(data["validation"], self.config["training"]["batch_size"] * 2, shuffle=False, pin_memory=device.type == "cuda")
            test_loader = DataLoader(data["test"], self.config["training"]["batch_size"] * 2, shuffle=False, pin_memory=device.type == "cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            best, state, hist = -np.inf, None, []
            for epoch in range(self.config["gate0"]["epochs"]):
                loss, tr = self._cnn_epoch(model, train_loader, optimizer, device)
                _, va = self._cnn_epoch(model, val_loader, None, device); hist.append({"epoch": epoch + 1, "loss": loss, "train": tr, "validation": va})
                if va["balanced_accuracy"] > best: best, state = va["balanced_accuracy"], {k: v.cpu() for k, v in model.state_dict().items()}
            model.load_state_dict(state); _, te = self._cnn_epoch(model, test_loader, None, device)
            rows.append({"seed": seed, "train_ba": hist[-1]["train"]["balanced_accuracy"], "validation_ba": best, "test_ba": te["balanced_accuracy"], "test_macro_f1": te["macro_f1"], "test_auroc": te.get("auroc"), "confusion_matrix": te["confusion_matrix"], "predicted_counts": te["predicted_counts"]}); histories.extend(hist)
        return {"rows": rows, "history": histories, "mean_test_ba": float(np.mean([r["test_ba"] for r in rows]))}

    def _run_transformer_gate0(self, data: dict[str, EEGWindowDataset], operators: dict[str, TorchOperator]) -> dict[str, Any]:
        train_config = TrainConfig(**{key: self.config["training"][key] for key in asdict(TrainConfig()).keys()})
        model_kwargs = {"canonical_dim": self.config["canonical_basis"]["dimension"], "patch_size": self.config["model"]["patch_size"], "hidden_dim": self.config["model"]["hidden_dim"], "layers": self.config["model"]["num_layers"], "heads": self.config["model"]["num_heads"], "dropout": self.config["model"]["dropout"], "temporal_stem": True}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
        loaders = {k: DataLoader(v, train_config.batch_size * (2 if k != "train" else 1), shuffle=False, pin_memory=device.type == "cuda") for k, v in data.items()}
        for seed in self.config["training"]["seeds"]:
            checkpoint = self.output / "checkpoints" / f"gate0_transformer_seed{seed}.pt"
            model, history = train_model("B2", seed, data["train"], data["validation"], [operators["source_car64"]], operators["source_car64"], 2, checkpoint, train_config, model_kwargs)
            model.to(device); tr = evaluate(model, loaders["train"], operators["source_car64"], device, 2); va = evaluate(model, loaders["validation"], operators["source_car64"], device, 2); te = evaluate(model, loaders["test"], operators["source_car64"], device, 2)
            rows.append({"seed": seed, "train_ba": tr["balanced_accuracy"], "validation_ba": va["balanced_accuracy"], "test_ba": te["balanced_accuracy"], "test_macro_f1": te["macro_f1"], "test_auroc": te.get("auroc"), "confusion_matrix": confusion_matrix(te["labels"], te["predictions"], labels=[0,1]).tolist(), "predicted_counts": np.bincount(te["predictions"], minlength=2).tolist(), "best_epoch": int(np.argmax([h["validation_ba"] for h in history]) + 1), "loss_curve": [{"epoch": h["epoch"], "train_loss": h["train_loss"], "validation_ba": h["validation_ba"]} for h in history]})
        return {"rows": rows, "mean_test_ba": float(np.mean([r["test_ba"] for r in rows]))}

    def _tiny_overfit(self, data: EEGWindowDataset, operator: TorchOperator) -> dict[str, Any]:
        seed_everything(0); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        indices = np.arange(min(self.config["gate0"]["tiny_trials"], len(data)))
        subset = Subset(data, indices.tolist()); loader = DataLoader(subset, self.config["gate0"]["tiny_batch_size"], shuffle=True)
        model = Stage0Transformer("B2", 2, canonical_dim=self.config["canonical_basis"]["dimension"], patch_size=self.config["model"]["patch_size"], hidden_dim=self.config["model"]["hidden_dim"], layers=self.config["model"]["num_layers"], heads=self.config["model"]["num_heads"], dropout=0.0, temporal_stem=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
        best = 0.0
        for _ in range(self.config["gate0"]["tiny_epochs"]):
            model.train()
            for signal, target, _ in loader:
                optimizer.zero_grad(set_to_none=True); loss = nn.functional.cross_entropy(model(signal.to(device), operator.to(device)), target.to(device)); loss.backward(); optimizer.step()
            _, metrics = self._transformer_predictions(model, loader, operator, device)
            best = max(best, metrics["accuracy"])
            if best >= self.config["gate0"]["tiny_target"]: break
        return {"trials": len(subset), "best_train_accuracy": best, "target": self.config["gate0"]["tiny_target"], "pass": bool(best >= self.config["gate0"]["tiny_target"])}

    def _transformer_predictions(self, model: nn.Module, loader: DataLoader, operator: TorchOperator, device: torch.device):
        model.eval(); ys, ls = [], []
        with torch.no_grad():
            for signal, target, _ in loader:
                ys.append(target.numpy()); ls.append(model(signal.to(device), operator.to(device)).cpu().numpy())
        y, logits = np.concatenate(ys), np.concatenate(ls); return y, _metrics(y, logits)

    def _feature_audit(self, operators: dict[str, TorchOperator]) -> None:
        rows = []
        for method in ("B2", "B3", "B4", "B5", "B6"):
            model = Stage0Transformer(method, 2, canonical_dim=self.config["canonical_basis"]["dimension"], hidden_dim=32, layers=1, heads=4, patch_size=160)
            for view in self.views:
                if view.operator_id == "source_car64" or view.split in {"train", "validation", "test"}:
                    op = operators[view.operator_id]
                    dummy = torch.zeros(1, 64, 640)
                    _, feat = model._representation(dummy, op)
                    raw = feat.detach().numpy(); norm = model.operator_feature_norm(feat).detach().numpy()
                    rows.append({"method": method, "operator_id": view.operator_id, "split": view.split, "raw_mean": float(raw.mean()), "raw_std": float(raw.std()), "raw_l2": float(np.linalg.norm(raw)), "layernorm_mean": float(norm.mean()), "layernorm_std": float(norm.std()), "layernorm_l2": float(np.linalg.norm(norm))})
        pd.DataFrame(rows).to_csv(self.output / "feature_scale_audit.csv", index=False)

    def _run_methods(self, data: dict[str, EEGWindowDataset], operators: dict[str, TorchOperator]) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_config = TrainConfig(**{key: self.config["training"][key] for key in asdict(TrainConfig()).keys()})
        model_kwargs = {"canonical_dim": self.config["canonical_basis"]["dimension"], "patch_size": self.config["model"]["patch_size"], "hidden_dim": self.config["model"]["hidden_dim"], "layers": self.config["model"]["num_layers"], "heads": self.config["model"]["num_heads"], "dropout": self.config["model"]["dropout"], "temporal_stem": True}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); loader = DataLoader(data["test"], train_config.batch_size * 2, shuffle=False, pin_memory=device.type == "cuda")
        primary_ids = [v.operator_id for v in self.views if v.split == "test" and v.operator_family != "polarity"]
        train_ids = [v.operator_id for v in self.views if v.split == "train"]
        validation_id = next(v.operator_id for v in self.views if v.split == "validation")
        all_rows, training_rows = [], []
        for method in ("B2", "B3", "B4", "B5", "B6"):
            for seed in self.config["training"]["seeds"]:
                checkpoint = self.output / "checkpoints" / f"eegmmidb_{method}_seed{seed}.pt"
                model, history = train_model(method, seed, data["train"], data["validation"], [operators[i] for i in train_ids], operators[validation_id], 2, checkpoint, train_config, model_kwargs)
                model.to(device); matched = evaluate(model, loader, operators["source_car64"], device, 2); matched_ba = float(matched["balanced_accuracy"])
                training_rows.append({"dataset": "eegmmidb", "method": method, "seed": seed, "status": "complete", "parameter_count": model.parameter_count, "best_epoch": int(np.argmax([h["validation_ba"] for h in history]) + 1), "best_validation_ba": max(h["validation_ba"] for h in history)})
                for v in [vv for vv in self.views if vv.split == "test"]:
                    result = evaluate(model, loader, operators[v.operator_id], device, 2); ba = float(result["balanced_accuracy"]); drop = matched_ba - ba
                    all_rows.append({"dataset": "eegmmidb", "method": method, "seed": seed, "operator_id": v.operator_id, "operator_family": v.operator_family, "evaluation": "unseen", "balanced_accuracy": ba, "macro_f1": result["macro_f1"], "operator_drop": drop, "rei": drop / max(1 - matched_ba, 1e-12)})
        all_frame = pd.DataFrame(all_rows); pd.DataFrame(training_rows).to_csv(self.output / "training_runs.csv", index=False); all_frame.to_csv(self.output / "matched_unseen_all.csv", index=False)
        agg = all_frame.groupby(["method", "operator_family"], as_index=False).agg(balanced_accuracy=("balanced_accuracy", "mean"), operator_drop=("operator_drop", "mean")); agg.insert(0, "dataset", "eegmmidb"); agg.to_csv(self.output / "operator_family_results.csv", index=False)
        composite = agg[agg.operator_family.isin(["subset_car", "local_average", "laplacian", "weighted_reference", "bipolar"])].copy(); composite.to_csv(self.output / "composite_operator_results.csv", index=False)
        overall = all_frame.groupby("method", as_index=False).agg(unseen_ba=("balanced_accuracy", "mean"), operator_drop=("operator_drop", "mean"), macro_f1=("macro_f1", "mean")); overall.insert(0, "dataset", "eegmmidb"); overall["matched_ba"] = 1.0 - overall["operator_drop"]
        overall.to_csv(self.output / "matched_unseen_agg.csv", index=False)
        return all_frame, agg

    def _plots(self, gate0: dict[str, Any], all_frame: pd.DataFrame | None = None) -> None:
        fig, ax = plt.subplots(figsize=(6, 4)); labels = ["CNN", "Transformer"]
        values = [gate0["cnn"]["mean_test_ba"], gate0["transformer"]["mean_test_ba"]]
        ax.bar(labels, values, color=["#4472c4", "#ed7d31"]); ax.axhline(.5, color="black", ls="--", lw=1); ax.set_ylabel("test balanced accuracy"); ax.set_title("Figure 0: Gate-0 sanity baseline"); fig.tight_layout(); fig.savefig(self.output / "plots/figure0_gate0_sanity.png", dpi=160); plt.close(fig)
        if all_frame is not None and not all_frame.empty:
            fig, ax = plt.subplots(figsize=(8, 4)); all_frame.groupby("method")["operator_drop"].mean().plot.bar(ax=ax); ax.axhline(0, color="black", lw=1); ax.set_ylabel("mean matched−unseen BA"); ax.set_title("Figure 1: held-out non-polarity degradation"); fig.tight_layout(); fig.savefig(self.output / "plots/figure1_matched_unseen.png", dpi=160); plt.close(fig)

    def run(self) -> str:
        self.audit(); data = self._datasets(); operators = self._torch_ops(); self._feature_audit(operators)
        tiny = self._tiny_overfit(data["train"], operators["source_car64"])
        cnn = self._run_cnn(data)
        transformer = self._run_transformer_gate0(data, operators)
        gate0 = {"tiny_overfit": tiny, "cnn": cnn, "transformer": transformer, "chance_ba": 0.5, "transformer_above_chance": transformer["mean_test_ba"] > 0.55, "cnn_above_chance": cnn["mean_test_ba"] > 0.55, "cnn_transformer_headroom": cnn["mean_test_ba"] - transformer["mean_test_ba"] < 0.20}
        _json(self.output / "gate0_task_qualification.md.json", gate0)
        lines = ["# Gate 0 task qualification", "", json.dumps(gate0, indent=2)]
        (self.output / "gate0_task_qualification.md").write_text("\n".join(lines), encoding="utf-8")
        self._plots(gate0)
        gate0_pass = bool(tiny["pass"] and cnn["mean_test_ba"] > 0.55 and transformer["mean_test_ba"] > 0.55 and gate0["cnn_transformer_headroom"])
        if not gate0_pass:
            status = "SCIENTIFIC_STOP_GATE_0"
            gate0["stop_reason"] = "STOP_STAGE0_TASK_PIPELINE_BUG"
            summary = {"status": status, "stop_reason": gate0["stop_reason"], "gates": {"0": "FAIL", "A": "NOT_RUN", "B": "NOT_RUN"}, "gate0": gate0}
            _json(self.output / "gate_summary.json", summary); self._write_report(status, gate0, None, None); return status
        all_frame, family = self._run_methods(data, operators); self._plots(gate0, all_frame)
        primary = all_frame[all_frame.operator_family != "polarity"]
        overall = primary.groupby("method")["operator_drop"].mean()
        family_drop = primary.groupby(["method", "operator_family"])["operator_drop"].mean()
        family_threshold = self.config["gates"]["A_family_drop_pp"] / 100
        gate_a_rows = {method: {"mean_drop": float(overall[method]), "families_meeting": int(sum(value >= family_threshold for (m, _), value in family_drop.items() if m == method))} for method in overall.index}
        gate_a_pass = bool(all(value["mean_drop"] >= self.config["gates"]["A_mean_drop_pp"] / 100 and value["families_meeting"] >= 2 for value in gate_a_rows.values()))
        if not gate_a_pass:
            status = "SCIENTIFIC_STOP_GATE_A"; _json(self.output / "gate_summary.json", {"status": status, "gates": {"0": "PASS", "A": "FAIL", "B": "NOT_RUN"}, "gate0": gate0, "gate_a": gate_a_rows}); self._write_report(status, gate0, gate_a_rows, None); return status
        best = overall.drop("B6").idxmin() if "B6" in overall else overall.idxmin(); b6 = float(overall.get("B6", np.nan)); best_value = float(overall[best]); composite_f = family[family.operator_family.isin(["subset_car", "local_average", "laplacian", "weighted_reference", "bipolar"])].groupby("method")["operator_drop"].mean(); b_pass = bool(b6 <= best_value - self.config["gates"]["B_gain_pp"] / 100 and b6 <= float(composite_f.drop("B6").min()) - self.config["gates"]["B_gain_pp"] / 100)
        status = "PROMISING_PENDING_EXPLICIT_REFERENCE_REPLICATION" if b_pass else "SCIENTIFIC_STOP_GATE_B"
        _json(self.output / "gate_summary.json", {"status": status, "gates": {"0": "PASS", "A": "PASS", "B": "PASS" if b_pass else "FAIL"}, "gate0": gate0, "gate_a": gate_a_rows, "gate_b": {"B6": b6, "strongest_B2_B5": best, "strongest_value": best_value, "composite": composite_f.to_dict()}})
        self._write_report(status, gate0, gate_a_rows, {"B6": b6, "strongest_B2_B5": best, "strongest_value": best_value, "composite": composite_f.to_dict()}); return status

    def _write_report(self, status: str, gate0: dict[str, Any], gate_a: Any, gate_b: Any) -> None:
        text = ["# Stage-0 v2 repair and re-falsification report", "", "The v1 artifacts are preserved under `../stage0_invalid_v1/` and explicitly invalidated.", "", f"Gate-0: `{json.dumps(gate0, ensure_ascii=False)}`", f"Gate-A: `{json.dumps(gate_a, ensure_ascii=False)}`", f"Gate-B: `{json.dumps(gate_b, ensure_ascii=False)}`", "", "ISRUC status: `DATA_NOT_LOCALLY_VERIFIED`; no strong Stage-1 GO is claimed.", "", f"TERMINAL_STATE: {status}"]
        (self.output / "stage0_v2_report.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    runner = Stage0V2(args.repository, args.config); status = runner.run() if args.run else (runner.audit() or "AUDIT_COMPLETE"); print(status)


if __name__ == "__main__":
    main()
