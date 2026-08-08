from __future__ import annotations

from dataclasses import asdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from .basis import CanonicalBasis, normalize_electrode_name, standard_1020_coordinates
from .data import EEGWindowDataset, fixed_subject_split
from .lifting import lifting_operators, numerical_audit
from .models import Stage0Transformer, TorchOperator, make_torch_operator
from .operators import OperatorView, generate_eegmmidb_operators
from .training import TrainConfig, evaluate, train_model


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_serialize), encoding="utf-8")


def _serialize(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


class Stage0Pipeline:
    def __init__(self, repository: Path, config_path: Path):
        self.repository = repository.resolve()
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.project = Path(self.config["project_root"])
        self.output = self.repository / "results/stage0"
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("plots", "checkpoints", "logs", "operators"):
            (self.output / name).mkdir(exist_ok=True)
        shutil.copy2(config_path, self.output / "stage0_config.yaml")
        self.basis: CanonicalBasis | None = None
        self.coordinates: dict[str, np.ndarray] = {}
        self.names: list[str] = []
        self.source_b: np.ndarray | None = None
        self.operators: list[OperatorView] = []

    def prepare(self) -> None:
        paths = sorted(self.project.glob(self.config["dataset"]["eegmmidb"]["processed_glob"]))
        if not paths: raise FileNotFoundError("EEGMMIDB processed caches are absent")
        with h5py.File(paths[0]) as handle:
            raw_names = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["channel_names"][:]]
        self.names = [normalize_electrode_name(name) for name in raw_names]
        if len(set(self.names)) != len(self.names): raise ValueError("channel normalization produced duplicate electrodes")
        self.coordinates = standard_1020_coordinates(self.names)
        basis_config = self.config["canonical_basis"]
        self.basis = CanonicalBasis.fixed(basis_config["dimension"], basis_config["bandwidth"], basis_config["grid_size"], basis_config["ridge"])
        electrode_evaluation = self.basis.evaluate(np.stack([self.coordinates[name] for name in self.names]))
        car = np.eye(len(self.names)) - np.ones((len(self.names), len(self.names))) / len(self.names)
        self.source_b = car @ electrode_evaluation
        self.operators = generate_eegmmidb_operators(self.names, self.source_b)

    def audit(self) -> None:
        self.prepare()
        assert self.basis is not None and self.source_b is not None
        _json(self.output / "basis_audit.json", self.basis.audit())
        reference_audit = {
            "dataset": "EEGMMIDB", "official_dataset_version": "1.0.0",
            "edf_channel_labels_are_electrode_names_only": True,
            "edf_reference_field_present": False,
            "exact_acquisition_reference": "UNRESOLVED",
            "prohibited_assumptions": ["M1", "M2", "average mastoid", "absolute scalp potential"],
            "working_transformation": "CAR64",
            "working_transformation_condition": "all 64 amplifier channels share a common acquisition reference",
            "evidence_status": "BORDERLINE: same-system literature supports a common right-ear reference, but the PhysioNet EDF metadata does not encode it",
            "scientific_restriction": "EEGMMIDB results are controlled supporting evidence, not the clean explicit-reference replication",
            "isruc_status": "official public download attempted; clean replication remains required",
        }
        _json(self.output / "d0_reference_audit.json", reference_audit)
        catalog_rows, numerical_rows = [], []
        for operator in self.operators:
            npz = self.output / "operators" / f"{operator.operator_id}.npz"
            np.savez_compressed(npz, A_matrix=operator.A, B_matrix=operator.B, electrode_coefficients=operator.electrode_coefficients)
            row = operator.audit_row(self.source_b)
            row.update({"A_matrix": str(npz.relative_to(self.output)) + "::A_matrix", "B_matrix": str(npz.relative_to(self.output)) + "::B_matrix"})
            catalog_rows.append(row)
            matrix = numerical_audit(operator.B, self.config["lifting"]["selected_alpha"])
            matrix["operator_id"] = operator.operator_id; numerical_rows.append(matrix)
        catalog = pd.DataFrame(catalog_rows)
        catalog.to_csv(self.output / "operator_catalog.csv", index=False)
        catalog.to_csv(self.output / "d0_operator_audit.csv", index=False)
        _json(self.output / "matrix_numerical_tests.json", numerical_rows)
        if not bool(catalog.is_legal.all()): raise RuntimeError("D0 operator audit failed")
        paths = sorted(self.project.glob(self.config["dataset"]["eegmmidb"]["processed_glob"]))
        split = fixed_subject_split(paths, self.config["split"]["seed"], self.config["split"]["train_fraction"], self.config["split"]["validation_fraction"])
        public_split = {key: [Path(path).stem for path in values] for key, values in split.items()}
        _json(self.output / "subject_splits.json", {"eegmmidb": public_split, "isruc": {"status": "data_unavailable"}})
        _json(self.output / "operator_splits.json", {key: [operator.operator_id for operator in self.operators if operator.split == key] for key in ("train", "validation", "test")})
        # Empty but schema-correct files exist before training and survive fail-fast.
        for filename, columns in {
            "training_runs.csv": ["dataset", "method", "seed", "status", "parameter_count", "best_epoch", "best_validation_ba"],
            "matched_unseen_all.csv": ["dataset", "method", "seed", "operator_id", "operator_family", "evaluation", "balanced_accuracy", "macro_f1", "operator_drop", "rei"],
            "matched_unseen_agg.csv": ["dataset", "method", "matched_ba", "unseen_ba", "operator_drop", "macro_f1"],
            "operator_family_results.csv": ["dataset", "method", "operator_family", "balanced_accuracy", "operator_drop"],
            "observability_scores.csv": ["operator_id", "O_dim", "O_eff", "O_task", "O_task_soft"],
            "observability_failure_correlation.csv": ["score", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
        }.items():
            pd.DataFrame(columns=columns).to_csv(self.output / filename, index=False)
        _json(self.output / "gate_summary.json", {"status": "AUDIT_COMPLETE", "gates": {name: "NOT_RUN" for name in "ABCD"}})
        self._write_report("AUDIT_COMPLETE")

    def _torch_operators(self) -> dict[str, TorchOperator]:
        assert self.basis is not None
        coordinates = np.stack([self.coordinates[name] for name in self.names])
        alpha = self.config["lifting"]["selected_alpha"]
        return {view.operator_id: make_torch_operator(view, coordinates, self.basis.centers, alpha) for view in self.operators}

    def _load_data(self) -> dict[str, EEGWindowDataset]:
        split_ids = json.loads((self.output / "subject_splits.json").read_text())["eegmmidb"]
        root = self.project / "data/processed/eegmmidb"
        return {key: EEGWindowDataset([root / f"{identifier}.h5" for identifier in values]) for key, values in split_ids.items() if key in {"train", "validation", "test"}}

    def _run_method(self, method: str, seeds: list[int], data: dict[str, EEGWindowDataset], operators: dict[str, TorchOperator]) -> None:
        train_config = TrainConfig(**{key: self.config["training"][key] for key in asdict(TrainConfig()).keys()})
        model_kwargs = {
            "canonical_dim": self.config["canonical_basis"]["dimension"], "patch_size": self.config["model"]["patch_size"],
            "hidden_dim": self.config["model"]["hidden_dim"], "layers": self.config["model"]["num_layers"],
            "heads": self.config["model"]["num_heads"], "dropout": self.config["model"]["dropout"],
        }
        training_rows = pd.read_csv(self.output / "training_runs.csv").to_dict("records")
        result_rows = pd.read_csv(self.output / "matched_unseen_all.csv").to_dict("records")
        training_ids = json.loads((self.output / "operator_splits.json").read_text())["train"]
        validation_id = json.loads((self.output / "operator_splits.json").read_text())["validation"][0]
        test_ids = json.loads((self.output / "operator_splits.json").read_text())["test"]
        # Frozen before any performance is read: dense32_b is the stable dense
        # training view (condition number ~138 versus ~1.9e4 for dense32_a).
        matched_id = "dense32_b"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_loader = DataLoader(data["test"], train_config.batch_size * 2, shuffle=False, pin_memory=device.type == "cuda")
        for seed in seeds:
            if any(row.get("method") == method and int(row.get("seed", -1)) == seed and row.get("status") == "complete" for row in training_rows):
                continue
            checkpoint = self.output / "checkpoints" / f"eegmmidb_{method}_seed{seed}.pt"
            model, history = train_model(method, seed, data["train"], data["validation"], [operators[key] for key in training_ids], operators[validation_id], 4, checkpoint, train_config, model_kwargs)
            model.to(device)
            matched = evaluate(model, test_loader, operators[matched_id], device, 4)
            matched_ba = float(matched["balanced_accuracy"])
            training_rows.append({"dataset": "eegmmidb", "method": method, "seed": seed, "status": "complete", "parameter_count": model.parameter_count, "best_epoch": int(np.argmax([row["validation_ba"] for row in history]) + 1), "best_validation_ba": max(row["validation_ba"] for row in history)})
            result_rows.append({"dataset": "eegmmidb", "method": method, "seed": seed, "operator_id": matched_id, "operator_family": "dense_subset", "evaluation": "matched", "balanced_accuracy": matched_ba, "macro_f1": matched["macro_f1"], "operator_drop": 0.0, "rei": 0.0})
            for identifier in test_ids:
                result = evaluate(model, test_loader, operators[identifier], device, 4)
                unseen = float(result["balanced_accuracy"]); drop = matched_ba - unseen
                rei = drop / max(1.0 - matched_ba, 1e-12)
                family = next(view.operator_family for view in self.operators if view.operator_id == identifier)
                result_rows.append({"dataset": "eegmmidb", "method": method, "seed": seed, "operator_id": identifier, "operator_family": family, "evaluation": "unseen", "balanced_accuracy": unseen, "macro_f1": result["macro_f1"], "operator_drop": drop, "rei": rei})
            pd.DataFrame(training_rows).to_csv(self.output / "training_runs.csv", index=False)
            pd.DataFrame(result_rows).to_csv(self.output / "matched_unseen_all.csv", index=False)

    def _aggregate(self) -> pd.DataFrame:
        rows = pd.read_csv(self.output / "matched_unseen_all.csv")
        output = []
        if rows.empty: return pd.DataFrame()
        for method, group in rows.groupby("method"):
            matched = group[group.evaluation == "matched"]
            unseen = group[group.evaluation == "unseen"]
            output.append({"dataset": "eegmmidb", "method": method, "matched_ba": matched.balanced_accuracy.mean(), "unseen_ba": unseen.balanced_accuracy.mean(), "operator_drop": matched.balanced_accuracy.mean() - unseen.balanced_accuracy.mean(), "macro_f1": unseen.macro_f1.mean()})
        aggregate = pd.DataFrame(output).sort_values("method")
        aggregate.to_csv(self.output / "matched_unseen_agg.csv", index=False)
        family = rows[rows.evaluation == "unseen"].groupby(["dataset", "method", "operator_family"], as_index=False).agg(balanced_accuracy=("balanced_accuracy", "mean"), operator_drop=("operator_drop", "mean"))
        family.to_csv(self.output / "operator_family_results.csv", index=False)
        return aggregate

    def _gate_a(self, aggregate: pd.DataFrame) -> dict[str, Any]:
        candidates = aggregate[aggregate.method.isin(["B4", "B5"])].sort_values("unseen_ba", ascending=False)
        strongest = candidates.iloc[0]
        rows = pd.read_csv(self.output / "matched_unseen_all.csv")
        unseen = rows[(rows.method == strongest.method) & (rows.evaluation == "unseen")]
        per_operator = unseen.groupby(["operator_id", "operator_family"], as_index=False).agg(
            operator_drop=("operator_drop", "mean"), rei=("rei", "mean"), unseen_ba=("balanced_accuracy", "mean"),
        )
        trigger = per_operator.sort_values(["operator_drop", "rei"], ascending=False).iloc[0]
        passed = bool(trigger.operator_drop >= self.config["gate"]["A_drop_pp"] / 100 or trigger.rei >= self.config["gate"]["A_rei"])
        return {
            "status": "PASS" if passed else "FAIL", "strongest_non_moa": strongest.method,
            "matched_ba": strongest.matched_ba, "mean_unseen_ba": strongest.unseen_ba,
            "mean_operator_drop": strongest.operator_drop,
            "trigger_operator": trigger.operator_id, "trigger_family": trigger.operator_family,
            "trigger_unseen_ba": trigger.unseen_ba, "trigger_operator_drop": trigger.operator_drop,
            "trigger_rei": trigger.rei,
            "scope_warning": "Gate is driven by one operator; inspect all families before claiming broad operator-shift headroom",
        }

    def run(self) -> None:
        if not (self.output / "d0_operator_audit.csv").exists(): self.audit()
        else: self.prepare()
        data, operators = self._load_data(), self._torch_operators()
        seeds = list(self.config["training"]["seeds"])
        existing_runs = pd.read_csv(self.output / "training_runs.csv")
        completed = {
            method for method, group in existing_runs[existing_runs.status == "complete"].groupby("method")
            if set(group.seed.astype(int)) >= set(seeds)
        } if not existing_runs.empty else set()
        for method in ("B4", "B5"):
            if method not in completed: self._run_method(method, seeds, data, operators)
        aggregate = self._aggregate(); gate_a = self._gate_a(aggregate)
        gates: dict[str, Any] = {"A": gate_a, "B": {"status": "NOT_RUN"}, "C": {"status": "NOT_RUN"}, "D": {"status": "NOT_RUN"}}
        if gate_a["status"] != "PASS":
            _json(self.output / "gate_summary.json", {"status": "SCIENTIFIC_STOP_GATE_A", "gates": gates})
            self._plots(); self._write_report("SCIENTIFIC_STOP_GATE_A"); return
        for method in ("B2", "B3", "B6"):
            if method not in completed: self._run_method(method, seeds, data, operators)
        aggregate = self._aggregate()
        baseline = aggregate[aggregate.method.isin(["B2", "B3", "B4", "B5"])].sort_values("unseen_ba", ascending=False).iloc[0]
        signed = aggregate[aggregate.method == "B6"].iloc[0]
        gain = float(signed.unseen_ba - baseline.unseen_ba)
        seed_rows = pd.read_csv(self.output / "matched_unseen_all.csv")
        per_seed_signed = seed_rows[(seed_rows.method == "B6") & (seed_rows.evaluation == "unseen")].groupby("seed").balanced_accuracy.mean()
        per_seed_base = seed_rows[(seed_rows.method == baseline.method) & (seed_rows.evaluation == "unseen")].groupby("seed").balanced_accuracy.mean()
        positive = int((per_seed_signed - per_seed_base > 0).sum())
        b_status = "PASS" if gain >= self.config["gate"]["B_gain_pp"] / 100 and positive >= 2 else ("BORDERLINE" if gain > 0 else "FAIL")
        gates["B"] = {"status": b_status, "strongest_baseline": baseline.method, "gain": gain, "positive_seeds": positive, "seed_count": len(seeds)}
        if b_status == "FAIL":
            _json(self.output / "gate_summary.json", {"status": "SCIENTIFIC_STOP_GATE_B", "gates": gates})
            self._plots(); self._write_report("SCIENTIFIC_STOP_GATE_B"); return
        for method in ("B7", "B8"):
            if method not in completed: self._run_method(method, seeds, data, operators)
        aggregate = self._aggregate()
        gates.update(self._observability(data, operators))
        strongest = aggregate[aggregate.method.isin(["B2", "B3", "B4", "B5"])].sort_values("matched_ba", ascending=False).iloc[0]
        b8 = aggregate[aggregate.method == "B8"].iloc[0]
        difference = float(b8.matched_ba - strongest.matched_ba)
        gates["D"] = {"status": "PASS" if difference >= self.config["gate"]["D_matched_tolerance_pp"] / 100 else "FAIL", "difference": difference, "strongest_baseline": strongest.method}
        final = "GO_TO_STAGE_1" if all(gates[key]["status"] == "PASS" for key in "ABCD") else "DO_NOT_PROCEED_YET"
        _json(self.output / "gate_summary.json", {"status": final, "gates": gates})
        self._plots(); self._write_report(final)

    def _observability(self, data: dict[str, EEGWindowDataset], operators: dict[str, TorchOperator]) -> dict[str, Any]:
        # The task subspace is estimated only from training subjects/operators.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.output / "checkpoints/eegmmidb_B8_seed0.pt", map_location="cpu", weights_only=False)
        kwargs = {"canonical_dim": 32, "patch_size": self.config["model"]["patch_size"], "hidden_dim": self.config["model"]["hidden_dim"], "layers": self.config["model"]["num_layers"], "heads": self.config["model"]["num_heads"], "dropout": self.config["model"]["dropout"]}
        model = Stage0Transformer("B8", 4, **kwargs).to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval()
        loader = DataLoader(data["train"], 16, shuffle=False)
        gradients = []
        train_ids = json.loads((self.output / "operator_splits.json").read_text())["train"]
        for batch_index, (signal, target, _) in enumerate(loader):
            if batch_index >= 20: break
            op = operators[train_ids[batch_index % len(train_ids)]].to(device)
            source = signal.to(device); observed = torch.einsum("oc,bct->bot", op.A, source) * 1e6
            lifted = torch.einsum("kc,bct->bkt", op.L, observed).detach().requires_grad_(True)
            logits = model.forward_representation(lifted, op.R)
            loss = torch.nn.functional.cross_entropy(logits, target.to(device))
            gradient = torch.autograd.grad(loss, lifted)[0].mean(dim=-1)
            gradients.append(gradient.detach().cpu().numpy())
        values = np.concatenate(gradients)
        covariance = values.T @ values / max(len(values), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        u = eigenvectors[:, np.argsort(eigenvalues)[-8:]]
        np.save(self.output / "task_subspace_U.npy", u)
        scores = []
        for view in self.operators:
            matrices = lifting_operators(view.B, self.config["lifting"]["selected_alpha"])
            scores.append({"operator_id": view.operator_id, "O_dim": matrices["rank"] / 32, "O_eff": np.trace(matrices["R"]) / 32, "O_task": np.trace(u.T @ matrices["Q"] @ u) / 8, "O_task_soft": np.trace(u.T @ matrices["R"] @ u) / 8})
        scores_frame = pd.DataFrame(scores); scores_frame.to_csv(self.output / "observability_scores.csv", index=False)
        results = pd.read_csv(self.output / "matched_unseen_all.csv")
        degradation = results[(results.method == "B8") & (results.evaluation == "unseen")].groupby("operator_id").operator_drop.mean()
        merged = scores_frame.merge(degradation.rename("degradation"), on="operator_id")
        correlations = []
        for score in ("O_dim", "O_eff", "O_task", "O_task_soft"):
            pearson = pearsonr(merged[score], merged.degradation); spearman = spearmanr(merged[score], merged.degradation)
            correlations.append({"score": score, "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue, "spearman_r": spearman.statistic, "spearman_p": spearman.pvalue})
        correlation_frame = pd.DataFrame(correlations); correlation_frame.to_csv(self.output / "observability_failure_correlation.csv", index=False)
        task = correlation_frame[correlation_frame.score == "O_task_soft"].iloc[0]
        c_status = "PASS" if task.spearman_r <= -self.config["gate"]["C_abs_spearman"] else "FAIL"
        aggregate = self._aggregate(); b6 = aggregate[aggregate.method == "B6"].iloc[0]; b7 = aggregate[aggregate.method == "B7"].iloc[0]; b8 = aggregate[aggregate.method == "B8"].iloc[0]
        return {
            "C": {"status": c_status, "soft_task_spearman": task.spearman_r, "n_test_operators": len(merged)},
            "lifting": {"B7_minus_B6_unseen": float(b7.unseen_ba - b6.unseen_ba)},
            "observability_conditioning": {"B8_minus_B7_unseen": float(b8.unseen_ba - b7.unseen_ba)},
        }

    def _plots(self) -> None:
        aggregate_path = self.output / "matched_unseen_agg.csv"
        if not aggregate_path.exists(): return
        aggregate = pd.read_csv(aggregate_path)
        if aggregate.empty: return
        x = np.arange(len(aggregate)); width = 0.38
        fig, axis = plt.subplots(figsize=(9, 4.8)); axis.bar(x - width/2, aggregate.matched_ba, width, label="Matched"); axis.bar(x + width/2, aggregate.unseen_ba, width, label="Unseen"); axis.set_xticks(x, aggregate.method); axis.set_ylabel("Balanced accuracy"); axis.legend(); fig.tight_layout(); fig.savefig(self.output / "plots/figure1_matched_unseen.png", dpi=180); plt.close(fig)
        family_path = self.output / "operator_family_results.csv"
        family = pd.read_csv(family_path)
        if not family.empty:
            pivot = family.pivot(index="operator_family", columns="method", values="operator_drop")
            axis = pivot.plot(kind="bar", figsize=(10, 5)); axis.set_ylabel("Operator drop (BA)"); plt.tight_layout(); plt.savefig(self.output / "plots/figure2_operator_family_drop.png", dpi=180); plt.close()
            pivot_ba = family.pivot(index="operator_family", columns="method", values="balanced_accuracy")
            axis = pivot_ba.plot(kind="bar", figsize=(11, 5)); axis.set_ylabel("Balanced accuracy"); plt.tight_layout(); plt.savefig(self.output / "plots/figure5_family_breakdown.png", dpi=180); plt.close()
        scores_path = self.output / "observability_scores.csv"; results_path = self.output / "matched_unseen_all.csv"
        if scores_path.exists() and results_path.exists() and (self.output / "observability_scores.csv").stat().st_size > 80:
            scores = pd.read_csv(scores_path); results = pd.read_csv(results_path)
            if not scores.empty and "B8" in set(results.method):
                degradation = results[(results.method == "B8") & (results.evaluation == "unseen")].groupby("operator_id").operator_drop.mean()
                merged = scores.merge(degradation.rename("degradation"), on="operator_id")
                fig, axis = plt.subplots(figsize=(5.5, 4.8)); axis.scatter(merged.O_task_soft, merged.degradation); axis.set_xlabel("Task-aware soft observability"); axis.set_ylabel("Operator drop"); fig.tight_layout(); fig.savefig(self.output / "plots/figure3_task_observability.png", dpi=180); plt.close(fig)
                correlations = pd.read_csv(self.output / "observability_failure_correlation.csv")
                fig, axis = plt.subplots(figsize=(6.5, 4)); axis.bar(correlations.score, correlations.spearman_r); axis.axhline(0, color="black", linewidth=.8); axis.set_ylabel("Spearman correlation with failure"); fig.tight_layout(); fig.savefig(self.output / "plots/figure4_observability_comparison.png", dpi=180); plt.close(fig)
        for filename, title in (
            ("figure3_task_observability.png", "Not run: Gate B failed before B7/B8"),
            ("figure4_observability_comparison.png", "Not run: task subspace was not estimated"),
        ):
            destination = self.output / "plots" / filename
            if not destination.exists():
                fig, axis = plt.subplots(figsize=(6.5, 4)); axis.axis("off"); axis.text(.5, .5, title, ha="center", va="center", fontsize=14); fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)

    def _write_report(self, status: str) -> None:
        aggregate = pd.read_csv(self.output / "matched_unseen_agg.csv") if (self.output / "matched_unseen_agg.csv").exists() else pd.DataFrame()
        labels = {"B2": "Coordinate", "B3": "Bipolar Midpoint", "B4": "Coord + Ref Metadata", "B5": "Interpolation", "B6": "Signed Operator", "B7": "+ Lifting", "B8": "+ Observability"}
        if not aggregate.empty:
            display = pd.DataFrame({"method": list(labels)}).merge(aggregate, on="method", how="left")
            display.insert(1, "Method name", display.method.map(labels))
            display.insert(2, "status", np.where(display.matched_ba.notna(), "complete", "not run (Gate B stop)"))
            table = display.to_markdown(index=False, floatfmt=".4f")
        else:
            table = "No model training result is available yet."
        gates = json.loads((self.output / "gate_summary.json").read_text()) if (self.output / "gate_summary.json").exists() else {}
        gate_values = gates.get("gates", {})
        gate_a = gate_values.get("A", {}); gate_b = gate_values.get("B", {})
        family_path = self.output / "operator_family_results.csv"
        hardest = "not available"
        if family_path.exists():
            family = pd.read_csv(family_path)
            if not family.empty:
                row = family.sort_values("operator_drop", ascending=False).iloc[0]
                hardest = f"{row.operator_family} for {row.method} (mean drop {row.operator_drop:.4f} BA)"
        final_wording = "GO TO STAGE-1" if status == "GO_TO_STAGE_1" else "DO NOT PROCEED YET"
        report = f"""# MOA-EEG Stage-0 report

## Scientific terminal state

`{status}`

This run is a falsification study. It does not use shared/self/cross consistency, a canonical decoder, rendering losses, uncertainty routing, conformal prediction, CVaR, or foundation-model adaptation.

## D0 reference and legality conclusion

The PhysioNet EEGMMIDB EDF metadata does not encode an exact acquisition reference. The pipeline therefore does not claim M1, M2, average mastoid, or absolute scalp potential. It uses an explicit CAR64 transformation under the documented condition that the amplifier channels shared one acquisition reference. Because direct dataset metadata is missing, EEGMMIDB is supporting controlled evidence, not the required clean explicit-reference replication. ISRUC remains mandatory before a strong Stage-1 decision.

All admitted synthetic views satisfy `Y_t=A_tY_0` by construction and `B_t=A_tB_0` numerically. Failed operators are excluded before training.

## Main table — EEGMMIDB

{table}

ISRUC is reported separately: no result is available until the official extracted-channel files and annotations are locally verified.

The absolute matched BA of the strongest non-MOA model is {gate_a.get('matched_ba', float('nan')):.4f}. This is only marginally above four-class chance and materially limits the scientific strength of any observed operator drop.

## Gates

```json
{json.dumps(gates, indent=2, ensure_ascii=False)}
```

## Required questions

1. Legal operator shift: Gate A was triggered by `{gate_a.get('trigger_operator', 'not run')}` with drop `{gate_a.get('trigger_operator_drop', float('nan')):.4f}` BA; the all-operator mean drop was `{gate_a.get('mean_operator_drop', float('nan')):.4f}`.
2. Strongest geometry/reference baseline: `{gate_b.get('strongest_baseline', gate_a.get('strongest_non_moa', 'not run'))}`.
3. Signed operator beyond geometry+metadata: `{gate_b.get('status', 'NOT_RUN')}`; mean unseen gain was `{gate_b.get('gain', float('nan')):.4f}` BA with `{gate_b.get('positive_seeds', 0)}/{gate_b.get('seed_count', 0)}` positive seeds.
4. Lifting beyond signed operator: not run because Gate B failed; running B7 would violate fail-fast.
5. Observability conditioning beyond lifting: not run because Gate B failed.
6. Hardest observed family/method combination: {hardest}.
7. Task-aware observability prediction: not estimated because B8 was prohibited by Gate B.
8. Matched-performance sacrifice for B8: not evaluated because B8 was prohibited.
9. EEGMMIDB/ISRUC direction consistency: unresolved; ISRUC was not locally available and its official MEGA delivery could not yet be verified on this server.
10. Stage-1 decision: `{final_wording}`. The blocking gate is `{status}`.

## Final decision

`{final_wording}`

## Reproduce

```bash
cd /root/autodl-tmp/hsc_tta_eeg/repo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hsc_gpu
PYTHONPATH=src python scripts/moa_stage0/run.py --config configs/moa_stage0.yaml --phase audit
PYTHONPATH=src python scripts/moa_stage0/run.py --config configs/moa_stage0.yaml --phase run
```
"""
        (self.output / "stage0_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("audit", "run", "all"), default="all")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    pipeline = Stage0Pipeline(repository, args.config)
    if args.phase in {"audit", "all"}: pipeline.audit()
    if args.phase in {"run", "all"}: pipeline.run()


if __name__ == "__main__": main()
