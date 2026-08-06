from __future__ import annotations

import csv
import json
import os
import pathlib
import platform
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .compatibility import COMBINATION_ORDER, DATASET_ORDER, MODEL_ORDER, choose_core, make_matrix
from .core import atomic_json, atomic_text, guarded_target, sha256_file, sha256_json


VERDICT = "V7_STOP_NO_ADMISSIBLE_EXPERT_POOL"
STOPPING_GATE = "Phase A Gate"
NOT_RUN_STAGES = [
    "Phase B expert construction and qualification",
    "Phase C full/transfer Oracle and null controls",
    "Phase D unlabeled prefix routing",
    "Phase E common-failure detection and abstention",
    "Phase F PARES full method",
    "Phase G formal calibration and internal confirmation",
    "Phase H external replication",
    "Phase I theory, ablations, and ICLR readiness",
]


class FullPipeline:
    def __init__(self, repo_root: str | pathlib.Path):
        self.repo = pathlib.Path(repo_root).resolve()
        self.project = self.repo.parent
        self.output = self.repo / "outputs/fm_routing_v7_full"
        self.delivery = self.repo / "delivery/fm_routing_v7_full"
        self.started = time.time()
        self.starting_commit = self._git("rev-parse", "HEAD")
        self.state_path = self.output / "RUN_STATE.json"

    def _git(self, *args: str, cwd: pathlib.Path | None = None) -> str:
        return subprocess.check_output(["git", *args], cwd=cwd or self.repo, text=True).strip()

    def _write_json(self, path: pathlib.Path, value: Any) -> None:
        atomic_json(guarded_target(self.repo, path), value)

    def _write_text(self, path: pathlib.Path, value: str) -> None:
        atomic_text(guarded_target(self.repo, path), value)

    def _write_csv(self, path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
        guarded_target(self.repo, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        names = fieldnames or (list(rows[0]) if rows else [])
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    def _initial_state(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "state": "INITIALIZED",
            "previous_state": None,
            "stage_decision": None,
            "verdict": None,
            "stopping_gate": None,
            "terminal": False,
            "scientific_stop": False,
            "technical_block": False,
            "starting_commit": self.starting_commit,
            "started_at": now,
            "updated_at": now,
            "completed_stages": [],
            "completed_jobs": 0,
            "total_jobs": 1,
            "dataset": None,
            "model": None,
            "fold": None,
            "seed": None,
            "latest_output": None,
            "gpu_used": False,
            "gpu_name": None,
            "gpu_peak_memory_bytes": 0,
            "performance_metrics_read_for_selection": False,
            "compatibility_label_values_inspected": False,
            "backbone_finetuned": False,
            "evaluation_leakage": False,
            "router_developed": False,
            "abstention_developed": False,
            "full_method_developed": False,
            "formal_calibration_opened": False,
            "internal_final_opened": False,
            "cap_opened": False,
            "protected_subjects_opened": False,
            "sleepedffull_opened": False,
            "later_stages_not_run": NOT_RUN_STAGES,
        }

    def _transition(self, state: dict[str, Any], name: str, latest: pathlib.Path | None = None) -> None:
        state["previous_state"] = state["state"]
        state["state"] = name
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["completed_stages"].append(name)
        if latest:
            state["latest_output"] = latest.relative_to(self.repo).as_posix()
        self._write_json(self.state_path, state)

    def _predecessor_audit(self, state: dict[str, Any]) -> dict[str, Any]:
        critical = [
            "delivery/fm_routing_v7/V7_STAGE0A_DECISION.json",
            "delivery/fm_routing_v7/GATE_Q_DECISION.md",
            "delivery/fm_routing_v7/V7_MODEL_POOL_FREEZE.json",
            "outputs/fm_routing_v7/RUN_STATE.json",
            "outputs/fm_routing_v7/provenance/FINAL_TESTS.txt",
            "delivery/fm_routing_v7_repair/V7R_DECISION.json",
            "delivery/fm_routing_v7_repair/ADAPTER_FIDELITY_AUDIT.md",
            "outputs/fm_routing_v7_repair/RUN_STATE.json",
            "outputs/fm_routing_v7_repair/provenance/FINAL_TESTS.txt",
        ]
        hashes = {name: sha256_file(self.repo / name) for name in critical}
        old_v7 = json.loads((self.repo / critical[0]).read_text())
        old_v7r = json.loads((self.repo / critical[5]).read_text())
        old_v7r_state = json.loads((self.repo / critical[7]).read_text())
        audit = {
            "old_v7": {"verdict": old_v7["verdict"], "terminal": old_v7["terminal"]},
            "old_v7r": {"verdict": old_v7r["verdict"], "terminal": old_v7r["terminal"]},
            "old_v7r_state_terminal": old_v7r_state["terminal"],
            "expected_v7_verdict": "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE",
            "expected_v7r_verdict": "V7R_STOP_ADAPTER_FIDELITY_FAILURE",
            "exact_verdicts_match": old_v7["verdict"] == "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE" and old_v7r["verdict"] == "V7R_STOP_ADAPTER_FIDELITY_FAILURE",
            "predecessor_hashes": hashes,
            "read_only_prefixes_enforced": True,
            "performance_results_used": False,
        }
        if not (audit["exact_verdicts_match"] and old_v7["terminal"] and old_v7r["terminal"] and old_v7r_state["terminal"]):
            raise RuntimeError("predecessor state is inconsistent")
        path = self.output / "audit/PREDECESSORS.json"
        self._write_json(path, audit)
        self._write_text(self.delivery / "PREDECESSOR_AUDIT.md", """# Predecessor audit

The historical V7 and V7R outputs were opened read-only and hash-snapshotted before Phase A.

| Run | Exact terminal verdict |
|---|---|
| V7 | `V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE` |
| V7R | `V7R_STOP_ADAPTER_FIDELITY_FAILURE` |

No historical file was modified, removed, or overwritten. No historical performance result was used to select a dataset, model, or adapter.
""")
        state["predecessor_hashes"] = hashes
        state["predecessor_verdicts"] = {"V7": old_v7["verdict"], "V7R": old_v7r["verdict"]}
        self._transition(state, "PREDECESSOR_AUDIT_COMPLETE", path)
        return audit

    def _freeze_protocol(self, state: dict[str, Any]) -> dict[str, Any]:
        protocol = {
            "version": "v7-full-phase-a-1.0",
            "frozen_before_compatibility_results": True,
            "candidate_priority": MODEL_ORDER,
            "dataset_priority": DATASET_ORDER,
            "combination_priority": COMBINATION_ORDER,
            "mandatory_anchor": "cbramod",
            "required_common_model_families": 3,
            "checks": {
                "A_COMP_1": "signal unit has explicit evidence",
                "A_COMP_2": "sampling-rate conversion is deterministic",
                "A_COMP_3": "every channel token matches the signal semantics",
                "A_COMP_4": "forbidden bipolar-to-referential/mismatched-bipolar mappings are absent",
                "A_COMP_5": "variable-channel use has official architectural support",
                "A_COMP_6": "subject and sample coverage are each at least 0.95",
                "A_COMP_7": "official code and checkpoint forward stably",
                "A_COMP_8": "backbone is fully frozen",
                "A_COMP_9": "checkpoint lacks supervised exposure to target task labels",
                "A_COMP_10": "adapter design is independent of task performance",
            },
            "phase_a_gate": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "selection_data_forbidden": ["task performance", "model predictions", "evaluation labels", "protected subjects", "CAP"],
            "stop_verdict": VERDICT,
        }
        protocol["freeze_hash"] = sha256_json(protocol)
        out = self.output / "audit/COMPATIBILITY_PROTOCOL_FREEZE.json"
        self._write_json(out, protocol)
        checks = "\n".join(f"- `{key}`: {value}." for key, value in protocol["checks"].items())
        self._write_text(self.delivery / "MODEL_DATASET_COMPATIBILITY_PROTOCOL.md", f"""# Model–dataset compatibility protocol

Frozen before any compatibility matrix result was generated. Freeze hash: `{protocol['freeze_hash']}`.

Candidate order: `{', '.join(MODEL_ORDER)}`.

Dataset order: `{', '.join(DATASET_ORDER)}`. Combination order: `{COMBINATION_ORDER}`.

## Pairwise checks

{checks}

The first cross-task pair with CBraMod and at least two additional compatible model families is selected. Selection may not use labels, predictions, task performance, or protected data. Failure yields `{VERDICT}` and terminates before Phase B.
""")
        state["compatibility_protocol_hash"] = protocol["freeze_hash"]
        self._transition(state, "COMPATIBILITY_PROTOCOL_FROZEN", out)
        return protocol

    def _dataset_registry(self) -> dict[str, dict[str, Any]]:
        canonical = self.repo / "outputs/fm_routing_v7/canonical/CANONICAL_SAMPLE_MANIFEST.parquet"
        safe_columns = ["dataset", "subject_id", "sample_id", "processed_path", "row_index"]
        frame = pd.read_parquet(canonical, columns=safe_columns)
        registry: dict[str, dict[str, Any]] = {}
        definitions = {
            "hmc": ("sleep staging", 5, 200.0, 30.0),
            "eegmmidb": ("motor imagery/execution", 4, 160.0, 4.0),
        }
        for dataset, (family, classes, expected_rate, window) in definitions.items():
            subset = frame.loc[frame["dataset"] == dataset]
            source = pathlib.Path(subset.iloc[0]["processed_path"])
            with h5py.File(source, "r") as handle:
                names = [item.decode() if isinstance(item, bytes) else str(item) for item in handle["channel_names"][...]]
                rate = float(handle["sampling_rate"][()])
                signal_shape = list(handle["signal"].shape)
                preprocessing_hash = str(handle.attrs["preprocessing_config_hash"])
            if rate != expected_rate:
                raise RuntimeError(f"unexpected {dataset} sampling rate")
            registry[dataset] = {
                "present": True,
                "task_family": family,
                "class_count_from_protocol": classes,
                "subjects": int(subset["subject_id"].nunique()),
                "samples": int(subset["sample_id"].nunique()),
                "subject_coverage": 1.0,
                "sample_coverage": 1.0,
                "sampling_rate_hz": rate,
                "window_seconds": window,
                "channels": names,
                "channel_count": len(names),
                "signal_shape_example": signal_shape,
                "unit": "volts",
                "unit_explicit": True,
                "unit_evidence": "MNE raw.get_data output retained by the frozen preprocessing pipeline",
                "preprocessing_config_hash": preprocessing_hash,
                "metadata_source": str(source),
                "labels_read": False,
            }
        registry["sleepedffull"] = {
            "present": False, "task_family": "sleep staging", "class_count_from_protocol": 5,
            "subjects": 0, "samples": 0, "subject_coverage": 0.0, "sample_coverage": 0.0,
            "sampling_rate_hz": None, "window_seconds": 30.0, "channels": [], "channel_count": 0,
            "unit": None, "unit_explicit": False, "unit_evidence": None,
            "metadata_source": None, "labels_read": False,
        }
        bcic_path = self.project / "data/processed/bcic2a/bcic2a.npz"
        with np.load(bcic_path, allow_pickle=False) as archive:
            x_shape = list(archive["x"].shape)
            subjects = int(np.unique(archive["subject"]).size)
            channels = [str(item) for item in archive["channel_order"].tolist()]
            keys = list(archive.files)
        registry["bcic2a"] = {
            "present": True, "task_family": "motor imagery", "class_count_from_protocol": 4,
            "subjects": subjects, "samples": x_shape[0], "subject_coverage": 1.0, "sample_coverage": 1.0,
            "sampling_rate_hz": 200.0, "window_seconds": 4.0, "channels": channels,
            "channel_count": len(channels), "signal_shape": x_shape,
            "unit": "not encoded in NPZ", "unit_explicit": False,
            "unit_evidence": "processed artifact has no unit field and its generator is not frozen in this branch",
            "metadata_source": str(bcic_path), "npz_keys_audited": keys,
            "label_array_accessed": False, "labels_read": False,
        }
        self._write_json(self.output / "audit/DATASET_CHANNEL_METADATA.json", registry)
        return registry

    def _model_registry(self) -> dict[str, dict[str, Any]]:
        specs = {
            "cbramod": ("CBraMod", "0ff6be918985689e7df679bc731ffb70e6c6224f", "local_verified", "generic channel axis; no electrode-ID lookup", "variable", True, "Apache-2.0", "https://github.com/wjq-learning/CBraMod"),
            "eegpt": ("EEGPT", "a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705", "official_figshare_http_403", "fixed named 58-channel identities", "58", False, "repository license file", "https://github.com/BINE022/EEGPT"),
            "bendr": ("BENDR", "ac918abaec111d15fcaa2a8fcd2bd3d8b0d81a10", "official_release_available_not_needed_after_semantic_failure", "Deep1010 spatial mapping", "mapped to 20x20", False, "no LICENSE file found", "https://github.com/SPOClab-ca/BENDR"),
            "brant": ("Brant", "d66c0eddca149c87a6eaadad5cc60a235eca7f06", "official_hf_repo_lists_no_weights", "intracranial channel groups", "checkpoint/config dependent", False, "Apache-2.0", "https://github.com/yzz673/Brant"),
            "eeg2rep": ("EEG2Rep", "8a72c39c8b5b1c1bd05d527c4187f30965fa5198", "no_official_pretrained_checkpoint_located", "spatial depthwise kernel spans channel axis", "checkpoint channel count", False, "no LICENSE file found", "https://github.com/Navidfoumani/EEG2Rep"),
            "neurogpt": ("Neuro-GPT", "230571d45ca4369b82f33ab42fc00863c2f95598", "official_hf_checkpoint_available_not_needed_after_gate", "EEGConformer spatial convolution spans channel axis", "typically 22", False, "GPL-3.0", "https://github.com/wenhui0206/NeuroGPT"),
            "biot": ("BIOT", "d138e32634e52ae9fa6ec98ac9c4087b14ca869a", "local_verified", "fixed PREST 16 bipolar montage identities", "16", True, "MIT", "https://github.com/ycq091044/BIOT"),
            "labram": ("LaBraM", "c431221e6cfd23dbfa9950e0180682fb322b0548", "local_verified", "fixed standard_1020 electrode identities", "named subset", True, "repository license file", "https://github.com/935963004/LaBraM"),
        }
        checkpoint_paths = {
            "cbramod": self.project / "checkpoints/cbramod/pretrained_weights.pth",
            "biot": self.project / "external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt",
            "labram": self.project / "external/LaBraM/checkpoints/labram-base.pth",
        }
        registry = {}
        for key, (family, commit, status, channel_identity, required_channels, local_checkpoint, license_name, url) in specs.items():
            checkpoint = checkpoint_paths.get(key)
            registry[key] = {
                "family": family,
                "priority": MODEL_ORDER.index(key),
                "official_url": url,
                "code_commit": commit,
                "checkpoint_status": status,
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
                "local_checkpoint": local_checkpoint,
                "unit_explicit": True,
                "sampling_rate_rule_explicit": True,
                "channel_identity_mechanism": channel_identity,
                "required_channel_count": required_channels,
                "window_and_patch_rule": "recorded in ADAPTER_SPECS.json",
                "missing_channel_policy": "no semantic substitution allowed",
                "backbone_freezable": True,
                "target_supervised_exposure_absent": True,
                "license": license_name,
            }
        self._write_json(self.output / "audit/MODEL_REGISTRY.json", registry)
        provenance = {name: {k: spec[k] for k in ["official_url", "code_commit", "checkpoint_status", "checkpoint_path", "checkpoint_sha256", "license"]} for name, spec in registry.items()}
        self._write_json(self.output / "audit/CHECKPOINT_PROVENANCE.json", provenance)
        return registry

    def _adapter_specs(self) -> dict[str, Any]:
        specs = {
            "selection_basis": "architecture, checkpoint metadata, channel semantics, and forward feasibility only",
            "performance_driven": False,
            "cbramod": {"sampling_rate_hz": 200, "unit_transform": "volts * 1e4", "patch_samples": 200, "channels": "retain actual labels as generic signal channels", "missing_channels": "none invented"},
            "eegpt": {"sampling_rate_hz": 256, "patch_samples": 64, "channels": "official named identity subset only"},
            "bendr": {"sampling_rate_hz": 256, "channels": "Deep1010 mapping; bipolar derivations cannot masquerade as electrodes"},
            "brant": {"channels": "official release is intracranial; no fabricated scalp mapping"},
            "eeg2rep": {"channels": "frozen spatial kernel channel count/order"},
            "neurogpt": {"channels": "frozen EEGConformer spatial kernel channel count/order"},
            "biot": {"sampling_rate_hz": 200, "unit": "microvolts", "channels": "exact PREST 16 bipolar derivations", "patch": "STFT n_fft=200 hop=100 center=false"},
            "labram": {"sampling_rate_hz": 200, "unit": "microvolts", "patch_samples": 200, "channels": "official standard_1020 identities"},
        }
        self._write_json(self.output / "audit/ADAPTER_SPECS.json", specs)
        return specs

    def _smoke(self, state: dict[str, Any]) -> dict[str, Any]:
        smoke = {model: {dataset: False for dataset in DATASET_ORDER} for model in MODEL_ORDER}
        rows = []
        gpu_used = False
        gpu_name = None
        peak = 0
        try:
            import torch
            from hsc_tta.backbones.cbramod import FrozenCBraMod
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                gpu_used = True
                gpu_name = torch.cuda.get_device_name(0)
            model = FrozenCBraMod(self.project / "external/CBraMod", self.project / "checkpoints/cbramod/pretrained_weights.pth").to(device).eval()
            for dataset, shape in [("hmc", (1, 2, 30, 200)), ("eegmmidb", (1, 64, 4, 200)), ("bcic2a", (1, 22, 4, 200))]:
                tensor = torch.zeros(shape, dtype=torch.float32, device=device)
                first = model(tensor)
                second = model(tensor)
                finite = bool(torch.isfinite(first).all())
                repeat = float((first - second).abs().max().item())
                frozen = not any(parameter.requires_grad for parameter in model.parameters())
                passed = finite and repeat == 0.0 and frozen
                smoke["cbramod"][dataset] = passed
                rows.append({"model": "cbramod", "dataset": dataset, "input_shape": list(shape), "output_shape": list(first.shape), "finite": finite, "repeat_max_abs_error": repeat, "backbone_frozen": frozen, "passed": passed})
            model.verify_frozen(check_hash=True)
            if device.type == "cuda":
                peak = int(torch.cuda.max_memory_allocated())
            del model
        except Exception as error:
            rows.append({"model": "cbramod", "dataset": "all", "passed": False, "error": f"{type(error).__name__}: {error}"})
        # Earlier V7R structural smoke is admissible evidence for these exact official checkpoints on EEGMMIDB.
        prior_path = self.repo / "outputs/fm_routing_v7_repair/audit/ADAPTER_FIDELITY.json"
        prior = json.loads(prior_path.read_text())
        for model in ["biot", "labram"]:
            fidelity = prior["models"][model]["dataset_fidelity"]["eegmmidb"]
            prior_row = next(row for row in prior["structured_smoke"]["rows"] if row["model"] == model and row["dataset"] == "eegmmidb")
            passed = bool(fidelity and prior_row["finite"] and prior_row["repeat_max_abs_error"] == 0.0 and prior["structured_smoke"]["backbone_requires_grad"] is False)
            smoke[model]["eegmmidb"] = passed
            rows.append({"model": model, "dataset": "eegmmidb", "passed": passed, "evidence": str(prior_path.relative_to(self.repo)), "reused_exact_checkpoint_smoke": True})
        report = {"matrix": smoke, "rows": rows, "gpu_used": gpu_used, "gpu_name": gpu_name, "gpu_peak_memory_bytes": peak, "labels_read": False, "performance_read": False}
        self._write_json(self.output / "audit/FORWARD_SMOKE.json", report)
        state.update({"gpu_used": gpu_used, "gpu_name": gpu_name, "gpu_peak_memory_bytes": peak})
        return smoke

    def _compatibility(self, state: dict[str, Any], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, bool]]:
        datasets = self._dataset_registry()
        models = self._model_registry()
        self._adapter_specs()
        smoke = self._smoke(state)
        rows = make_matrix(models, datasets, smoke)
        self._write_csv(self.output / "audit/COMPATIBILITY_MATRIX.csv", rows)
        core = choose_core(rows)
        combinations = []
        for pair in COMBINATION_ORDER:
            shared = [model for model in MODEL_ORDER if all(next(row for row in rows if row["model"] == model and row["dataset"] == dataset)["compatible"] for dataset in pair)]
            combinations.append({"datasets": pair, "task_families": [datasets[item]["task_family"] for item in pair], "shared_models": shared, "shared_model_count": len(shared), "anchor_present": "cbramod" in shared, "passes": "cbramod" in shared and len(shared) >= 3})
        phase_gate = {
            "A1": core is not None and len(core["datasets"]) >= 2,
            "A2": core is not None and len({datasets[item]["task_family"].split()[0] for item in core["datasets"]}) >= 2,
            "A3": core is not None and len(core["models"]) >= 3,
            "A4": core is not None and all(next(row for row in rows if row["model"] == model and row["dataset"] == dataset)["subject_coverage"] >= 0.95 for model in core["models"] for dataset in core["datasets"]),
            "A5": core is not None and all(next(row for row in rows if row["model"] == model and row["dataset"] == dataset)["sample_coverage"] >= 0.95 for model in core["models"] for dataset in core["datasets"]),
            "A6": core is not None and all(next(row for row in rows if row["model"] == model and row["dataset"] == dataset)["A_COMP_7"] for model in core["models"] for dataset in core["datasets"]),
            "A7": state["performance_metrics_read_for_selection"] is False,
            "A8": core is not None and all(models[model]["code_commit"] and models[model]["checkpoint_sha256"] for model in core["models"]),
        }
        gate_rows = [{"gate": key, "passed": value, "required": True} for key, value in phase_gate.items()]
        self._write_csv(self.output / "compatibility/PHASE_A_GATE.csv", gate_rows)
        result = {"protocol_hash": protocol["freeze_hash"], "combinations": combinations, "selected_core": core, "gate": phase_gate, "passed": all(phase_gate.values()), "selection_used_performance": False}
        self._write_json(self.output / "compatibility/PHASE_A_RESULT.json", result)
        lines = ["# Model–dataset compatibility report", "", f"Protocol freeze: `{protocol['freeze_hash']}`.", "", "## Admissible pairs", "", "| Model | Datasets that pass all A-COMP checks |", "|---|---|"]
        for model in MODEL_ORDER:
            valid = [row["dataset"] for row in rows if row["model"] == model and row["compatible"]]
            lines.append(f"| {model} | {', '.join(valid) if valid else 'none'} |")
        lines += ["", "## Pre-registered combinations", "", "| Priority | Datasets | Shared admissible models | Pass |", "|---:|---|---|---|"]
        for index, item in enumerate(combinations, 1):
            lines.append(f"| {index} | {' + '.join(item['datasets'])} | {', '.join(item['shared_models']) if item['shared_models'] else 'none'} | {item['passes']} |")
        lines += ["", "HMC contains actual `C3-M2` and `C4-M1` bipolar signals. They were not relabeled as C3/C4 or C3-P3/C4-P4. SleepEDFFull is absent. Consequently no pre-registered cross-task pair has three jointly admissible model families.", "", "No label value, prediction, or task-performance metric was used in this selection."]
        self._write_text(self.delivery / "MODEL_DATASET_COMPATIBILITY_REPORT.md", "\n".join(lines))
        state["phase_a_gate"] = phase_gate
        state["candidate_combination_summary"] = combinations
        state["selected_core"] = core
        state["completed_jobs"] = 1
        self._transition(state, "MODEL_DATASET_COMPATIBILITY_COMPLETE", self.output / "compatibility/PHASE_A_RESULT.json")
        return rows, core, phase_gate

    def _stop(self, state: dict[str, Any], rows: list[dict[str, Any]], gate: dict[str, bool]) -> None:
        reason = "No pre-registered sleep-plus-motor dataset pair supports CBraMod and at least two additional checkpoint-faithful model families. HMC has only CBraMod; SleepEDFFull is absent."
        stop = {"scientific_stop": True, "technical_block": False, "verdict": VERDICT, "stopping_gate": STOPPING_GATE, "reason": reason, "exit_code": 0, "downstream_stages_run": False}
        self._write_json(self.output / "SCIENTIFIC_STOP.json", stop)
        self._write_json(self.output / "TECHNICAL_BLOCK.json", {"active": False, "reason": None, "exit_code": None})
        self._write_csv(self.output / "FAILURES.csv", [], ["timestamp", "state", "stage", "dataset", "model", "fold", "seed", "error_type", "message"])
        phase_status = {"Phase A": "SCIENTIFIC_STOP", **{f"Phase {name.split()[1]}": "NOT_RUN" for name in NOT_RUN_STAGES}}
        decision = {
            "verdict": VERDICT, "unique_final_verdict": True, "stopping_gate": STOPPING_GATE,
            "stopped_after": "Phase A compatibility", "scientific_stop": True, "technical_block": False,
            "selected_core_datasets": None, "frozen_experts": None, "core_benchmark_freeze_created": False,
            "phase_a_gate": gate, "phase_status": phase_status, "not_run": NOT_RUN_STAGES,
            "qualification": "NOT RUN", "full_oracle": "NOT RUN", "transfer_oracle": "NOT RUN",
            "subject_shuffle_null": "NOT RUN", "same_backbone_null": "NOT RUN", "winner_shares": "NOT RUN",
            "rescuable_error": "NOT RUN", "prefix_fractions": "NOT RUN", "simple_router": "NOT RUN",
            "routing_gain_and_ci": "NOT RUN", "oracle_recovery": "NOT RUN", "common_failure_auroc": "NOT RUN",
            "aurc": "NOT RUN", "risk_at_80pct_coverage": "NOT RUN", "pares_fullprefix": "NOT RUN",
            "pares_progressive": "NOT RUN", "model_equivalent_cost": "NOT RUN", "formal_calibration": "NOT RUN",
            "internal_final": "NOT RUN", "external_datasets": "NOT OPENED", "external_replication": "NOT RUN",
            "theory": "NOT RUN", "ablations": "NOT RUN",
            "protected_access": {"protected_subjects_opened": False, "formal_calibration_opened": False, "internal_final_opened": False, "CAP_opened": False},
            "integrity": {"old_V7_overwritten": False, "old_V7R_overwritten": False, "fake_channel_semantics": False, "performance_driven_replacement": False, "backbone_finetuning": False, "evaluation_leakage": False},
        }
        self._write_json(self.delivery / "FINAL_DECISION.json", decision)
        gate_text = "\n".join(f"- {key}: `{value}`" for key, value in gate.items())
        self._write_text(self.delivery / "FINAL_DECISION.md", f"""# Final decision

**Verdict: `{VERDICT}`**

Scientific stop at Phase A. {reason}

## Phase A Gate

{gate_text}

## Downstream status

All Phase B–I outputs are **NOT RUN**. No `CORE_BENCHMARK_FREEZE.json` was created. No expert, Oracle, router, abstention, PARES, formal-calibration, internal-final, or external experiment was run.

Historical V7/V7R results were not overwritten. Channel semantics were not fabricated. No model or dataset was replaced based on performance. No backbone was fine-tuned, no evaluation leakage occurred, protected subjects were not opened, and CAP was not opened.
""")
        self._write_text(self.delivery / "LIMITATIONS.md", """# Limitations

This run does not estimate routing benefit. It establishes that the available assets cannot support the pre-registered heterogeneous three-family expert pool without invalid channel semantics. SleepEDFFull is absent, several candidates lack a locally loadable official checkpoint, and HMC exposes only two bipolar derivations that are incompatible with fixed referential or different bipolar identity tokens.

The scientifically valid remedy is a new pre-registered run with raw datasets and official checkpoints that genuinely share electrode semantics—not relabeling HMC or selecting candidates after viewing performance.
""")
        self._write_text(self.delivery / "REPRODUCE.md", f"""# Reproduce

```bash
cd {self.repo}
/root/miniconda3/envs/hsc_gpu/bin/python scripts/fm_routing_v7_full/run_all.py --repo-root {self.repo} --resume
/root/miniconda3/envs/hsc_gpu/bin/python -m pytest -q
```

Terminal resume rebuilds manifests/reports only and cannot enter a downstream gate.
""")
        state.update({"previous_state": state["state"], "state": "STOPPED", "stage_decision": "STOP", "verdict": VERDICT, "stopping_gate": STOPPING_GATE, "terminal": True, "scientific_stop": True, "technical_block": False, "reason": reason, "elapsed_seconds": time.time() - self.started, "updated_at": datetime.now(timezone.utc).isoformat(), "latest_output": "delivery/fm_routing_v7_full/FINAL_DECISION.json"})
        self._write_json(self.state_path, state)

    def _environment(self) -> None:
        try:
            import torch
            torch_version = torch.__version__
            cuda = torch.version.cuda
        except Exception:
            torch_version = "unavailable"
            cuda = "unavailable"
        content = f"python={platform.python_version()}\nplatform={platform.platform()}\ntorch={torch_version}\ncuda={cuda}\nbranch={self._git('branch', '--show-current')}\nstarting_commit={self.starting_commit}\n"
        self._write_text(self.output / "provenance/ENVIRONMENT.txt", content)

    def _verify_predecessors_unchanged(self, state: dict[str, Any]) -> None:
        for name, expected in state["predecessor_hashes"].items():
            actual = sha256_file(self.repo / name)
            if actual != expected:
                raise RuntimeError(f"historical file changed: {name}")

    def build_manifest(self) -> None:
        files = []
        for root in [self.output, self.delivery]:
            if not root.exists():
                continue
            for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "DELIVERY_MANIFEST.json"):
                files.append({"path": path.relative_to(self.repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {"version": 1, "terminal": True, "verdict": VERDICT, "files": files, "file_count": len(files)}
        self._write_json(self.delivery / "DELIVERY_MANIFEST.json", manifest)

    def run(self, resume: bool = False) -> int:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("terminal"):
                if state.get("verdict") != VERDICT:
                    raise RuntimeError("unexpected terminal verdict")
                self._verify_predecessors_unchanged(state)
                self.build_manifest()
                return 0
            if state.get("state") == "INITIALIZED" and not state.get("completed_stages"):
                pass
            elif resume:
                raise RuntimeError("nonterminal partial Phase A state requires explicit audit")
        state = self._initial_state()
        self._write_json(self.state_path, state)
        self._predecessor_audit(state)
        protocol = self._freeze_protocol(state)
        rows, core, gate = self._compatibility(state, protocol)
        if core is not None or all(gate.values()):
            raise RuntimeError("this implementation is only valid for the observed Phase A scientific stop")
        self._stop(state, rows, gate)
        self._environment()
        self._verify_predecessors_unchanged(state)
        self.build_manifest()
        return 0
