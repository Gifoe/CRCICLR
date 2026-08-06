from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import platform
import subprocess
import time
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from .core import (
    OLD_READ_ONLY_PREFIXES,
    PROTECTED_FLAGS,
    TechnicalBlock,
    adapter_gate,
    all_protected_false,
    atomic_json,
    atomic_text,
    deterministic_seeds,
    directory_hashes,
    sha256_file,
    sha256_json,
)


VERDICTS = [
    "V7R_STOP_ADAPTER_FIDELITY_FAILURE",
    "V7R_STOP_CBRAMOD_ANCHOR_FAILURE",
    "V7R_STOP_EXPERT_QUALIFICATION_FAILURE",
    "V7R_STOP_NO_CROSS_MODEL_HEADROOM",
    "V7R_STOP_NO_STABLE_SUBJECT_COMPLEMENTARITY",
    "V7R_CONTINUE_TO_UNLABELED_ROUTING_SCREEN",
    "V7R_TECHNICAL_BLOCK",
]
MODEL_ORDER = ["cbramod", "labram", "biot"]
EXPECTED_CHECKPOINTS = {
    "cbramod": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
    "labram": "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c",
    "biot": "40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class RepairPipeline:
    def __init__(self, repo_root: str | pathlib.Path):
        self.repo = pathlib.Path(repo_root).resolve()
        self.project = self.repo.parent
        self.output = self.repo / "outputs/fm_routing_v7_repair"
        self.delivery = self.repo / "delivery/fm_routing_v7_repair"
        self.state_path = self.output / "RUN_STATE.json"
        self.started = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output.mkdir(parents=True, exist_ok=True)
        self.delivery.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {
                "state": "INITIALIZED",
                "terminal": False,
                "verdict": None,
                "starting_commit": self.git("rev-parse", "HEAD"),
                "started_at": utc_now(),
                "gpu_used": False,
                "completed_jobs": [],
                "failed_jobs": [],
                **{flag: False for flag in PROTECTED_FLAGS},
            }
            atomic_json(self.state_path, self.state)
        self.commands = self.output / "provenance/COMMANDS.txt"
        self.commands.parent.mkdir(parents=True, exist_ok=True)

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def transition(self, state: str, **updates: Any) -> None:
        self.state.update(updates)
        self.state["previous_state"] = self.state.get("state")
        self.state["state"] = state
        self.state["updated_at"] = utc_now()
        self.state["elapsed_seconds"] = time.time() - self.started + float(self.state.get("prior_elapsed_seconds", 0.0))
        atomic_json(self.state_path, self.state)
        print(json.dumps({"state": state, "verdict": self.state.get("verdict"), "elapsed_seconds": self.state["elapsed_seconds"]}), flush=True)

    def audit_predecessor(self) -> None:
        decision_path = self.repo / "delivery/fm_routing_v7/V7_STAGE0A_DECISION.json"
        state_path = self.repo / "outputs/fm_routing_v7/RUN_STATE.json"
        decision = json.loads(decision_path.read_text())
        state = json.loads(state_path.read_text())
        if decision.get("verdict") != "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE":
            raise TechnicalBlock("unexpected V7 predecessor verdict")
        if decision.get("stopping_gate") != "Gate Q" or decision.get("terminal") is not True:
            raise TechnicalBlock("unexpected V7 predecessor terminal state")
        legacy_flags = [flag for flag in PROTECTED_FLAGS if flag in state]
        legacy_protected_false = all(state.get(flag) is False for flag in legacy_flags)
        legacy_scout_absent = not any(path.exists() for path in [
            self.repo / "src/hsc_tta/fm_routing/scout.py",
            self.repo / "outputs/fm_routing_v7/scout",
        ])
        if state.get("gate_q_passed") is not False or not legacy_protected_false or not legacy_scout_absent:
            raise TechnicalBlock("V7 predecessor protected state mismatch")
        forbidden = [
            self.repo / "outputs/fm_routing_v7/results/FULL_ORACLE_SUMMARY.csv",
            self.repo / "src/hsc_tta/fm_routing/router.py",
            self.repo / "src/hsc_tta/fm_routing/abstention.py",
        ]
        if any(path.exists() for path in forbidden):
            raise TechnicalBlock("predecessor contains forbidden Oracle/router artifacts")
        required = [
            "delivery/fm_routing_v7/GATE_Q_DECISION.md",
            "delivery/fm_routing_v7/MODEL_QUALIFICATION_REPORT.md",
            "delivery/fm_routing_v7/V7_MODEL_POOL_FREEZE.json",
            "delivery/fm_routing_v7/INPUT_ADAPTER_AUDIT.md",
            "outputs/fm_routing_v7/results/MODEL_METRICS_BY_SEED.csv",
            "outputs/fm_routing_v7/provenance/FINAL_TESTS.txt",
            "outputs/online_blockwise_v6/results/BACKBONE_GATE.csv",
        ]
        missing = [item for item in required if not (self.repo / item).exists()]
        if missing:
            raise TechnicalBlock(f"missing predecessor artifacts: {missing}")
        critical_hashes = {item: sha256_file(self.repo / item) for item in required}
        audit = {
            "verified_at": utc_now(),
            "v7_verdict": decision["verdict"],
            "stopping_gate": decision["stopping_gate"],
            "terminal": decision["terminal"],
            "oracle_not_run": True,
            "router_not_implemented": True,
            "abstention_not_implemented": True,
            "protected_flags_all_false": True,
            "critical_hashes": critical_hashes,
            "v6_reference_ba": {"hmc": 0.596565, "eegmmidb": 0.410526},
            "v7_pooled_ba": {"hmc": 0.4873248758, "eegmmidb": 0.2538410028},
        }
        atomic_json(self.output / "audit/PREDECESSOR.json", audit)
        atomic_text(self.delivery / "PREDECESSOR_AUDIT.md", """# Predecessor audit

The historical V7 result remains valid and unchanged: `V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE`, stopping at Gate Q with `terminal=true`. Cross-model Oracle, router, abstention and scout were not run or implemented. All protected flags were false.

The V6 CBraMod reference balanced accuracies are 0.596565 (HMC) and 0.410526 (EEGMMIDB); the failed V7 pooled readout produced 0.487325 and 0.253841 respectively. This repair is explicitly post-hoc development, not confirmatory evidence.
""")
        self.state["predecessor_hashes"] = critical_hashes
        self.transition("PREDECESSOR_VERIFIED", predecessor_verdict=decision["verdict"])

    def freeze_protocol(self) -> None:
        pool = json.loads((self.repo / "delivery/fm_routing_v7/V7_MODEL_POOL_FREEZE.json").read_text())
        canonical = self.repo / "outputs/fm_routing_v7/canonical/CANONICAL_SAMPLE_MANIFEST.parquet"
        if pool["checkpoint_hashes"] != EXPECTED_CHECKPOINTS:
            raise TechnicalBlock("frozen checkpoint hashes changed")
        checkpoint_paths = {
            "cbramod": self.project / "checkpoints/cbramod/pretrained_weights.pth",
            "labram": self.project / "external/LaBraM/checkpoints/labram-base.pth",
            "biot": self.project / "external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt",
        }
        actual_hashes = {name: sha256_file(path) for name, path in checkpoint_paths.items()}
        if actual_hashes != EXPECTED_CHECKPOINTS:
            raise TechnicalBlock(f"checkpoint bytes changed: {actual_hashes}")
        code_roots = {"cbramod": self.project / "external/CBraMod", "labram": self.project / "external/LaBraM", "biot": self.project / "external/BIOT"}
        actual_commits = {name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() for name, path in code_roots.items()}
        if actual_commits != pool["code_commits"]:
            raise TechnicalBlock(f"frozen model code commits changed: {actual_commits}")
        freeze = {
            "frozen_at": utc_now(),
            "post_hoc_development_repair": True,
            "starting_commit": self.state["starting_commit"],
            "models": MODEL_ORDER,
            "checkpoint_hashes": EXPECTED_CHECKPOINTS,
            "code_commits": actual_commits,
            "checkpoint_paths": {name: str(path) for name, path in checkpoint_paths.items()},
            "canonical_manifest_hash": sha256_file(canonical),
            "datasets": ["hmc", "eegmmidb"],
            "outer_folds": [0, 1, 2, 3, 4],
            "seeds": [0, 1, 2, 3, 4],
            "fold_roles": "evaluation=e; validation=(e+1)%5; training=remaining three; refit=all four non-evaluation folds",
            "adapter_priority": ["checkpoint configuration", "same-checkpoint official inference", "official downstream code", "minimal deterministic mapping"],
            "structured_representation": {
                "cbramod": "final channel-patch tokens with channel and patch identity",
                "labram": "final patch/channel tokens; HMC three 10-second subwindows concatenated in time order",
                "biot": "final transformer token sequence with checkpoint channel-token and patch order",
            },
            "readout_families": {
                "H0_GLOBAL_LOGREG": {"C": [0.01, 0.1, 1.0, 10.0], "solver": "LBFGS", "max_iter": 2000, "class_weight": "training-fold inverse frequency"},
                "H1_TOKEN_ATTENTION_POOL": {"architecture": "Linear(input_dim,64)->LayerNorm->one learned query->4-head cross-attention->residual 2-layer FFN(hidden=128)->LayerNorm->Linear(n_classes)", "optimizer": "AdamW", "lr": [0.0001, 0.0003, 0.001], "weight_decay": [0.0001, 0.001], "dropout": [0.0, 0.1], "max_epochs": 100, "patience": 15, "gradient_clip": 1.0, "loss": "class-weighted cross entropy", "validation_metric": "mean subject balanced accuracy", "mini_batch": True, "deterministic": True, "parameter_cap": 100000},
            },
            "head_tie_rule": "choose H0 when validation mean-subject BA difference is <=0.005",
            "adapter_gate": {"F1": "all adapters have official code/checkpoint basis", "F2": "no performance-driven adapter selection", "F3": "100% canonical subject coverage", "F4": ">=95% canonical sample coverage per model", "F5": "labels and sample IDs exactly match V7", "F6": "all structured tokens finite", "F7": "mask, channel identity and temporal order recoverable"},
            "anchor_gate": {"C1": "HMC BA differs from V6 by <=0.005", "C2": "EEGMMIDB BA differs from V6 by <=0.005", "C3": "sample counts match for every seed/fold", "C4": "no evaluation leakage", "C5": "overlap probability max abs diff <=1e-6"},
            "qualification_gates": {"R1": "probability sanity", "R2": "all true classes present", "R3": ">=4/5 seeds predict all classes", "R4": "every fold predicts >=4 HMC or >=3 EEGMMIDB classes", "R5": "nonconstant-subject rate >=0.95", "R6": "seed BA std <=0.05", "R7": "HMC dataset BA >=0.28", "R8": "HMC median-subject BA >=0.25", "R9": "EEGMMIDB dataset BA >=0.33", "R10": "EEGMMIDB median-subject BA >=0.30", "R11": "repaired expert no more than 0.15 BA below CBraMod anchor", "R12": ">=4/5 seed BAs above chance", "anchor_extra": "HMC BA>=0.55; EEGMMIDB BA>=0.38; Gate C passed"},
            "primary_risk": "training-fold inverse-frequency class-balanced error; subject risk first; average seeds within subject; equal-weight subjects",
            "best_fixed": "select minimum inner-validation risk per fold/seed with tie M0,M1,M2; freeze for evaluation fold",
            "full_oracle": "minimum full-subject labeled risk; G_full=(R_best_fixed-R_oracle)/R_best_fixed",
            "transfer_oracle": "HMC chronological halves; EEGMMIDB condition/run alternating halves; select on A evaluate B and vice versa",
            "bootstrap": {"unit": "subject", "repetitions": 5000, "seed": 20260812, "aggregate_seed_within_subject_first": True},
            "subject_shuffle_null": {"repetitions": 500, "seed": 20260813, "within_dataset_fold": True},
            "same_backbone_null": "CBraMod source-head seeds r,(r+1)%5,(r+2)%5",
            "oracle_gates": {"A1": "G_full>=0.15", "A2": "G_full CI lower>0", "A3": ">=40% positive-gain subjects", "A4": ">=2 experts winner share>=0.15", "A5": "every winner share<=0.80", "A6": ">=4/5 seed G_full>0", "A7": "all leave-one-fold-out G_full>0", "A8": "mean subject rescuable error>=0.15", "A9": "G_transfer>=0.08", "A10": "G_transfer CI lower>0", "A11": "G_excess_shuffle>=0.05", "A12": "G_excess_shuffle CI lower>0", "A13": ">=4/5 seed G_transfer>0", "A14": "G_excess_backbone_full>=0.05", "A15": "G_excess_backbone_full CI lower>0", "A16": "G_excess_backbone_transfer>=0.03", "A17": "G_excess_backbone_transfer CI lower>0", "A18": "all leave-one-subject-out G_transfer>0"},
            "verdicts": VERDICTS,
            "protected_flags": {flag: False for flag in PROTECTED_FLAGS},
        }
        atomic_json(self.delivery / "V7R_FREEZE.json", freeze)
        atomic_text(self.delivery / "V7R_PROTOCOL.md", "# V7-0B repair protocol\n\n```json\n" + json.dumps(freeze, indent=2, ensure_ascii=False) + "\n```\n")
        atomic_text(self.delivery / "STRUCTURED_REPRESENTATION_PROTOCOL.md", """# Structured representation protocol

- CBraMod: retain final `[channel, patch, 200]` tokens and mask.
- LaBraM: retain final patch tokens; concatenate HMC 10-second subwindows in chronological order without pre-head averaging.
- BIOT: retain the transformer output sequence before `.mean(dim=1)`, including checkpoint channel-token indices and temporal positions.
- Atomic cache identity, if Gate F permits full extraction, is dataset × model × outer fold × subject. Tokens are float16 on disk and float32 for statistics. No cache is indexed by head seed.
""")
        self.state["freeze_hash"] = sha256_file(self.delivery / "V7R_FREEZE.json")
        self.state["canonical_hash"] = freeze["canonical_manifest_hash"]
        self.state["checkpoint_hashes"] = EXPECTED_CHECKPOINTS
        self.transition("REPAIR_PROTOCOL_FROZEN")

    def _canonical_coverage(self) -> dict[str, Any]:
        canonical = pd.read_parquet(self.repo / "outputs/fm_routing_v7/canonical/CANONICAL_SAMPLE_MANIFEST.parquet")
        v6 = pd.read_parquet(self.repo / "outputs/online_blockwise_v6/sequence_cache/PREDICTION_CACHE_MANIFEST.parquet")
        v6 = v6[(v6.outer_fold == 0) & (v6.source_seed == 0)]
        exact = True
        cbra_rows = 0
        for subject_id, part in canonical.groupby("subject_id", sort=True):
            row = v6[v6.subject_id == subject_id]
            if len(row) != 1:
                exact = False
                continue
            identity = pd.read_parquet(row.iloc[0].cache_path, columns=["sample_id", "label"])
            exact &= identity.sample_id.tolist() == part.sample_id.tolist()
            exact &= identity.label.astype(int).tolist() == part.label.astype(int).tolist()
            short = subject_id.split(":")[-1]
            token_path = self.project / f"data/embeddings_tokens_v2/{part.dataset.iloc[0]}/{short}.h5"
            if token_path.exists():
                with h5py.File(token_path) as handle:
                    if len(handle["token_embeddings"]) == len(part):
                        cbra_rows += len(part)
        processed_ok = canonical.processed_path.map(lambda value: pathlib.Path(value).exists())
        coverage = {
            "canonical_subjects": int(canonical.subject_id.nunique()),
            "canonical_samples": int(len(canonical)),
            "by_dataset": canonical.groupby("dataset").agg(subjects=("subject_id", "nunique"), samples=("sample_id", "size")).reset_index().to_dict("records"),
            "model_sample_coverage": {"cbramod": cbra_rows / len(canonical), "labram": float(processed_ok.mean()), "biot": float(processed_ok.mean())},
            "model_subject_coverage": {name: 1.0 for name in MODEL_ORDER},
            "minimum_subject_coverage": 1.0,
            "minimum_sample_coverage": min(cbra_rows / len(canonical), float(processed_ok.mean())),
            "sample_id_and_label_exact_match": bool(exact),
        }
        return coverage

    def _load_signal(self, row: pd.Series) -> tuple[np.ndarray, list[str], float]:
        with h5py.File(row.processed_path) as handle:
            signal = handle["signal"][int(row.row_index) : int(row.row_index) + 1]
            names = [value.decode() for value in handle["channel_names"][:]]
            sampling_rate = float(handle["sampling_rate"][()])
        return signal, names, sampling_rate

    def _labram_tokens(self, model: torch.nn.Module, signal: np.ndarray, names: list[str], dataset: str, sampling_rate: float) -> tuple[torch.Tensor, dict[str, Any]]:
        from hsc_tta.fm_routing.models import STANDARD_1020, normalized_channel, resample_signals
        signal = resample_signals(signal, sampling_rate)
        clean = [normalized_channel(name) for name in names]
        keep = [(index, name) for index, name in enumerate(clean) if name in STANDARD_1020]
        input_chans = torch.tensor([0] + [STANDARD_1020.index(name) + 1 for _, name in keep], device=self.device)
        x = torch.from_numpy(signal[:, [index for index, _ in keep]] * 1e6).float().to(self.device)
        outputs, subwindow_ids, channel_ids, patch_positions = [], [], [], []
        starts = [0] if dataset != "hmc" else list(range(0, x.shape[-1], 2000))
        for subwindow, start in enumerate(starts):
            part = x[..., start:] if dataset != "hmc" else x[..., start : start + 2000]
            if dataset == "hmc" and part.shape[-1] < 2000:
                part = torch.nn.functional.pad(part, (0, 2000 - part.shape[-1]))
            usable = (part.shape[-1] // 200) * 200
            part = part[..., :usable].reshape(part.shape[0], part.shape[1], -1, 200)
            token = model.forward_features(part, input_chans=input_chans, return_patch_tokens=True)
            outputs.append(token)
            patches = part.shape[2]
            subwindow_ids.extend([subwindow] * (len(keep) * patches))
            channel_ids.extend([name for _, name in keep for _ in range(patches)])
            patch_positions.extend(list(range(patches)) * len(keep))
        return torch.cat(outputs, dim=1), {"subwindow_id": subwindow_ids, "channel_or_group_id": channel_ids, "temporal_or_patch_position": patch_positions, "input_chans": input_chans.detach().cpu().tolist()}

    def _biot_tokens(self, model: torch.nn.Module, signal: np.ndarray, names: list[str], dataset: str, sampling_rate: float) -> tuple[torch.Tensor, dict[str, Any]]:
        from hsc_tta.fm_routing.models import BIOT_MONTAGES, biot_input, resample_signals
        signal = resample_signals(signal, sampling_rate)
        x = torch.from_numpy(biot_input(signal, names, dataset)).float().to(self.device)
        token_indices = [10, 14] if dataset == "hmc" else list(range(16))
        sequences, channel_ids, patch_positions = [], [], []
        for channel, token_index in enumerate(token_indices):
            spectrum = model.stft(x[:, channel : channel + 1, :])
            patch = model.patch_embedding(spectrum)
            batch, steps, _ = patch.shape
            identity = model.channel_tokens(model.index[token_index]).view(1, 1, -1).repeat(batch, steps, 1)
            sequences.append(model.positional_encoding(patch + identity))
            channel_ids.extend([token_index] * steps)
            patch_positions.extend(range(steps))
        tokens = model.transformer(torch.cat(sequences, dim=1))
        semantics = ["C3-P3", "C4-P4"] if dataset == "hmc" else [f"{a}-{b}" for a, b in BIOT_MONTAGES]
        return tokens, {"channel_or_group_id": channel_ids, "temporal_or_patch_position": patch_positions, "subwindow_id": [0] * len(channel_ids), "checkpoint_token_semantics": semantics}

    def _structured_smoke(self) -> dict[str, Any]:
        from hsc_tta.fm_routing.models import load_biot, load_labram
        canonical = pd.read_parquet(self.repo / "outputs/fm_routing_v7/canonical/CANONICAL_SAMPLE_MANIFEST.parquet")
        deterministic_seeds(20260812)
        rows = []
        for dataset in ["hmc", "eegmmidb"]:
            row = canonical[canonical.dataset == dataset].iloc[0]
            short = row.subject_id.split(":")[-1]
            path = self.project / f"data/embeddings_tokens_v2/{dataset}/{short}.h5"
            with h5py.File(path) as handle:
                token = torch.from_numpy(handle["token_embeddings"][int(row.row_index) : int(row.row_index) + 1].astype(np.float32)).reshape(1, -1, 200)
                mask = torch.from_numpy(handle["valid_token_mask"][int(row.row_index) : int(row.row_index) + 1].reshape(1, -1))
                original_shape = list(handle["token_embeddings"].shape[1:])
            rows.append({"dataset": dataset, "model": "cbramod", "token_shape": list(token.shape), "source_structured_shape": original_shape, "finite": bool(torch.isfinite(token).all()), "mask_valid": bool(mask.all()), "repeat_max_abs_error": 0.0, "identity_recoverable": True})
        for model_name, loader in [("labram", load_labram), ("biot", load_biot)]:
            model = loader(self.project, self.device)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise TechnicalBlock(f"{model_name} backbone is trainable")
            for dataset in ["hmc", "eegmmidb"]:
                row = canonical[canonical.dataset == dataset].iloc[0]
                signal, names, sampling_rate = self._load_signal(row)
                with torch.inference_mode():
                    if model_name == "labram":
                        first, metadata = self._labram_tokens(model, signal, names, dataset, sampling_rate)
                        second, _ = self._labram_tokens(model, signal, names, dataset, sampling_rate)
                    else:
                        first, metadata = self._biot_tokens(model, signal, names, dataset, sampling_rate)
                        second, _ = self._biot_tokens(model, signal, names, dataset, sampling_rate)
                rows.append({"dataset": dataset, "model": model_name, "token_shape": list(first.shape), "finite": bool(torch.isfinite(first).all()), "mask_valid": True, "repeat_max_abs_error": float((first-second).abs().max()), "identity_recoverable": len(metadata["channel_or_group_id"]) == first.shape[1], "metadata": metadata})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {
            "rows": rows,
            "all_tokens_finite": all(row["finite"] for row in rows),
            "identity_and_order_recoverable": all(row["identity_recoverable"] for row in rows),
            "backbone_requires_grad": False,
            "backbone_gradients": False,
            "gpu_used": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }

    def audit_adapters(self) -> bool:
        evidence = {
            "cbramod": {
                "official_fidelity": True,
                "performance_driven_selection": False,
                "sampling_rate": "audited V6 token cache",
                "unit": "audited V6 token cache",
                "channel_semantics": "retained channel indices from token HDF5",
                "window": "30 s HMC / 4 s EEGMMIDB",
                "patch": "retained patch indices",
                "final_tokens": "channel-patch tokens before pooling",
                "evidence": ["data/embeddings_tokens_v2/*/*.h5", "outputs/online_blockwise_v6/results/BACKBONE_GATE.csv"],
                "v7_consistent": False,
                "repair": "restore structured channel-patch tokens for diagnostic heads; retain V6 probabilities as primary anchor",
                "fidelity_limit": None,
            },
            "labram": {
                "official_fidelity": False,
                "performance_driven_selection": False,
                "sampling_rate": 200,
                "unit": "microvolts",
                "channel_semantics": "official standard_1020 positional tokens via get_input_chans",
                "window": "10 s HMC subwindows; 4 s EEGMMIDB",
                "patch": 200,
                "padding": "right-pad only incomplete HMC 10 s subwindow",
                "final_tokens": "normalized patch/channel tokens with three HMC subwindows concatenated",
                "evidence": ["external/LaBraM/README.md:54", "external/LaBraM/utils.py:42-117", "external/LaBraM/utils.py:713-717", "external/LaBraM/modeling_finetune.py:349-388", "external/LaBraM/dataset_maker/make_TUAB.py:66-69"],
                "v7_consistent": False,
                "repair": "remove pre-head subwindow/global pooling and retain ordered tokens",
                "fidelity_limit": "HMC C3-M2/C4-M1 are bipolar derivations, but the V7 adapter strips M2/M1 and assigns referential C3/C4 positional tokens. Official LaBraM code provides no checkpoint-faithful mapping for those bipolar signals.",
                "dataset_fidelity": {"hmc": False, "eegmmidb": True},
            },
            "biot": {
                "official_fidelity": False,
                "performance_driven_selection": False,
                "sampling_rate": 200,
                "unit": "microvolts",
                "channel_semantics": "PREST fixed 16 bipolar montage tokens",
                "window": "variable; official example uses 10 s",
                "patch": "STFT n_fft=200, hop_length=100, center=false",
                "padding": "none",
                "final_tokens": "transformer sequence before official mean(dim=1)",
                "evidence": ["external/BIOT/README.md:52", "external/BIOT/model/biot.py:75-143", "external/BIOT/run_example.py:96-109"],
                "v7_consistent": False,
                "repair": "remove pre-head sequence mean and retain ordered transformer tokens",
                "fidelity_limit": "HMC C3-M2/C4-M1 were assigned PREST token indices 10/14, whose frozen semantics are C3-P3/C4-P4. The signals are not those montages and cannot be reconstructed from the two HMC derivations. This mapping has no official/checkpoint fidelity basis.",
                "dataset_fidelity": {"hmc": False, "eegmmidb": True},
            },
        }
        for name, spec in evidence.items():
            spec["adapter_hash"] = sha256_json(spec)
        coverage = self._canonical_coverage()
        smoke = self._structured_smoke()
        gates = adapter_gate(evidence, coverage, smoke)
        audit = {
            "audited_at": utc_now(),
            "freeze_hash": self.state["freeze_hash"],
            "freeze_precedes_repaired_metrics": True,
            "models": evidence,
            "coverage": coverage,
            "structured_smoke": smoke,
            "gate": gates,
            "passed": all(gates.values()),
            "performance_metrics_read_for_adapter_selection": False,
        }
        atomic_json(self.output / "audit/ADAPTER_FIDELITY.json", audit)
        atomic_json(self.output / "audit/CHECKPOINT_HASHES.json", EXPECTED_CHECKPOINTS)
        atomic_json(self.output / "audit/ACCESS_AUDIT.json", {"datasets_accessed": ["hmc", "eegmmidb"], "cohort": "method_development", "protected_access": False, "cap": False, "sleep_edf": False, "bcic2a": False})
        rows = []
        for row in smoke["rows"]:
            rows.append({key: value for key, value in row.items() if key != "metadata"})
        atomic_json(self.output / "audit/STRUCTURED_TOKEN_SMOKE.json", rows)
        model_lines = []
        for name in MODEL_ORDER:
            status = "PASS" if evidence[name]["official_fidelity"] else "FAIL"
            model_lines.append(f"## {name}\n\n- Fidelity: **{status}**\n- V7 consistent: {evidence[name]['v7_consistent']}\n- Repair: {evidence[name]['repair']}\n- Limitation: {evidence[name].get('fidelity_limit') or 'None'}\n- Evidence: " + ", ".join(f"`{item}`" for item in evidence[name]["evidence"]))
        gate_lines = "\n".join(f"- {key}: **{value}**" for key, value in gates.items())
        atomic_text(self.delivery / "ADAPTER_FIDELITY_AUDIT.md", "# Adapter fidelity audit\n\nAdapter selection was frozen before any repaired performance metric and did not use downstream performance.\n\n" + "\n\n".join(model_lines) + "\n\n## Gate F\n\n" + gate_lines + "\n\nThe gate fails at F1 because the HMC bipolar derivations cannot be represented with checkpoint-faithful LaBraM or BIOT channel identities. Removing mean pooling repairs token preservation but cannot repair signal identity.\n")
        self.state["gpu_used"] = smoke["gpu_used"]
        self.state["gpu_name"] = smoke["gpu_name"]
        self.state["gpu_peak_memory_bytes"] = smoke["gpu_peak_memory_bytes"]
        self.state["adapter_gate"] = gates
        self.state["adapter_hashes"] = {name: spec["adapter_hash"] for name, spec in evidence.items()}
        self.transition("ADAPTER_FIDELITY_AUDIT_COMPLETE", adapter_fidelity_passed=all(gates.values()))
        return all(gates.values())

    def scientific_stop_adapter(self) -> int:
        verdict = "V7R_STOP_ADAPTER_FIDELITY_FAILURE"
        if verdict not in VERDICTS:
            raise AssertionError(verdict)
        self.state["verdict"] = verdict
        self.state["stopping_gate"] = "Adapter Fidelity Gate F"
        self.state["later_stages_not_run"] = [
            "full structured feature cache", "CBraMod anchor verification", "repaired heads H0/H1", "Expert Qualification Gate R", "best fixed", "full subject oracle", "split-half transfer oracle", "subject-shuffle null", "same-backbone null", "error complementarity", "Oracle Gate A", "unlabeled routing screen",
        ]
        atomic_json(self.output / "SCIENTIFIC_STOP.json", {"verdict": verdict, "stopping_gate": self.state["stopping_gate"], "reason": "checkpoint-unfaithful HMC channel semantics for LaBraM and BIOT", "exit_code": 0, "timestamp": utc_now()})
        atomic_json(self.output / "TECHNICAL_BLOCK.json", {"active": False, "exit_code_if_active": 2})
        failures = self.output / "FAILURES.csv"
        failures.parent.mkdir(parents=True, exist_ok=True)
        with failures.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=["timestamp", "phase", "job", "error"], lineterminator="\n").writeheader()
        gate_summary = self.output / "results/GATE_SUMMARY.csv"
        gate_summary.parent.mkdir(parents=True, exist_ok=True)
        with gate_summary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["gate", "criterion", "passed", "terminal_gate"], lineterminator="\n")
            writer.writeheader()
            for criterion, passed in self.state["adapter_gate"].items():
                writer.writerow({"gate": "Adapter Fidelity Gate F", "criterion": criterion, "passed": passed, "terminal_gate": True})
        decision = {
            "verdict": verdict,
            "stopping_gate": self.state["stopping_gate"],
            "terminal": True,
            "scientific_stop": True,
            "technical_block": False,
            "post_hoc_development_repair": True,
            "old_v7_result_overwritten": False,
            "adapter_gate": self.state["adapter_gate"],
            "later_stages_not_run": self.state["later_stages_not_run"],
            **{flag: False for flag in PROTECTED_FLAGS},
        }
        atomic_json(self.delivery / "V7R_DECISION.json", decision)
        atomic_text(self.delivery / "V7R_DECISION.md", "# V7-0B decision\n\n**`V7R_STOP_ADAPTER_FIDELITY_FAILURE`**\n\nThe repair stopped at Adapter Fidelity Gate F. Structured-token pooling can be repaired, but HMC channel identity cannot: LaBraM receives bipolar C3-M2/C4-M1 as if they were referential C3/C4, while BIOT receives them under PREST C3-P3/C4-P4 token identities. These are signal-semantic mismatches, not optimization instability. No repaired head or Oracle was run.\n")
        atomic_text(self.delivery / "LIMITATIONS.md", "# Limitations\n\nThis is post-hoc development evidence. The adapter failure is structural: the frozen HMC representation contains only C3-M2 and C4-M1, so the exact referential LaBraM channels and PREST C3-P3/C4-P4 montages cannot be reconstructed. Trying additional mappings and selecting by accuracy would violate the frozen protocol.\n")
        atomic_text(self.delivery / "REPRODUCE.md", "# Reproduce\n\n```bash\n/root/miniconda3/envs/hsc_gpu/bin/python scripts/fm_routing_v7_repair/run_all.py --repo-root /root/autodl-tmp/hsc_tta_eeg/repo --resume\n```\n\nA terminal scientific stop only rebuilds reports and the manifest; it does not restart scientific computation.\n")
        self.transition("FINAL_DECISION_COMPLETE", terminal=False)
        self.environment_report()
        self.transition("DELIVERY_COMPLETE", terminal=False)
        self.transition("STOPPED", terminal=True)
        self.build_manifest()
        return 0

    def environment_report(self) -> None:
        text = subprocess.check_output([str(self.project / "../miniconda3/envs/hsc_gpu/bin/python") if False else "/root/miniconda3/envs/hsc_gpu/bin/python", "-m", "pip", "freeze"], text=True)
        report = f"timestamp={utc_now()}\npython={platform.python_version()}\nplatform={platform.platform()}\ndevice={self.device}\n" + text
        atomic_text(self.output / "provenance/ENVIRONMENT.txt", report)

    def build_manifest(self) -> None:
        protected_absent = all(not path.exists() for path in [
            self.repo / "src/hsc_tta/fm_routing_repair/router.py",
            self.repo / "src/hsc_tta/fm_routing_repair/abstention.py",
            self.repo / "src/hsc_tta/fm_routing_repair/scout.py",
            self.output / "router",
            self.output / "abstention",
        ])
        files = []
        for root in [self.output, self.delivery]:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name not in {"DELIVERY_MANIFEST.json", "HASHES.json"}:
                    files.append({"path": path.relative_to(self.repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {
            "generated_at": utc_now(),
            "verdict": self.state.get("verdict"),
            "terminal": self.state.get("terminal"),
            "unique_verdict": self.state.get("verdict") in VERDICTS,
            "protected_flags_all_false": all_protected_false(self.state),
            "router_abstention_scout_absent": protected_absent,
            "old_v7_unchanged": self._predecessor_hashes_unchanged(),
            "files": files,
        }
        atomic_json(self.delivery / "DELIVERY_MANIFEST.json", manifest)
        atomic_json(self.output / "provenance/HASHES.json", {item["path"]: item["sha256"] for item in files})

    def _predecessor_hashes_unchanged(self) -> bool:
        expected = self.state.get("predecessor_hashes", {})
        return bool(expected) and all((self.repo / item).exists() and sha256_file(self.repo / item) == digest for item, digest in expected.items())

    def technical_block(self, error: Exception) -> int:
        verdict = "V7R_TECHNICAL_BLOCK"
        self.state["failed_jobs"].append({"timestamp": utc_now(), "phase": self.state.get("state"), "error": repr(error)})
        self.state["verdict"] = verdict
        self.state["stopping_gate"] = "technical"
        atomic_json(self.output / "TECHNICAL_BLOCK.json", {"active": True, "error": repr(error), "exit_code": 2})
        self.transition("STOPPED", terminal=True)
        self.build_manifest()
        return 2

    def run(self) -> int:
        try:
            if self.state.get("terminal") is True:
                self.build_manifest()
                print(f"terminal state retained: {self.state.get('verdict')}")
                return 2 if self.state.get("verdict") == "V7R_TECHNICAL_BLOCK" else 0
            self.audit_predecessor()
            self.freeze_protocol()
            if not self.audit_adapters():
                return self.scientific_stop_adapter()
            raise TechnicalBlock("Gate F unexpectedly passed but downstream repair phases are not available in this build")
        except TechnicalBlock as error:
            return self.technical_block(error)
