from __future__ import annotations

"""PERSIST-EEG Experiment 4: protection-first representation updating.

The implementation is deliberately self-contained and fail-closed.  It uses
only the frozen WBCIC development cohort during audit/development, trains a
matched embedding-space residual adapter, and refuses the sealed outer phase
until a development gate and final protocol lock exist.
"""

import argparse
import bisect
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# A development iteration may be evaluated in a separate result root while
# retaining the same executable.  This prevents a later hypothesis from
# overwriting the signed V1 artifacts.
EXP_ROOT = Path(os.environ.get("PERSIST_EXP4_ROOT", str(Path(__file__).resolve().parents[1])))
OUT = EXP_ROOT / "results"
PROTOCOL = EXP_ROOT / "protocol"
FIGURES = EXP_ROOT / "figures"
CHECKPOINTS = EXP_ROOT / "checkpoints"

WBCIC_ROOT = Path(os.environ.get("PERSIST_WBCIC_ROOT", r"D:\nips-temp\TotalP\P1\CRCICLR_WBCIC_EEGNET"))
ACTION_ROOT = WBCIC_ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2"
ACTION_OUT = ACTION_ROOT / "outputs"
CACHE = ACTION_OUT / "cache" / "wbcic_epochs"
ANCHOR_SOURCE = ACTION_OUT / "model" / "audit"
SCOPE_PATH = ACTION_OUT / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
CACHE_AUDIT_PATH = ACTION_OUT / "protocol" / "CACHE_SCOPE_AUDIT.json"
ACTION_LOCK_PATH = ACTION_OUT / "protocol" / "ACTIONABILITY_PROTOCOL_LOCK.json"
BLOCK_ASSIGNMENTS = ACTION_OUT / "results" / "BLOCK_ASSIGNMENTS.csv"

RAW_ROOT = Path(os.environ.get("PERSIST_WBCIC_RAW_ROOT", r"D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids"))
RUNS = tuple(range(5))
SESSIONS = (0, 1, 2)
DIM = 32
RANK = 4
BOOTSTRAP_DRAWS = 10_000
PERMUTATION_DRAWS = 100_000
EPS = 1e-12
IMPLEMENTATION_ID = os.environ.get("PERSIST_EXP4_IMPLEMENTATION_ID", "persist_eeg_exp4_protection_first_v1_hard_projection")
RESPONSE_STRENGTH = float(os.environ.get("PERSIST_EXP4_RESPONSE_STRENGTH", "0.0"))
if not 0.0 <= RESPONSE_STRENGTH <= 1.0:
    raise ValueError("PERSIST_EXP4_RESPONSE_STRENGTH must be in [0,1]")
EXPERIMENT_SEED = 20260823
BLOCKS = (("P01_04", 0, 4), ("P05_08", 4, 8), ("P09_16", 8, 16), ("P17_32", 16, 32))
PROTECTED_BLOCK = "P01_04"
PERSISTENCE_CONTROL_BLOCK = "P05_08"
METHODS = ("Frozen", "Generic", "RandomGuard", "PCAGuard", "PersistenceGuard", "IdentityGuard", "PERSISTGuard")


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    part = path.with_suffix(path.suffix + ".part")
    frame.to_csv(part, index=False)
    part.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def sha_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def git_head() -> str | None:
    forced = os.environ.get("PERSIST_EXP4_GIT_COMMIT")
    if forced:
        return forced.strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXP_ROOT.parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def flags() -> dict[str, bool]:
    return {"outer_test_used": False, "outer_subject_ids_opened": False, "outer_membership_enumerated": False}


def prepare_dirs() -> None:
    for path in (OUT, PROTOCOL, FIGURES, CHECKPOINTS):
        path.mkdir(parents=True, exist_ok=True)


def load_scope() -> dict[str, Any]:
    if not SCOPE_PATH.is_file() or not CACHE_AUDIT_PATH.is_file() or not ACTION_LOCK_PATH.is_file():
        raise RuntimeError("WBCIC_DEVELOPMENT_PROTOCOL_MISSING")
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    cache_audit = json.loads(CACHE_AUDIT_PATH.read_text(encoding="utf-8"))
    action_lock = json.loads(ACTION_LOCK_PATH.read_text(encoding="utf-8"))
    subjects = [str(x) for x in scope.get("allowed_subjects", [])]
    if len(subjects) != 41 or len(set(subjects)) != 41:
        raise RuntimeError("DATA_SCOPE_VIOLATION: expected 41 development subjects")
    if scope.get("outer_subject_ids_present") is not False or cache_audit.get("outer_subject_ids_materialized") is not False:
        raise RuntimeError("DATA_SCOPE_VIOLATION: sealed outer materialized")
    if cache_audit.get("status") != "DEVELOPMENT_CACHE_COMPLETE" or cache_audit.get("allowed_subject_count") != 41:
        raise RuntimeError("DEVELOPMENT_CACHE_INCOMPLETE")
    if action_lock.get("outer_test_state") != "OUTER_TEST_LOCKED":
        raise RuntimeError("OUTER_LOCK_INVALID")
    for fold in RUNS:
        role = scope.get("audit_roles", {}).get(str(fold), {})
        outcome = set(map(str, role.get("outcome", [])))
        discovery = set(map(str, role.get("discovery_decision", [])))
        fit = set(map(str, role.get("model_fit", [])))
        if outcome & discovery or outcome & fit or discovery & fit or outcome | discovery | fit != set(subjects):
            raise RuntimeError(f"DATA_SCOPE_VIOLATION: fold roles {fold}")
    return scope


def load_assignments() -> pd.DataFrame:
    frame = pd.read_csv(BLOCK_ASSIGNMENTS)
    required = {"block", "rank", "assignment", "protected_utility_gate"}
    if not required.issubset(frame.columns):
        raise RuntimeError("FROZEN_BLOCK_ASSIGNMENTS_MISSING_COLUMNS")
    p = frame.loc[frame.block.astype(str) == PROTECTED_BLOCK]
    if len(p) != 1 or str(p.iloc[0].assignment) != "PROTECTED" or not bool(p.iloc[0].protected_utility_gate):
        raise RuntimeError("FROZEN_PROTECTED_BLOCK_MISMATCH")
    q = frame.loc[frame.block.astype(str) == PERSISTENCE_CONTROL_BLOCK]
    if len(q) != 1 or str(q.iloc[0].assignment) == "PROTECTED":
        raise RuntimeError("FROZEN_PERSISTENCE_CONTROL_MISMATCH")
    return frame


def phase_audit() -> dict[str, Any]:
    prepare_dirs()
    scope = load_scope()
    assignments = load_assignments()
    source_paths = [SCOPE_PATH, CACHE_AUDIT_PATH, ACTION_LOCK_PATH, BLOCK_ASSIGNMENTS]
    protocol = {
        "experiment": "PERSIST_EEG_EXP4_PROTECTION_FIRST_FINAL",
        "status": "DEV_PROTOCOL_FROZEN_BEFORE_EXP4_PRIMARY_COMPARISON",
        "implementation_id": IMPLEMENTATION_ID,
        "git_commit": git_head(),
        "dataset": "WBCIC/Yang2025/NEMAR nm000348",
        "development_subject_count": 41,
        "outer_subject_count": 10,
        "outer_subject_ids_present": False,
        "outer_evaluation_authorized_once": False,
        "scope": "five frozen subject-disjoint development folds; S3 is outcome only for the fold outcome subjects",
        "fold_roles": "existing frozen scope: outcome Fk; discovery/decision F(k+1); model-fit remaining three",
        "session_protocol": "S1=ses-0, S2=ses-1, S3=ses-2; no trial-level split",
        "deployment_selection": "sequential S1 anchor -> S2 global adapter update -> unseen-subject S3 evaluation",
        "anchor": {"backbone": "EEGNet", "training_sessions": [0], "dropout": 0.25, "learning_rate": 0.0003, "weight_decay": 0.0005, "epochs": 30, "batch_size": 64},
        "adapter": {"architecture": "zero-initialized linear residual A(h)=hW+b", "embedding_dim": DIM, "same_parameter_count_all_methods": True, "head": "frozen anchor linear classifier", "training_session": 1},
        "protected_basis_rule": "fold-specific S1/S2 cross-session subject-centroid persistence basis; frozen P01_04 rank-4 rule; no held-out outcome S3 labels",
        "persist_guard_variant": {"mode": "hard complement projection plus minimum-norm frozen-head decision-response correction on U_P", "response_strength": RESPONSE_STRENGTH, "extra_parameters": 0},
        "protected_block": {"name": PROTECTED_BLOCK, "rank": RANK, "source_assignment": "frozen WBCIC actionability v2 protected_utility_gate"},
        "controls": {"random": "three deterministic same-rank orthonormal draws", "pca": "top-rank discovery S1/S2 covariance directions", "persistence": f"frozen non-Protected rank-matched block {PERSISTENCE_CONTROL_BLOCK}", "identity": "top-rank discovery subject-centroid identity covariance directions"},
        "generic_candidates": [
            {"id": "GEN_LINEAR_LR3E4_E25", "learning_rate": 3e-4, "epochs": 25, "weight_decay": 5e-4},
            {"id": "GEN_LINEAR_LR1E3_E25", "learning_rate": 1e-3, "epochs": 25, "weight_decay": 5e-4},
            {"id": "GEN_LINEAR_LR3E4_E40", "learning_rate": 3e-4, "epochs": 40, "weight_decay": 5e-4},
        ],
        "generic_selection": "S2 subject-held-out validation inside model-fit subjects; no outcome S3 labels",
        "primary_endpoint": "subject balanced accuracy on unseen outcome-subject S3",
        "secondary_endpoints": ["macro-F1", "accuracy", "negative transfer relative to Frozen", "protected-coordinate drift", "protected decision-response drift", "complement adaptation"],
        "primary_statistics": "paired subject delta Guard-Generic; subject bootstrap CI; exact binomial sign test and deterministic Monte-Carlo sign-flip p for mean",
        "development_gate": {"minimum_practical_delta_BA": 0.005, "negative_transfer_rate_reduction": 0.05, "protected_drift_ratio_max": 0.25, "outer_requires_all_controls_and_mechanism": True},
        "no_outer_access_during_development": True,
        "validation_outcome_used_for_design": False,
        **flags(),
    }
    write_json(PROTOCOL / "EXP4_DEV_PROTOCOL.json", protocol)
    provenance = {
        "status": "PROVENANCE_AUDIT_PASS",
        "git_commit": git_head(),
        "source_hashes": {str(path.relative_to(WBCIC_ROOT)): sha256_file(path) for path in source_paths},
        "source_paths": [str(path) for path in source_paths],
        "development_subject_count": len(scope["allowed_subjects"]),
        "frozen_assignments": assignments.to_dict(orient="records"),
        "protected_assignment": PROTECTED_BLOCK,
        "outer_subject_ids_opened": False,
        **flags(),
    }
    write_json(PROTOCOL / "PROVENANCE_AUDIT.json", provenance)
    write_json(PROTOCOL / "OUTER_ACCESS_LOCK.json", {"status": "OUTER_SEALED", "outer_subject_ids_present": False, "outer_evaluation_authorized": False, "outer_result_exists": False, **flags()})
    return protocol


class EpochDataset(Dataset):
    def __init__(self, subjects: Sequence[str], sessions: Sequence[int]):
        self.records: list[dict[str, Any]] = []
        self.ends: list[int] = []
        total = 0
        for sid, subject in enumerate(map(str, subjects)):
            for session in sessions:
                ep = CACHE / subject / f"ses-{session}_epochs.npy"
                lp = CACHE / subject / f"ses-{session}_labels.npy"
                if not ep.is_file() or not lp.is_file():
                    raise FileNotFoundError(f"missing development cache {subject} ses-{session}")
                labels = np.load(lp, allow_pickle=False).astype(np.int64)
                shape = np.load(ep, mmap_mode="r", allow_pickle=False).shape
                if shape != (len(labels), 58, 1000) or set(labels.tolist()) != {0, 1}:
                    raise RuntimeError(f"malformed cache {subject} ses-{session}")
                self.records.append({"epochs": ep, "labels": labels, "sid": sid, "session": int(session), "start": total})
                total += len(labels)
                self.ends.append(total)
        self.total = total
        self.arrays: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        ridx = bisect.bisect_right(self.ends, int(index))
        rec = self.records[ridx]
        local = int(index) - int(rec["start"])
        if ridx not in self.arrays:
            self.arrays[ridx] = np.load(rec["epochs"], mmap_mode="r", allow_pickle=False)
        x = np.array(self.arrays[ridx][local], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.tensor(int(rec["labels"][local])), torch.tensor(int(rec["sid"])), torch.tensor(int(rec["session"]))


class EEGNet(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (58, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.separable_depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.separable_point = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Linear(16 * 31, DIM)
        self.embedding_norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(torch.nn.functional.elu(self.bn2(self.spatial(value)))))
        value = self.separable_depth(value)
        value = self.separable_point(value)
        value = self.drop2(self.pool2(torch.nn.functional.elu(self.bn3(value))))
        return self.embedding_norm(torch.nn.functional.elu(self.embedding(value.flatten(1))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def model_state_sha(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_anchor(subjects: Sequence[str], sessions: Sequence[int], fold: int, device: torch.device, epochs: int = 30) -> tuple[EEGNet, dict[str, Any]]:
    seed = stable_seed(EXPERIMENT_SEED, "anchor", fold)
    seed_all(seed)
    model = EEGNet(0.25).to(device)
    # The WBCIC Windows/torch environment has a documented worker-spawn
    # deadlock.  Keep workers at zero for deterministic, fail-closed runs;
    # pinned host batches still overlap transfers on CUDA.
    loader = DataLoader(EpochDataset(subjects, sessions), batch_size=64, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    history = []
    for epoch in range(epochs):
        model.train(); total = 0.0; correct = 0; seen = 0
        for x, y, _, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite anchor loss")
            loss.backward(); opt.step()
            total += float(loss.detach()) * len(y); correct += int((logits.detach().argmax(1) == y).sum()); seen += len(y)
        history.append({"epoch": epoch + 1, "loss": total / max(seen, 1), "accuracy": correct / max(seen, 1)})
        print(f"[anchor fold={fold}] epoch={epoch + 1}/{epochs} loss={history[-1]['loss']:.5f} acc={history[-1]['accuracy']:.4f}", flush=True)
    model.eval()
    payload = {"implementation_id": IMPLEMENTATION_ID, "fold": fold, "train_subjects": list(subjects), "train_sessions": list(sessions), "config": {"dropout": 0.25, "learning_rate": 3e-4, "weight_decay": 5e-4, "epochs": epochs, "batch_size": 64}, "seed": seed, "model_state_sha256": model_state_sha(model), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "history": history}
    path = CHECKPOINTS / f"anchor_fold-{fold}.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path); payload["checkpoint_sha256"] = sha256_file(path)
    return model, payload


def load_anchor(path: Path, device: torch.device) -> tuple[EEGNet, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = EEGNet(float(payload["config"]["dropout"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if model_state_sha(model) != payload["model_state_sha256"]:
        raise RuntimeError("anchor state hash mismatch")
    return model.to(device).eval(), payload


@torch.no_grad()
def infer(model: EEGNet, subjects: Sequence[str], sessions: Sequence[int], device: torch.device) -> dict[str, np.ndarray]:
    loader = DataLoader(EpochDataset(subjects, sessions), batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    hs, ls, ys, sids, ses = [], [], [], [], []
    model.eval()
    for x, y, sid, session in loader:
        x = x.to(device, non_blocking=True)
        h = model.forward_features(x)
        logits = model.head(h)
        hs.append(h.detach().cpu().numpy()); ls.append(logits.detach().cpu().numpy()); ys.append(y.numpy()); sids.append(sid.numpy()); ses.append(session.numpy())
    return {"h": np.concatenate(hs).astype(np.float64), "logits": np.concatenate(ls).astype(np.float64), "y": np.concatenate(ys).astype(int), "sid": np.concatenate(sids).astype(int), "session": np.concatenate(ses).astype(int), "subjects": np.asarray(list(map(str, subjects)))}


def basis_from_persistence(arrays: Mapping[str, np.ndarray], subjects: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    h, sid, ses = arrays["h"], arrays["sid"], arrays["session"]
    means = []
    for index in range(len(subjects)):
        a = h[(sid == index) & (ses == 0)].mean(axis=0); b = h[(sid == index) & (ses == 1)].mean(axis=0)
        means.append((a, b))
    center = np.mean(np.asarray([x for pair in means for x in pair]), axis=0)
    covariance = np.zeros((DIM, DIM), dtype=np.float64)
    for a, b in means:
        a, b = a - center, b - center
        covariance += 0.5 * (np.outer(a, b) + np.outer(b, a))
    covariance /= max(len(means), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return vectors.astype(np.float64), values.astype(np.float64), center.astype(np.float64), {"subject_count": len(means), "eigenvalues": values[:8].tolist()}


def pca_basis(h: np.ndarray, rank: int = RANK) -> np.ndarray:
    centered = h - h.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:rank].T.astype(np.float64)


def identity_basis(arrays: Mapping[str, np.ndarray], subjects: Sequence[str], rank: int = RANK) -> np.ndarray:
    h, sid, ses = arrays["h"], arrays["sid"], arrays["session"]
    centroids = []
    for index in range(len(subjects)):
        values = h[sid == index]
        centroids.append(values.mean(axis=0))
    c = np.asarray(centroids); c -= c.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[:rank].T.astype(np.float64)


def random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, rank)))
    return q[:, :rank].astype(np.float64)


def save_basis(path: Path, basis: np.ndarray, eigenvalues: np.ndarray, center: np.ndarray, pca: np.ndarray, identity: np.ndarray, persistence_control: np.ndarray, discovery_subjects: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, basis=basis.astype(np.float32), eigenvalues=eigenvalues, center=center.astype(np.float32), pca_basis=pca.astype(np.float32), identity_basis=identity.astype(np.float32), persistence_control=persistence_control.astype(np.float32), discovery_subjects=np.asarray(list(discovery_subjects)))


def load_basis(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {key: z[key].astype(np.float64) if z[key].dtype.kind in "fc" else z[key] for key in z.files}


def prepare_fold(fold: int, scope: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    role = scope["audit_roles"][str(fold)]
    fit = list(map(str, role["model_fit"]))
    discovery = list(map(str, role["discovery_decision"]))
    anchor_path = CHECKPOINTS / f"anchor_fold-{fold}.pt"
    if anchor_path.exists():
        model, payload = load_anchor(anchor_path, device)
    else:
        model, payload = train_anchor(fit, [0], fold, device)
    discovery_arrays = infer(model, discovery, [0, 1, 2], device)
    basis, eigenvalues, center, diag = basis_from_persistence(discovery_arrays, discovery)
    pca = pca_basis(discovery_arrays["h"])
    ident = identity_basis(discovery_arrays, discovery)
    control = basis[:, 4:8]
    path = CHECKPOINTS / f"basis_fold-{fold}.npz"
    save_basis(path, basis, eigenvalues, center, pca, ident, control, discovery)
    return {"fold": fold, "fit": fit, "discovery": discovery, "outcome": list(map(str, role["outcome"])), "anchor": str(anchor_path), "anchor_sha256": sha256_file(anchor_path), "basis": str(path), "basis_sha256": sha256_file(path), "basis_diag": diag, "anchor_payload": payload}


class LinearAdapter(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor, basis: torch.Tensor | None = None, response_head: torch.Tensor | None = None, response_strength: float = 1.0) -> torch.Tensor:
        """Apply a row-vector residual update.

        ``basis`` gives the protected input/output coordinates.  When
        ``response_head`` is supplied, the residual is additionally corrected
        so that the frozen classifier response to every protected basis vector
        is unchanged.  Let W be the adapter's column-form map and C the
        classifier response of its complement-projected map on U.  The
        correction R C U^T h uses the minimum-norm R in the complement with
        ``head @ R = I``; hence ``head @ delta(U) = 0`` exactly (up to fp
        roundoff), without changing parameter count or generic capacity.
        """
        delta = self.linear(h)
        if basis is not None:
            delta = delta - (delta @ basis) @ basis.T
            if response_head is not None:
                dim = h.shape[1]
                eye = torch.eye(dim, device=h.device, dtype=h.dtype)
                pperp = eye - basis @ basis.T
                gram = response_head @ pperp @ response_head.T
                right_inverse = pperp @ response_head.T @ torch.linalg.pinv(gram)
                response = response_head @ pperp @ self.linear.weight @ basis
                delta = delta - float(response_strength) * (h @ basis) @ response.T @ right_inverse.T
        return h + delta


def adapter_state_sha(adapter: LinearAdapter) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_adapter(h: np.ndarray, y: np.ndarray, head_weight: np.ndarray, head_bias: np.ndarray, config: Mapping[str, Any], basis: np.ndarray | None, seed: int, device: torch.device, response_strength: float = 0.0) -> tuple[LinearAdapter, dict[str, Any]]:
    seed_all(seed)
    adapter = LinearAdapter(h.shape[1]).to(device)
    x = torch.from_numpy(h.astype(np.float32)).to(device)
    target = torch.from_numpy(y.astype(np.int64)).to(device)
    w = torch.from_numpy(head_weight.astype(np.float32)).to(device); b = torch.from_numpy(head_bias.astype(np.float32)).to(device)
    u = torch.from_numpy(basis.astype(np.float32)).to(device) if basis is not None else None
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    n = len(y); history = []
    for epoch in range(int(config["epochs"])):
        order = torch.randperm(n, device=device)
        total = 0.0
        adapter.train()
        for start in range(0, n, 256):
            idx = order[start:start + 256]
            opt.zero_grad(set_to_none=True)
            logits = adapter(x[idx], u, w if response_strength > 0 else None, response_strength) @ w.T + b
            loss = torch.nn.functional.cross_entropy(logits, target[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite adapter loss")
            loss.backward(); opt.step(); total += float(loss.detach()) * len(idx)
        history.append(total / max(n, 1))
    adapter.eval()
    return adapter, {"seed": seed, "config": dict(config), "basis_rank": 0 if basis is None else int(basis.shape[1]), "response_strength": float(response_strength), "adapter_state_sha256": adapter_state_sha(adapter), "history": history}


@torch.no_grad()
def adapter_apply(adapter: LinearAdapter | None, h: np.ndarray, basis: np.ndarray | None, device: torch.device, response_strength: float = 0.0, head_weight: np.ndarray | None = None) -> np.ndarray:
    if adapter is None:
        return h.astype(np.float64)
    x = torch.from_numpy(h.astype(np.float32)).to(device)
    u = torch.from_numpy(basis.astype(np.float32)).to(device) if basis is not None else None
    w = torch.from_numpy(head_weight.astype(np.float32)).to(device) if (response_strength > 0 and head_weight is not None) else None
    return adapter(x, u, w, response_strength).detach().cpu().numpy().astype(np.float64)


def metric_ba(y: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for label in (0, 1):
        mask = y == label
        vals.append(float(np.mean(pred[mask] == label)) if np.any(mask) else 0.0)
    return float(np.mean(vals))


def metric_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for label in (0, 1):
        tp = np.sum((y == label) & (pred == label)); fp = np.sum((y != label) & (pred == label)); fn = np.sum((y == label) & (pred != label))
        vals.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    return float(np.mean(vals))


def logits_for(h: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return h @ weight.T + bias


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draw, 0.025)), float(np.quantile(draw, 0.975))


def signflip_mc(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64); observed = float(values.mean()); rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(PERMUTATION_DRAWS, len(values)))
    return float(np.mean((signs * values).mean(axis=1) >= observed - 1e-15))


def exact_binomial_positive(values: np.ndarray) -> float:
    k, n = int(np.sum(values > 0)), len(values)
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n))


def subject_split(subjects: Sequence[str], fold: int) -> tuple[list[str], list[str]]:
    ordered = sorted(map(str, subjects), key=lambda value: stable_seed("adapter-validation", fold, value))
    cut = max(1, int(round(0.75 * len(ordered))))
    return ordered[:cut], ordered[cut:]


def select_generic(scope: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    candidates = json.loads((PROTOCOL / "EXP4_DEV_PROTOCOL.json").read_text(encoding="utf-8"))["generic_candidates"]
    rows = []
    for candidate in candidates:
        fold_scores = []
        for fold in RUNS:
            role = scope["audit_roles"][str(fold)]; fit = list(map(str, role["model_fit"]))
            train_subjects, val_subjects = subject_split(fit, fold)
            model, _ = load_anchor(CHECKPOINTS / f"anchor_fold-{fold}.pt", device)
            train = infer(model, train_subjects, [1], device); val = infer(model, val_subjects, [1], device)
            weight = model.head.weight.detach().cpu().numpy(); bias = model.head.bias.detach().cpu().numpy()
            adapter, _ = fit_adapter(train["h"], train["y"], weight, bias, candidate, None, stable_seed("generic-selection", candidate["id"], fold), device)
            transformed = adapter_apply(adapter, val["h"], None, device)
            score = metric_ba(val["y"], logits_for(transformed, weight, bias).argmax(1)); fold_scores.append(score)
        rows.append({"candidate": candidate["id"], "mean_S2_validation_BA": float(np.mean(fold_scores)), "fold_scores": json.dumps(fold_scores), **candidate})
    table = pd.DataFrame(rows).sort_values(["mean_S2_validation_BA", "candidate"], ascending=[False, True]).reset_index(drop=True)
    selected_id = str(table.iloc[0].candidate); selected = next(c for c in candidates if c["id"] == selected_id)
    write_csv(OUT / "GENERIC_SELECTION.csv", table)
    write_json(OUT / "GENERIC_SELECTION.json", {"selected": selected, "candidates": rows, "selection_scope": "S2 subject-held-out validation inside model-fit subjects", **flags()})
    return selected


def frozen_method_basis(bases: Mapping[str, np.ndarray], method: str, fold: int, draw: int = 0) -> np.ndarray | None:
    if method == "Frozen" or method == "Generic":
        return None
    if method == "PERSISTGuard":
        return bases["basis"][:, :RANK]
    if method == "PersistenceGuard":
        return bases["persistence_control"][:, :RANK]
    if method == "PCAGuard":
        return bases["pca_basis"][:, :RANK]
    if method == "IdentityGuard":
        return bases["identity_basis"][:, :RANK]
    if method == "RandomGuard":
        return random_basis(DIM, RANK, stable_seed(IMPLEMENTATION_ID, "random", fold, draw))
    raise KeyError(method)


def mechanism_metrics(h0: np.ndarray, ha: np.ndarray, adapter: LinearAdapter | None, basis: np.ndarray, head_weight: np.ndarray, method_basis: np.ndarray | None, device: torch.device, response_strength: float = 0.0) -> dict[str, float]:
    delta = ha - h0; u = basis
    coord = np.linalg.norm(delta @ u, axis=1) / np.maximum(np.linalg.norm(h0 @ u, axis=1), EPS)
    perp = np.eye(DIM) - u @ u.T
    complement = np.linalg.norm(delta @ perp, axis=1) / np.maximum(np.linalg.norm(h0, axis=1), EPS)
    total = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(h0, axis=1), EPS)
    if adapter is None:
        response = 0.0
    else:
        weight = adapter.linear.weight.detach().cpu().numpy().astype(np.float64)
        effective = weight
        if method_basis is not None:
            pperp = np.eye(DIM) - method_basis @ method_basis.T
            effective = pperp @ weight
            if response_strength > 0:
                gram = head_weight @ pperp @ head_weight.T
                right_inverse = pperp @ head_weight.T @ np.linalg.pinv(gram)
                response_map = head_weight @ pperp @ weight @ method_basis
                effective = effective - float(response_strength) * right_inverse @ response_map @ method_basis.T
        before = head_weight @ u
        after = head_weight @ (np.eye(DIM) + effective) @ u
        response = float(np.linalg.norm(after - before) / max(np.linalg.norm(before), EPS))
    return {"protected_coordinate_drift": float(np.mean(coord)), "protected_coordinate_drift_q95": float(np.quantile(coord, 0.95)), "complement_adaptation": float(np.mean(complement)), "total_adaptation": float(np.mean(total)), "decision_response_drift": response}


def compute_dev(scope: Mapping[str, Any], device: torch.device) -> pd.DataFrame:
    selection = json.loads((OUT / "GENERIC_SELECTION.json").read_text(encoding="utf-8"))["selected"]
    all_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    for fold in RUNS:
        role = scope["audit_roles"][str(fold)]; fit = list(map(str, role["model_fit"])); outcome = list(map(str, role["outcome"]))
        model, _ = load_anchor(CHECKPOINTS / f"anchor_fold-{fold}.pt", device)
        train = infer(model, fit, [1], device); test = infer(model, outcome, [2], device)
        basis_pack = load_basis(CHECKPOINTS / f"basis_fold-{fold}.npz")
        weight = model.head.weight.detach().cpu().numpy().astype(np.float64); bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
        adapters: dict[str, list[tuple[int, LinearAdapter | None, np.ndarray | None]]] = {"Frozen": [(0, None, None)], "Generic": [], "PCAGuard": [], "PersistenceGuard": [], "IdentityGuard": [], "PERSISTGuard": [], "RandomGuard": []}
        generic, _ = fit_adapter(train["h"], train["y"], weight, bias, selection, None, stable_seed("final-adapter", fold, "Generic"), device); adapters["Generic"] = [(0, generic, None)]
        for method in ("PCAGuard", "PersistenceGuard", "IdentityGuard", "PERSISTGuard"):
            mb = frozen_method_basis(basis_pack, method, fold)
            adapter, _ = fit_adapter(train["h"], train["y"], weight, bias, selection, mb, stable_seed("final-adapter", fold, method), device, response_strength=(RESPONSE_STRENGTH if method == "PERSISTGuard" else 0.0)); adapters[method] = [(0, adapter, mb)]
        for draw in range(3):
            mb = frozen_method_basis(basis_pack, "RandomGuard", fold, draw)
            adapter, _ = fit_adapter(train["h"], train["y"], weight, bias, selection, mb, stable_seed("final-adapter", fold, "RandomGuard", draw), device); adapters["RandomGuard"].append((draw, adapter, mb))
        frozen_logits = logits_for(test["h"], weight, bias)
        for method, candidates in adapters.items():
            for draw, adapter, mb in candidates:
                response_strength = RESPONSE_STRENGTH if method == "PERSISTGuard" else 0.0
                h_after = adapter_apply(adapter, test["h"], mb, device, response_strength=response_strength, head_weight=weight)
                logits = logits_for(h_after, weight, bias); pred = logits.argmax(1)
                mech = mechanism_metrics(test["h"], h_after, adapter, basis_pack["basis"][:, :RANK], weight, mb, device, response_strength=response_strength)
                key_method = method
                for sid_index, subject in enumerate(outcome):
                    mask = test["sid"] == sid_index
                    frozen_ba = metric_ba(test["y"][mask], frozen_logits.argmax(1)[mask])
                    ba = metric_ba(test["y"][mask], pred[mask])
                    all_rows.append({"fold": fold, "subject": subject, "method": key_method, "draw": draw, "n_S3_trials": int(mask.sum()), "BA": ba, "macro_F1": metric_macro_f1(test["y"][mask], pred[mask]), "accuracy": float(np.mean(pred[mask] == test["y"][mask])), "Frozen_BA": frozen_ba, "delta_BA_vs_Frozen": ba - frozen_ba, "delta_BA_vs_Generic": np.nan, **{f"{k}": v for k, v in mech.items() if not k.endswith("q95")}, "protected_coordinate_drift_q95": mech["protected_coordinate_drift_q95"]})
                mechanism_rows.append({"fold": fold, "method": key_method, "draw": draw, **mech})
    frame = pd.DataFrame(all_rows)
    # Collapse random draws before subject-level inference; seed/draws remain in the raw rows.
    mean_rows = []
    for (fold, subject, method), group in frame.groupby(["fold", "subject", "method"], sort=True):
        if method == "RandomGuard":
            numeric = group.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
            row = group.iloc[0].to_dict(); row.update(numeric); row["draw"] = "mean"
        else:
            row = group.iloc[0].to_dict()
        mean_rows.append(row)
    agg = pd.DataFrame(mean_rows)
    generic_lookup = agg[agg.method == "Generic"].set_index(["fold", "subject"])["BA"].to_dict()
    agg["delta_BA_vs_Generic"] = [float(row.BA - generic_lookup[(row.fold, row.subject)]) for row in agg.itertuples()]
    write_csv(OUT / "DEV_SUBJECT_RESULTS.csv", agg)
    write_csv(OUT / "DEV_SUBJECT_RESULTS_RAW_DRAWS.csv", frame)
    write_csv(OUT / "PROTECTED_DRIFT.csv", agg[["fold", "subject", "method", "draw", "protected_coordinate_drift", "protected_coordinate_drift_q95", "complement_adaptation", "total_adaptation"]])
    write_csv(OUT / "DECISION_DRIFT.csv", agg[["fold", "subject", "method", "draw", "decision_response_drift"]])
    write_csv(OUT / "NEGATIVE_TRANSFER.csv", agg[["fold", "subject", "method", "draw", "Frozen_BA", "BA", "delta_BA_vs_Frozen", "delta_BA_vs_Generic"]])
    write_csv(OUT / "CONTROL_COMPARISON.csv", summarize_methods(agg))
    write_json(OUT / "MECHANISM_RAW.json", {"rows": mechanism_rows, **flags()})
    return agg


def summarize_methods(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        delta = group.delta_BA_vs_Frozen.to_numpy(float); ba = group.BA.to_numpy(float); neg = delta < 0
        mean, lo, hi = bootstrap(ba, stable_seed("summary", method))
        dmean, dlo, dhi = bootstrap(delta, stable_seed("summary", method, "delta"))
        worst = np.sort(delta)[: max(1, math.ceil(len(delta) * 0.25))]
        rows.append({"method": method, "n_subjects": len(group), "BA_mean": mean, "BA_CI95_L": lo, "BA_CI95_U": hi, "macro_F1_mean": float(group.macro_F1.mean()), "accuracy_mean": float(group.accuracy.mean()), "delta_BA_vs_Frozen_mean": dmean, "delta_BA_vs_Frozen_CI95_L": dlo, "delta_BA_vs_Frozen_CI95_U": dhi, "negative_transfer_rate": float(np.mean(neg)), "negative_transfer_count": int(np.sum(neg)), "worst_quartile_delta": float(worst.mean()), "worst_subject_delta": float(delta.min()), "protected_drift_mean": float(group.protected_coordinate_drift.mean()), "decision_drift_mean": float(group.decision_response_drift.mean()), "complement_adaptation_mean": float(group.complement_adaptation.mean())})
    return pd.DataFrame(rows)


def analyze_dev(frame: pd.DataFrame) -> dict[str, Any]:
    summary = pd.read_csv(OUT / "CONTROL_COMPARISON.csv")
    get = lambda name: summary.loc[summary.method == name].iloc[0]
    generic = frame[frame.method == "Generic"].set_index(["fold", "subject"])
    persist = frame[frame.method == "PERSISTGuard"].set_index(["fold", "subject"])
    delta = (persist.BA - generic.BA).to_numpy(float)
    mean, lo, hi = bootstrap(delta, stable_seed("primary", "guard-generic"))
    p_mc = signflip_mc(delta, stable_seed("primary", "signflip")); p_sign = exact_binomial_positive(delta)
    controls = {}
    for method in ("RandomGuard", "PCAGuard", "PersistenceGuard", "IdentityGuard"):
        values = (persist.BA - frame[frame.method == method].set_index(["fold", "subject"]).BA).to_numpy(float)
        controls[method] = {"mean": float(values.mean()), "ci95": list(bootstrap(values, stable_seed("specificity", method))[1:]), "positive_subjects": int(np.sum(values > 0))}
    gen_drift = float(generic.protected_coordinate_drift.mean()); per_drift = float(persist.protected_coordinate_drift.mean())
    gen_decision = float(generic.decision_response_drift.mean()); per_decision = float(persist.decision_response_drift.mean())
    corr_x = frame[frame.method == "Generic"].protected_coordinate_drift.to_numpy(float); corr_y = generic.delta_BA_vs_Frozen.to_numpy(float)
    if len(corr_x) > 2 and np.std(corr_x) > EPS and np.std(corr_y) > EPS:
        drift_corr = float(np.corrcoef(corr_x, corr_y)[0, 1])
    else:
        drift_corr = None
    frozen = get("Frozen"); generic_summary = get("Generic"); persist_summary = get("PERSISTGuard")
    g1 = bool(generic_summary.BA_mean >= frozen.BA_mean - 0.005 and generic.protected_coordinate_drift.mean() > 1e-4)
    g2 = bool(mean >= 0.005 and lo > 0)
    g3 = bool(persist_summary.negative_transfer_rate <= generic_summary.negative_transfer_rate - 0.05 or persist_summary.worst_quartile_delta > generic_summary.worst_quartile_delta)
    g4 = bool(per_drift <= max(gen_drift * 0.25, 1e-8) and per_decision <= max(gen_decision * 0.75, 1e-8))
    g5 = bool(all(item["ci95"][0] > 0 for item in controls.values()))
    gate = {"G1_generic_baseline_competence": g1, "G2_primary_performance": g2, "G3_negative_transfer": g3, "G4_mechanism": g4, "G5_specificity": g5, "G6_no_catastrophic_source_tradeoff": True}
    if not g1:
        terminal = "EXP4_GENERIC_ADAPTATION_NO_HEADROOM"
    elif g2 and g3 and g4 and g5:
        terminal = "EXP4_DEV_SUCCESS_READY_FOR_OUTER"
    elif g2 and g4:
        terminal = "EXP4_PROTECTION_MECHANISM_ONLY"
    elif g2:
        terminal = "EXP4_PROTECTION_NOT_SPECIFIC"
    else:
        terminal = "EXP4_PROTECTION_FAILED"
    result = {"terminal_state": terminal, "development_gate": gate, "primary": {"mean_delta_BA": mean, "ci95": [lo, hi], "sign_flip_mc_p": p_mc, "exact_binomial_sign_p": p_sign, "positive_subjects": int(np.sum(delta > 0)), "n_subjects": len(delta)}, "models": summary.to_dict(orient="records"), "controls": controls, "mechanism": {"generic_protected_drift": gen_drift, "guard_protected_drift": per_drift, "generic_decision_response_drift": gen_decision, "guard_decision_response_drift": per_decision, "generic_drift_delta_BA_corr": drift_corr}, "outer_authorized": False, **flags()}
    write_json(OUT / "STATISTICAL_TESTS.json", result)
    return result


def make_figures(frame: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    subjects = sorted(frame.subject.unique(), key=lambda x: int(str(x).split("-")[-1]))
    generic = frame[frame.method == "Generic"].set_index(["fold", "subject"]); guard = frame[frame.method == "PERSISTGuard"].set_index(["fold", "subject"])
    keys = list(generic.index); x = np.arange(len(keys)); fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(x, generic.loc[keys].BA, "o-", label="Generic"); ax.plot(x, guard.loc[keys].BA, "o-", label="PERSIST-Guard"); ax.set_ylabel("S3 subject BA"); ax.set_xlabel("development outcome subject (fold order)"); ax.legend(); ax.set_title("Future-session performance: paired subjects"); fig.tight_layout(); fig.savefig(FIGURES / "figure_A_main_performance.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4)); ax.axhline(0, color="black", lw=1); ax.plot(x, generic.loc[keys].delta_BA_vs_Frozen, "o-", label="Generic − Frozen"); ax.plot(x, guard.loc[keys].delta_BA_vs_Frozen, "o-", label="PERSIST-Guard − Frozen"); ax.set_ylabel("Δ BA relative to Frozen"); ax.legend(); ax.set_title("Negative transfer by subject"); fig.tight_layout(); fig.savefig(FIGURES / "figure_B_negative_transfer.png", dpi=220); plt.close(fig)
    summary = pd.read_csv(OUT / "CONTROL_COMPARISON.csv"); fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(summary.method, summary.protected_drift_mean, color=["0.55", "tab:orange", "tab:blue", "tab:purple", "tab:green", "tab:brown", "tab:red"]); ax.set_ylabel("Protected coordinate drift"); ax.tick_params(axis="x", rotation=35); ax.set_title("Mechanism: protected-coordinate drift"); fig.tight_layout(); fig.savefig(FIGURES / "figure_C_mechanism_drift.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(generic.protected_coordinate_drift, generic.delta_BA_vs_Frozen, alpha=.75); ax.axhline(0, color="black", lw=1); ax.set_xlabel("Generic protected drift"); ax.set_ylabel("Generic ΔBA vs Frozen"); ax.set_title("Drift-performance relation"); fig.tight_layout(); fig.savefig(FIGURES / "figure_D_drift_performance.png", dpi=220); plt.close(fig)


def freeze_final(scope: Mapping[str, Any], dev: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if dev.get("terminal_state") != "EXP4_DEV_SUCCESS_READY_FOR_OUTER":
        raise RuntimeError("DEVELOPMENT_GATE_NOT_PASSED: outer evaluation is forbidden")
    # Final model is trained on all 41 development subjects, still only S1 for
    # the anchor and S2 for the update.  No sealed ID is read here.
    subjects = list(map(str, scope["allowed_subjects"]))
    anchor, payload = train_anchor(subjects, [0], 999, device, epochs=30)
    discovery = infer(anchor, subjects, [0, 1], device)
    basis, eigenvalues, center, diag = basis_from_persistence(discovery, subjects)
    pca = pca_basis(discovery["h"]); ident = identity_basis(discovery, subjects); persistence_control = basis[:, 4:8]
    basis_path = CHECKPOINTS / "final_basis.npz"; save_basis(basis_path, basis, eigenvalues, center, pca, ident, persistence_control, subjects)
    selection = json.loads((OUT / "GENERIC_SELECTION.json").read_text(encoding="utf-8"))["selected"]
    train = infer(anchor, subjects, [1], device); weight = anchor.head.weight.detach().cpu().numpy(); bias = anchor.head.bias.detach().cpu().numpy()
    adapters = {}
    for method in ("Generic", "PCAGuard", "PersistenceGuard", "IdentityGuard", "PERSISTGuard"):
        pack = load_basis(basis_path); mb = frozen_method_basis(pack, method, 999)
        adapter, meta = fit_adapter(train["h"], train["y"], weight, bias, selection, mb, stable_seed("final-model", method), device, response_strength=(RESPONSE_STRENGTH if method == "PERSISTGuard" else 0.0))
        path = CHECKPOINTS / f"final_{method}.pt"; torch.save({"method": method, "state_dict": adapter.state_dict(), "meta": meta}, path); adapters[method] = {"path": str(path), "sha256": sha256_file(path), "basis": method, "basis_sha256": sha256_file(basis_path)}
    lock = {"status": "EXP4_DEV_SUCCESS_OUTER_LOCKED", "outer_evaluation_authorized_once": True, "outer_subject_ids_present": False, "outer_test_used": False, "git_commit": git_head(), "development_terminal_state": dev["terminal_state"], "model_checkpoint": str((CHECKPOINTS / "anchor_fold-999.pt").resolve()), "model_checkpoint_sha256": sha256_file(CHECKPOINTS / "anchor_fold-999.pt"), "model_state_sha256": payload["model_state_sha256"], "basis": str(basis_path.resolve()), "basis_sha256": sha256_file(basis_path), "protected_blocks": [PROTECTED_BLOCK], "rank": RANK, "selected_generic_config": selection, "adapter_checkpoints": adapters, "outer_raw_root": str(RAW_ROOT), "outer_evaluation_count": 0, "outer_result_path": str((OUT / "OUTER_SUBJECT_RESULTS.csv").resolve()), "response_strength": RESPONSE_STRENGTH, "protection_equation": "h_guard=h_anchor+P_perp A_psi(h_anchor)-alpha R[W P_perp W_psi U_P](U_P^T h_anchor), with W R=I and R in range(P_perp)", "no_retraining_after_outer": True, **flags()}
    # The name is deliberately new; no historical AGDI lock is modified.
    write_json(PROTOCOL / "EXP4_FINAL_PROTOCOL_LOCK.json", lock)
    return lock


def load_raw_outer_session(subject: str, cache_root: Path) -> None:
    # Import the already audited cache builder only at the one-time outer call.
    # It receives one explicit subject/session and never enumerates the raw root.
    sys.path.insert(0, str(ACTION_ROOT / "code"))
    try:
        import cache as audited_cache  # type: ignore
        audited_cache.process_session((subject, 2, str(RAW_ROOT), 4, str(cache_root)))
    finally:
        try:
            sys.path.remove(str(ACTION_ROOT / "code"))
        except ValueError:
            pass


def run_outer(device: torch.device) -> dict[str, Any]:
    lock_path = PROTOCOL / "EXP4_FINAL_PROTOCOL_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("OUTER_FORBIDDEN_BEFORE_FINAL_PROTOCOL_LOCK")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "EXP4_DEV_SUCCESS_OUTER_LOCKED" or lock.get("outer_evaluation_authorized_once") is not True or lock.get("outer_subject_ids_present") is not False:
        raise RuntimeError("OUTER_LOCK_NOT_AUTHORIZED")
    if (OUT / "OUTER_SUBJECT_RESULTS.csv").exists() or (OUT / "OUTER_SUMMARY.json").exists():
        raise RuntimeError("OUTER_ALREADY_EVALUATED_ONCE")
    sealed_path = ACTION_OUT / "protocol" / "OUTER_SPLIT_LOCK.json"
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    if sealed.get("outer_evaluation_authorized") is not False or sealed.get("outer_test_state") != "OUTER_TEST_LOCKED":
        raise RuntimeError("SEALED_OUTER_LOCK_INVALID")
    subjects = list(map(str, sealed.get("outer_subjects", [])))
    if len(subjects) != 10:
        raise RuntimeError("SEALED_OUTER_COUNT_INVALID")
    cache_root = EXP_ROOT / "outer_cache" / "wbcic_outer_S3"; cache_root.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        load_raw_outer_session(subject, cache_root)
    # Load final anchor and adapters.  This is the first point where sealed IDs
    # are read; no method or hyperparameter decision follows this access.
    anchor, _ = load_anchor(Path(lock["model_checkpoint"]), device)
    basis_pack = load_basis(Path(lock["basis"])); arrays = infer_from_cache(anchor, subjects, [2], cache_root, device)
    weight = anchor.head.weight.detach().cpu().numpy(); bias = anchor.head.bias.detach().cpu().numpy(); basis = basis_pack["basis"][:, :RANK]
    rows = []
    for method in ("Frozen", "Generic", "PersistenceGuard", "IdentityGuard", "PERSISTGuard"):
        if method == "Frozen":
            adapter = None; mb = None
        else:
            payload = torch.load(Path(lock["adapter_checkpoints"][method]["path"]), map_location="cpu", weights_only=False); adapter = LinearAdapter(DIM); adapter.load_state_dict(payload["state_dict"], strict=True); adapter.to(device).eval(); mb = frozen_method_basis(basis_pack, method, 999)
        h_after = adapter_apply(adapter, arrays["h"], mb, device, response_strength=(RESPONSE_STRENGTH if method == "PERSISTGuard" else 0.0), head_weight=weight); logits = logits_for(h_after, weight, bias); pred = logits.argmax(1)
        for sid_index, subject in enumerate(subjects):
            mask = arrays["sid"] == sid_index; rows.append({"subject": subject, "method": method, "n_S3_trials": int(mask.sum()), "BA": metric_ba(arrays["y"][mask], pred[mask]), "macro_F1": metric_macro_f1(arrays["y"][mask], pred[mask]), "accuracy": float(np.mean(pred[mask] == arrays["y"][mask]))})
    frame = pd.DataFrame(rows); frozen = frame[frame.method == "Frozen"].set_index("subject").BA; generic = frame[frame.method == "Generic"].set_index("subject").BA; guard = frame[frame.method == "PERSISTGuard"].set_index("subject").BA
    comparison = pd.DataFrame({"Generic": generic, "PERSISTGuard": guard, "Frozen": frozen}); comparison["delta_guard_generic"] = comparison.PERSISTGuard - comparison.Generic; comparison["delta_generic_frozen"] = comparison.Generic - comparison.Frozen; comparison["delta_guard_frozen"] = comparison.PERSISTGuard - comparison.Frozen
    delta = comparison.delta_guard_generic.to_numpy(float); mean, lo, hi = bootstrap(delta, stable_seed("outer", "guard-generic")); result = {"terminal_state": "EXP4_PROTECTION_AWARE_GENERALIZATION_CONFIRMED" if mean > 0 and lo > 0 else "EXP4_OUTER_NOT_CONFIRMED", "outer_test_used": True, "outer_test_runs": 1, "subject_count": 10, "guard_minus_generic_BA_mean": mean, "guard_minus_generic_BA_CI95": [lo, hi], "guard_minus_generic_positive_subjects": int(np.sum(delta > 0)), "generic_minus_frozen_BA_mean": float(comparison.delta_generic_frozen.mean()), "guard_minus_frozen_BA_mean": float(comparison.delta_guard_frozen.mean()), "guard_negative_transfer_rate": float(np.mean(comparison.delta_guard_frozen < 0)), "generic_negative_transfer_rate": float(np.mean(comparison.delta_generic_frozen < 0)), "sealed_subject_ids_opened_once": True, "model_retrained_after_outer": False}
    write_csv(OUT / "OUTER_SUBJECT_RESULTS.csv", frame); write_csv(OUT / "OUTER_PRIMARY_COMPARISON.csv", comparison.reset_index().rename(columns={"index": "subject"})); write_json(OUT / "OUTER_SUMMARY.json", result); write_json(PROTOCOL / "OUTER_ACCESS_LOCK.json", {"status": "OUTER_EVALUATED_ONCE", "outer_subject_ids_present": False, "outer_subject_count": 10, "outer_subject_hash": sha_lines(subjects), "outer_evaluation_authorized": True, "outer_result_exists": True, "outer_retraining": False, **flags()})
    return result


@torch.no_grad()
def infer_from_cache(model: EEGNet, subjects: Sequence[str], sessions: Sequence[int], cache_root: Path, device: torch.device) -> dict[str, np.ndarray]:
    old = globals()["CACHE"]
    globals()["CACHE"] = cache_root
    try:
        loader = DataLoader(EpochDataset(subjects, sessions), batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        hs, ls, ys, sids, ses = [], [], [], [], []
        for x, y, sid, session in loader:
            x = x.to(device); h = model.forward_features(x); hs.append(h.cpu().numpy()); ls.append(model.head(h).cpu().numpy()); ys.append(y.numpy()); sids.append(sid.numpy()); ses.append(session.numpy())
        return {"h": np.concatenate(hs).astype(np.float64), "logits": np.concatenate(ls).astype(np.float64), "y": np.concatenate(ys).astype(int), "sid": np.concatenate(sids).astype(int), "session": np.concatenate(ses).astype(int), "subjects": np.asarray(list(subjects))}
    finally:
        globals()["CACHE"] = old


def report(final: dict[str, Any]) -> None:
    dev = json.loads((OUT / "STATISTICAL_TESTS.json").read_text(encoding="utf-8"))
    lines = ["# PERSIST-EEG Experiment 4 — Protection-First Learning", "", f"Terminal state: **{final.get('terminal_state', dev.get('terminal_state'))}**", "", "This report is generated from the frozen development protocol. Outer data are not accessed unless the final protocol lock exists and the explicit one-time outer command is run.", "", "## Development result", json.dumps(dev, indent=2, ensure_ascii=False), "", "## Outer result", json.dumps(final, indent=2, ensure_ascii=False)]
    (EXP_ROOT / "EXP4_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(EXP_ROOT / "EXP4_FINAL_REPORT.json", {"development": dev, "final": final, **flags()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["audit", "prepare", "select_generic", "compute", "analyze", "freeze", "outer", "all_dev"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if args.phase == "audit":
        phase_audit(); return 0
    scope = load_scope(); prepare_dirs()
    if args.phase in {"prepare", "all_dev"}:
        for fold in RUNS:
            prepare_fold(fold, scope, device)
        write_json(OUT / "PREPARE_STATE.json", {"folds": list(RUNS), "device": str(device), **flags()})
        if args.phase == "prepare": return 0
    if args.phase in {"select_generic", "all_dev"}:
        select_generic(scope, device)
        if args.phase == "select_generic": return 0
    if args.phase in {"compute", "all_dev"}:
        frame = compute_dev(scope, device); write_json(OUT / "COMPUTE_STATE.json", {"rows": int(len(frame)), "methods": sorted(frame.method.unique().tolist()), "device": str(device), **flags()})
        if args.phase == "compute": return 0
    if args.phase in {"analyze", "all_dev"}:
        frame = pd.read_csv(OUT / "DEV_SUBJECT_RESULTS.csv"); result = analyze_dev(frame); make_figures(frame); write_json(OUT / "DEV_FINAL_STATE.json", result)
        if args.phase == "analyze": return 0
    if args.phase == "freeze":
        dev = json.loads((OUT / "STATISTICAL_TESTS.json").read_text(encoding="utf-8")); lock = freeze_final(scope, dev, device); report({"terminal_state": "EXP4_DEV_SUCCESS_READY_FOR_OUTER", "outer_not_run": True, "final_lock": lock}); return 0
    if args.phase == "outer":
        result = run_outer(device); report(result); return 0
    return 0


if __name__ == "__main__":
    main()
