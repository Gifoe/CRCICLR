"""AMSE-v2 seed-0 frozen five-fold experiment.

The script deliberately separates protocol locking from outcome evaluation.  Run
``--lock-only`` (and commit the generated protocol files) before ``--run``.
Only V8 SEARCH subjects and the already frozen carrier checkpoints are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
EXP = REPO / "experiments" / "persist_eeg_amse_seed0_v2"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("AMSE_RUNTIME", REPO.parent / "amse_seed0_runtime")).resolve()
SPLIT = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"
CARRIER_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", REPO.parent / "carrier_5fold_multiseed_stability_runtime")).resolve()
SRC = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"
STAGE = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code"
sys.path[:0] = [str(SRC), str(STAGE)]

# Imported after sys.path setup; these are the frozen loader and carriers.
import run_stage1 as base  # noqa: E402
import run_carrier_screen as carrier  # noqa: E402
from eegnet_locked import EEGNet  # noqa: E402

SEED = 0
EPOCHS = 60
MIN_EPOCH = 10
LR = 3e-4
WD = 5e-4
CLIP = 5.0
BOOTSTRAPS = 10000


def clean(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def state_hash(state: dict[str, Any]) -> str:
    buf = io.BytesIO(); torch.save(state, buf); return sha_bytes(buf.getvalue())


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    s: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def restore_rng(s: dict[str, Any]) -> None:
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"].detach().cpu())
    if "cuda" in s and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.detach().cpu() for x in s["cuda"]])


class FastGPUCache:
    """Materialize one normalized SEARCH cache per fold; no holdout paths are touched."""
    def __init__(self, bundle: Any, mean: np.ndarray, std: np.ndarray, device: torch.device):
        pieces: list[np.ndarray] = []
        n = len(bundle.search_rows)
        for start in range(0, n, 256):
            idx = np.arange(start, min(n, start + 256), dtype=np.int64)
            x = bundle.accessor.batch(idx).astype(np.float32, copy=False)
            x = (x - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
            pieces.append(np.ascontiguousarray(x))
        self.x = torch.from_numpy(np.concatenate(pieces, axis=0)).to(device, non_blocking=True)
        self.y = torch.from_numpy(bundle.labels(np.arange(n, dtype=np.int64))).to(device, non_blocking=True)
        self.device = device

    def batch(self, indices: Any) -> tuple[torch.Tensor, torch.Tensor]:
        ii = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        return self.x.index_select(0, ii), self.y.index_select(0, ii)


class AMSE(torch.nn.Module):
    """AMSE-v2: exact canonical EEGNet stable carrier plus isolated expressive path."""
    def __init__(self, channels: int, classes: int = 2):
        super().__init__()
        # Keeping this module literally canonical makes state copying and the
        # stable equivalence test unambiguous.
        self.stable = EEGNet(channels, samples=1000, dropout=.25)
        def stem(k: int) -> torch.nn.ModuleList:
            return torch.nn.ModuleList([
                torch.nn.Conv2d(1, 8, (1, k), padding="same", bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
                torch.nn.BatchNorm2d(16),
            ])
        self.short = stem(15); self.long = stem(127)
        self.adapter_depth = torch.nn.Conv2d(16, 16, (1, 15), padding="same", groups=16, bias=False)
        self.adapter_point = torch.nn.Conv2d(16, 16, 1, bias=False)
        self.adapter_bn = torch.nn.BatchNorm2d(16)
        self.exp_depth1 = torch.nn.Conv2d(48, 48, (1, 15), padding="same", groups=48, bias=False)
        self.exp_point1 = torch.nn.Conv2d(48, 64, 1, bias=False)
        self.exp_bn1 = torch.nn.BatchNorm2d(64)
        self.exp_depth2 = torch.nn.Conv2d(64, 64, (1, 31), padding="same", groups=64, bias=False)
        self.exp_point2 = torch.nn.Conv2d(64, 64, 1, bias=False)
        self.exp_bn2 = torch.nn.BatchNorm2d(64)
        self.exp_fc = torch.nn.Linear(512, 64)
        self.exp_ln = torch.nn.LayerNorm(64)
        self.exp_head = torch.nn.Linear(64, classes)
        self.drop_stem = 0.20

    def stable_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stable logits, embedding, and pre-dropout post-pool4 anchor."""
        s = self.stable; v = x.unsqueeze(1)
        v = s.bn1(s.temporal(v))
        a = torch.nn.functional.elu(s.bn2(s.spatial(v)))
        a = s.pool1(a)  # exact A64 anchor, before canonical dropout
        v = s.drop1(a)
        v = s.drop2(s.pool2(torch.nn.functional.elu(s.bn3(s.point(s.depth(v))))))
        h = s.embedding(v.flatten(1)); return s.head(h), h, a

    @staticmethod
    def _stem(x: torch.Tensor, s: torch.nn.ModuleList, p: float, training: bool) -> torch.Tensor:
        y = torch.nn.functional.elu(s[1](s[0](x)))
        y = torch.nn.functional.elu(s[3](s[2](y)))
        y = torch.nn.functional.avg_pool2d(y, (1, 4))
        return torch.nn.functional.dropout(y, p=p, training=training)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zs, hs, anchor = self.stable_forward(x)
        vx = x.unsqueeze(1)
        h15 = self._stem(vx, self.short, self.drop_stem, self.training)
        h127 = self._stem(vx, self.long, self.drop_stem, self.training)
        a = anchor.detach()
        delta = self.adapter_depth(a); delta = self.adapter_point(delta)
        h64 = a + torch.nn.functional.elu(self.adapter_bn(delta))
        h64 = torch.nn.functional.dropout(h64, self.drop_stem, self.training)
        he = torch.cat((h15, h64, h127), dim=1)
        he = self.exp_depth1(he); he = self.exp_point1(he); he = torch.nn.functional.elu(self.exp_bn1(he))
        he = torch.nn.functional.avg_pool2d(he, (1, 2)); he = torch.nn.functional.dropout(he, .15, self.training)
        he = self.exp_depth2(he); he = self.exp_point2(he); he = torch.nn.functional.elu(self.exp_bn2(he))
        he = torch.nn.functional.avg_pool2d(he, (1, 2)); he = torch.nn.functional.dropout(he, .15, self.training)
        he = torch.nn.functional.adaptive_avg_pool2d(he, (1, 8)).flatten(1)
        he = torch.nn.functional.dropout(torch.nn.functional.elu(self.exp_ln(self.exp_fc(he))), .25, self.training)
        ze = self.exp_head(he)
        return .5 * zs + .5 * ze, zs, ze


def count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_split() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], str]:
    raw = json.loads(SPLIT.read_text(encoding="utf-8"))
    if raw.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1" or raw.get("split_seed") != 0:
        raise RuntimeError("unexpected frozen split provenance")
    folds = {d: raw["folds"][d] for d in ("OpenBMI", "WBCIC")}
    search = {d: [str(x) for x in raw["search_subjects"][d]] for d in folds}
    if len(folds["OpenBMI"]) != 5 or len(folds["WBCIC"]) != 5: raise RuntimeError("five folds required")
    if len(search["OpenBMI"]) != 40 or len(search["WBCIC"]) != 31: raise RuntimeError("frozen SEARCH size mismatch")
    for d in folds:
        seen: set[str] = set()
        for f in folds[d]:
            a, b, c = map(set, (map(str, f["inner_train_subjects"]), map(str, f["inner_val_subjects"]), map(str, f["outer_dev_subjects"])))
            if a & b or a & c or b & c or (a | b | c) != set(search[d]): raise RuntimeError(f"split invariant failed {d} f{f['fold_id']}")
            seen |= c
        if seen != set(search[d]): raise RuntimeError(f"outer coverage failed {d}")
    return folds, search, sha_file(SPLIT)


def metric(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    p = logits.argmax(1)
    return {"BA": float(balanced_accuracy_score(y, p)), "macro_F1": float(f1_score(y, p, average="macro")), "accuracy": float(accuracy_score(y, p)), "trials": int(len(y))}


def evaluate(model: torch.nn.Module, bundle: Any, subjects: list[str], cache: FastGPUCache, sessions: tuple[int, ...] = (2,)) -> dict[str, dict[str, Any]]:
    model.eval(); result: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for subject in subjects:
            idx = bundle.indices([subject], sessions); y = bundle.labels(idx)
            outs: list[list[np.ndarray]] = [[], [], []]
            for start in range(0, len(idx), 256):
                f, s, e = model(cache.batch(idx[start:start + 256])[0])
                outs[0].append(f.float().cpu().numpy()); outs[1].append(s.float().cpu().numpy()); outs[2].append(e.float().cpu().numpy())
            result[str(subject)] = {"y": y, "final": np.concatenate(outs[0]), "stable": np.concatenate(outs[1]), "expressive": np.concatenate(outs[2])}
    return result


def evaluate_single(model: torch.nn.Module, bundle: Any, subjects: list[str], cache: FastGPUCache, sessions: tuple[int, ...] = (2,)) -> dict[str, dict[str, Any]]:
    model.eval(); result: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for subject in subjects:
            idx = bundle.indices([subject], sessions); y = bundle.labels(idx); logs = []
            for start in range(0, len(idx), 256):
                logs.append(model(cache.batch(idx[start:start + 256])[0])[0].float().cpu().numpy())
            result[str(subject)] = {"y": y, "logits": np.concatenate(logs)}
    return result


def train(model: AMSE, dataset: str, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], minfo: dict[str, Any], bundle: Any, cache: FastGPUCache, device: torch.device) -> dict[str, Any]:
    cell = RUNTIME / dataset.lower() / f"fold{fold['fold_id']}"; cell.mkdir(parents=True, exist_ok=True)
    path = cell / "checkpoint_latest.pt"; selected_path = cell / "selected_best.pt"
    init_hash = state_hash(copy.deepcopy(model.state_dict())); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["init_sha256"] != init_hash or saved["manifest_sha256"] != minfo["sha256"]: raise RuntimeError("AMSE resume invariant mismatch")
        model.load_state_dict(saved["current_state"]); opt.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start, history, best, best_epoch, best_state = saved["epoch"] + 1, saved["history"], saved["best_val_BA"], saved["best_epoch"], saved["best_state"]
        restore_rng(saved["rng"])
    began = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train(); losses = []
        for ep in manifest[epoch - 1]:
            idx = ep["support_indices"] + ep["query_indices"]; x, y = cache.batch(idx); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _zf, zs, ze = model(x)
                # V2 trains individually competent carriers.  The fixed final
                # average is evaluation-only and never enters the objective.
                loss = torch.nn.functional.cross_entropy(zs, y) + torch.nn.functional.cross_entropy(ze, y)
            if not torch.isfinite(loss): raise RuntimeError("non-finite AMSE loss")
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        val = evaluate(model, bundle, [str(x) for x in fold["inner_val_subjects"]], cache)
        val_ba = float(np.mean([metric(v["y"], v["final"])["BA"] for v in val.values()]))
        val_stable = float(np.mean([metric(v["y"], v["stable"])["BA"] for v in val.values()]))
        val_expressive = float(np.mean([metric(v["y"], v["expressive"])["BA"] for v in val.values()]))
        selected = epoch >= MIN_EPOCH and val_ba > best + 1e-12
        if selected: best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        row = {"epoch": epoch, "CE_total": float(np.mean(losses)), "inner_val_subject_BA": val_ba,
               "inner_val_stable_BA": val_stable, "inner_val_expressive_BA": val_expressive, "selected": selected}; history.append(row)
        torch.save({"epoch": epoch, "history": history, "best_val_BA": best, "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "manifest_sha256": minfo["sha256"], "init_sha256": init_hash}, path)
        if epoch == 1 or epoch % 5 == 0 or selected: print(f"[{dataset} fold={fold['fold_id']}] epoch={epoch:02d} loss={row['CE_total']:.4f} valBA={val_ba:.4f}", flush=True)
    if best_state is None: raise RuntimeError("no eligible AMSE checkpoint")
    model.load_state_dict(best_state); torch.save(model.state_dict(), selected_path)
    return {"selected_epoch": int(best_epoch), "best_inner_val_BA": float(best), "history": history, "checkpoint_path": str(selected_path), "checkpoint_sha256": sha_file(selected_path), "manifest_sha256": minfo["sha256"], "init_sha256": init_hash, "elapsed_seconds": time.perf_counter() - began, "amp": amp}


def smoke_test(device: torch.device) -> dict[str, Any]:
    out = []
    for c in (62, 58):
        set_seed(0); m = AMSE(c).to(device); x = torch.randn(4, c, 1000, device=device); y = torch.tensor([0, 1, 0, 1], device=device)
        zf, zs, ze = m(x); loss = torch.nn.functional.cross_entropy(zf, y) + .25 * torch.nn.functional.cross_entropy(zs, y) + .25 * torch.nn.functional.cross_entropy(ze, y); loss.backward()
        assert zf.shape == zs.shape == ze.shape == (4, 2) and torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        b = io.BytesIO(); torch.save(m.state_dict(), b); m.load_state_dict(torch.load(io.BytesIO(b.getvalue()), weights_only=True)); out.append({"channels": c, "params": count(m), "shape": [4, 2], "finite_loss_gradient": True, "save_load": True})
    return {"tests": out, "stop_gradient_declared": True, "heldout_loaded": False}


def stable_exactness_test(device: torch.device) -> dict[str, Any]:
    """Numerically prove that the V2 stable path is canonical EEGNet."""
    rows = []
    for channels in (62, 58):
        set_seed(0)
        canonical = EEGNet(channels, samples=1000, dropout=.25).to(device).eval()
        amse = AMSE(channels).to(device).eval()
        amse.stable.load_state_dict(copy.deepcopy(canonical.state_dict()))
        x = torch.randn(7, channels, 1000, device=device)
        with torch.no_grad():
            zc, hc = canonical(x)
            zs, hs, _ = amse.stable_forward(x)
        dz = float((zc - zs).abs().max().cpu()); dh = float((hc - hs).abs().max().cpu())
        if dz > 1e-6 or dh > 1e-6:
            raise RuntimeError(f"stable exactness failed C={channels}: logits={dz} hidden={dh}")
        rows.append({"channels": channels, "max_abs_logits": dz, "max_abs_hidden": dh, "tolerance": 1e-6, "passed": True})
    return {"protocol": "AMSE_V2_STABLE_EXACTNESS", "tests": rows, "passed": True}


def gradient_firewall_test(device: torch.device) -> dict[str, Any]:
    """Check that each branch objective backpropagates only to its carrier."""
    set_seed(0); model = AMSE(62).to(device).train()
    x = torch.randn(8, 62, 1000, device=device); y = torch.tensor([0, 1] * 4, device=device)
    amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        _zf, zs, ze = model(x)
        stable_loss = torch.nn.functional.cross_entropy(zs, y)
    stable_loss.backward()
    names = dict(model.named_parameters())
    stable_grads = {n: p.grad for n, p in names.items() if n.startswith("stable.")}
    private_grads = {n: p.grad for n, p in names.items() if not n.startswith("stable.")}
    stable_ok = all(g is not None and bool(torch.isfinite(g).all()) for g in stable_grads.values())
    private_zero = all(g is None or bool((g == 0).all()) for g in private_grads.values())
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        _zf, zs, ze = model(x)
        expressive_loss = torch.nn.functional.cross_entropy(ze, y)
    expressive_loss.backward()
    stable_grads_b = {n: p.grad for n, p in names.items() if n.startswith("stable.")}
    private_grads_b = {n: p.grad for n, p in names.items() if not n.startswith("stable.")}
    stable_zero = all(g is None or bool((g == 0).all()) for g in stable_grads_b.values())
    private_ok = all(g is not None and bool(torch.isfinite(g).all()) for g in private_grads_b.values())
    if not (stable_ok and private_zero and stable_zero and private_ok):
        raise RuntimeError("gradient firewall test failed")
    return {"protocol": "AMSE_V2_GRADIENT_FIREWALL", "stable_objective": {"stable_finite": stable_ok, "expressive_private_zero": private_zero}, "expressive_objective": {"stable_zero": stable_zero, "expressive_private_finite": private_ok}, "passed": True}


def lock_only(device: torch.device) -> None:
    folds, search, split_hash = read_split(); PROTOCOL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    refs = {}
    for d in ("OpenBMI", "WBCIC"):
        refs[d] = {}
        for f in range(5):
            refs[d][str(f)] = {}
            for name in ("EEGNet", "LiteBN"):
                p = CARRIER_RUNTIME / f"{d.lower()}_fold{f}_seed0_{name.lower()}" / "selected_best.pt"
                if not p.is_file(): raise FileNotFoundError(p)
                refs[d][str(f)][name] = {"path": str(p), "sha256": sha_file(p)}
    exact = stable_exactness_test(device); firewall = gradient_firewall_test(device)
    write_json(PROTOCOL / "STABLE_EXACTNESS_TEST.json", exact); write_json(PROTOCOL / "GRADIENT_FIREWALL_TEST.json", firewall)
    write_json(PROTOCOL / "SEED_FIDELITY.json", {"protocol": "AMSE_V2_SEED_FIDELITY", "model_initialization_seed": 0, "training_rng_seed": 100000, "per_fold": True, "fold_seed_offset": 0, "dataset_seed_offset": 0, "forbidden": ["seed+fold", "dataset-specific seed", "fold-dependent seed"], "implemented_order": ["set_seed(0)", "construct AMSE-v2", "set_seed(100000)", "start epoch training"]})
    spec = {"name": "AMSE-v2", "input": "[B,C,T]", "stable": {"architecture": "canonical EEGNet", "temporal_kernel": 64, "depthwise_spatial": [8, 16], "pool": [4, 8], "embedding": 64, "dropout": [.25, .25], "anchor": "post-pool4 pre-dropout A64", "state_module": "stable"}, "expressive": {"private_stems": [{"kernel": 15, "out": 16}, {"kernel": 127, "out": 16}], "anchor_adapter": {"input": "detach(A64)", "depthwise_kernel": 15, "channels": 16}, "concat_channels": 48, "refinement_depthwise": [15, 31], "pointwise_channels": [64, 64], "pool": [2, 2], "adaptive": [1, 8], "embedding": 64, "dropout": [.20, .20, .15, .15, .25]}, "objective": "CE(z_stable,y)+CE(z_expressive,y)", "inference": "0.5*z_stable + 0.5*z_expressive", "learned_gate": False, "learned_alpha": False, "classes": 2}
    write_json(PROTOCOL / "AMSE_MODEL_SPEC.json", spec); (PROTOCOL / "AMSE_MODEL_SPEC.md").write_text("# AMSE-v2 model lock\n\nStable is the canonical EEGNet module exactly. The expressive path receives detached post-pool4 A64 through a private trainable residual adapter and concatenates private k=15/k=127 stems. Training is CE(z_stable)+CE(z_expressive); the fixed 50/50 raw-logit average is evaluation-only.\n", encoding="utf-8")
    training = {"seed": 0, "model_init_seed": 0, "training_rng_seed": 100000, "datasets": ["OpenBMI", "WBCIC"], "folds": 5, "epochs": 60, "minimum_selection_epoch": 10, "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "gradient_clip": CLIP, "loss": "CE(z_stable)+CE(z_expressive)", "selection": "inner_val future-session subject-equal BA of fixed final logits", "amp": device.type == "cuda", "batch": "frozen carrier episode manifest", "holdout_access": False}
    write_json(PROTOCOL / "TRAINING_PROTOCOL.json", training); write_json(PROTOCOL / "SPLIT_PROVENANCE.json", {"source": str(SPLIT), "sha256": split_hash, "protocol": "CARRIER_5FOLD_MULTISEED_STABILITY_V1", "seed": 0, "datasets": {d: {"search_subjects": search[d], "folds": folds[d]} for d in folds}})
    write_json(PROTOCOL / "REFERENCE_PROVENANCE.json", {"carrier_runtime": str(CARRIER_RUNTIME), "checkpoints": refs, "retrained": False, "reference_models": ["EEGNet", "LiteBN"]})
    write_json(PROTOCOL / "OUTCOME_VISIBILITY_AUDIT.json", {"outer_dev_outcome_inspected_before_lock": False, "internal_holdout_loaded": False, "wbcic_true_outer_loaded": False, "sealed_outcome_loaded": False, "only_metadata_hashes_read": True})
    write_json(PROTOCOL / "SMOKE_TEST.json", smoke_test(device))
    (PROTOCOL / "BUGFIX_LEDGER.md").write_text("# AMSE bug-fix ledger\n\nNo outcome-driven changes.\n", encoding="utf-8")
    entries = []
    for p in (PROTOCOL / "AMSE_MODEL_SPEC.json", PROTOCOL / "TRAINING_PROTOCOL.json", PROTOCOL / "SPLIT_PROVENANCE.json", PROTOCOL / "SEED_FIDELITY.json", PROTOCOL / "REFERENCE_PROVENANCE.json", PROTOCOL / "OUTCOME_VISIBILITY_AUDIT.json", PROTOCOL / "SMOKE_TEST.json", PROTOCOL / "STABLE_EXACTNESS_TEST.json", PROTOCOL / "GRADIENT_FIREWALL_TEST.json"):
        entries.append(f"{sha_file(p)}  {p.name}")
    (PROTOCOL / "PROTOCOL_LOCK.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print("AMSE_PROTOCOL_LOCK_READY", flush=True)


def bootstrap_delta(delta_pp: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(0); draw = rng.choice(delta_pp, size=(BOOTSTRAPS, len(delta_pp)), replace=True).mean(1)
    return {"n_subjects": int(len(delta_pp)), "mean_delta_pp": float(delta_pp.mean()), "median_delta_pp": float(np.median(delta_pp)), "ci_low_pp": float(np.quantile(draw, .025)), "ci_high_pp": float(np.quantile(draw, .975)), "positive_subjects": int((delta_pp > 0).sum()), "negative_subjects": int((delta_pp < 0).sum())}


def run_experiment(device: torch.device) -> None:
    folds, search, split_hash = read_split(); OUT.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True); base.RUNTIME = RUNTIME / "manifest_runtime"; base.RUNTIME.mkdir(parents=True, exist_ok=True)
    bundles = {d: base.load_bundle(d, search[d]) for d in ("OpenBMI", "WBCIC")}
    fold_rows: list[dict[str, Any]] = []; subject_rows: list[dict[str, Any]] = []; training_rows: list[dict[str, Any]] = []; complement_rows: list[dict[str, Any]] = []; ref_audit: list[dict[str, Any]] = []
    for dataset in ("OpenBMI", "WBCIC"):
        bundle = bundles[dataset]
        for fold in folds[dataset]:
            f = int(fold["fold_id"]); mean, std, norm = base.normalizer(bundle, [str(x) for x in fold["inner_train_subjects"]]); manifest, minfo = base.make_manifest(bundle, fold); cache = FastGPUCache(bundle, mean, std, device)
            # Protocol fidelity: every fold initializes with seed 0, then the
            # training RNG is reset to the single fixed training seed 100000.
            set_seed(SEED); model = AMSE(bundle.channels).to(device); set_seed(100000)
            info = train(model, dataset, fold, manifest, minfo, bundle, cache, device); training_rows.append({"dataset": dataset, "fold": f, "normalizer": norm, "manifest": minfo, **info})
            amse = evaluate(model, bundle, [str(x) for x in fold["outer_dev_subjects"]], cache)
            ref: dict[str, dict[str, Any]] = {}
            for name, cls in (("EEGNet", EEGNet), ("LiteBN", lambda c: carrier.CompactLite(c, "bn"))):
                p = CARRIER_RUNTIME / f"{dataset.lower()}_fold{f}_seed0_{name.lower()}" / "selected_best.pt"
                obj = cls(bundle.channels).to(device); obj.load_state_dict(torch.load(p, map_location=device, weights_only=True)); ref[name] = evaluate_single(obj, bundle, [str(x) for x in fold["outer_dev_subjects"]], cache); ref[name]["_path"] = str(p); ref[name]["_sha256"] = sha_file(p); ref_audit.append({"dataset": dataset, "fold": f, "model": name, "path": str(p), "sha256": sha_file(p), "loaded": True})
            for subject in [str(x) for x in fold["outer_dev_subjects"]]:
                y = amse[subject]["y"]; final_l, stable_l, exp_l = amse[subject]["final"], amse[subject]["stable"], amse[subject]["expressive"]
                eeg_l, lite_l = ref["EEGNet"][subject]["logits"], ref["LiteBN"][subject]["logits"]; ref50 = .5 * eeg_l + .5 * lite_l
                row = {"dataset": dataset, "fold": f, "subject_id": subject}
                for n, l in (("EEGNet", eeg_l), ("LiteBN", lite_l), ("REF50", ref50), ("AMSE_Stable", stable_l), ("AMSE_Expressive", exp_l), ("AMSE_Final", final_l)):
                    m = metric(y, l); row.update({f"{n}_{k}": v for k, v in m.items()})
                row["Subject_oracle_ceiling_delta_pp"] = (row["AMSE_Final_BA"] - max(row["EEGNet_BA"], row["LiteBN_BA"])) * 100
                # Filled with the realizable dataset-level strongest carrier
                # after all folds are collected; this temporary value is only
                # retained in the separate oracle diagnostic.
                row["AMSE_Final_minus_best_ref_pp"] = row["Subject_oracle_ceiling_delta_pp"]
                row["AMSE_Final_minus_REF50_pp"] = (row["AMSE_Final_BA"] - row["REF50_BA"]) * 100
                row["Stable_minus_EEGNet_pp"] = (row["AMSE_Stable_BA"] - row["EEGNet_BA"]) * 100
                row["Expressive_minus_LiteBN_pp"] = (row["AMSE_Expressive_BA"] - row["LiteBN_BA"]) * 100
                row["Final_minus_strongest_internal_pp"] = (row["AMSE_Final_BA"] - max(row["AMSE_Stable_BA"], row["AMSE_Expressive_BA"])) * 100
                subject_rows.append(row)
                ps, pe = stable_l.argmax(1), exp_l.argmax(1); complement_rows.append({"dataset": dataset, "fold": f, "subject_id": subject, "both_correct": int(((ps == y) & (pe == y)).sum()), "both_wrong": int(((ps != y) & (pe != y)).sum()), "stable_only_correct": int(((ps == y) & (pe != y)).sum()), "expressive_only_correct": int(((ps != y) & (pe == y)).sum()), "trials": int(len(y))})
            fr = pd.DataFrame([r for r in subject_rows if r["dataset"] == dataset and r["fold"] == f]); fold_rows.append({"dataset": dataset, "fold": f, **{k: float(fr[k].mean()) for k in ("EEGNet_BA", "LiteBN_BA", "REF50_BA", "AMSE_Stable_BA", "AMSE_Expressive_BA", "AMSE_Final_BA", "AMSE_Final_minus_best_ref_pp", "AMSE_Final_minus_REF50_pp", "Final_minus_strongest_internal_pp", "Stable_minus_EEGNet_pp", "Expressive_minus_LiteBN_pp")}})
            del cache; torch.cuda.empty_cache() if device.type == "cuda" else None
            print(f"[{dataset} fold={f}] AMSE final={fold_rows[-1]['AMSE_Final_BA']:.4f} REF50={fold_rows[-1]['REF50_BA']:.4f}", flush=True)
    sdf, fdf, cdf = pd.DataFrame(subject_rows), pd.DataFrame(fold_rows), pd.DataFrame(complement_rows)
    # Resolve the fixed, realizable strongest single model per dataset before
    # any primary delta or bootstrap is computed.  The old subject-wise oracle
    # remains descriptive only.
    best_models: dict[str, str] = {}
    for dataset in ("OpenBMI", "WBCIC"):
        ds0 = sdf[sdf.dataset == dataset]
        best_models[dataset] = "EEGNet" if float(ds0.EEGNet_BA.mean()) >= float(ds0.LiteBN_BA.mean()) else "LiteBN"
        bcol = f"{best_models[dataset]}_BA"
        mask = sdf.dataset == dataset
        sdf.loc[mask, "AMSE_Final_minus_best_ref_pp"] = (sdf.loc[mask, "AMSE_Final_BA"] - sdf.loc[mask, bcol]) * 100
        fmask = fdf.dataset == dataset
        # Dataset-level model choice is fixed; fold deltas compare fold means.
        fdf.loc[fmask, "AMSE_Final_minus_best_ref_pp"] = (fdf.loc[fmask, "AMSE_Final_BA"] - fdf.loc[fmask, f"{best_models[dataset]}_BA"]) * 100
    summaries = []; branch_rows = []; fusion_rows = []
    for dataset in ("OpenBMI", "WBCIC"):
        ds = sdf[sdf.dataset == dataset]; ff = fdf[fdf.dataset == dataset]; best_model = best_models[dataset]; best_ref = float(ds[f"{best_model}_BA"].mean()); ref50 = float(ds.REF50_BA.mean()); final = float(ds.AMSE_Final_BA.mean()); stable = float(ds.AMSE_Stable_BA.mean()); expressive = float(ds.AMSE_Expressive_BA.mean()); gref = (ref50 - best_ref) * 100; gamse = (final - best_ref) * 100
        boot_best = bootstrap_delta((ds.AMSE_Final_BA.to_numpy() - ds[f"{best_model}_BA"].to_numpy()) * 100); boot_ref50 = bootstrap_delta((ds.AMSE_Final_BA.to_numpy() - ds.REF50_BA.to_numpy()) * 100)
        r = gamse / gref if gref > 0 else None; posfold = int((ff.AMSE_Final_minus_best_ref_pp > 0).sum())
        summaries.append({"dataset": dataset, "EEGNet_BA": float(ds.EEGNet_BA.mean()), "LiteBN_BA": float(ds.LiteBN_BA.mean()), "Best_Single_Model": best_model, "REF50_BA": ref50, "AMSE_Stable_BA": stable, "AMSE_Expressive_BA": expressive, "AMSE_Final_BA": final, "best_single_ref_BA": best_ref, "G_REF_pp": gref, "G_AMSE_pp": gamse, "R": r, "G_INTERNAL_pp": (final - max(stable, expressive)) * 100, "final_positive_folds": posfold, "final_bootstrap_ci_low_pp": boot_best["ci_low_pp"], "final_bootstrap_ci_high_pp": boot_best["ci_high_pp"], "final_vs_REF50_ci_low_pp": boot_ref50["ci_low_pp"], "final_vs_REF50_ci_high_pp": boot_ref50["ci_high_pp"], "positive_subjects_fixed_best": boot_best["positive_subjects"], "negative_subjects_fixed_best": boot_best["negative_subjects"]})
        branch_rows.append({"dataset": dataset, "stable_minus_EEGNet_pp": float(ds.Stable_minus_EEGNet_pp.mean()), "expressive_minus_LiteBN_pp": float(ds.Expressive_minus_LiteBN_pp.mean()), "stable_positive_folds": int((ff.Stable_minus_EEGNet_pp >= 0).sum()), "expressive_positive_folds": int((ff.Expressive_minus_LiteBN_pp >= 0).sum()), "final_minus_strongest_internal_pp": float(ds.Final_minus_strongest_internal_pp.mean())})
        fusion_rows.append({"dataset": dataset, "G_REF_pp": gref, "G_AMSE_pp": gamse, "R": r, "G_INTERNAL_pp": (final - max(stable, expressive)) * 100, "final_vs_REF50": boot_ref50})
    summary_df = pd.DataFrame(summaries); both = {r["dataset"]: r for r in summaries}; stable_comp = all(r["stable_minus_EEGNet_pp"] >= -.5 for r in branch_rows); exp_comp = all(r["expressive_minus_LiteBN_pp"] >= -.5 for r in branch_rows); strong = all(both[d]["G_AMSE_pp"] > 0 and (both[d]["R"] is not None and both[d]["R"] >= .8) and both[d]["final_positive_folds"] >= 3 for d in both) and stable_comp and exp_comp and all(r["final_minus_strongest_internal_pp"] > 0 for r in branch_rows)
    promising = all(both[d]["G_AMSE_pp"] > 0 and (both[d]["R"] is not None and both[d]["R"] >= .5) for d in both) and any((both[d]["R"] or 0) >= .8 for d in both) and stable_comp and exp_comp
    terminal = "AMSE_SEED0_STRONG" if strong else "AMSE_SEED0_PROMISING" if promising else "AMSE_SEED0_MIXED" if sum(both[d]["G_AMSE_pp"] > 0 for d in both) == 1 else "AMSE_SEED0_FAIL"
    sdf.to_csv(OUT / "SUBJECT_METRICS.csv", index=False); fdf.to_csv(OUT / "FOLD_METRICS.csv", index=False); summary_df.to_csv(OUT / "DATASET_SUMMARY.csv", index=False); pd.DataFrame(branch_rows).to_csv(OUT / "BRANCH_COMPETENCE.csv", index=False); pd.DataFrame(fusion_rows).to_csv(OUT / "FUSION_GAIN_RETENTION.csv", index=False); cdf.to_csv(OUT / "COMPLEMENTARITY.csv", index=False); sdf[["dataset", "fold", "subject_id", "Subject_oracle_ceiling_delta_pp"]].to_csv(OUT / "SUBJECT_ORACLE_DIAGNOSTIC.csv", index=False); write_json(OUT / "MODEL_COST.json", {"EEGNet_params_62": count(EEGNet(62)), "EEGNet_params_58": count(EEGNet(58)), "LiteBN_params_62": count(carrier.CompactLite(62, "bn")), "LiteBN_params_58": count(carrier.CompactLite(58, "bn")), "AMSE_v1_params_62": 57044, "AMSE_v1_params_58": 56852, "AMSE_v2_params_62": count(AMSE(62)), "AMSE_v2_params_58": count(AMSE(58)), "AMSE_v2_over_EEGNet_plus_LiteBN": {"62": count(AMSE(62)) / (count(EEGNet(62)) + count(carrier.CompactLite(62, "bn"))), "58": count(AMSE(58)) / (count(EEGNet(58)) + count(carrier.CompactLite(58, "bn")))}}); write_json(OUT / "REFERENCE_AUDIT.json", {"entries": ref_audit, "holdout_loaded": False}); write_json(OUT / "CHECKPOINT_SELECTION.json", training_rows); write_json(OUT / "TRAINING_LOGS.json", training_rows)
    lines = ["# AMSE-v2 seed-0 decision", "", f"Final terminal: **{terminal}**", "", "| Dataset | EEGNet | LiteBN | Best single | REF50 | Stable | Expressive | V2 Final | Δ vs best ref | Δ vs REF50 | R |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        rtxt = "NA" if r["R"] is None else f"{r['R']:.3f}"
        lines.append(f"| {r['dataset']} | {r['EEGNet_BA']*100:.2f}% | {r['LiteBN_BA']*100:.2f}% | {r['Best_Single_Model']} | {r['REF50_BA']*100:.2f}% | {r['AMSE_Stable_BA']*100:.2f}% | {r['AMSE_Expressive_BA']*100:.2f}% | {r['AMSE_Final_BA']*100:.2f}% | {r['G_AMSE_pp']:+.2f} pp | {(r['AMSE_Final_BA']-r['REF50_BA'])*100:+.2f} pp | {rtxt} |")
    lines += ["", "1. Stable path competence: " + ("retained" if stable_comp else "collapse risk"), "2. Expressive path competence: " + ("retained" if exp_comp else "collapse risk"), "3. Final over strongest internal path: see `BRANCH_COMPETENCE.csv`.", "4. Final over strongest existing single: see Δ vs best ref.", "5. Independent fusion gain retention: see `FUSION_GAIN_RETENTION.csv`.", "6. OpenBMI/WBCIC direction: " + ("consistent" if all(r['G_AMSE_pp'] > 0 for r in summaries) else "not consistent"), "7. Multi-seed/ERP/SSVEP: STOP and await user decision; no automatic extension.", "8. Branch collapse: " + ("no evidence" if stable_comp and exp_comp else "possible"), "9. Cost: see `MODEL_COST.json`.", "10. This is SEARCH-only; no OpenBMI final holdout or WBCIC true outer data were loaded."]
    (OUT / "FINAL_SEED0_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(terminal, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--lock-only", action="store_true"); ap.add_argument("--smoke-test", action="store_true"); ap.add_argument("--run", action="store_true"); ap.add_argument("--device", default="auto"); a = ap.parse_args()
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else a.device); set_seed(0); CODE.mkdir(parents=True, exist_ok=True)
    if a.smoke_test: print(json.dumps(smoke_test(device), indent=2), flush=True); return 0
    if a.lock_only: lock_only(device); return 0
    if a.run: run_experiment(device); return 0
    ap.error("select --smoke-test, --lock-only, or --run"); return 2


if __name__ == "__main__": raise SystemExit(main())
