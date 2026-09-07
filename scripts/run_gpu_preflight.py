#!/usr/bin/env python
from __future__ import annotations

import hashlib
from pathlib import Path
import traceback

import h5py
import numpy as np
import torch

from hsc_tta.actions import EntropyAdapter, T3A
from hsc_tta.backbones import CBraModInputAdapter, FrozenCBraMod
from hsc_tta.gpu.embeddings import extract_subject, load_embedding
from hsc_tta.models import TaskHead


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
CHECKPOINT = ROOT / "checkpoints" / "cbramod" / "pretrained_weights.pth"
SOURCE = ROOT / "external" / "CBraMod"
EXPECTED = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def sample(dataset: str) -> tuple[Path, np.ndarray, list[str], float]:
    source = sorted((ROOT / "data" / "processed" / dataset).glob("*.h5"))[0]
    with h5py.File(source, "r") as handle:
        x = handle["signal"][:1]
        names = [v.decode() for v in handle["channel_names"][...]]
        rate = float(handle["sampling_rate"][()])
    return source, x, names, rate


def main() -> int:
    report = ROOT / "outputs" / "full_experiment" / "environment" / "preflight_report.md"
    blocker = ROOT / "outputs" / "full_experiment" / "BLOCKER_REPORT.md"
    checks: list[str] = []
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (12, 0):
            raise RuntimeError("RTX 5090 sm_120 CUDA gate failed")
        checks.append("GPU available and capability sm_120")
        if file_hash(CHECKPOINT) != EXPECTED:
            raise RuntimeError("checkpoint hash mismatch")
        checks.append("official checkpoint SHA256")
        adapter = CBraModInputAdapter()
        backbone = FrozenCBraMod(SOURCE, CHECKPOINT).cuda().eval()
        original_hash = backbone.frozen_hash
        representations = {}
        for dataset in ("hmc", "cap", "eegmmidb"):
            _, x, names, rate = sample(dataset)
            adapted = adapter.adapt(dataset, x, names, rate, device="cuda")
            y1 = backbone(adapted.tensor)
            y2 = backbone(adapted.tensor)
            if not torch.equal(y1, y2) or not torch.isfinite(y1).all():
                raise RuntimeError(f"{dataset} deterministic forward/finiteness failed")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                amp = backbone(adapted.tensor)
            if not torch.isfinite(amp).all():
                raise RuntimeError(f"{dataset} bfloat16 AMP failed")
            # Materialize a regular tensor; inference-mode tensors cannot participate in the head backward check.
            representations[dataset] = torch.from_numpy(y1.float().cpu().numpy().copy()).to("cuda")
            checks.append(f"{dataset} input adaptation and CBraMod forward")
        head = TaskHead(200, 5).cuda()
        logits, hidden = head(representations["hmc"], return_hidden=True)
        torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device="cuda")).backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise RuntimeError("task-head backward failed")
        checks.append("task-head float32 forward/backward")
        head.eval()
        weight = head.classifier.weight.detach().cpu().numpy()
        T3A(weight, filter_k=20).adapt(hidden.detach().cpu().numpy(), logits.detach().cpu().numpy()).predict_proba(hidden.detach().cpu().numpy())
        entropy = EntropyAdapter(head, steps=1, device="cuda")
        entropy.adapt(representations["hmc"].cpu().numpy())
        if entropy.diagnostics["gradient_nan_flag"]:
            raise RuntimeError("entropy-adapter gradient failed")
        checks.append("T3A and entropy adapter subject-local adaptation")
        mi_source, _, _, _ = sample("eegmmidb")
        mi_subject = f"eegmmidb:{mi_source.stem.split('_', 1)[1]}"
        episodes = __import__("pandas").read_parquet(ROOT / "data" / "episodes_main120" / "eegmmidb" / "seed_0.parquet")
        episode = episodes[episodes["subject_id"] == mi_subject].iloc[0]
        destination = ROOT / "outputs" / "full_experiment" / "environment" / "preflight_resume_eegmmidb.h5"
        first = extract_subject("eegmmidb", mi_source, destination, backbone, adapter,
            (np.asarray(episode.context_indices, int), np.asarray(episode.future_indices, int)), EXPECTED,
            "0ff6be918985689e7df679bc731ffb70e6c6224f", device="cuda", batch_size=4, resume=False)
        second = extract_subject("eegmmidb", mi_source, destination, backbone, adapter,
            (np.asarray(episode.context_indices, int), np.asarray(episode.future_indices, int)), EXPECTED,
            "0ff6be918985689e7df679bc731ffb70e6c6224f", device="cuda", batch_size=4, resume=True)
        cached = load_embedding(destination)
        if first["status"] != "complete" or second["status"] != "resumed":
            raise RuntimeError("one-subject resume gate failed")
        if not np.array_equal(cached["original_index"], np.arange(len(cached["label"]))) or not np.isfinite(cached["embedding"]).all():
            raise RuntimeError("embedding/index/label alignment gate failed")
        checks.append("one-subject atomic embedding, alignment, and resume")
        backbone.verify_frozen(check_hash=True)
        if backbone.frozen_hash != original_hash:
            raise RuntimeError("backbone hash changed")
        checks.append("frozen backbone before/after hash")
        free_gb = int(__import__("shutil").disk_usage(ROOT).free / 2**30)
        if free_gb < 60:
            raise RuntimeError(f"disk hard gate: {free_gb} GiB free")
        checks.append(f"disk space {free_gb} GiB")
        report.write_text("# GPU preflight: PASS\n\n" + "\n".join(f"- PASS: {x}" for x in checks) + "\n",
                          encoding="utf-8")
        if blocker.exists():
            blocker.unlink()
        print(report.read_text())
        return 0
    except Exception as error:
        message = f"# GPU preflight: BLOCKED\n\n{type(error).__name__}: {error}\n\n```\n{traceback.format_exc()}\n```\n"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(message, encoding="utf-8")
        blocker.write_text(message, encoding="utf-8")
        print(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
