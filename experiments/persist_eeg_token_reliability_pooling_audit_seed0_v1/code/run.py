"""Frozen CBraMod token extraction for the OpenBMI MI reliability audit.

The `train` command is deliberately incapable of loading OUTER rows. `outer`
requires a pre-existing, immutable TRAIN-selection seal for its fold.
Neither command imports a final-heldout loader or accepts heldout paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
BENCH = REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"
MODERN = REPO / "experiments" / "persist_eeg_outcome_blind_modern_backbone_seed0_v1" / "code" / "modern_common.py"
RUNTIME = Path(os.environ["TOKEN_AUDIT_RUNTIME"]).resolve()
SOURCE_RUNTIME = Path(os.environ["SEVEN_RUNTIME"]).resolve()
CACHE = Path(os.environ["FULL_OPENBMI_CACHE"]).resolve()
CHANNEL_LOCK = REPO / "experiments" / "persist_eeg_fm_rescue_stage0" / "protocol" / "FM_INPUT_PROTOCOL_LOCK.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        x = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(str(x.dtype).encode() + b"\0")
        digest.update(np.asarray(x.shape, dtype=np.int64).tobytes())
        digest.update(x.tobytes())
    return digest.hexdigest()


def module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_split(fold_id: int):
    modern = module_from_path("token_audit_modern", MODERN)
    folds, _, split_hash = modern.load_split()
    fold = next(row for row in folds["OpenBMI"] if row["fold_id"] == fold_id)
    a, b, c = (set(fold[key]) for key in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects"))
    if not a or (a & b) or (a & c) or (b & c):
        raise RuntimeError("biological subject role overlap")
    return fold, split_hash


def load_source_module():
    sys.path.insert(0, str(BENCH))
    return module_from_path("token_audit_benchmark_data", BENCH / "tech_recipe_selection.py")


def data_rows(fold: dict, *, role: str):
    source = load_source_module()
    subjects = fold["inner_train_subjects"] if role == "train" else fold["outer_dev_subjects"]
    sessions = (1, 2) if role == "train" else (2,)
    x, y, owner, mapping = source._openbmi_rows(CACHE, subjects, sessions, "mi")
    if mapping != {1: 0, 2: 1} or x.shape[1:] != (62, 1000):
        raise RuntimeError("unexpected OpenBMI MI cache/label contract")
    # The original source helper groups by subject and, within subject, by
    # increasing session; verify this contract rather than infer from labels.
    observed = []
    for subject in source._sort_subjects(subjects):
        for session in sessions:
            base = CACHE / f"sub-{int(subject):02d}" / f"ses-{session}" / "mi_1train_codes.npy"
            codes = np.load(base, mmap_mode="r", allow_pickle=False)
            observed.extend([session] * len(codes))
    sess = np.asarray(observed, dtype=np.int8)
    if len(sess) != len(y) or len(set(map(str, owner))) != len(subjects):
        raise RuntimeError("session or subject inventory mismatch")
    return x, y, owner.astype(str), sess


def normalizer(fold: dict):
    source = load_source_module()
    x, _, _, _ = source._openbmi_rows(CACHE, fold["inner_train_subjects"], (1,), "mi")
    n = x.shape[0] * x.shape[2]
    total = x.sum(axis=(0, 2), dtype=np.float64)
    square = np.square(x, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
    mean = (total / n).astype(np.float32)
    std = np.sqrt(np.maximum(square / n - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    digest = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    return mean, std, digest


def model_for_fold(fold_id: int, *, channels: int = 62):
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    from backbone_models import build_model
    selected = SOURCE_RUNTIME / "search_cells" / "openbmi_mi" / "cbramod" / f"fold{fold_id}_seed0" / "selected.pt"
    record_path = selected.with_name("record.json")
    if not selected.is_file() or not record_path.is_file():
        raise FileNotFoundError("exact task-trained CBraMod checkpoint/record absent")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record["task"], record["model"], record["fold"], record["seed"]) != ("OpenBMI_MI", "CBraMod", fold_id, 0):
        raise RuntimeError("selected checkpoint record identity mismatch")
    if sha(selected) != record["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint SHA mismatch")
    model = build_model("CBraMod", dataset="OpenBMI", channels=channels, samples=1000, classes=2)
    saved = torch.load(selected, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    if saved["invariant"] != record["invariant_sha256"] or saved["selected_epoch"] != record["selected_epoch"]:
        raise RuntimeError("checkpoint/record invariant mismatch")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, selected, record


def token_index() -> list[dict]:
    channel_doc = json.loads(CHANNEL_LOCK.read_text(encoding="utf-8"))
    channels = channel_doc["OpenBMI"]["channels"]
    if len(channels) != 62:
        raise RuntimeError("frozen OpenBMI channel index mismatch")
    return [{"token_index": channel * 4 + patch, "channel_index": channel,
             "channel_name": name, "temporal_patch_index": patch,
             "approx_start_seconds": float(patch), "approx_end_seconds": float(patch + 1),
             "embedding_dimension": 200}
            for channel, name in enumerate(channels) for patch in range(4)]


def extract(args: argparse.Namespace):
    torch.set_num_threads(4)
    fold, split_hash = load_split(args.fold)
    out = RUNTIME / f"fold{args.fold}" / args.role
    if out.exists():
        raise FileExistsError(f"refusing duplicate extraction: {out}")
    if args.role == "outer":
        seal = RUNTIME / f"fold{args.fold}" / "TRAIN_SELECTION_SEAL.json"
        if not seal.is_file():
            raise RuntimeError("OUTER extraction forbidden before TRAIN selection seal")
        sealed = json.loads(seal.read_text(encoding="utf-8"))
        if sealed.get("fold") != args.fold or sealed.get("status") != "FROZEN_TRAIN_ONLY":
            raise RuntimeError("invalid TRAIN selection seal")
    model, selected, record = model_for_fold(args.fold)
    if record["split_sha256"] != split_hash:
        raise RuntimeError("selected checkpoint uses a different subject split")
    mean, std, norm_hash = normalizer(fold)
    if record["normalizer"]["mean_std_sha256"] != norm_hash:
        raise RuntimeError("selected checkpoint uses a different source normalizer")
    before = state_hash(model)
    if args.preflight:
        dummy = torch.zeros((2, 62, 4, 200))
        with torch.inference_mode():
            tokens = model.model.encoder(model.model.patch_embedding(dummy))
            logits = model.head(tokens.mean(dim=(1, 2)))
            reference = model(dummy.reshape(2, 62, 800))
        if tuple(tokens.shape) != (2, 62, 4, 200) or tuple(logits.shape) != (2, 2):
            raise RuntimeError("CBraMod pre-pooling hook shape failure")
        if not torch.allclose(logits, reference, atol=1e-6, rtol=1e-6):
            raise RuntimeError("pre-pooling hook differs from benchmark downstream forward")
        print(json.dumps({"fold": args.fold, "checkpoint_sha256": sha(selected),
                          "split_sha256": split_hash, "normalizer_sha256": norm_hash,
                          "state_sha256": before, "token_shape": list(tokens.shape),
                          "role": "PREFLIGHT_TRAIN_SESSION1_EEG_ONLY",
                          "formal_final_heldout_eeg_reads": 0}, sort_keys=True))
        return
    device = torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12_000 * 1024**2:
        raise RuntimeError("GPU extraction gate: fewer than 12000 MiB free")
    x, y, owner, sessions = data_rows(fold, role=args.role)
    out.mkdir(parents=True, exist_ok=False)
    # Runtime-only cache: no trial token arrays are ever committed to Git.
    z_path = out / "tokens.npy"
    z = np.lib.format.open_memmap(z_path, mode="w+", dtype=np.float32, shape=(len(y), 248, 200))
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(y), args.batch):
            stop = min(start + args.batch, len(y))
            normalized = ((x[start:stop] - mean[None, :, None]) /
                          np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
            native = resample_poly(normalized.astype(np.float64), 4, 5, axis=-1).astype(np.float32)
            if native.shape[1:] != (62, 800):
                raise RuntimeError("native CBraMod resampling shape mismatch")
            value = torch.from_numpy(native.reshape(stop-start, 62, 4, 200)).to(device)
            tokens = model.model.encoder(model.model.patch_embedding(value))
            if tuple(tokens.shape[1:]) != (62, 4, 200):
                raise RuntimeError("pre-pooling token shape changed")
            z[start:stop] = tokens.reshape(stop-start, 248, 200).cpu().numpy()
            if start % (10 * args.batch) == 0:
                print(f"fold{args.fold}/{args.role}: {stop}/{len(y)}", flush=True)
    z.flush()
    after = state_hash(model)
    if before != after:
        raise RuntimeError("CBraMod parameter or buffer changed during extraction")
    np.save(out / "labels.npy", y, allow_pickle=False)
    np.save(out / "subjects.npy", owner.astype("U16"), allow_pickle=False)
    np.save(out / "sessions.npy", sessions, allow_pickle=False)
    metadata = {"fold": args.fold, "role": args.role, "subjects": sorted(set(owner)),
                "sessions": sorted(set(sessions.tolist())), "trials": len(y),
                "token_shape": list(z.shape), "hook": "CBraModAdapter.model.encoder output, before Identity proj_out and mean(dim=(1,2))",
                "checkpoint": str(selected), "checkpoint_sha256": sha(selected),
                "source_code_sha256": sha(BENCH / "backbone_models.py"),
                "official_code_sha256": sha(Path(os.environ["OFFICIAL_BACKBONE_ROOT"]) / "CBraMod" / "models" / "cbramod.py"),
                "preprocessing_code_sha256": sha(BENCH / "benchmark_data.py"),
                "split_sha256": split_hash, "normalizer_sha256": norm_hash,
                "model_state_before_sha256": before, "model_state_after_sha256": after,
                "formal_final_heldout_eeg_reads": 0,
                "files": {name: sha(out / name) for name in ("tokens.npy", "labels.npy", "subjects.npy", "sessions.npy")}}
    (out / "EXTRACTION_PROVENANCE.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "fold": args.fold, "role": args.role,
                      "provenance_sha256": sha(out / "EXTRACTION_PROVENANCE.json")}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--role", choices=("train", "outer"), default="train")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > 64:
        raise ValueError("batch out of locked range")
    extract(args)


if __name__ == "__main__":
    main()
