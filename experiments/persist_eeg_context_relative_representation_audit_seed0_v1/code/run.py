"""Frozen EEGNet stage extraction. Only named role subjects can reach the EEG reader.

The context-reference analysis lives in aggregate.py. OUTER extraction requires a
persisted TRAIN selection lock and never participates in selection.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("REL_RUNTIME", r"D:\nips-temp\TotalP\P1\context_relative_representation_runtime"))
SEVEN = Path(os.environ.get("SEVEN_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK"))
SDG = Path(os.environ.get("SDG_SOURCE", r"D:\nips-temp\TotalP\P1\shared_decision_geometry_full_runtime\code\run_full_v1.py"))
LOCK = json.loads((HERE / "protocol/PROTOCOL_LOCK.json").read_text(encoding="utf-8"))


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def stage_forward(model: torch.nn.Module, x: torch.Tensor):
    # Same four canonical boundaries as protected_pathway_mechanism.Stages.all.
    a = model.bn1(model.temporal(x.unsqueeze(1)))
    b = model.drop1(model.pool1(F.elu(model.bn2(model.spatial(a)))))
    c = model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(b))))))
    h = model.embedding(c.flatten(1))
    logits = model.head(h)
    # TEMPORAL pooling is a declared secondary storage/decoder reduction. The
    # canonical tensor boundary is unchanged; SPATIAL/SHARED are flattened.
    values = {
        "TEMPORAL": F.adaptive_avg_pool2d(a, (a.shape[2], 16)).flatten(1),
        "SPATIAL": b.flatten(1),
        "SHARED": c.flatten(1),
        "EMBEDDING": h,
    }
    shapes = {"TEMPORAL": list(a.shape[1:]), "SPATIAL": list(b.shape[1:]),
              "SHARED": list(c.shape[1:]), "EMBEDDING": list(h.shape[1:])}
    return values, logits, shapes


def extract(model, x: np.ndarray, device: torch.device):
    features = {name: [] for name in LOCK["stages"]}
    raw_shapes = None
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            batch = torch.from_numpy(np.ascontiguousarray(x[start:start + 32])).to(device)
            # Audit native equivalence and immutable state on every data role.
            vals, logits, shapes = stage_forward(model, batch)
            native = model(batch)
            if not torch.allclose(logits, native, rtol=1e-5, atol=1e-6):
                raise RuntimeError("manual canonical stages diverge from native logits")
            if raw_shapes is None:
                raw_shapes = shapes
            for key, tensor in vals.items():
                features[key].append(tensor.float().cpu().numpy())
    return {key: np.concatenate(parts).astype(np.float32) for key, parts in features.items()}, raw_shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--role", required=True, choices=("TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"))
    parser.add_argument("--attempt", default="v1", choices=("v1", "v2"))
    args = parser.parse_args()
    if args.role == "OUTER_DEVELOPMENT":
        selection = ROOT / "train_lock" / f"fold{args.fold}.json"
        if not selection.is_file():
            raise RuntimeError("OUTER EEG read forbidden before TRAIN-only selection lock")
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        if chosen.get("role") != "TRAIN_GEOMETRY" or chosen.get("fold") != args.fold:
            raise RuntimeError("selection identity mismatch")
    suffix = "" if args.attempt == "v1" else f"_{args.attempt}"
    out = ROOT / "features" / f"fold{args.fold}_{args.role.lower()}{suffix}"
    if out.exists():
        raise FileExistsError(out)
    sdg = import_file(f"rel_sdg_{args.fold}", SDG)
    inner_path = sdg.SEVEN_CODE / "tech_recipe_selection.py"
    inner = sdg.module(f"rel_inner_{args.fold}", inner_path)
    ids, sessions, split_sha, cache_name = sdg.split_role("OpenBMI_MI", args.fold)
    assert cache_name == "mi" and len(ids["TRAIN_GEOMETRY"]) == 26 and len(ids["OUTER_DEVELOPMENT"]) == 8
    source, sy, ss, mapping = inner._openbmi_rows(sdg.CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[0],), cache_name)
    mean, std, norm_sha = sdg.historical_normalizer(source)
    model, head, cp = sdg.checkpoint("EEGNet", "OpenBMI_MI", args.fold, split_sha, norm_sha, source.shape[1:])
    del source, sy, ss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    before = sdg.state_sha(model)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"fold": args.fold, "role": args.role, "attempt": args.attempt,
                "role_subjects": list(ids[args.role]),
                "sessions": list(sessions), "split_sha256": split_sha, "normalizer_sha256": norm_sha,
                "normalizer_source": "all TRAIN_GEOMETRY S1 trials", "model_source_sha256": cp["model_source_sha256"],
                "checkpoint_sha256": cp["checkpoint_sha256"], "checkpoint_record_sha256": cp["record_sha256"],
                "checkpoint_path": cp["checkpoint_path"], "model_state_before": before,
                "sdg_source_sha256": sha(SDG), "inner_source_sha256": sha(inner_path), "final_heldout_eeg_reads": 0,
                "stage_files": {}, "stage_audit": {}}
    for session in sessions:
        raw, y, owner, new_mapping = inner._openbmi_rows(
            sdg.CACHE, list(ids[args.role]), (session,), cache_name, mapping)
        if new_mapping != mapping or set(owner.astype(str)) != set(ids[args.role]):
            raise RuntimeError("dataset label mapping or role drift")
        x = ((raw - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
        del raw
        vals, raw_shapes = extract(model, x, device)
        del x
        if sdg.state_sha(model) != before:
            raise RuntimeError("neural parameters or normalization state changed")
        # The reader preserves cache acquisition order within each subject.
        for stage, h in vals.items():
            path = out / f"session{session}_{stage.lower()}.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, h=h, y=y.astype(np.int64), subject=owner.astype(str),
                                    session=np.full(len(y), session, dtype=np.int64))
            metadata["stage_files"][f"session{session}_{stage}"] = {"path": str(path), "sha256": sha(path),
                "feature_sha256": array_sha(h), "rows": int(len(h)), "dim": int(h.shape[1])}
            metadata["stage_audit"][stage] = {"canonical_boundary": LOCK["stages"][stage],
                "raw_tensor_shape_excluding_batch": raw_shapes[stage], "feature_dim": int(h.shape[1]),
                "operation": "adaptive average time pool to 16 bins, then flatten" if stage == "TEMPORAL" else "flatten canonical tensor",
                "primary": stage == "EMBEDDING"}
    metadata["model_state_after"] = sdg.state_sha(model)
    if metadata["model_state_after"] != before:
        raise RuntimeError("model state drift")
    save_json_new(out / "PROVENANCE.json", metadata)
    print(json.dumps({"fold": args.fold, "role": args.role, "rows": metadata["stage_files"],
                      "provenance_sha256": sha(out / "PROVENANCE.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
