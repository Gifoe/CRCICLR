"""One frozen EEGNet fold/stage analysis, after independently verified geometry."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

import data_geometry as dg
import decoders
import semantics


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty output {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def verify_geometry(fold: int, stage: str, checkpoint_sha: str, context: dict) -> tuple[dict, dict]:
    path = dg.RUNTIME / "geometry" / f"fold{fold}_seed0"
    provenance = json_read(path / "GEOMETRY_PROVENANCE.json")
    if provenance["schema"] != "P_SEMANTICS_EEGNET_INNER_TRAIN_GEOMETRY_V1":
        raise RuntimeError("unknown geometry schema")
    if provenance["fold"] != fold or provenance["population"] != "TRAIN_GEOMETRY only":
        raise RuntimeError("geometry fold or fit population mismatch")
    if provenance["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("geometry/checkpoint provenance mismatch")
    if provenance["checkpoint_split_sha256"] != context["split_sha256"]:
        raise RuntimeError("geometry/split provenance mismatch")
    if provenance["checkpoint_normalizer_sha256"] != context["normalizer_sha256"]:
        raise RuntimeError("geometry/normalizer provenance mismatch")
    for key in ("checkpoint_validation_eeg_reads", "outer_development_eeg_reads", "final_heldout_eeg_reads"):
        if provenance[key] != 0:
            raise RuntimeError(f"geometry fitting leaked role: {key}")
    artifact = np.load(path / f"{stage}.npz", allow_pickle=False)
    entry = {key: artifact[key] for key in ("mu", "q")}
    if dg.array_sha(entry["mu"], entry["q"]) != provenance["stage_geometry_sha256"][stage]:
        raise RuntimeError("stage geometry content hash mismatch")
    if entry["q"].shape[1] != provenance["final_P_rank"]:
        raise RuntimeError("stage rank does not match final P rank")
    if np.max(np.abs(entry["q"].T @ entry["q"] - np.eye(entry["q"].shape[1]))) > 1e-5:
        raise RuntimeError("stage projector not orthonormal")
    return entry, provenance


def cap_held(pop: dg.Population, fold: int, een) -> dg.Population:
    chosen = []
    for session in sorted(set(pop.session)):
        ix = np.flatnonzero(pop.session == session)
        kept = een.capped_indices(pop.subject[ix], pop.y[ix], int(session), "OpenBMI_MI", fold, "evaluation")
        chosen.extend(ix[kept].tolist())
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    return dg.Population(pop.x[chosen], pop.y[chosen], pop.subject[chosen], pop.session[chosen], pop.role)


def extract_stage(pop: dg.Population, stages, stage: str, *, batch: int = 32) -> np.ndarray:
    values = []
    stages.m.eval()
    with torch.inference_mode():
        for start in range(0, len(pop.x), batch):
            x = stages.tensor(pop.x[start:start + batch])
            by_stage, _, native = stages.all(x)
            if not torch.allclose(native, stages.m(x), rtol=1e-5, atol=1e-6):
                raise RuntimeError("stage/native forward mismatch")
            values.append(by_stage[stage].reshape(len(native), -1).float().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def head_rows(fold: int, geometry: dict, populations: dict, head, device) -> list[dict]:
    mu, q = geometry["mu"], geometry["q"]
    train_pop, train = populations["TRAIN_GEOMETRY"]
    # The class-common reference is fitted on TRAIN only. Since mu is the
    # TRAIN mean, both reference residual components should be near zero.
    train_centered = train - mu
    p_reference = ((train_centered @ q) @ q.T).mean(axis=0)
    c_reference = (train_centered - (train_centered @ q) @ q.T).mean(axis=0)
    rows = []
    for family in ("FULL_NATIVE", "P_RETAINED", "C_RETAINED", "RANDOM_P_RETAINED"):
        for draw in (range(20) if family == "RANDOM_P_RETAINED" else (-1,)):
            basis = dg.random_subspace(len(mu), q.shape[1], fold, "embedding_64d", draw) if draw >= 0 else q
            for role, (pop, raw) in populations.items():
                for session in sorted(set(pop.session)):
                    ix = pop.session == session
                    source = raw[ix]
                    z = source - mu
                    if family == "FULL_NATIVE":
                        intervention = source
                    elif family == "P_RETAINED":
                        intervention = mu + ((z @ q) @ q.T) + c_reference
                    elif family == "C_RETAINED":
                        intervention = mu + p_reference + z - ((z @ q) @ q.T)
                    else:
                        intervention = mu + ((z @ basis) @ basis.T) + c_reference
                    with torch.inference_mode():
                        logits = head(torch.from_numpy(np.ascontiguousarray(intervention, dtype=np.float32)).to(device)).cpu().numpy()
                        native = head(torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32)).to(device)).cpu().numpy()
                    softmax = lambda a: np.exp(a - a.max(axis=1, keepdims=True)) / np.exp(a - a.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)
                    probability = softmax(logits)
                    pred = probability.argmax(1)
                    native_pred = native.argmax(1)
                    rows.append({"fold": fold, "stage": "embedding_64d", "population": role,
                        "session": session, "family": family, "random_draw": draw if draw >= 0 else "",
                        "BA": float(balanced_accuracy_score(pop.y[ix], pred)),
                        "macro_F1": float(f1_score(pop.y[ix], pred, average="macro", zero_division=0)),
                        "NLL": float(log_loss(pop.y[ix], probability, labels=[0, 1])),
                        "native_prediction_agreement": float(np.mean(pred == native_pred)),
                        "rows": int(ix.sum()), "head_frozen": True,
                        "reference_population": "TRAIN_GEOMETRY only",
                        "causal_attribution_claim": False})
    return rows


def run(fold: int, stage: str, part: str) -> None:
    if stage not in dg.STAGES:
        raise KeyError(stage)
    target = dg.RUNTIME / "analysis" / f"fold{fold}_seed0" / stage / part
    if target.exists():
        raise FileExistsError(f"analysis output already exists: {target}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    state_sha = dg.model_sha(model)
    geometry, provenance = verify_geometry(fold, stage, dg.file_sha(ckpt), context)
    # The first held-role EEG read is strictly after all geometry checks.
    held = {role: cap_held(dg.load_held(role, context, een, up), fold, een)
            for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT")}
    stages = pw.Stages(model, head, "EEGNet", device)
    populations = {"TRAIN_GEOMETRY": (train, extract_stage(train, stages, stage))}
    populations.update({role: (pop, extract_stage(pop, stages, stage)) for role, pop in held.items()})
    for role, (pop, x) in populations.items():
        if x.shape[1] != len(geometry["mu"]) or set(pop.subject) != set(context["ids"][role]):
            raise RuntimeError(f"stage dimension or role mismatch {role}")
    outputs = {}
    if part == "semantics":
        outputs = semantics.summarize_centroids(fold, stage, geometry, populations, context["sessions"])
    elif part == "decoders":
        outputs = {name: [] for name in ("P_SEMANTIC_PROBES", "FROZEN_DECODER_SESSION_TRANSFER",
                                            "FROZEN_DECODER_CROSS_SUBJECT_TRANSFER", "DECODER_BOUNDARY_STABILITY")}
        for family in ("FULL", "P", "C", "RANDOM"):
            for draw in (range(20) if family == "RANDOM" else (-1,)):
                a, b, c = decoders.decoder_rows(fold, stage, geometry, populations, context["sessions"], family, draw)
                outputs["FROZEN_DECODER_SESSION_TRANSFER"].extend(a)
                outputs["FROZEN_DECODER_CROSS_SUBJECT_TRANSFER"].extend(b)
                outputs["DECODER_BOUNDARY_STABILITY"].extend(c)
                outputs["P_SEMANTIC_PROBES"].extend(decoders.probe_rows(fold, stage, geometry, populations,
                                                                         context["sessions"], family, draw))
        if stage == "embedding_64d":
            outputs["ORIGINAL_HEAD_P_FUNCTION"] = head_rows(fold, geometry, populations, head, device)
    else:
        raise KeyError(part)
    if dg.model_sha(model) != state_sha:
        raise RuntimeError("frozen neural parameter/BatchNorm state changed")
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in outputs.items():
        path = target / f"{name}.csv"
        write_rows(path, rows)
        hashes[path.name] = {"sha256": dg.file_sha(path), "rows": len(rows)}
    audit = {"schema": "P_SEMANTICS_FOLD_STAGE_ANALYSIS_V1", "fold": fold, "stage": stage, "part": part,
             "checkpoint_sha256": dg.file_sha(ckpt), "geometry_provenance_sha256": dg.file_sha(
                 dg.RUNTIME / "geometry" / f"fold{fold}_seed0" / "GEOMETRY_PROVENANCE.json"),
             "stage_geometry_sha256": provenance["stage_geometry_sha256"][stage],
             "model_state_sha256_before_after": state_sha,
             "role_subject_counts": {role: len(set(pop.subject)) for role, (pop, _) in populations.items()},
             "role_trial_rows": {role: len(pop.y) for role, (pop, _) in populations.items()},
             "fits_use_train_only": True, "outer_development_fit_rows": 0,
             "checkpoint_validation_fit_rows": 0, "final_heldout_eeg_reads": 0,
             "files": hashes}
    with (target / "AUDIT.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"fold": fold, "stage": stage, "part": part, "files": hashes,
                      "final_heldout_eeg_reads": 0}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("semantics", "decoders"))
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--stage", choices=dg.STAGES, required=True)
    args = parser.parse_args()
    run(args.fold, args.stage, args.part)
