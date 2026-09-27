"""Paired biological-subject bootstrap: G minus frozen rank-matched random."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

import run as core
from run import dg, dec, sem


def relation_by_subject(train_x, train_pop, outer_x, outer_pop, sessions):
    tg = sem.centroid_grid(train_x, train_pop)
    og = sem.centroid_grid(outer_x, outer_pop)
    _, td = sem.baseline_relation(tg, sessions)
    _, od = sem.baseline_relation(og, sessions)
    reference = {t: np.mean([td[(s, t)] for s in sorted(set(train_pop.subject))], axis=0)
                 for t in sessions}
    return {s: float(np.mean([sem.sim(od[(s, t)], reference[t])["cosine"] for t in sessions]))
            for s in sorted(set(outer_pop.subject))}


def ba_by_subject(train_x, train_pop, outer_x, outer_pop, sessions):
    grid = sem.centroid_grid(train_x, train_pop)
    keys = sorted(grid)
    x = np.stack([grid[k] for k in keys])
    labels = np.asarray([k[2] for k in keys], dtype=np.int64)
    scaler, model = dec.fit_linear(x, labels)
    probabilities = model.predict_proba(scaler.transform(outer_x))
    pred = model.classes_[probabilities.argmax(axis=1)]
    return {s: float(balanced_accuracy_score(outer_pop.y[outer_pop.subject == s],
                                             pred[outer_pop.subject == s]))
            for s in sorted(set(outer_pop.subject))}


def run(fold):
    target = core.RUNTIME / "subject_bootstrap" / f"fold{fold}.json"
    if target.exists():
        raise FileExistsError(target)
    if not (core.RUNTIME / "evaluation" / f"fold{fold}" / "AUDIT.json").is_file():
        raise RuntimeError("main evaluation absent")
    torch.set_num_threads(4)
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    state = dg.model_sha(model)
    stages = pw.Stages(model, head, "EEGNet", device)
    h = dg.extract(train, stages)[0][core.STAGE]
    spec, gate = core.gated_spec(fold, h, train, een)
    if dg.file_sha(ckpt) != gate["checkpoint_sha256"]:
        raise RuntimeError("checkpoint drift")
    from analysis_runner import cap_held, extract_stage
    outer = cap_held(dg.load_held("OUTER_DEVELOPMENT", context, een, up), fold, een)
    oh = extract_stage(outer, stages, core.STAGE)
    mu = h.mean(0, dtype=np.float64).astype(np.float32)
    qg = core.family_basis(spec, gate["families"]["PROTECTED_G"])
    if dg.array_sha(mu, qg) != gate["source_geometry_sha256"]:
        raise RuntimeError("G geometry drift")
    def measurements(q):
        tx, ox = (h - mu) @ q, (oh - mu) @ q
        return relation_by_subject(tx, train, ox, outer, context["sessions"]), ba_by_subject(
            tx, train, ox, outer, context["sessions"])
    g_relation, g_ba = measurements(qg)
    owners = sorted(g_relation)
    random_relation = {s: [] for s in owners}
    random_ba = {s: [] for s in owners}
    for draw in range(100):
        qr = dg.random_subspace(len(mu), qg.shape[1], fold, core.STAGE, draw)
        rr, rb = measurements(qr)
        for s in owners:
            random_relation[s].append(rr[s])
            random_ba[s].append(rb[s])
    rows = []
    for name, actual, random in (("cross_subject_relation_cosine", g_relation, random_relation),
                                 ("frozen_cross_subject_decoder_BA", g_ba, random_ba)):
        diff = np.asarray([actual[s] - float(np.mean(random[s])) for s in owners], dtype=np.float64)
        rng = np.random.default_rng(dg.seed("taskcore-G-random-subject-bootstrap", fold, name))
        ix = rng.integers(0, len(owners), size=(20_000, len(owners)))
        means = diff[ix].mean(axis=1)
        rows.append({"metric": name, "G_subject_equal_mean": float(np.mean([actual[s] for s in owners])),
                     "rank_random_subject_equal_mean": float(np.mean([np.mean(random[s]) for s in owners])),
                     "G_minus_random_subject_equal_mean": float(diff.mean()),
                     "bootstrap_CI_low": float(np.quantile(means, .025)),
                     "bootstrap_CI_high": float(np.quantile(means, .975)),
                     "bootstrap_draws": 20_000, "subjects": len(owners), "random_subspaces": 100})
    if dg.model_sha(model) != state:
        raise RuntimeError("neural/BatchNorm drift")
    core.write_new(target, {"fold": fold, "rank": qg.shape[1], "rows": rows,
                            "checkpoint_sha256": dg.file_sha(ckpt), "source_geometry_sha256": gate["source_geometry_sha256"],
                            "model_state_sha256_before_after": state,
                            "fit_population": "TRAIN_GEOMETRY", "outer_fit_rows": 0,
                            "final_heldout_eeg_reads": 0})
    print(json.dumps({"fold": fold, "status": "PAIRED_SUBJECT_BOOTSTRAP_COMPLETE"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    run(parser.parse_args().fold)
