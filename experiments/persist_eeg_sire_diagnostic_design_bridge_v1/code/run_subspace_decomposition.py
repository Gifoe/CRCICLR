#!/usr/bin/env python3
"""Frozen Part A: Protected-only versus P-removed utility, with exact P/controls."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from source_audit import EXP, OLD, sha

sys.path.insert(0, str(OLD / "code"))
import run_peeh_bridge as peeh  # noqa: E402
import run_pswa_bridge as pswa  # noqa: E402

OUT = EXP / "outputs/subspace_decomposition"
RUNTIME = EXP / "runtime/part_a_embeddings"
MODEL_NAME = {"OFFICIAL_FINAL_LITEBN_REFERENCE": "Full SIRE-EEG (historical)",
              "B1_SAME_SCALE_63": "B1 SameScale63"}
METRICS = ("protected_only_WSBA", "complement_retained_WSBA", "intact_probe_WSBA",
           "full_model_WSBA", "PEEH_pp", "PSWA_pp")


def old_run(task, model, fold):
    return json.loads(pswa.peeh_result_path(task, model, fold, 0).read_text())


def replay_check() -> None:
    """Fail before computation if the source Table-12 bridge cannot be replayed."""
    old_task = pd.read_csv(OLD / "outputs/p3_temporal_bridge/PEEH_PSWA_TASK_SUMMARY.csv")
    old_pswa = pd.read_csv(OLD / "outputs/p3_temporal_bridge/pswa/SUBJECT_LEVEL_PSWA_SEED0.csv")
    old_peeh = pd.read_csv(OLD / "outputs/p3_temporal_bridge/SUBJECT_LEVEL_PEEH.csv")
    rows = []
    for task in peeh.TASKS:
        for model in peeh.MODELS:
            subject_pswa = []
            subject_peeh = []
            for fold in range(5):
                p = old_run(task, model, fold)
                r = json.loads(pswa.run_cache_path(task, model, fold, 0).read_text())
                assert p["protected_coordinates"] == r["protected_coordinates"]
                sets = pswa.random_sets(task, model, fold, 0, p["protected_rank"], p["active_rank"])
                assert pswa.sha256_json(sets) == r["random_sets_sha256"]
                for subject in pswa.subjects_sessions(task)[0]:
                    pp = [x["BA"] for x in r["protected_rows"] if str(x["subject_id"]) == subject]
                    rr = pd.DataFrame([x for x in r["random_rows"] if str(x["subject_id"]) == subject])
                    if not pp or rr.draw.nunique() != 100:
                        raise RuntimeError("PSWA replay cell incomplete")
                    random_ws = rr.groupby("draw").BA.min().mean()
                    subject_pswa.append((subject, fold, 100 * (min(pp) - random_ws)))
                subject_peeh.extend((str(x["subject_id"]), fold, x["PEEH_pp"]) for x in p["subject_rows"])
            sp = pd.DataFrame(subject_pswa, columns=["subject_id", "fold", "value"]).groupby("subject_id").value.mean()
            sh = pd.DataFrame(subject_peeh, columns=["subject_id", "fold", "value"]).groupby("subject_id").value.mean()
            prior = old_task[(old_task.task == task) & (old_task.model == model)].iloc[0]
            prior_sp = old_pswa[(old_pswa.task == task) & (old_pswa.model == model)].copy()
            prior_sh = old_peeh[(old_peeh.task == task) & (old_peeh.model == model)].copy()
            for frame, series, col in ((prior_sp, sp, "PSWA_pp"), (prior_sh, sh, "PEEH_pp")):
                frame.subject_id = frame.subject_id.astype(str)
                ref = frame.set_index("subject_id")[col].sort_index()
                if not np.allclose(ref.to_numpy(), series.reindex(ref.index).to_numpy(), atol=1e-8):
                    raise RuntimeError(f"Table-12 subject replay mismatch {task} {model} {col}")
            if abs(sp.mean() - prior.PSWA_mean_pp) > 1e-8 or abs(sh.mean() - prior.PEEH_pp) > 1e-8:
                raise RuntimeError(f"Table-12 aggregate replay mismatch {task} {model}")
            rows.append({"task": task, "model": model, "PEEH_pp": sh.mean(), "PSWA_pp": sp.mean(), "status": "PASS"})
    pswa.atomic_csv(OUT / "REPLAY_AUDIT.csv", rows)


def infer_missing(runtime, task, model_name, fold, bundle, mean, std, device):
    path = RUNTIME / task / model_name / f"fold{fold}_seed0.npz"
    checkpoint = peeh.checkpoint_path(model_name, task, fold, 0)
    if path.is_file():
        z = pswa.load_npz(path)
        if z["metadata"]["checkpoint_sha256"] != sha(checkpoint):
            raise RuntimeError("stale Part-A embedding")
        return z
    model = peeh.build_model(runtime, model_name, task, checkpoint, device)
    values = []
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None]
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None]
    with torch.inference_mode():
        for start in range(0, len(bundle.rows), 64):
            ids = np.arange(start, min(start + 64, len(bundle.rows)), dtype=np.int64)
            raw = bundle.signal_batch(ids)
            x = torch.as_tensor(np.ascontiguousarray(raw), dtype=torch.float32, device=device)
            values.append(model((x - mean_t) / torch.clamp(std_t, min=1e-6))[1].float().cpu().numpy())
    meta = peeh.bundle_metadata(bundle)
    result = {"h": np.concatenate(values).astype(np.float32), "y": meta.label.to_numpy(int),
              "subject": meta.subject_id.astype(str).to_numpy(dtype="U16"),
              "session": meta.session_id.to_numpy(int),
              "metadata": {"checkpoint_sha256": sha(checkpoint), "normalizer_sha256": sha(peeh.normalizer_path(task, fold)),
                           "missing_sessions_only": True, "eval_mode": True}}
    pswa.save_session_cache(path, result["h"], result["y"], result["subject"], result["session"], result["metadata"])
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def probe_session_scores(train_x, train_y, eval_x, eval_y, subject, session, task):
    classes = len(np.unique(train_y))
    pack = pswa.ridge_fit(train_x, train_y, classes)
    pred = pswa.ridge_predict(eval_x, pack)
    rows = {}
    for s in pswa.subjects_sessions(task)[0]:
        rows[s] = {}
        for t in pswa.subjects_sessions(task)[1]:
            mask = (subject.astype(str) == s) & (session == t)
            if not mask.any():
                raise RuntimeError(f"missing subject/session {task} {s} S{t}")
            rows[s][t] = pswa.ba(eval_y[mask], pred[mask], classes)
    return rows


def ci(values, *key):
    arr = np.asarray(values, float)
    rng = np.random.default_rng(peeh.stable_seed("P3-bridge-bootstrap", *key))
    draws = arr[rng.integers(0, len(arr), (20_000, len(arr)))].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main() -> None:
    if not (EXP / "protocol/SOURCE_MANIFEST.json").is_file():
        raise RuntimeError("run source_audit first")
    OUT.mkdir(parents=True, exist_ok=True)
    replay_check()
    runtime = peeh.load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_sub = pd.read_csv(OLD / "outputs/SUBJECT_RESULTS.csv")
    old_sub.subject_id = old_sub.subject_id.astype(str)
    old_session = pd.read_csv(OLD / "runtime/SESSION_RESULTS.csv")
    old_session.subject_id = old_session.subject_id.astype(str)
    old_peeh_subject = pd.read_csv(OLD / "outputs/p3_temporal_bridge/SUBJECT_LEVEL_PEEH.csv")
    old_pswa_subject = pd.read_csv(OLD / "outputs/p3_temporal_bridge/pswa/SUBJECT_LEVEL_PSWA_SEED0.csv")
    for frame in (old_peeh_subject, old_pswa_subject):
        frame.subject_id = frame.subject_id.astype(str)
    table12 = pd.read_csv(OLD / "outputs/p3_temporal_bridge/PEEH_PSWA_TASK_SUMMARY.csv")
    cell_rows, subject_rows, checkpoint_rows = [], [], []
    csgd = pswa.import_path("bridge_csgd", pswa.CSGD_CODE)
    for task in peeh.TASKS:
        if task == "WBCIC_MI":
            full_bundle, _ = csgd.build_wbcic_outer_bundle(runtime)
        else:
            full_bundle = runtime.base.build_bundle(task, pswa.subjects_sessions(task)[0])
        missing_bundle = peeh.subset_bundle(runtime, full_bundle, pswa.subjects_sessions(task)[1][:-1])
        for fold in range(5):
            mean, std, _ = runtime.base.load_tensor_pair(peeh.normalizer_path(task, fold))
            for model_name in peeh.MODELS:
                old = old_run(task, model_name, fold)
                emb = pswa.load_npz(pswa.peeh_embedding_path(task, model_name, fold, 0))
                basis = pswa.canonical_basis(emb["train_h"], emb["train_subject"], emb["train_session"], emb["train_y"], old)
                miss = infer_missing(runtime, task, model_name, fold, missing_bundle, mean, std, device)
                ev_h = np.concatenate([miss["h"], emb["eval_h"]])
                ev_y = np.concatenate([miss["y"], emb["eval_y"]])
                ev_subject = np.concatenate([miss["subject"], emb["eval_subject"]]).astype(str)
                ev_session = np.concatenate([miss["session"], emb["eval_session"]]).astype(int)
                q_frozen = pswa.load_npz(pswa.session_cache_path(task, model_name, fold, 0))
                if not np.array_equal(ev_y, q_frozen["y"]) or not np.array_equal(ev_subject, q_frozen["subject"].astype(str)) or not np.array_equal(ev_session, q_frozen["session"]):
                    raise RuntimeError("Part-A session embedding alignment mismatch")
                q_replayed = pswa.z_coordinates(ev_h, basis)
                if not np.allclose(q_replayed, q_frozen["h"], atol=1e-4, rtol=1e-4):
                    raise RuntimeError("Part-A canonical basis/session replay mismatch")
                p = old["protected_coordinates"]
                sets = pswa.random_sets(task, model_name, fold, 0, len(p), old["active_rank"])
                pswa_run = json.loads(pswa.run_cache_path(task, model_name, fold, 0).read_text())
                assert pswa.sha256_json(sets) == pswa_run["random_sets_sha256"]
                intact = probe_session_scores(emb["train_h"], emb["train_y"], ev_h, ev_y, ev_subject, ev_session, task)
                complement = probe_session_scores(peeh.erase(emb["train_h"], basis, p), emb["train_y"],
                    peeh.erase(ev_h, basis, p), ev_y, ev_subject, ev_session, task)
                random_complement = {s: [] for s in pswa.subjects_sessions(task)[0]}
                for selected in sets:
                    current = probe_session_scores(peeh.erase(emb["train_h"], basis, selected), emb["train_y"],
                        peeh.erase(ev_h, basis, selected), ev_y, ev_subject, ev_session, task)
                    for s, sessions in current.items():
                        random_complement[s].append(min(sessions.values()))
                p_frame, r_frame = pd.DataFrame(pswa_run["protected_rows"]), pd.DataFrame(pswa_run["random_rows"])
                p_frame.subject_id = p_frame.subject_id.astype(str)
                r_frame.subject_id = r_frame.subject_id.astype(str)
                session_frame = old_session[(old_session.task == task) & (old_session.variant == model_name) & (old_session.fold == fold)]
                checkpoint = peeh.checkpoint_path(model_name, task, fold, 0)
                checkpoint_rows.append({"task": task, "model": model_name, "fold": fold, "seed": 0,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                    "normalizer_sha256": sha(peeh.normalizer_path(task, fold)),
                    "protected_sha256": sha(pswa.peeh_result_path(task, model_name, fold, 0)),
                    "random_sets_sha256": pswa_run["random_sets_sha256"],
                    "basis_sha256": q_frozen["metadata"]["basis_sha256"],
                    "canonical_session_replay": "PASS"})
                current_subject = []
                for s in pswa.subjects_sessions(task)[0]:
                    p_rows = p_frame[p_frame.subject_id == s]
                    r_rows = r_frame[r_frame.subject_id == s]
                    model_rows = session_frame[session_frame.subject_id == s]
                    if len(model_rows) != len(pswa.subjects_sessions(task)[1]):
                        raise RuntimeError("full-decoder session result missing")
                    p_ws = p_rows.BA.min()
                    random_ws = r_rows.groupby("draw").BA.min().mean()
                    peeh_row = next(x for x in old["subject_rows"] if str(x["subject_id"]) == s)
                    row = {"task": task, "model": model_name, "fold": fold, "seed": 0, "subject_id": s,
                           "active_rank": old["active_rank"], "protected_rank": len(p),
                           "protected_only_WSBA": float(p_ws), "random_only_WSBA": float(random_ws),
                           "PSWA_pp": 100 * (p_ws - random_ws), "PEEH_pp": peeh_row["PEEH_pp"],
                           "complement_retained_WSBA": min(complement[s].values()),
                           "random_complement_WSBA": float(np.mean(random_complement[s])),
                           "intact_probe_WSBA": min(intact[s].values()),
                           "full_model_WSBA": float(model_rows.BA.min())}
                    current_subject.append(row)
                    subject_rows.append(row)
                frame = pd.DataFrame(current_subject)
                cell_rows.append({"task": task, "model": model_name, "fold": fold, "seed": 0,
                                  "active_rank": old["active_rank"], "protected_rank": len(p),
                                  "protected_fraction": len(p) / old["active_rank"],
                                  "selected_blocks": sum(x.get("protected", False) for x in old["block_selection"]),
                                  "protected_coverage": int(len(p) > 0),
                                  **{m: frame[m].mean() for m in (*METRICS, "random_only_WSBA", "random_complement_WSBA")}})
                pswa.atomic_csv(OUT / "CELL_RESULTS.csv", cell_rows)
                pswa.atomic_csv(OUT / "SUBJECT_RESULTS.csv", subject_rows)
                print(f"PART_A {task} {model_name} f{fold}", flush=True)
        del full_bundle, missing_bundle
        gc.collect()
    if len(cell_rows) != 40:
        raise RuntimeError("Part A incomplete")
    cells, raw = pd.DataFrame(cell_rows), pd.DataFrame(subject_rows)
    biological = raw.groupby(["task", "model", "subject_id"], as_index=False)[list(METRICS) + ["random_only_WSBA", "random_complement_WSBA"]].mean()
    # Match Table 12 full-model estimator: session BA is fold-averaged before the session minimum.
    for row in biological.itertuples():
        ref = old_sub[(old_sub.task == row.task) & (old_sub.variant == row.model) & (old_sub.subject_id == row.subject_id)]
        if len(ref) != 1:
            raise RuntimeError("seed0 full-decoder subject result missing")
        biological.loc[(biological.task == row.task) & (biological.model == row.model) &
                       (biological.subject_id == row.subject_id), "full_model_WSBA"] = float(ref.iloc[0].WS_BA)
    pswa.atomic_csv(OUT / "BIOLOGICAL_SUBJECT_RESULTS.csv", biological)
    summary, effects = [], []
    for task in peeh.TASKS:
        for model in peeh.MODELS:
            group = biological[(biological.task == task) & (biological.model == model)]
            ranks = cells[(cells.task == task) & (cells.model == model)]
            ref = table12[(table12.task == task) & (table12.model == model)].iloc[0]
            if abs(group.PEEH_pp.mean() - ref.PEEH_pp) > 1e-8 or abs(group.PSWA_pp.mean() - ref.PSWA_mean_pp) > 1e-8:
                raise RuntimeError("Table12 aggregate replay lost after decomposition")
            if abs(group.full_model_WSBA.mean() - ref.full_model_WS_BA) > 1e-8:
                raise RuntimeError("Table12 full-model WSBA replay mismatch")
            summary.append({"task": task, "model": model, "protected_rank_mean": ranks.protected_rank.mean(),
                            "protected_coverage": f"{ranks.protected_coverage.sum()}/5",
                            **{m: group[m].mean() for m in (*METRICS, "random_only_WSBA", "random_complement_WSBA")}})
        a = biological[(biological.task == task) & (biological.model == "OFFICIAL_FINAL_LITEBN_REFERENCE")].set_index("subject_id")
        b = biological[(biological.task == task) & (biological.model == "B1_SAME_SCALE_63")].set_index("subject_id")
        if set(a.index) != set(b.index):
            raise RuntimeError("paired biological subjects mismatch")
        for metric in METRICS:
            delta = (b[metric] - a[metric]).to_numpy(float)
            lo, hi = ci(delta, task, metric)
            effects.append({"task": task, "effect": "B1_minus_Full", "metric": metric,
                            "n_subjects": len(delta), "mean": delta.mean(), "CI95_low": lo, "CI95_high": hi,
                            "unit": "pp" if metric.endswith("_pp") else "proportion", "bootstrap_draws": 20_000})
    pswa.atomic_csv(OUT / "CHECKPOINT_AUDIT.csv", checkpoint_rows)
    pswa.atomic_csv(OUT / "TASK_SUMMARY.csv", summary)
    pswa.atomic_csv(OUT / "FULL_VS_B1_PAIRED_EFFECTS.csv", effects)
    pswa.atomic_csv(OUT / "DECOMPOSITION_TABLE_MANUSCRIPT.csv", summary)
    lines = ["# SIRE-EEG Protected/complement-retained decomposition", "",
             "40/40 frozen seed0 analyses; no neural retraining. The official Full checkpoint is not a matched B0.",
             "Prior PEEH/PSWA and full-decoder Table-12 values replay exactly at subject and task level.",
             "Complement-retained denotes E_P(h): it retains non-P active coordinates and residual, not a strict raw P orthogonal complement.",
             "Probe utilities are not additive; no causal mediation claim follows.", "",
             "| Task | Model | rank(P) | Protected-only WSBA | Complement-retained WSBA | Intact-probe WSBA | PEEH pp | PSWA pp | Full-model WSBA |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['task']} | {MODEL_NAME[r['model']]} | {r['protected_rank_mean']:.2f} | {100*r['protected_only_WSBA']:.2f} | {100*r['complement_retained_WSBA']:.2f} | {100*r['intact_probe_WSBA']:.2f} | {r['PEEH_pp']:.2f} | {r['PSWA_pp']:.2f} | {100*r['full_model_WSBA']:.2f} |")
    lines += ["", "Paired B1−Full effects and 20,000-draw biological-subject CIs: `FULL_VS_B1_PAIRED_EFFECTS.csv`.",
              "All four tasks are retained regardless of direction. Protected rank is descriptive, not a quality score.",
              "The full-decoder task/subject WSBA follows the historical Table-12 estimator (fold-average session BA, then min); cell-level columns are per-fold min."]
    pswa.atomic_text(OUT / "FINAL_SUBSPACE_DECOMPOSITION_REPORT.md", "\n".join(lines))
    pswa.atomic_json(OUT / "COMPLETION.json", {"status": "PASS", "cells": 40, "neural_training_runs": 0,
        "table12_replay": "PASS", "protected_random_assignments": "exact existing", "bootstrap_draws": 20_000})
    print("PART_A_COMPLETE 40/40", flush=True)


if __name__ == "__main__":
    main()
