"""Conditional Random-PC follow-up, explicitly exploratory after final heldout.

The decision rule is frozen in source before the main FINAL_EVAL_LOCK: run only
tasks with positive Protected-PC minus matched-baseline primary heldout BA.
Random coordinates and context bases were generated from the refit development
pool and hashed in the main final lock, before heldout was read. No architecture,
rank, PCA dimension, LR, or epoch count is selected from heldout performance.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

import run as v1

OUT = v1.OUTPUT / "exploratory_random"


def decision() -> list[str]:
    lock = v1.check_final_lock()
    path = v1.OUTPUT / "HELDOUT_MODEL_TASK_SUMMARY.csv"
    if not path.is_file(): raise RuntimeError("primary V1 final result must be complete first")
    rows = {(r["task"], r["variant"]): r for r in csv.DictReader(path.open(newline="", encoding="utf-8"))}
    if len(rows) != 8: raise RuntimeError("primary V1 task/model summary incomplete")
    gains = {task: float(rows[task, "PROTECTED_PC_REFINE"]["BA"]) - float(rows[task, "BASELINE"]["BA"])
             for task in v1.TASKS}
    eligible = [task for task in v1.TASKS if gains[task] > 0]
    v1.json_write(OUT / "EXPLORATORY_RANDOM_DECISION.json", {
        "phase": "EXPLORATORY_POST_HELDOUT", "trigger": "per-task primary heldout BA point difference > 0",
        "primary_final_lock_sha256": v1.sha(v1.PROTOCOL / "FINAL_EVAL_LOCK.json"),
        "primary_result_sha256": v1.sha(path), "source_sha256": v1.sha(Path(__file__)),
        "gains": gains, "eligible_tasks": eligible, "random_bases_prelocked": True,
        "epoch_schedule": "reuse corresponding Protected-PC discovery-selected phase1/phase2 epochs; no random-control outcome selection"})
    return eligible


def train_cell(task: str, fold: int) -> None:
    v1.check_final_lock()
    refit = v1.cell_dir(task, fold, "refit")
    target = v1.cell_dir(task, fold, "exploratory_random")
    if (target / "COMPLETE.json").exists(): return
    cell = json.loads((refit / "COMPLETE.json").read_text(encoding="utf-8"))
    data = v1.development(task, fold, refit=True)
    if data["normalizer"]["mean_std_sha256"] != cell["normalizer_sha256"]: raise RuntimeError("normalizer mismatch")
    base = v1.eegnet(data)
    base.load_state_dict(torch.load(refit / "BASELINE.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    base.eval()
    stored = np.load(refit / "bases.npz", allow_pickle=False)
    basis = {key: stored[key] for key in stored.files}
    v1.seed_all(v1.stable_seed("adapter-initial", task, fold, "RANDOM_PC_REFINE"))
    model = v1.make_variant(base, basis, "RANDOM_PC_REFINE")
    identity = v1.identity_audit(base, model, data["source"])
    epochs = cell["selected_epochs"]["PROTECTED_PC_REFINE"]
    model, _, history1 = v1.train_adapter(model, data, task, fold, 1, int(epochs[0]), False, "RANDOM_PC_REFINE")
    model, _, history2 = v1.train_adapter(model, data, task, fold, 2, int(epochs[1]), False, "RANDOM_PC_REFINE")
    ck = target / "RANDOM_PC_REFINE.pt"
    v1.torch_write(ck, {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epochs": epochs})
    v1.csv_write(target / "TRAINING_AUDIT.csv", [{"task": task, "fold": fold, **r} for r in history1+history2])
    v1.json_write(target / "COMPLETE.json", {"phase": "EXPLORATORY_POST_HELDOUT", "task": task, "fold": fold,
                                              "checkpoint_sha256": v1.sha(ck), "identity": identity,
                                              "random_subspace_hash": cell["basis_hashes"]["qr"],
                                              "random_PCA_hash": cell["basis_hashes"]["ur"]})


def evaluate_task(task: str) -> None:
    lock = v1.check_final_lock()
    sessions = (0,1,2) if task == "WBCIC_MI" else (1,2)
    held_ids = lock["heldout_subjects"]["WBCIC_true_outer" if task == "WBCIC_MI" else "OpenBMI"]
    for fold in v1.FOLDS:
        target = v1.cell_dir(task, fold, "exploratory_random")
        if (target / "predictions.npz").exists(): continue
        rec = json.loads((target / "COMPLETE.json").read_text(encoding="utf-8"))
        data = v1.development(task, fold, refit=True)
        stored = np.load(v1.cell_dir(task, fold, "refit") / "bases.npz", allow_pickle=False)
        basis = {key: stored[key] for key in stored.files}
        model = v1.make_variant(v1.eegnet(data), basis, "RANDOM_PC_REFINE")
        ck = target / "RANDOM_PC_REFINE.pt"
        if v1.sha(ck) != rec["checkpoint_sha256"]: raise RuntimeError("exploratory checkpoint hash mismatch")
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        model.eval()
        output = {}
        for session in sessions:
            x,y,s,_ = v1.rows(task, held_ids, (session,), data["cache_name"], data["mapping"], final=(task == "WBCIC_MI"))
            x = ((x-data["mu"][None,:,None])/np.maximum(data["sd"][None,:,None],1e-6)).astype(np.float32)
            output[f"S{session}_z"] = v1.logits(model,x)
            output[f"S{session}_y"] = y.astype(np.int64)
            output[f"S{session}_subjects"] = s.astype("U")
        np.savez_compressed(target / "predictions.npz", **output)
        print("EXPLORATORY_RANDOM_EVAL_DONE", task, fold, flush=True)
    subject_rows = []
    for session in sessions:
        chunks = [np.load(v1.cell_dir(task, fold, "exploratory_random") / "predictions.npz", allow_pickle=False) for fold in v1.FOLDS]
        y = chunks[0][f"S{session}_y"]
        subjects = chunks[0][f"S{session}_subjects"].astype(str)
        for part in chunks[1:]:
            if not np.array_equal(y, part[f"S{session}_y"]) or not np.array_equal(subjects, part[f"S{session}_subjects"].astype(str)):
                raise RuntimeError("exploratory fold trial order mismatch")
        z = np.stack([part[f"S{session}_z"] for part in chunks])
        p = np.exp(z-z.max(2,keepdims=True)); p /= p.sum(2,keepdims=True)
        pred = p.mean(0).argmax(1)
        for sub in sorted(set(subjects), key=lambda t:int(t.replace("sub-",""))):
            ix = subjects == sub
            subject_rows.append({"phase":"EXPLORATORY_POST_HELDOUT","task":task,"subject":sub,"session":f"S{session}",
                                 "variant":"RANDOM_PC_REFINE","BA":float(balanced_accuracy_score(y[ix],pred[ix]))})
        for part in chunks: part.close()
    v1.csv_write(OUT / f"{task}_RANDOM_PC_HELDOUT_SUBJECT_RESULTS.csv", subject_rows)
    primary = [r for r in subject_rows if r["session"] == "S2"]
    base = {(r["task"],r["subject"],r["variant"]):float(r["BA"]) for r in csv.DictReader((v1.OUTPUT / "HELDOUT_SUBJECT_RESULTS.csv").open(newline="",encoding="utf-8")) if r["session"] == "S2"}
    values = np.asarray([float(r["BA"]) for r in primary]); subjects = [r["subject"] for r in primary]
    if len(subjects) != len(held_ids): raise RuntimeError("exploratory subject count mismatch")
    comparison=[]
    for variant in v1.VARIANTS:
        matched = np.asarray([base[task,sub,variant] for sub in subjects])
        diff, low, high = v1.paired_ci(values,matched,("exploratory-random",task,variant))
        comparison.append({"phase":"EXPLORATORY_POST_HELDOUT","task":task,"contrast":f"RANDOM_PC_REFINE - {variant}",
                           "BA_difference":diff,"CI95_low":low,"CI95_high":high,"bootstrap_draws":20000})
    v1.csv_write(OUT / f"{task}_RANDOM_PC_PAIRED_CONTRASTS.csv",comparison)
    print("EXPLORATORY_RANDOM_TASK_DONE",task,flush=True)


def main() -> None:
    eligible = decision()
    report = ["# Conditional Random-PC follow-up", "", "Status: EXPLORATORY_POST_HELDOUT.",
              "Random-PC was triggered per task by a positive Protected-PC versus Baseline primary heldout BA point difference.",
              "Random subspace and PCA were fitted and hashed before V1 final heldout; epoch budgets were inherited from Protected-PC.",
              "This follow-up cannot turn the selected task results into a confirmatory mechanism-specific claim.", ""]
    for task in eligible:
        for fold in v1.FOLDS: train_cell(task, fold)
        evaluate_task(task)
        rows = list(csv.DictReader((OUT / f"{task}_RANDOM_PC_PAIRED_CONTRASTS.csv").open(newline="",encoding="utf-8")))
        report += [f"## {task}", ""]
        for row in rows:
            report.append(f"{row['contrast']}: ΔBA={float(row['BA_difference']):+.4f}, 95% CI [{float(row['CI95_low']):+.4f}, {float(row['CI95_high']):+.4f}].")
        report.append("")
    if not eligible: print("EXPLORATORY_RANDOM_NOT_TRIGGERED", flush=True)
    if not eligible: report.append("No task met the positive point-difference trigger; Random-PC was not trained.")
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / "EXPLORATORY_RANDOM_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
