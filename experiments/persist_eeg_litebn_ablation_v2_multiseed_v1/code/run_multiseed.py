#!/usr/bin/env python3
"""Three-seed LiteBN B1--B4 architecture study.

Seed-0 B1--B4 checkpoints are reused from the completed v2 study.  Per the
revised user scope, B0 is not trained for any seed.  This runner trains only
B1--B4 for seeds 1 and 2, freezes every checkpoint, and only then evaluates
the frozen heldout cohorts across all sessions.  The official frozen LiteBN
is retained solely as a non-matched descriptive reference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(os.environ.get("ABLATION_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_ablation_v2_multiseed_v1"
OUT, RUN, PROTOCOL = EXP / "outputs", EXP / "runtime", EXP / "protocol"
SEED0_EXP = REPO / "experiments/persist_eeg_litebn_ablation_v2_seed0"
V2_PATH = SEED0_EXP / "code/run_litebn_ablation_v2.py"
CSGD_PATH = REPO / "experiments/persist_eeg_litebn_tfformer_csgd_v1/code/run_csgd.py"
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
SEEDS = (0, 1, 2)
B0 = "B0_FULL_MATCHED"
REFERENCE = "OFFICIAL_FINAL_LITEBN_REFERENCE"
BOOTSTRAPS = 20_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = load_module("litebn_v2_seed0_source", V2_PATH)
csgd = load_module("litebn_v2_multiseed_csgd", CSGD_PATH)
VARIANTS = tuple(v2.VARIANTS)
EVAL_MODELS = (REFERENCE,) + VARIANTS


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(task: str, variant: str, seed: int, device=None):
    set_seed(seed)
    if variant in (B0, REFERENCE):
        model = v2.base.build_model("LiteBN_BASELINE", task)
    else:
        reference = v2.base.build_model("LiteBN_BASELINE", task)
        channels = int(reference.spatial[0].kernel_size[0])
        classes = int(reference.head.out_features)
        del reference
        model = v2.LiteBNAblationV2(channels, classes, variant)
    return model if device is None else model.to(device)


def install_training_context(seed: int) -> None:
    v2.SEED = seed
    v2.RUN = RUN / f"seed{seed}"
    v2.build_variant = lambda task, variant, device: make_model(task, variant, seed, device)


def seed0_reuse_records() -> list[dict]:
    path = SEED0_EXP / "runtime/TRAINING_LOGS.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    source = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in source:
        if row.get("variant") not in v2.VARIANTS:
            continue
        copied = dict(row)
        copied["seed"] = 0
        copied["reused_from_completed_seed0"] = True
        if not Path(copied["checkpoint"]).is_file():
            raise FileNotFoundError(copied["checkpoint"])
        rows.append(copied)
    expected = len(TASKS) * len(v2.VARIANTS) * 5
    if len(rows) != expected:
        raise RuntimeError(f"seed0 reuse count {len(rows)} != {expected}")
    return rows


def train_all(requested_seeds: tuple[int, ...], device) -> list[dict]:
    _, fold_map, _ = v2.base.load_folds()
    log_path = RUN / "TRAINING_LOGS.json"
    records = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else seed0_reuse_records()
    done = {(r["task"], r["variant"], int(r["seed"]), int(r["fold"])) for r in records}
    for seed in requested_seeds:
        install_training_context(seed)
        for task in TASKS:
            for fold in fold_map[v2.base.TASKS[task]["dataset"]]:
                fold_id = int(fold["fold_id"])
                pending = [variant for variant in VARIANTS if (task, variant, seed, fold_id) not in done]
                if not pending:
                    continue
                bundle = v2.base.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
                mean, std, _ = v2.base.load_tensor_pair(v2.baseline_runner.normalizer_source(task, fold_id))
                raw = v2.base.RawGPUCache(bundle, device)
                cache = v2.NormalizedCache(raw, mean, std)
                weight, _ = v2.base.class_weights(bundle, fold["inner_train_subjects"])
                print(f"MSARCH_CACHE seed={seed} {task} f{fold_id} pending={pending}", flush=True)
                for variant in pending:
                    set_seed(seed)
                    row = v2.train_one(task, fold, variant, bundle, cache, weight, device)
                    row["seed"] = seed
                    records.append(row)
                    done.add((task, variant, seed, fold_id))
                    atomic_json(log_path, records)
                    atomic_csv(OUT / "TRAINING_RUNS.csv", [
                        {k: value for k, value in r.items() if k != "history"} for r in records
                    ])
                    atomic_csv(OUT / "TRAINING_TRAJECTORY.csv", [
                        {**entry, "variant": r["variant"], "task": r["task"], "seed": r["seed"], "fold": r["fold"]}
                        for r in records for entry in r.get("history", [])
                    ])
                del cache, raw, bundle
                gc.collect(); torch.cuda.empty_cache()
    expected = len(TASKS) * len(VARIANTS) * len(SEEDS) * 5
    complete = {(r["task"], r["variant"], int(r["seed"]), int(r["fold"])) for r in records}
    if len(complete) != expected:
        missing = [(t, v, s, f) for t in TASKS for v in VARIANTS for s in SEEDS for f in range(5)
                   if (t, v, s, f) not in complete]
        raise RuntimeError(f"training incomplete: {missing[:10]} (n={len(missing)})")
    return records


def checkpoint_map(records: list[dict]) -> dict[tuple, Path]:
    result = {}
    for row in records:
        key = (row["task"], row["variant"], int(row["seed"]), int(row["fold"]))
        path = Path(row["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(path)
        result[key] = path
    return result


def evaluate_all(records: list[dict], device) -> list[dict]:
    runtime = csgd.load_runtime()
    paths = checkpoint_map(records)
    result_path = RUN / "SESSION_RESULTS.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.is_file() else []
    complete = {(r["task"], r["variant"], int(r["seed"]), int(r["fold"])) for r in rows}
    for task in TASKS:
        subjects, sessions = csgd.subjects_and_sessions(task)
        bundle = csgd.build_wbcic_outer_bundle(runtime)[0] if task == "WBCIC_MI" else runtime.base.build_bundle(task, subjects)
        raw = runtime.base.RawGPUCache(bundle, device)
        for fold in range(5):
            mean, std, metadata = runtime.base.load_tensor_pair(csgd.normalizer_path(task, fold))
            for seed in SEEDS:
                for variant in EVAL_MODELS:
                    key = (task, variant, seed, fold)
                    if key in complete:
                        continue
                    path = csgd.litebn_path(task, seed, fold) if variant == REFERENCE else paths[key]
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    model = make_model(task, variant, seed)
                    state = payload if variant == REFERENCE else payload["state_dict"]
                    model.load_state_dict(state, strict=True)
                    model = model.to(device).eval()
                    current = csgd.evaluate_sessions(runtime, model, "LiteBN", bundle, raw, None, subjects, sessions, mean, std)
                    digest = sha(path)
                    for row in current:
                        row.update({"task": task, "variant": variant, "seed": seed, "fold": fold,
                                    "checkpoint": str(path), "checkpoint_sha256": digest,
                                    "normalizer_mean_std_sha256": metadata.get("mean_std_sha256")})
                    rows.extend(current)
                    complete.add(key)
                    atomic_csv(result_path, rows)
                    print(f"MSARCH_EVAL {task} {variant} seed={seed} f{fold}", flush=True)
                    del model
                    gc.collect(); torch.cuda.empty_cache()
        del raw, bundle
        gc.collect(); torch.cuda.empty_cache()
    return rows


def bootstrap(values, key: str):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable("architecture-bootstrap", key))
    means = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 2000):
        stop = min(BOOTSTRAPS, start + 2000)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return float(values.mean()), float(low), float(high)


def complexity(task: str, variant: str):
    model = make_model(task, variant, 0).eval()
    channels = int(v2.base.TASKS[task]["channels"])
    macs = 0
    hooks = []
    def hook(module, _inputs, output):
        nonlocal macs
        if isinstance(module, torch.nn.Conv2d):
            per = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            macs += int(output.numel() * per)
        elif isinstance(module, torch.nn.Linear):
            macs += int(output.numel() * module.in_features)
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(torch.zeros(1, channels, 1000))
    for item in hooks:
        item.remove()
    return int(sum(p.numel() for p in model.parameters())), int(macs)


def aggregate(rows: list[dict]):
    frame = pd.DataFrame(rows)
    session = frame.groupby(["task", "variant", "seed", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), folds=("fold", "nunique")
    )
    if not (session.folds == 5).all():
        raise RuntimeError("not every seed/subject/session has five fold evaluations")
    subject_seed = []
    for (task, variant, seed, subject), part in session.groupby(["task", "variant", "seed", "subject_id"]):
        cells = {r.session: r for r in part.itertuples(index=False)}
        required = ("S0", "S1", "S2") if task == "WBCIC_MI" else ("S1", "S2")
        if set(cells) != set(required):
            raise RuntimeError(f"session mismatch {task}/{variant}/{seed}/{subject}")
        subject_seed.append({"task": task, "variant": variant, "seed": seed, "subject_id": subject,
                             "future_BA": cells["S2"].BA, "future_macro_F1": cells["S2"].macro_F1,
                             "WS_BA": min(cells[s].BA for s in required)})
    ss = pd.DataFrame(subject_seed)
    seed_summary = ss.groupby(["task", "variant", "seed"], as_index=False).agg(
        biological_subjects=("subject_id", "nunique"), future_BA=("future_BA", "mean"),
        future_macro_F1=("future_macro_F1", "mean"), WS_BA=("WS_BA", "mean")
    )
    biological_session = session.groupby(["task", "variant", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), seeds=("seed", "nunique")
    )
    if not (biological_session.seeds == 3).all():
        raise RuntimeError("not every biological subject has three seeds")
    biological_rows = []
    for (task, variant, subject), part in biological_session.groupby(["task", "variant", "subject_id"]):
        cells = {r.session: r for r in part.itertuples(index=False)}
        required = ("S0", "S1", "S2") if task == "WBCIC_MI" else ("S1", "S2")
        if set(cells) != set(required):
            raise RuntimeError(f"biological session mismatch {task}/{variant}/{subject}")
        biological_rows.append({"task": task, "variant": variant, "subject_id": subject,
                                "seeds": 3, "future_BA": cells["S2"].BA,
                                "future_macro_F1": cells["S2"].macro_F1,
                                "WS_BA": min(cells[s].BA for s in required)})
    biological = pd.DataFrame(biological_rows)
    summary = []
    effects = []
    for task in TASKS:
        ref = biological[(biological.task == task) & (biological.variant == REFERENCE)].set_index("subject_id")
        for variant in EVAL_MODELS:
            part = biological[(biological.task == task) & (biological.variant == variant)]
            params, macs = complexity(task, variant)
            row = {"task": task, "variant": variant, "biological_subjects": len(part),
                   "params": params, "MACs_T1000": macs}
            for metric in ("future_BA", "future_macro_F1", "WS_BA"):
                mean, low, high = bootstrap(part[metric].to_numpy(), f"{task}/{variant}/{metric}/absolute")
                row[metric] = mean; row[f"{metric}_CI95_low"] = low; row[f"{metric}_CI95_high"] = high
            summary.append(row)
            if variant == REFERENCE:
                continue
            cur = part.set_index("subject_id")
            ids = sorted(set(ref.index) & set(cur.index))
            effect = {"task": task, "variant": variant, "reference": REFERENCE, "biological_subjects": len(ids)}
            for metric in ("future_BA", "future_macro_F1", "WS_BA"):
                delta = 100 * (cur.loc[ids, metric].to_numpy() - ref.loc[ids, metric].to_numpy())
                mean, low, high = bootstrap(delta, f"{task}/{variant}/{metric}/delta")
                effect[f"delta_{metric}_pp"] = mean
                effect[f"delta_{metric}_CI95_low_pp"] = low
                effect[f"delta_{metric}_CI95_high_pp"] = high
            effects.append(effect)
    return session.to_dict("records"), subject_seed, seed_summary.to_dict("records"), biological.to_dict("records"), summary, effects


def report(summary, effects) -> str:
    sm = pd.DataFrame(summary)
    ef = pd.DataFrame(effects)
    lines = ["# LiteBN B1--B4 architecture study (seeds 0, 1, 2)", "",
             "B0 was not trained for any seed by explicit user instruction. Seed-0 B1--B4 checkpoints were reused exactly and seeds 1/2 were newly trained; no heldout labels entered training or selection. The official frozen LiteBN is a non-matched descriptive reference only.", "",
             "OpenBMI V8 is an internal-heldout diagnostic cohort. The WBCIC true-outer cohort has already been accessed for prior ablation diagnostics and is therefore an exposed benchmark, not untouched final confirmation.", "",
             "Fold and seed repetitions were averaged within each biological subject and session; WS-BA is the minimum of those session means. The 20,000-draw bootstrap resamples biological subjects.", "",
             "| Variant | MI BA | ERP BA | SSVEP BA | WBCIC BA | Mean BA | Mean WS-BA |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in EVAL_MODELS:
        p = sm[sm.variant == variant].set_index("task")
        lines.append(f"| {variant} | {p.loc['OpenBMI_MI','future_BA']:.4f} | {p.loc['OpenBMI_ERP','future_BA']:.4f} | {p.loc['OpenBMI_SSVEP','future_BA']:.4f} | {p.loc['WBCIC_MI','future_BA']:.4f} | {p.future_BA.mean():.4f} | {p.WS_BA.mean():.4f} |")
    lines += ["", "## Paired descriptive effects versus official frozen LiteBN", "", "| Variant | Task | Delta BA pp [95% CI] | Delta Macro-F1 pp [95% CI] | Delta WS-BA pp [95% CI] |", "|---|---|---:|---:|---:|"]
    for r in ef.itertuples(index=False):
        lines.append(f"| {r.variant} | {r.task} | {r.delta_future_BA_pp:+.3f} [{r.delta_future_BA_CI95_low_pp:+.3f}, {r.delta_future_BA_CI95_high_pp:+.3f}] | {r.delta_future_macro_F1_pp:+.3f} [{r.delta_future_macro_F1_CI95_low_pp:+.3f}, {r.delta_future_macro_F1_CI95_high_pp:+.3f}] | {r.delta_WS_BA_pp:+.3f} [{r.delta_WS_BA_CI95_low_pp:+.3f}, {r.delta_WS_BA_CI95_high_pp:+.3f}] |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    requested = tuple(int(x) for x in args.seeds.split(",") if x)
    if any(s not in SEEDS for s in requested):
        raise ValueError(requested)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    for directory in (OUT, RUN, PROTOCOL):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_json(PROTOCOL / "PROTOCOL.json", {"seeds": SEEDS, "folds": list(range(5)), "tasks": TASKS,
        "variants": VARIANTS, "B0_trained": False, "seed0_B1_B4_reused": True,
        "official_reference": REFERENCE, "official_reference_is_matched": False, "optimizer": "AdamW",
        "lr": 3e-4, "weight_decay": 5e-4, "gradient_clip": 5.0, "max_epochs": 60,
        "minimum_checkpoint_epoch": 10, "early_stop_patience": 8,
        "checkpoint_metric": "subject-equal inner-validation BA", "heldout_used_for_selection": False,
        "bootstrap_unit": "biological subject", "bootstrap_draws": BOOTSTRAPS})
    if args.preflight:
        rows = seed0_reuse_records()
        print(f"MSARCH_PREFLIGHT_OK reused_seed0={len(rows)} variants={VARIANTS}", flush=True)
        return
    if args.aggregate_only:
        path = RUN / "SESSION_RESULTS.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = pd.read_csv(path).to_dict("records")
    else:
        device = torch.device("cuda")
        records = train_all(requested, device)
        rows = evaluate_all(records, device)
    session, subject_seed, seed_summary, biological, summary, effects = aggregate(rows)
    atomic_csv(OUT / "SESSION_SUBJECT_RESULTS.csv", session)
    atomic_csv(OUT / "SUBJECT_SEED_RESULTS.csv", subject_seed)
    atomic_csv(OUT / "TASK_SEED_SUMMARY.csv", seed_summary)
    atomic_csv(OUT / "BIOLOGICAL_SUBJECT_RESULTS.csv", biological)
    atomic_csv(OUT / "MULTISEED_TASK_SUMMARY.csv", summary)
    atomic_csv(OUT / "PAIRED_EFFECTS_VS_OFFICIAL_REFERENCE.csv", effects)
    atomic_text(OUT / "FINAL_B1_B4_MULTISEED_REPORT.md", report(summary, effects))
    atomic_json(OUT / "COMPLETION.json", {"status": "COMPLETE", "training_cells": len(TASKS)*len(VARIANTS)*len(SEEDS)*5,
        "new_training_cells": len(TASKS)*len(VARIANTS)*2*5, "reused_seed0_cells": len(TASKS)*len(VARIANTS)*5,
        "missing_cells": 0, "seeds": SEEDS, "B0_trained": False,
        "heldout_evaluated_after_all_checkpoints_frozen": True})
    print("LITEBN_ARCHITECTURE_MULTISEED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
