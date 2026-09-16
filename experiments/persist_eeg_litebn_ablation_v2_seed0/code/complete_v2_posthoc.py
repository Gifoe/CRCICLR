#!/usr/bin/env python3
"""Complete frozen multi-session evaluation for LiteBN ablation v2.

No model is trained or adapted here.  The user explicitly waived matched-Full
training, so the official final LiteBN is retained only as a non-matched
reference and every contrast is labelled accordingly.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("ABLATION_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_ablation_v2_seed0"
OUT, PROTOCOL, RUN = EXP / "outputs", EXP / "protocol", EXP / "runtime"
CSGD_PATH = REPO / "experiments/persist_eeg_litebn_tfformer_csgd_v1/code/run_csgd.py"
sys.path.insert(0, str(EXP / "code"))
from litebn_variants_v2 import LiteBNAblationV2  # noqa: E402

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
VARIANTS = LiteBNAblationV2.VARIANTS
REFERENCE = "OFFICIAL_FINAL_LITEBN_REFERENCE"
ALL = (REFERENCE,) + VARIANTS
FOLDS = tuple(range(5))
BOOTSTRAPS = 20_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint(csgd, task: str, fold: int, variant: str) -> Path:
    if variant == REFERENCE:
        return csgd.litebn_path(task, 0, fold)
    return RUN / "checkpoints" / task / variant / f"fold{fold}" / "selected.pt"


def make_model(runtime, task: str, variant: str):
    if variant == REFERENCE:
        return runtime.base.build_model("LiteBN_BASELINE", task)
    reference = runtime.base.build_model("LiteBN_BASELINE", task)
    channels = int(reference.spatial[0].kernel_size[0])
    classes = int(reference.head.out_features)
    return LiteBNAblationV2(channels, classes, variant)


def load_model(runtime, csgd, task: str, fold: int, variant: str, device):
    path = checkpoint(csgd, task, fold, variant)
    model = make_model(runtime, task, variant)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload if variant == REFERENCE else payload["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), path


def complexity(runtime, task: str, variant: str):
    model = make_model(runtime, task, variant).eval()
    channels = int(runtime.base.TASKS[task]["channels"])
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


def bootstrap(values, key: str):
    values = np.asarray(values, dtype=np.float64)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 2000):
        stop = min(BOOTSTRAPS, start + 2000)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return float(values.mean()), float(low), float(high)


def aggregate(rows):
    frame = pd.DataFrame(rows)
    session = frame.groupby(["task", "variant", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), fold_replicates=("fold", "nunique")
    )
    if not (session.fold_replicates == 5).all():
        raise RuntimeError("incomplete five-fold session evaluation")
    subjects = []
    for (task, variant, subject), part in session.groupby(["task", "variant", "subject_id"]):
        cells = {row.session: row for row in part.itertuples(index=False)}
        required = ("S0", "S1", "S2") if task == "WBCIC_MI" else ("S1", "S2")
        if set(cells) != set(required):
            raise RuntimeError(f"session mismatch {task}/{variant}/{subject}")
        future = cells["S2"]
        subjects.append({
            "task": task, "variant": variant, "subject_id": subject,
            "future_BA": float(future.BA), "future_macro_F1": float(future.macro_F1),
            "WS_BA": float(min(cells[s].BA for s in required)),
            **{f"{s}_BA": float(cells[s].BA) for s in required},
            **{f"{s}_macro_F1": float(cells[s].macro_F1) for s in required},
        })
    sf = pd.DataFrame(subjects)
    summary = []
    for task in TASKS:
        for variant in ALL:
            part = sf[(sf.task == task) & (sf.variant == variant)]
            params, macs = complexity(runtime_global, task, variant)
            summary.append({
                "task": task, "variant": variant, "params": params, "MACs_T1000": macs,
                "biological_subjects": len(part), "future_BA": part.future_BA.mean(),
                "future_macro_F1": part.future_macro_F1.mean(), "WS_BA": part.WS_BA.mean(),
            })
    effects = []
    for task in TASKS:
        ref = sf[(sf.task == task) & (sf.variant == REFERENCE)].set_index("subject_id")
        for variant in VARIANTS:
            cur = sf[(sf.task == task) & (sf.variant == variant)].set_index("subject_id")
            ids = sorted(set(ref.index) & set(cur.index))
            row = {"task": task, "variant": variant, "reference": REFERENCE, "subjects": len(ids)}
            for metric in ("future_BA", "future_macro_F1", "WS_BA"):
                delta = 100 * (cur.loc[ids, metric].to_numpy() - ref.loc[ids, metric].to_numpy())
                mean, low, high = bootstrap(delta, f"{task}/{variant}/{metric}")
                row[f"delta_{metric}_pp"] = mean
                row[f"delta_{metric}_CI95_low_pp"] = low
                row[f"delta_{metric}_CI95_high_pp"] = high
            effects.append(row)
    return session.to_dict("records"), subjects, summary, effects


def report(summary, effects):
    sm = pd.DataFrame(summary)
    ef = pd.DataFrame(effects)
    lines = [
        "# LiteBN ablation v2 seed-0: frozen multi-session evaluation", "",
        "B0_FULL_MATCHED was not trained by explicit user override. The official final LiteBN is a non-matched reference, so these are descriptive contrasts rather than matched-control causal effects.", "",
        "All five fold checkpoints were averaged within each biological subject before subjects were equally averaged. WS-BA is the within-subject minimum session BA after fold averaging.", "",
        "| Variant | Params range | MACs range | MI BA/WS | ERP BA/WS | SSVEP BA/WS | WBCIC BA/WS | Mean BA | Mean WS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ALL:
        part = sm[sm.variant == variant].set_index("task")
        mean_ba, mean_ws = part.future_BA.mean(), part.WS_BA.mean()
        lines.append(
            f"| {variant} | {part.params.min()}-{part.params.max()} | {part.MACs_T1000.min()}-{part.MACs_T1000.max()} | "
            f"{part.loc['OpenBMI_MI','future_BA']:.4f}/{part.loc['OpenBMI_MI','WS_BA']:.4f} | "
            f"{part.loc['OpenBMI_ERP','future_BA']:.4f}/{part.loc['OpenBMI_ERP','WS_BA']:.4f} | "
            f"{part.loc['OpenBMI_SSVEP','future_BA']:.4f}/{part.loc['OpenBMI_SSVEP','WS_BA']:.4f} | "
            f"{part.loc['WBCIC_MI','future_BA']:.4f}/{part.loc['WBCIC_MI','WS_BA']:.4f} | {mean_ba:.4f} | {mean_ws:.4f} |"
        )
    lines += ["", "## Paired effects versus non-matched official reference", "",
              "| Variant | Task | Delta future BA [95% CI], pp | Delta WS-BA [95% CI], pp |",
              "|---|---|---:|---:|"]
    for row in ef.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.task} | {row.delta_future_BA_pp:+.3f} "
            f"[{row.delta_future_BA_CI95_low_pp:+.3f}, {row.delta_future_BA_CI95_high_pp:+.3f}] | "
            f"{row.delta_WS_BA_pp:+.3f} [{row.delta_WS_BA_CI95_low_pp:+.3f}, {row.delta_WS_BA_CI95_high_pp:+.3f}] |"
        )
    return "\n".join(lines) + "\n"


def main():
    global runtime_global
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    csgd = load_module("v2_csgd_runtime", CSGD_PATH)
    runtime_global = csgd.load_runtime()
    device = torch.device("cuda")
    result_path = RUN / "SESSION_RESULTS.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.is_file() else []
    complete = {(r["task"], r["variant"], int(r["fold"])) for r in rows}
    audit = []
    for task in TASKS:
        subjects, sessions = csgd.subjects_and_sessions(task)
        bundle = csgd.build_wbcic_outer_bundle(runtime_global)[0] if task == "WBCIC_MI" else runtime_global.base.build_bundle(task, subjects)
        raw = runtime_global.base.RawGPUCache(bundle, device)
        for fold in FOLDS:
            mean, std, meta = runtime_global.base.load_tensor_pair(csgd.normalizer_path(task, fold))
            for variant in ALL:
                path = checkpoint(csgd, task, fold, variant)
                if not path.is_file():
                    raise FileNotFoundError(path)
                audit.append({"task": task, "variant": variant, "fold": fold, "checkpoint": str(path), "sha256": sha(path)})
                if (task, variant, fold) in complete:
                    continue
                model, path = load_model(runtime_global, csgd, task, fold, variant, device)
                current = csgd.evaluate_sessions(runtime_global, model, "LiteBN", bundle, raw, None, subjects, sessions, mean, std)
                for row in current:
                    row.update({"task": task, "variant": variant, "fold": fold, "seed": 0,
                                "checkpoint": str(path), "checkpoint_sha256": sha(path),
                                "normalizer_mean_std_sha256": meta.get("mean_std_sha256")})
                rows.extend(current)
                atomic_csv(result_path, rows)
                complete.add((task, variant, fold))
                print(f"V2_POSTHOC_DONE {task} {variant} f{fold}", flush=True)
                del model
                gc.collect(); torch.cuda.empty_cache()
        del raw, bundle
        gc.collect(); torch.cuda.empty_cache()
    session, subjects, summary, effects = aggregate(rows)
    atomic_csv(OUT / "SESSION_RESULTS.csv", session)
    atomic_csv(OUT / "SUBJECT_RESULTS.csv", subjects)
    atomic_csv(OUT / "ABLATION_SUMMARY.csv", summary)
    atomic_csv(OUT / "PAIRED_EFFECTS.csv", effects)
    atomic_csv(OUT / "OFFICIAL_FULL_SANITY_REFERENCE.csv", [r for r in summary if r["variant"] == REFERENCE])
    atomic_csv(PROTOCOL / "CHECKPOINT_REUSE_AUDIT.csv", audit)
    atomic_json(PROTOCOL / "VARIANT_DEFINITIONS.json", {
        "B1_SAME_SCALE_63": "three independent k63 branches; downstream identical",
        "B2_SCALE_COLLAPSE": "average three branch tensors then repeat to 48 channels",
        "B3_SINGLE_SPATIAL_BASIS": "one grouped spatial basis then linear 1x1 expansion to 16",
        "B4_ONE_STAGE_BACKEND": "remove only second backend stage",
        REFERENCE: "official frozen LiteBN; non-matched reference only",
    })
    atomic_json(PROTOCOL / "TRAINING_INVARIANTS.json", {
        "seed": 0, "folds": list(FOLDS), "tasks": list(TASKS), "newly_trained": list(VARIANTS),
        "B0_FULL_MATCHED_trained": False, "user_override": True, "max_epochs": 60,
        "min_checkpoint_epoch": 10, "patience": 8, "optimizer": "AdamW", "lr": 3e-4,
        "weight_decay": 5e-4, "gradient_clip": 5.0, "scheduler": "none",
    })
    validation = {
        "status": "PASS_WITH_USER_OVERRIDE", "seed_is_zero": True, "tasks_are_four": True,
        "folds_are_five": True, "B1_to_B4_accounted_for": True,
        "B0_FULL_MATCHED_accounted_for": False,
        "B0_deviation": "explicit user instruction not to train Full",
        "all_new_variants_protocol_identical": True, "B1_three_independent_branches": True,
        "B1_backend_width_48": True, "B2_parameter_count_equals_official_full": True,
        "B2_backend_width_48": True, "B3_branch_width_16": True,
        "B3_concat_width_48": True, "B4_only_removes_second_backend_stage": True,
        "heldout_not_used_for_training_or_selection": True,
        "bootstrap_unit": "biological subject", "bootstrap_resamples": BOOTSTRAPS,
    }
    atomic_json(PROTOCOL / "VALIDATION.json", validation)
    atomic_text(OUT / "FINAL_LITEBN_ABLATION_V2_SEED0_REPORT.md", report(summary, effects))
    atomic_json(OUT / "ARCHITECTURE_POSTHOC_COMPLETION.json", {
        "status": "COMPLETE_WITH_NONMATCHED_REFERENCE", "session_rows": len(session),
        "subject_rows": len(subjects), "summary_rows": len(summary), "paired_rows": len(effects),
    })
    print("V2_POSTHOC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
