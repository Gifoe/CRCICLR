from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import p4_persist_ct as base
import p4_persist_ct_v2 as audit_v2

ROOT = base.ROOT
AUDIT = ROOT / "outputs" / "persist_eeg_p4_ct_v2_1"
OUT = ROOT / "outputs" / "persist_eeg_iga_v0"
TASKS, CLASSES = base.TASKS, base.CLASSES

class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 128))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return x + self.net(x)

def labels(meta, task):
    return meta.event_label.astype(str).map(base.label_maps(meta)[task]).to_numpy(np.int64)

def proj_torch(h, spec, ids, device):
    if not ids: return torch.zeros_like(h)
    mu = torch.as_tensor(spec["mean"], dtype=torch.float32, device=device)
    W = torch.as_tensor(spec["whitener"], dtype=torch.float32, device=device)
    V = torch.as_tensor(spec["directions"], dtype=torch.float32, device=device)
    D = torch.as_tensor(spec["dewhitener"], dtype=torch.float32, device=device)
    q = (h - mu) @ W @ V
    z = torch.zeros_like(q); ix = torch.as_tensor(ids, dtype=torch.long, device=device); z[:, ix] = q[:, ix]
    return (z @ V.T) @ D

def macro_ba(logits, y, classes):
    pred = logits.argmax(1).detach().cpu().numpy(); y = np.asarray(y)
    return float(np.mean([np.mean(pred[y == k] == k) for k in range(classes) if np.any(y == k)]))

def run(fold, seed, device):
    torch.manual_seed(1000 * fold + seed); np.random.seed(1000 * fold + seed)
    manifest = base.load_manifest(); split = next(x for x in base.load_splits() if int(x["fold"]) == fold)
    ckpt, mean, std = base.historical(fold, seed); model = base.load_model(ckpt, manifest, device)
    trm, trh, _ = base.extract(model, manifest, split["train_subjects"], mean, std, device, 190000 + fold * 101 + seed, cap=3000)
    vam, vah, _ = base.extract(model, manifest, split["validation_subjects"], mean, std, device, 200000 + fold * 101 + seed, cap=1500)
    spec = audit_v2.build_spectrum_v2(trm, trh, 30000 + fold * 101 + seed)
    audit = json.loads((AUDIT / "audit" / f"fold-{fold}" / f"seed-{seed}" / "AUDIT_V2_1.json").read_text())
    train_data, val_data = [], []
    for task in TASKS:
        ti = np.flatnonzero((trm.paradigm == task).to_numpy()); vi = np.flatnonzero((vam.paradigm == task).to_numpy())
        prot = sorted(set(sum((spec["blocks"][b] for b in audit["assignments"][task]["protected"]), [])))
        nuis = sorted(set(sum((spec["blocks"][b] for b in audit["assignments"][task]["nuisance"]), [])))
        train_data.append((task, trh[ti].astype("float32"), labels(trm, task)[ti], trm.iloc[ti].subject_id.astype(str).to_numpy(), prot, nuis))
        val_data.append((task, vah[vi].astype("float32"), labels(vam, task)[vi]))

    init_adapter = ResidualAdapter().state_dict()
    init_heads = {t: nn.Linear(128, CLASSES[t]).state_dict() for t in TASKS}
    def fit(use_iga):
        adapter = ResidualAdapter().to(device)
        heads = nn.ModuleDict({t: nn.Linear(128, CLASSES[t]) for t in TASKS}).to(device)
        adapter.load_state_dict(init_adapter)
        for t in TASKS: heads[t].load_state_dict(init_heads[t])
        opt = torch.optim.Adam(list(adapter.parameters()) + list(heads.parameters()), lr=1e-3, weight_decay=1e-4)
        for _ in range(12):
            opt.zero_grad(); total = torch.zeros((), device=device)
            for task, x_np, y_np, subjects, prot, nuis in train_data:
                x = torch.as_tensor(x_np, device=device); y = torch.as_tensor(y_np, device=device)
                h = adapter(x); total = total + nn.functional.cross_entropy(heads[task](h), y)
                if use_iga:
                    p0 = proj_torch(x, spec, prot, device).detach(); p = proj_torch(h, spec, prot, device)
                    n = proj_torch(h, spec, nuis, device); align = torch.zeros((), device=device)
                    for lab in np.unique(y_np):
                        ix_lab = np.flatnonzero(y_np == lab); means = []
                        for subj in np.unique(subjects[ix_lab]):
                            ix = ix_lab[subjects[ix_lab] == subj]
                            if len(ix): means.append(n[torch.as_tensor(ix, device=device)].mean(0))
                        if means:
                            mm = torch.stack(means); align = align + ((mm - mm.mean(0)) ** 2).mean()
                    total = total + 0.01 * align + 0.2 * ((p - p0) ** 2).mean()
            total.backward(); opt.step()
        metrics = {}
        adapter.eval(); heads.eval()
        with torch.no_grad():
            for task, x_np, y_np in val_data:
                metrics[task] = macro_ba(heads[task](adapter(torch.as_tensor(x_np, device=device))), y_np, CLASSES[task])
        return metrics
    control = fit(False); iga = fit(True)
    return {"fold": fold, "seed": seed, "control": control, "iga": iga,
            "delta_task": {t: float(iga[t] - control[t]) for t in TASKS},
            "delta_macro_BA": float(np.mean(list(iga.values())) - np.mean(list(control.values()))),
            "outer_test_used": False}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--seed", type=int); a = ap.parse_args()
    folds = (a.fold,) if a.fold is not None else (0, 1, 2); seeds = (a.seed,) if a.seed is not None else (0, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); results = []
    for fold in folds:
        for seed in seeds:
            print(f"[IGA-V1] fold={fold} seed={seed}", flush=True); results.append(run(fold, seed, device))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"version": "IGA-V1", "parent": "IGA-V0", "observed_failure": "IGA-V0 mean macro BA gain -0.001388 and only 2/6 positive; MI harm was dominant in several runs", "modification": "lambda_N 0.10 -> 0.01 and protected anchor 0.10 -> 0.20", "runs": results, "mean_delta_macro_BA": float(np.mean([r["delta_macro_BA"] for r in results])), "positive_runs": int(sum(r["delta_macro_BA"] > 0 for r in results)), "outer_test_used": False}
    (OUT / "P4_IGA_V0_RESULTS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "P4_IGA_ADAPTATION_LOG.json").write_text(json.dumps({"version": "IGA-V1", "parent": "IGA-V0", "observed_failure": "IGA-V0 failed clean development gate", "diagnostic_evidence": {"mean_delta_macro_BA": -0.0013882473808648303, "positive_runs": 2}, "modification": "lambda_N 0.10 -> 0.01; lambda_P 0.10 -> 0.20", "scientific_reason": "reduce nuisance gradient conflict and strengthen protected teacher retention", "trainable_parameters": ["adapter", "task_heads"], "data_used": ["TRAIN", "VALIDATION"], "outer_test_used": False}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
if __name__ == "__main__": main()
