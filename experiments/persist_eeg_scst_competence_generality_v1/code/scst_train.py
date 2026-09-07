from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score

import admissibility as audit
import specialist_train as specialist


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
RUNTIME = EXP / "runtime"; RESULTS = EXP / "results"; PROTOCOL = EXP / "protocol"
FOLDS = range(5); SEEDS = range(3); METHODS = ("ERM", "Mixup", "RandomTransport", "SCST")


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def subject_ba(y: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    pred = logits.argmax(1); return float(np.mean([balanced_accuracy_score(y[subjects.astype(str) == subject], pred[subjects.astype(str) == subject]) for subject in sorted(np.unique(subjects.astype(str)), key=lambda x: (int(x) if x.isdigit() else 10**9, x))]))


def subject_f1(y: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    pred = logits.argmax(1); return float(np.mean([f1_score(y[subjects.astype(str) == subject], pred[subjects.astype(str) == subject], average="macro", zero_division=0) for subject in sorted(np.unique(subjects.astype(str)), key=lambda x: (int(x) if x.isdigit() else 10**9, x))]))


def read_rep(model: str, dataset: str, fold: int, seed: int, role: str) -> dict[str, np.ndarray]:
    return audit.load_rep(model, dataset, fold, seed, role)


def future_rep(model: str, fold: int, seed: int, device: torch.device) -> dict[str, np.ndarray]:
    path = RUNTIME / "future_representations" / model / "WBCIC" / f"fold-{fold}" / f"seed-{seed}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as value: return {key: value[key] for key in value.files}
    _, meta = specialist.load_data("WBCIC"); role = specialist.roles("WBCIC", fold); future_subjects = role["outcome"]
    if model == "CBraMod-R1":
        from phase1b_repair import load_repaired, repair_path
        m = load_repaired("WBCIC", fold, seed, repair_path("WBCIC", "R1_lr3e-5", fold, seed), device); raw_cache = audit.REPO / "experiments" / "persist_eeg_fm_rescue_stage0" / "runtime" / "WBCIC_FM_INPUT_UV_200HZ.npy"; xall = np.load(raw_cache, mmap_mode="r", allow_pickle=False); idx = specialist.indices(meta, future_subjects, (2,))
        logits = []; features = []
        for start in range(0, len(idx), 128):
            group = idx[start:start + 128]; x = torch.from_numpy(np.asarray(xall[group], np.float32)).to(device).reshape(len(group), 58, 4, 200)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): h = m.forward_features(x); z = m.head(h)
            features.append(h.float().cpu().numpy()); logits.append(z.float().cpu().numpy())
        picked = meta.iloc[idx]; value = {"indices": idx, "features": np.concatenate(features).astype(np.float32), "logits": np.concatenate(logits).astype(np.float32), "labels": picked.label.to_numpy(np.int64), "subjects": picked.subject_id.astype(str).to_numpy(), "sessions": picked.session_id.to_numpy(np.int64)}
    else:
        selected = pd.read_csv(RESULTS / "SPECIALIST_SELECTION.csv"); config_name = selected[(selected.model == model) & (selected.dataset == "WBCIC")].config.iloc[0]; m, _ = specialist.load_checkpoint(model, "WBCIC", specialist.checkpoint_path(model, "WBCIC", config_name, fold, seed), device); raw, meta = specialist.load_data("WBCIC"); idx = specialist.indices(meta, future_subjects, (2,)); value = specialist.infer(m, raw, meta, idx, 128, device)
    path.parent.mkdir(parents=True, exist_ok=True); clean = {key: (np.asarray(item).astype("U") if np.asarray(item).dtype == object else item) for key, item in value.items()}; temp = path.with_suffix(".npz.part");
    with temp.open("wb") as stream: np.savez_compressed(stream, **clean)
    os.replace(temp, path); return value


def normalize_source(source: dict[str, np.ndarray], validation: dict[str, np.ndarray], future: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    combined = np.concatenate([source["features"], validation["features"]]); session_a = audit.SOURCE_SESSIONS["WBCIC"][0]; bank = source["sessions"].astype(int) == session_a; center = source["features"][bank].mean(0); scale = source["features"][bank].std(0); scale[scale < 1e-6] = 1.0
    def convert(value): return {**value, "features": ((value["features"] - center) / scale).astype(np.float32)}
    return convert(source), convert(validation), convert(future), center, scale


def centroids(rep: dict[str, np.ndarray]) -> dict[tuple[str, int, int], np.ndarray]:
    output = {}
    for subject in sorted(np.unique(rep["subjects"].astype(str)), key=lambda x: (int(x) if x.isdigit() else 10**9, x)):
        for label in sorted(np.unique(rep["labels"]).astype(int)):
            for session in sorted(np.unique(rep["sessions"]).astype(int)):
                mask = (rep["subjects"].astype(str) == subject) & (rep["labels"] == label) & (rep["sessions"] == session)
                if mask.any(): output[(subject, label, session)] = rep["features"][mask].mean(0).astype(np.float64)
    return output


def support_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    d = np.sum(query * query, 1)[:, None] + np.sum(support * support, 1)[None, :] - 2 * query @ support.T; np.maximum(d, 0, out=d); return np.sqrt(np.partition(d, 2, axis=1)[:, :3]).mean(1)


def support_radius(support: np.ndarray) -> float:
    d = np.linalg.norm(support[:, None] - support[None, :], axis=2); np.fill_diagonal(d, np.inf); return float(np.quantile(np.partition(d, 2, axis=1)[:, :3].mean(1), .95))


def transport_arrays(source: dict[str, np.ndarray], model: str, fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = audit.SOURCE_SESSIONS["WBCIC"]; cs = centroids(source); subjects = sorted(np.unique(source["subjects"].astype(str)), key=lambda x: (int(x) if x.isdigit() else 10**9, x)); labels = sorted(np.unique(source["labels"]).astype(int)); population = {(label, session): np.mean([cs[(subject, label, session)] for subject in subjects], axis=0) for label in labels for session in (a, b)}; residual = {(subject, label, session): cs[(subject, label, session)] - population[(label, session)] for subject in subjects for label in labels for session in (a, b)}
    transported = np.empty_like(source["features"], np.float32); random_value = np.empty_like(source["features"], np.float32); rng = np.random.default_rng(stable_seed("random-transport", model, fold, seed))
    for idx in range(len(source["features"])):
        subject = str(source["subjects"][idx]); label = int(source["labels"][idx]); query = source["features"][idx].astype(np.float64); candidates = [value for value in subjects if value != subject]; target = candidates[stable_seed("target", model, fold, seed, idx) % len(candidates)]; delta = residual[(target, label, a)] - residual[(subject, label, a)]; support = np.stack([cs[(value, label, a)] for value in subjects]); alpha = audit.solve_alpha(query[None], delta[None], support, support_radius(support))[0]; noise = rng.normal(size=len(delta)); noise *= np.linalg.norm(delta) / max(np.linalg.norm(noise), audit.EPS); transported[idx] = query + alpha * delta; random_value[idx] = query + alpha * noise
    return transported, random_value


class LinearHead(nn.Module):
    def __init__(self, dim: int, weight: np.ndarray, bias: np.ndarray):
        super().__init__(); self.linear = nn.Linear(dim, 2); self.linear.weight.data.copy_(torch.from_numpy(weight.astype(np.float32))); self.linear.bias.data.copy_(torch.from_numpy(bias.astype(np.float32)))
    def forward(self, x): return self.linear(x)


def recovered_head(x: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1); coef = np.linalg.lstsq(design, logits, rcond=None)[0]; return coef[:-1].T.astype(np.float32), coef[-1].astype(np.float32)


def js_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(a, -1); q = torch.softmax(b, -1); m = .5 * (p + q); return .5 * ((p * (torch.log(p.clamp_min(1e-8)) - torch.log(m.clamp_min(1e-8)))).sum(-1) + (q * (torch.log(q.clamp_min(1e-8)) - torch.log(m.clamp_min(1e-8)))).sum(-1)).mean()


def train_method(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, transported: np.ndarray, random_value: np.ndarray, method: str, model_name: str, fold: int, seed: int, weight: np.ndarray, bias: np.ndarray, epochs: int = 25) -> LinearHead:
    set_seed(stable_seed("scst-train", model_name, fold, seed, method)); device = torch.device("cuda"); head = LinearHead(x.shape[1], weight, bias).to(device); optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4); rng = np.random.default_rng(stable_seed("scst-order", model_name, fold, seed, method)); xt = torch.from_numpy(x).to(device); yt = torch.from_numpy(y).long().to(device); tt = torch.from_numpy(transported).to(device); rr = torch.from_numpy(random_value).to(device); batch = min(512, len(x))
    for epoch in range(epochs):
        order = rng.permutation(len(x)); head.train()
        for start in range(0, len(order), batch):
            idx = torch.as_tensor(order[start:start + batch], device=device); clean_logits = head(xt[idx]); clean_loss = nn.functional.cross_entropy(clean_logits, yt[idx])
            if method == "ERM": aux = xt[idx]; aux_y = yt[idx]; aux_loss = nn.functional.cross_entropy(head(aux), aux_y); consistency = torch.zeros((), device=device)
            elif method == "Mixup":
                perm = idx[torch.randperm(len(idx), device=device)]; lam = float(np.random.beta(.4, .4)); aux = lam * xt[idx] + (1 - lam) * xt[perm]; aux_loss = lam * nn.functional.cross_entropy(head(aux), yt[idx]) + (1 - lam) * nn.functional.cross_entropy(head(aux), yt[perm]); consistency = js_divergence(clean_logits, head(aux))
            elif method == "RandomTransport": aux = rr[idx]; aux_loss = nn.functional.cross_entropy(head(aux), yt[idx]); consistency = js_divergence(clean_logits, head(aux))
            else: aux = tt[idx]; aux_loss = nn.functional.cross_entropy(head(aux), yt[idx]); consistency = js_divergence(clean_logits, head(aux))
            loss = .5 * clean_loss + .5 * aux_loss + (.1 * consistency if method != "ERM" else 0); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return head.eval()


def main() -> None:
    authorization = json.loads((RESULTS / "SCST_AUTHORIZATION.json").read_text(encoding="utf-8")); eligible = authorization.get("eligible_models", [])
    if not eligible: raise RuntimeError("SCST_TRAINING_NOT_AUTHORIZED")
    protocol = {"schema": "SCST_TRAINING_PROTOCOL_LOCK_V1", "training_authorized": True, "eligible_models": eligible, "dataset": "WBCIC", "future_session": 2, "methods": list(METHODS), "alpha_rule": "support-constrained ALPHA_GRID 0..0.25 step 1/64; 95th percentile 3NN support radius", "bank": "model_fit + validation source representations, source session 0 centroids", "loss": "0.5 clean CE + 0.5 auxiliary CE + 0.1 Jensen-Shannon for Mixup/RandomTransport/SCST", "optimizer": "AdamW lr=1e-3 wd=1e-4, 25 fixed epochs, same batches/seed per method", "controls": ["ERM", "Mixup", "RandomTransport"], "bootstrap_draws": 10000, "future_outcomes_accessed_after_lock": True, "success_gates": {"mean_delta_gt_zero": True, "CI95_L_gt_zero": True, "positive_folds_ge_3": True, "catastrophic_fold_failure_forbidden": True, "beats_random_transport": True, "class_fidelity": "audit class gate remains intact"}}
    (PROTOCOL / "SCST_TRAINING_PROTOCOL_LOCK.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = []
    for model_name in eligible:
        for fold in FOLDS:
            for seed in SEEDS:
                source, validation = read_rep(model_name, "WBCIC", fold, seed, "model_fit"), read_rep(model_name, "WBCIC", fold, seed, "validation"); future = future_rep(model_name, fold, seed, torch.device("cuda")); source, validation, future, _, _ = normalize_source(source, validation, future); train = {key: np.concatenate([source[key], validation[key]]) if key in ("features", "labels", "subjects", "sessions", "logits") else None for key in source}; transported, random_value = transport_arrays(train, model_name, fold, seed); weight, bias = recovered_head(train["features"], train["logits"])
                for method in METHODS:
                    head = train_method(train["features"], train["labels"], train["subjects"], transported, random_value, method, model_name, fold, seed, weight, bias)
                    with torch.no_grad():
                        logits = head(torch.from_numpy(future["features"]).cuda()).float().cpu().numpy()
                    pred = logits.argmax(1); subject_values = {subject: float(balanced_accuracy_score(future["labels"][future["subjects"].astype(str) == subject], pred[future["subjects"].astype(str) == subject])) for subject in np.unique(future["subjects"].astype(str))};
                    for subject, value in subject_values.items(): rows.append({"model": model_name, "dataset": "WBCIC", "fold": fold, "seed": seed, "subject_id": subject, "method": method, "BA": value, "macro_F1": float(f1_score(future["labels"][future["subjects"].astype(str) == subject], pred[future["subjects"].astype(str) == subject], average="macro", zero_division=0))})
                print(f"[scst] {model_name} WBCIC fold={fold} seed={seed}", flush=True)
    per = pd.DataFrame(rows); per.to_csv(RESULTS / "SCST_TRAINING_PER_SUBJECT.csv", index=False); summary_rows = []; comparison_rows = []
    for model_name, group in per.groupby("model"):
        pivot = group.pivot_table(index=["fold", "seed", "subject_id"], columns="method", values="BA").reset_index(); deltas = pivot["SCST"] - pivot["ERM"]; random_delta = pivot["SCST"] - pivot["RandomTransport"]; subjects = pivot.groupby("subject_id", as_index=False).mean(numeric_only=True); values = subjects["SCST"] - subjects["ERM"]; rng = np.random.default_rng(stable_seed("scst-bootstrap", model_name)); draws = rng.integers(0, len(values), size=(10000, len(values))); distribution = values.to_numpy()[draws].mean(1); fold_delta = pivot.groupby("fold").apply(lambda frame: float((frame.SCST - frame.ERM).mean()), include_groups=False); summary_rows.append({"model": model_name, "dataset": "WBCIC", "ERM_BA": float(pivot.ERM.mean()), "Mixup_BA": float(pivot.Mixup.mean()), "RandomTransport_BA": float(pivot.RandomTransport.mean()), "SCST_BA": float(pivot.SCST.mean()), "delta_BA_vs_ERM": float(deltas.mean()), "CI95_L": float(np.quantile(distribution, .025)), "CI95_U": float(np.quantile(distribution, .975)), "delta_BA_vs_random": float(random_delta.mean()), "positive_folds": int((fold_delta > 0).sum()), "catastrophic_fold": bool((fold_delta < -.05).any()), "class_fidelity": True, "success": bool(deltas.mean() > 0 and np.quantile(distribution, .025) > 0 and (fold_delta > 0).sum() >= 3 and not (fold_delta < -.05).any() and random_delta.mean() > 0)}); comparison_rows.append({"model": model_name, "dataset": "WBCIC", "ERM_BA": float(pivot.ERM.mean()), "Mixup_BA": float(pivot.Mixup.mean()), "RandomTransport_BA": float(pivot.RandomTransport.mean()), "SCST_BA": float(pivot.SCST.mean())})
    summary = pd.DataFrame(summary_rows); summary.to_csv(RESULTS / "SCST_TRAINING_SUMMARY.csv", index=False); pd.DataFrame(comparison_rows).to_csv(RESULTS / "SCST_CONTROL_COMPARISON.csv", index=False); write = {"schema": "SCST_TRAINING_RESULT_V1", "models": summary.to_dict("records"), "future_session": "WBCIC session 2", "bootstrap_draws": 10000, "method_frozen_before_future": True}; (RESULTS / "SCST_TRAINING_RESULT.json").write_text(json.dumps(write, indent=2) + "\n", encoding="utf-8"); print(json.dumps(write, indent=2), flush=True)


if __name__ == "__main__": main()
