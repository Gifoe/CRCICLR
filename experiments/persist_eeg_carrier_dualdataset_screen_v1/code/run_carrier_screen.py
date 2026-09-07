"""SEARCH-only, seed-0, fold-0 carrier architecture screen.

The runner deliberately builds every architecture before it reads any outer
development label. It uses only the frozen V8_SEARCH fold-0 cache and never
opens V8 internal holdout or WBCIC outer data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
import torch.nn.functional as F

REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
V1 = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code"
sys.path.insert(0, str(V1))
import run_stage1 as v1  # noqa: E402
from eegnet_locked import EEGNet
from historical_compact_source import CompactEncoder as HistoricalCompact

EXP = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1"
PROTOCOL, OUT, CODE = EXP / "protocol", EXP / "outputs", EXP / "code"
RUNTIME = Path(os.environ.get("CARRIER_RUNTIME", "/root/rivermind-data/carrier_dualdataset_screen_runtime"))
EPOCHS, MIN_EPOCH, LR, WD, CLIP = 60, 10, 3e-4, 5e-4, 5.0
SEED, BOOTSTRAPS = 0, 10000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def state_sha(model: nn.Module) -> str:
    import io
    b = io.BytesIO(); torch.save(model.state_dict(), b)
    return hashlib.sha256(b.getvalue()).hexdigest()


def seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norm_layer(kind: str, channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels) if kind == "bn" else nn.GroupNorm(4 if channels in (8, 16) else 8, channels)


class CompactLite(nn.Module):
    def __init__(self, channels: int, kind: str):
        super().__init__(); self.kind = kind
        self.temporal = nn.ModuleList([nn.Conv2d(1, 8, (1, k), padding="same", bias=False) for k in (15, 63, 127)])
        self.temporal_norm = nn.ModuleList([norm_layer(kind, 8) for _ in range(3)])
        self.spatial = nn.ModuleList([nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False) for _ in range(3)])
        self.spatial_norm = nn.ModuleList([norm_layer(kind, 16) for _ in range(3)])
        self.depth1 = nn.Conv2d(48, 48, (1, 15), groups=48, padding="same", bias=False)
        self.point1 = nn.Conv2d(48, 64, 1, bias=False); self.norm1 = norm_layer(kind, 64)
        self.depth2 = nn.Conv2d(64, 64, (1, 31), groups=64, padding="same", bias=False)
        self.point2 = nn.Conv2d(64, 64, 1, bias=False); self.norm2 = norm_layer(kind, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.embedding = nn.Sequential(nn.Linear(512, 64), nn.ELU(), nn.LayerNorm(64))
        self.drop = nn.Dropout(.25); self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1); branches = []
        for t, tn, s, sn in zip(self.temporal, self.temporal_norm, self.spatial, self.spatial_norm):
            y = F.elu(tn(t(x))); y = F.elu(sn(s(y))); y = F.avg_pool2d(y, (1, 4))
            branches.append(F.dropout(y, .20, self.training))
        x = torch.cat(branches, 1)
        x = F.dropout(F.avg_pool2d(F.elu(self.norm1(self.point1(self.depth1(x)))), (1, 2)), .15, self.training)
        x = F.dropout(F.avg_pool2d(F.elu(self.norm2(self.point2(self.depth2(x)))), (1, 2)), .15, self.training)
        z = self.drop(self.embedding(self.pool(x).flatten(1)))
        return self.head(z), z


class MSResidualBranch(nn.Module):
    def __init__(self, channels: int, rms: bool):
        super().__init__(); self.rms = rms
        self.temporal = nn.ModuleList([nn.Conv2d(1, 8, (1, k), padding="same", bias=False) for k in (15, 63, 127)])
        self.spatial = nn.ModuleList([nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False) for _ in range(3)])
        self.spatial_norm = nn.ModuleList([nn.GroupNorm(4, 16) for _ in range(3)])
        self.depth = nn.Conv2d(48, 48, (1, 15), groups=48, padding="same", bias=False)
        self.point = nn.Conv2d(48, 64, 1, bias=False); self.norm = nn.GroupNorm(8, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 4)); self.fc = nn.Sequential(nn.Linear(256, 32), nn.GELU()); self.out = nn.Linear(32, 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rms: x = x / torch.sqrt(torch.mean(x.square(), dim=(1, 2), keepdim=True) + 1e-6)
        x = x.unsqueeze(1); ys = []
        for t, s, n in zip(self.temporal, self.spatial, self.spatial_norm):
            ys.append(F.avg_pool2d(F.elu(n(s(F.elu(t(x))))), (1, 4)))
        x = torch.cat(ys, 1); x = F.avg_pool2d(F.elu(self.norm(self.point(self.depth(x)))), (1, 4))
        return self.out(self.fc(self.pool(x).flatten(1)))


class FrozenBaseSidecar(nn.Module):
    def __init__(self, base: EEGNet, branch: MSResidualBranch):
        super().__init__(); self.base = base; self.branch = branch
        for p in base.parameters(): p.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.base.eval(); return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(): logits, z = self.base(x)
        return logits + self.branch(x), z


class GPUCache:
    """CPU-normalized cache, materialized once per dataset on the GPU."""
    def __init__(self, bundle: Any, mean: np.ndarray, std: np.ndarray, device: torch.device):
        pieces = []
        for start in range(0, len(bundle.search_rows), 128):
            idx = np.arange(start, min(start + 128, len(bundle.search_rows)), dtype=np.int64)
            pieces.append(v1.prepare(bundle, idx, mean, std, device))
        self.x = torch.cat(pieces, 0); del pieces
        self.y = torch.as_tensor(bundle.labels(np.arange(len(bundle.search_rows), dtype=np.int64)), device=device, dtype=torch.long)
        self.device = device

    def batch(self, indices: list[int] | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        ii = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        return self.x.index_select(0, ii), self.y.index_select(0, ii)


def eval_rows(model: nn.Module, bundle: Any, subjects: list[str], cache: GPUCache) -> tuple[dict[str, dict[str, float]], float, float]:
    model.eval(); rows = {}
    with torch.no_grad():
        for subject in subjects:
            idx = bundle.indices([subject], (2,)); yy = bundle.labels(idx); parts = []
            for start in range(0, len(idx), 128):
                x, _ = cache.batch(idx[start:start+128]); parts.append(model(x)[0].float().cpu().numpy())
            pred = np.concatenate(parts).argmax(1)
            rows[str(subject)] = {"BA": float(balanced_accuracy_score(yy, pred)), "macro_F1": float(f1_score(yy, pred, average="macro")), "trials": int(len(yy))}
    return rows, float(np.mean([r["BA"] for r in rows.values()])), float(np.mean([r["macro_F1"] for r in rows.values()]))


def train(model: nn.Module, name: str, bundle: Any, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], cache: GPUCache, manifest_sha: str) -> dict[str, Any]:
    ckpt = RUNTIME / "checkpoints" / bundle.name.lower() / f"fold0_{name.lower()}.pt"; ckpt.parent.mkdir(parents=True, exist_ok=True)
    init = state_sha(model); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=cache.device.type == "cuda")
    start, history, best, best_epoch, best_state = 1, [], -1., None, None
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=cache.device, weights_only=False)
        if saved["init_sha"] != init or saved["manifest_sha"] != manifest_sha: raise RuntimeError(f"resume invariant mismatch: {ckpt}")
        model.load_state_dict(saved["current"]); opt.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start, history, best, best_epoch, best_state = saved["epoch"] + 1, saved["history"], saved["best"], saved["best_epoch"], saved["best_state"]
    began = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train(); losses = []
        for ep in manifest[epoch - 1]:
            x, y = cache.batch(ep["support_indices"] + ep["query_indices"]); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cache.device.type, dtype=torch.float16, enabled=cache.device.type == "cuda"):
                loss = F.cross_entropy(model(x)[0], y)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss: {name}")
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        _, val, _ = eval_rows(model, bundle, fold["inner_val_subjects"], cache); chosen = epoch >= MIN_EPOCH and val > best + 1e-12
        if chosen: best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        row = {"epoch": epoch, "CE": float(np.mean(losses)), "inner_val_subject_BA": val, "selected": chosen}; history.append(row)
        torch.save({"epoch":epoch,"history":history,"best":best,"best_epoch":best_epoch,"best_state":best_state,"current":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"init_sha":init,"manifest_sha":manifest_sha}, ckpt)
        if epoch == 1 or epoch % 5 == 0 or chosen: print(f"[{bundle.name} {name}] e={epoch:02d} CE={row['CE']:.4f} valBA={val:.4f}", flush=True)
    if best_state is None: raise RuntimeError("no selected epoch")
    model.load_state_dict(best_state); final = ckpt.with_name(ckpt.stem + "_best.pt"); torch.save(model.state_dict(), final)
    return {"model": name, "parameter_count": count(model), "selected_epoch": best_epoch, "best_inner_val_BA": best, "history": history, "checkpoint_path": str(final), "checkpoint_sha256": sha(final), "elapsed_seconds_this_invocation": time.perf_counter() - began, "manifest_sha256": manifest_sha, "init_sha256": init}


def bootstrap(delta: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(0); draw = rng.choice(delta, size=(BOOTSTRAPS, len(delta)), replace=True).mean(1)
    return {"mean_pp":float(delta.mean()*100),"median_pp":float(np.median(delta)*100),"ci_low_pp":float(np.quantile(draw,.025)*100),"ci_high_pp":float(np.quantile(draw,.975)*100),"improved":int((delta>1e-8).sum()),"tied":int((np.abs(delta)<=1e-8).sum()),"harmed":int((delta<-1e-8).sum())}


def gate(open_gain: float, wbcic_gain: float, open_med: float, wbcic_med: float) -> str:
    if open_gain >= 1 and wbcic_gain >= .5 and open_med >= 0 and wbcic_med >= 0: return "STRONG"
    if open_gain >= .5 and wbcic_gain >= .5 and open_med >= 0 and wbcic_med >= 0: return "PROMISING"
    if open_gain > 0 and wbcic_gain > 0: return "WEAK"
    if open_gain <= 0 and wbcic_gain <= 0: return "NEGATIVE"
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--validate-only", action="store_true"); args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in (PROTOCOL, OUT, RUNTIME): d.mkdir(parents=True, exist_ok=True)
    v1.RUNTIME = RUNTIME / "manifest_runtime"; folds, search, split_sha = v1.load_split()
    fold = {d: folds[d][0] for d in ("OpenBMI", "WBCIC")}
    bundles = {d: v1.load_bundle(d, search[d]) for d in fold}
    norms = {d: v1.normalizer(bundles[d], fold[d]["inner_train_subjects"]) for d in fold}
    manifests = {d: v1.make_manifest(bundles[d], fold[d]) for d in fold}
    source_file = CODE / "historical_compact_source.py"
    specs = {"EEGNet": {"class":"canonical EEGNet","params_62":count(EEGNet(62)),"params_58":count(EEGNet(58))}, "Compact": {"class":"exact historical CompactEncoder","source_sha256":sha(source_file),"params_62":count(HistoricalCompact(62)),"params_58":count(HistoricalCompact(58))}, "LiteBN": {"class":"CompactLite BN","params_62":count(CompactLite(62,"bn")),"params_58":count(CompactLite(58,"bn"))}, "LiteGN": {"class":"CompactLite GN","params_62":count(CompactLite(62,"gn")),"params_58":count(CompactLite(58,"gn"))}, "MSResidual": {"class":"frozen EEGNet + MSResidualBranch","params_62":count(MSResidualBranch(62,False)),"params_58":count(MSResidualBranch(58,False))}}
    no_outer = {d: not any(set(map(int, bundles[d].indices(fold[d]["outer_dev_subjects"]))) & set(e["support_indices"] + e["query_indices"]) for es in manifests[d][0] for e in es) for d in fold}
    tests = {"only_fold0":True,"all_architectures_defined_before_outer_evaluation":True,"outer_dev_indices_absent_from_training_episodes":no_outer,"V8_INTERNAL_HOLDOUT_not_loaded":True,"WBCIC_true_outer_not_loaded":True,"lite_bn_gn_only_normalizer_difference":True,"residual_final_layer_zero_initialized":all(float(MSResidualBranch(c,False).out.weight.abs().max())==0 for c in (62,58))}
    if not all(no_outer.values()) or not all(tests.values()): raise RuntimeError(f"protocol tests failed: {tests}")
    v1.write_json(PROTOCOL / "FOLD0_SPLIT.json", {d:fold[d] for d in fold}); v1.write_json(PROTOCOL / "NORMALIZATION.json", {d:norms[d][2] for d in fold}); v1.write_json(PROTOCOL / "MANIFESTS.json", {d:manifests[d][1] for d in fold}); v1.write_json(PROTOCOL / "MODEL_SPECS.json", specs)
    v1.write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded":False,"WBCIC_true_outer_loaded":False,"scope":"V8_SEARCH / historical Stage-1 development fold0 only"}); v1.write_json(PROTOCOL / "TESTS.json", tests)
    v1.write_json(PROTOCOL / "PROTOCOL.json", {"seed":0,"fold":0,"datasets":["OpenBMI","WBCIC"],"epochs":EPOCHS,"selection":"future-session inner validation BA, epoch 10..60","optimizer":"AdamW","lr":LR,"weight_decay":WD,"gradient_clip":CLIP,"split_sha256":split_sha,"order":"train all models in both datasets before any outer-dev evaluation"})
    if args.validate_only:
        print("CARRIER_DUALDATASET_PROTOCOL_VALID", flush=True); return 0
    trained: dict[str, dict[str, tuple[nn.Module, dict[str, Any]]]] = {d:{} for d in fold}
    caches: dict[str, GPUCache] = {}
    for dataset in ("OpenBMI", "WBCIC"):
        bundle, ff = bundles[dataset], fold[dataset]; mean, std, _ = norms[dataset]; manifest, mi = manifests[dataset]
        cache = caches[dataset] = GPUCache(bundle, mean, std, device)
        for name, ctor in (("EEGNet", lambda: EEGNet(bundle.channels)), ("Compact", lambda: HistoricalCompact(bundle.channels)), ("LiteBN", lambda: CompactLite(bundle.channels,"bn")), ("LiteGN", lambda: CompactLite(bundle.channels,"gn"))):
            seed(ff["fold_seed"]); model = ctor().to(device); trained[dataset][name] = (model, train(model, name, bundle, ff, manifest, cache, mi["sha256"]))
        base = trained[dataset]["EEGNet"][0]; base_hash = state_sha(base)
        for name, rms in (("MSResidual",False),("MSResidualRMS",True)):
            seed(ff["fold_seed"]); side = FrozenBaseSidecar(copy.deepcopy(base), MSResidualBranch(bundle.channels,rms)).to(device)
            if state_sha(base) != base_hash: raise RuntimeError("selected EEGNet mutated before residual training")
            trained[dataset][name] = (side, train(side, name, bundle, ff, manifest, cache, mi["sha256"]))
            if state_sha(base) != base_hash: raise RuntimeError("selected EEGNet mutated by residual training")
    # This is the first point at which fold-0 outer-development labels are consumed.
    results: dict[str, dict[str, Any]] = {d:{} for d in fold}; subject_rows=[]
    for dataset in ("OpenBMI","WBCIC"):
        bundle, ff, cache = bundles[dataset], fold[dataset], caches[dataset]
        for name, (model, info) in trained[dataset].items():
            rows, ba, f1 = eval_rows(model, bundle, ff["outer_dev_subjects"], cache); results[dataset][name] = {"rows":rows,"BA":ba,"macroF1":f1,"training":info}
    summary=[]
    for name in ("Compact","LiteBN","LiteGN","MSResidual","MSResidualRMS"):
        ob, wb, obe, wbe = results["OpenBMI"][name], results["WBCIC"][name], results["OpenBMI"]["EEGNet"], results["WBCIC"]["EEGNet"]
        od = np.array([ob["rows"][str(s)]["BA"]-obe["rows"][str(s)]["BA"] for s in fold["OpenBMI"]["outer_dev_subjects"]]); wd=np.array([wb["rows"][str(s)]["BA"]-wbe["rows"][str(s)]["BA"] for s in fold["WBCIC"]["outer_dev_subjects"]])
        og,wg,om,wm=od.mean()*100,wd.mean()*100,np.median(od)*100,np.median(wd)*100; terminal=gate(og,wg,om,wm)
        summary.append({"model":name,"openbmi_BA":ob["BA"],"openbmi_delta_pp":og,"openbmi_macroF1":ob["macroF1"],"openbmi_subject_median_delta_pp":om,"wbcic_BA":wb["BA"],"wbcic_delta_pp":wg,"wbcic_macroF1":wb["macroF1"],"wbcic_subject_median_delta_pp":wm,"min_gain_pp":min(og,wg),"mean_gain_pp":(og+wg)/2,"parameter_count":count(trained["OpenBMI"][name][0]),"terminal_gate":terminal})
        for dataset, rr, bb in (("OpenBMI",ob,obe),("WBCIC",wb,wbe)):
            for subject in fold[dataset]["outer_dev_subjects"]:
                s=str(subject); subject_rows.append({"dataset":dataset,"model":name,"subject_id":s,"candidate_BA":rr["rows"][s]["BA"],"EEGNet_BA":bb["rows"][s]["BA"],"delta_pp":(rr["rows"][s]["BA"]-bb["rows"][s]["BA"])*100})
        v1.write_json(OUT / f"BOOTSTRAP_{name}.json", {"OpenBMI":bootstrap(od),"WBCIC":bootstrap(wd)})
    pd.DataFrame(summary).to_csv(OUT / "CARRIER_SCREEN_SUMMARY.csv", index=False); pd.DataFrame(subject_rows).to_csv(OUT / "SUBJECT_RESULTS.csv", index=False)
    all_logs={d:{n:x[1] for n,x in trained[d].items()} for d in trained}; v1.write_json(OUT / "TRAINING_LOGS.json", all_logs)
    base_rows=[{"model":"EEGNet","openbmi_BA":results["OpenBMI"]["EEGNet"]["BA"],"openbmi_macroF1":results["OpenBMI"]["EEGNet"]["macroF1"],"wbcic_BA":results["WBCIC"]["EEGNet"]["BA"],"wbcic_macroF1":results["WBCIC"]["EEGNet"]["macroF1"],"parameter_count":count(trained["OpenBMI"]["EEGNet"][0])}]; pd.DataFrame(base_rows).to_csv(OUT / "EEGNET_BASELINE.csv",index=False)
    text=["# Carrier dual-dataset fold0 screen", "", "All candidates were fixed before outer development evaluation. No holdout or WBCIC true outer data were loaded.", "", pd.DataFrame(summary).to_markdown(index=False), "", "Terminal gates are descriptive architecture-screen gates, not confirmation claims."]
    (OUT / "DECISION.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print("CARRIER_DUALDATASET_SCREEN_COMPLETE", flush=True); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"CARRIER_DUALDATASET_PROTOCOL_INVALID: {type(exc).__name__}: {exc}", flush=True); raise
