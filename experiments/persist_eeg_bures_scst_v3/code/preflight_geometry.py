"""Compute source-only transport realization diagnostics before model training."""
from __future__ import annotations

import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as c
from bures import BuresBank
from source_v3 import AdapterHead, _candidate_geometry, _geometry_subject_frame, _matched_masks, _subject_batches


def _warmup(train: dict[str, np.ndarray], dataset: str, fold: int, seed: int, device: torch.device) -> AdapterHead:
    c.set_seed(c.stable_seed("bures-preflight-warmup", dataset, fold, seed)); model = AdapterHead(int(train["features"].shape[1])).to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=c.HEAD_LR, weight_decay=c.WEIGHT_DECAY)
    for epoch in range(c.WARMUP_EPOCHS):
        model.train()
        for positions in _subject_batches(train["subjects"], seed, epoch):
            if not len(positions): continue
            x = torch.from_numpy(train["features"][positions]).to(device); y = torch.from_numpy(train["labels"][positions]).long().to(device)
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model.logits(x).float(), y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.head.parameters(), 3.0); opt.step()
    model.eval(); return model


def _one(dataset: str, fold: int, seed: int, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = c.load_feature_cache(dataset, fold, seed, "train"); teacher = _warmup(train, dataset, fold, seed, device)
    with torch.inference_mode(): features = teacher.features(torch.from_numpy(train["features"]).to(device)).float().cpu().numpy()
    bank = BuresBank(features, train["labels"], train["subjects"], train["indices"], dataset=dataset, fold=fold, seed=seed)
    structured = _candidate_geometry(features, train["labels"], train["subjects"], train["indices"], dataset, fold, seed, teacher, mode="structured", bank=bank)
    random = _candidate_geometry(features, train["labels"], train["subjects"], train["indices"], dataset, fold, seed, teacher, mode="random", bank=bank)
    (mask_s, mask_r), matching = _matched_masks(structured, random, dataset, fold, seed)
    structured.valid &= mask_s; random.valid &= mask_r
    gs = _geometry_subject_frame(structured, train, dataset, fold, seed, "Bures-HardSCST", 0.50, 0.50); gr = _geometry_subject_frame(random, train, dataset, fold, seed, "Bures-HardRandom", 0.50, 0.50)
    return pd.concat([gs, gr], ignore_index=True), matching.assign(method="Bures-HardSCST", q=0.50, lambda_T=0.50), pd.DataFrame([{"dataset": dataset, "fold": fold, "seed": seed, "zero_anchor_self_neighbor_rate": 1.0, "structured_valid_candidates": int(structured.valid.sum()), "random_valid_candidates": int(random.valid.sum()), "matched_pairs": int(((mask_s & mask_r)).sum())}])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--sample", action="store_true", help="run one authorized fold/seed per development dataset"); args = parser.parse_args()
    c.ensure_dirs(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); geometry = []; matches = []; audit = []
    datasets = c.DATASETS; folds = (0,) if args.sample else c.FOLDS; seeds = (0,) if args.sample else c.SEEDS
    for dataset in datasets:
        for fold in folds:
            for seed in seeds:
                print(f"[preflight] START {dataset} f={fold} s={seed}", flush=True); g, m, a = _one(dataset, fold, seed, device); geometry.append(g); matches.append(m); audit.append(a); print(f"[preflight] DONE {dataset} f={fold} s={seed}", flush=True)
    c.write_csv(c.RESULTS / "GEOMETRY_PER_SUBJECT.csv", pd.concat(geometry, ignore_index=True)); c.write_csv(c.RESULTS / "RANDOM_AFFINE_MATCHING.csv", pd.concat(matches, ignore_index=True)); c.write_csv(c.RESULTS / "PREFLIGHT_AUDIT.csv", pd.concat(audit, ignore_index=True)); c.write_json(c.RESULTS / "PREFLIGHT_STATUS.json", {"source_units": len(audit), "s3_opened": False, "outer_or_sealed_opened": False})


if __name__ == "__main__": main()
