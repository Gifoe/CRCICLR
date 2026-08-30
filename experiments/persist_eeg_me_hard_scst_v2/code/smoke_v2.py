"""Deterministic source-only integration smoke test on actual WBCIC loaders."""
from __future__ import annotations

import json

import numpy as np
import torch

import v2_common as c
from candidate_engine import AdmissibilityEngine
from mixed_effects import MixedEffectsBank


def main() -> None:
    c.ensure_dirs()
    device = torch.device("cuda")
    raw, metadata, root = c.load_development_data("WBCIC")
    role = c.roles("WBCIC", 0)
    allowed = c.S1.row_indices(metadata, role["model_fit"], (c.SOURCE_TRAIN_SESSION["WBCIC"],))
    subjects = c.subject_sort(metadata.iloc[allowed].subject_id.unique())[:3]
    positions = []
    for subject in subjects:
        for label in (0, 1):
            mask = allowed[(metadata.iloc[allowed].subject_id.astype(str).to_numpy() == subject) & (metadata.iloc[allowed].label.to_numpy() == label)]
            positions.extend(mask[:4].tolist())
    indices = np.asarray(sorted(positions), np.int64)
    if len(indices) != 24:
        raise RuntimeError(f"SMOKE_BALANCE_FAILURE:{len(indices)}")
    net, checkpoint = c.load_anchor("ATCNet-CleanRoom", "WBCIC", 0, 0, device)
    net.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = torch.from_numpy(c.normalize_raw(raw[indices])).to(device)
        features = c.model_features("ATCNet-CleanRoom", net, x).float().cpu().numpy()
    labels = metadata.iloc[indices].label.to_numpy(np.int64)
    subject_values = metadata.iloc[indices].subject_id.astype(str).to_numpy()
    bank = MixedEffectsBank(features, labels, subject_values, indices)

    def head(values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(np.asarray(values, np.float32)).to(device)
            return c.feature_logits("ATCNet-CleanRoom", net, tensor).float().cpu().numpy()

    engine = AdmissibilityEngine(features, labels, subject_values, indices, bank)
    candidates = engine.generate(0, head, k_targets=c.K_TARGETS, alphas=c.ALPHAS, seed=c.stable_seed("smoke", 0), factorized=True, include_random=True)
    result = {
        "pass": bool(len(candidates.alphas) == 12 and np.isfinite(candidates.deltas).all()),
        "dataset": "WBCIC",
        "root": str(root),
        "checkpoint": str(checkpoint),
        "sessions_opened": [0],
        "subjects": subjects,
        "rows": int(len(indices)),
        "representation_dim": int(features.shape[1]),
        "candidate_budget": int(len(candidates.alphas)),
        "valid_candidates": int(candidates.valid.sum()),
        "bank_norm_radius": float(bank.norm_radius),
        "outer_or_future_opened": False,
    }
    if not result["pass"]:
        raise RuntimeError(result)
    c.write_json(c.RESULTS / "SMOKE_TEST.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

