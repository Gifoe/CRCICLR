from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import p4d_common as c


P4A_CODE = c.P4A / "code"
sys.path.insert(0, str(P4A_CODE))
import common as p4a_common  # noqa: E402
import train as p4a_train  # noqa: E402


def selected_configs() -> list[tuple[str, float]]:
    payload = c.read_json(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json")
    if payload.get("future_BA_F1_CE_accessed_for_selection") is not False:
        raise RuntimeError("canonical config purity flag failed")
    selected = [
        (str(row["method"]), float(row["lambda_star"]))
        for row in payload["methods"]
        if row["status"] == "IDENTITY_MANIPULATION_COMPETENT"
    ]
    if not selected:
        raise RuntimeError("no manipulation-competent method")
    return selected


def missing(configs: list[tuple[str, float]]) -> list[str]:
    result = []
    for fold in c.FOLDS:
        for seed in c.SEEDS:
            for method, lam in configs:
                path = c.source_complete("S6", fold, seed, method, lam)
                if not path.is_file() or c.read_json(path).get("pass") is not True:
                    result.append(f"S6/fold-{fold}/seed-{seed}/{c.slug(method, lam)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard specification")
    protocol = c.read_json(c.EXP / "P4D_PROTOCOL_FROZEN.json")
    if protocol.get("method_future_task_outcomes_accessed_before_freeze") is not False:
        raise RuntimeError("P4D protocol was not frozen cleanly")
    configs = selected_configs()
    before = missing(configs)
    started = time.time()
    status_path = c.EXP / "runtime" / ("P4D_S6_CANONICAL_TRAINING_STATUS.json" if args.shard_count == 1 else f"P4D_S6_CANONICAL_TRAINING_STATUS_SHARD_{args.shard_index}.json")
    c.write_json(
        status_path,
        {
            "schema": "PERSIST_EEG_P4D_S6_CANONICAL_TRAINING_STATUS_V1",
            "status": "RUNNING",
            "timestamp_utc": c.now_utc(),
            "configs": [{"method": method, "lambda": lam} for method, lam in configs],
            "missing_before": before,
            "task_outcomes_accessed": False,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
    )
    bundle = p4a_common.load_data("S6")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("canonical S6 training requires the server GPU")
    raw = torch.from_numpy(np.asarray(bundle.x)).to(device=device, non_blocking=False)
    print(f"[S6] loaded {tuple(raw.shape)} on {torch.cuda.get_device_name(0)}; configs={configs}", flush=True)
    unit_number = -1
    for fold in c.FOLDS:
        roles = p4a_common.roles_for("S6", fold)
        train_indices = p4a_common.row_indices(bundle.metadata, roles["model_fit"], bundle.source_sessions)
        validation_indices = p4a_common.row_indices(bundle.metadata, roles["validation"], bundle.source_sessions)
        outcome_indices = p4a_common.row_indices(bundle.metadata, roles["outcome"], (bundle.future_session,))
        mean, std = p4a_common.compute_normalizer("S6", raw, train_indices)
        scope_hashes = {
            "source": p4a_common.array_sha256(train_indices.astype(np.int64)),
            "validation": p4a_common.array_sha256(validation_indices.astype(np.int64)),
            "outcome": p4a_common.array_sha256(outcome_indices.astype(np.int64)),
        }
        for seed in c.SEEDS:
            unit = p4a_common.run_dir("S6", fold, seed)
            unit.mkdir(parents=True, exist_ok=True)
            normalizer = unit / "normalizer.npz"
            if not normalizer.is_file():
                p4a_train.save_npz(normalizer, mean=mean.detach().cpu().numpy(), std=std.detach().cpu().numpy())
            initialization_seed = p4a_common.stable_seed("P4A-init", "S6", fold, seed)
            loader_seed = p4a_common.stable_seed("P4A-loader", "S6", fold, seed)
            bandwidths = [1.0] if all(method != "MMD" for method, _ in configs) else p4a_common.determine_mmd_bandwidths("S6", initialization_seed, raw, train_indices, mean, std)
            for method, lam in configs:
                unit_number += 1
                if unit_number % args.shard_count != args.shard_index:
                    continue
                p4a_train.run_configuration(
                    "S6",
                    fold,
                    seed,
                    method,
                    lam,
                    bundle,
                    raw,
                    train_indices,
                    validation_indices,
                    outcome_indices,
                    mean,
                    std,
                    initialization_seed,
                    loader_seed,
                    bandwidths,
                    scope_hashes,
                )
            remaining = missing(configs)
            c.write_json(
                status_path,
                {
                    "schema": "PERSIST_EEG_P4D_S6_CANONICAL_TRAINING_STATUS_V1",
                    "status": "RUNNING" if remaining else "COMPLETE",
                    "timestamp_utc": c.now_utc(),
                    "configs": [{"method": method, "lambda": lam} for method, lam in configs],
                    "missing_before": before,
                    "missing_now": remaining,
                    "completed_new": len(before) - len(remaining),
                    "elapsed_seconds": time.time() - started,
                    "task_outcomes_accessed": False,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                },
            )
    after = missing(configs)
    if after:
        c.write_json(
            status_path,
            {
                "schema": "PERSIST_EEG_P4D_S6_CANONICAL_TRAINING_STATUS_V1",
                "status": "SHARD_COMPLETE_WAITING_FOR_PEERS",
                "timestamp_utc": c.now_utc(),
                "missing_before": before,
                "missing_now": after,
                "elapsed_seconds": time.time() - started,
                "task_outcomes_accessed": False,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            },
        )
        print(f"P4D_S6_CANONICAL_SHARD_{args.shard_index}_COMPLETE_WAITING_FOR_PEERS")
        return
    completion = {
        "schema": "PERSIST_EEG_P4D_S6_CANONICAL_TRAINING_COMPLETE_V1",
        "pass": True,
        "timestamp_utc": c.now_utc(),
        "configs": [{"method": method, "lambda": lam} for method, lam in configs],
        "new_training_runs": len(before),
        "expected_balanced_runs": 15 * len(configs),
        "elapsed_seconds": time.time() - started,
        "task_outcomes_accessed": False,
        "P4A_405_grid_resumed": False,
        "shard_count": args.shard_count,
    }
    c.write_json(c.RESULTS / "P4D_S6_CANONICAL_TRAINING_COMPLETE.json", completion)
    print(json.dumps(completion, indent=2))
    print("P4D_S6_CANONICAL_TRAINING_COMPLETE_NO_TASK_OUTCOME_ACCESSED")


if __name__ == "__main__":
    main()
