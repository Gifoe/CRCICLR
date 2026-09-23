"""TRAIN-only audit: orthogonal Q_final versus frozen canonical P projector.

This resolves whether the prompt's Q Q.T formula reproduces the frozen final
Protected object. It is diagnostic evidence, not an alternative P definition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=C.MODELS, required=True)
    parser.add_argument("--task", choices=C.TASKS, required=True)
    parser.add_argument("--fold", type=int, choices=C.FOLDS, required=True)
    args = parser.parse_args()
    model, task, fold = args.model, args.task, args.fold
    tag = f"{model}_{task}_fold{fold}"
    result_path = EXP / "gate" / f"FINAL_PROJECTOR_CONSISTENCY_{tag}.json"
    failure_path = EXP / "gate" / f"FINAL_PROJECTOR_CONSISTENCY_{tag}_FAIL_CLOSED.json"
    if result_path.exists() or failure_path.exists():
        raise RuntimeError("projector audit evidence already exists; no overwrite")
    try:
        gate_path = EXP / "gate" / "GATE_AUDIT.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASSED" or gate.get("final_heldout_accessed"):
            raise RuntimeError("upstream gate failed")
        source = C.prior_record(model, task, fold)
        protocol_sha = C.assert_lock(model, task, fold, source)
        if protocol_sha != gate["upstream_protocol_sha256"]:
            raise RuntimeError("frozen protocol SHA mismatch")
        record, stored, checkpoint, data = source["record"], source["stored"], source["checkpoint"], source["data"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = C.UP.helper(model).build_model({
            "Model": model, "Task": task, "fold": fold, "seed": 0,
            "channels": int(record.get("channels") or 62),
            "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]),
            "checkpoint_path": str(checkpoint),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        cap = C.UP.capped_train(data, task, model, fold)
        h, _, _ = C.UP.hook_representations(net, head, cap[0], model, device)
        spec = C.UP.helper(model).spectrum(h, *cap[1:], task, model, fold)
        if C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"]) != source["hashes"]["basis_sha256"]:
            raise RuntimeError("frozen canonical basis mismatch")
        dims = np.asarray(stored["protected_blocks"], int)
        q, mu = C.final_q(spec, dims)
        left = (spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]
        right = C.UP.raw_base(spec)[dims]
        exact = left.astype(np.float64) @ right.astype(np.float64)
        orthogonal = q.astype(np.float64) @ q.astype(np.float64).T
        subset = h[: min(64, len(h))].astype(np.float64) - mu.astype(np.float64)
        p_exact = subset @ exact
        p_orthogonal = subset @ orthogonal
        difference = p_exact - p_orthogonal
        C.write_json(result_path, {
            "schema": "PC_MECHANISM_CLOSURE_FINAL_PROJECTOR_CONSISTENCY_V1",
            "status": "AUDITED", "model": model, "task": task, "fold": fold, "seed": 0,
            "scope": "TRAIN_ONLY_CAPPED_REPRESENTATIONS",
            "protected_rank": int(len(dims)),
            "classifier_input_dim": int(h.shape[1]),
            "orthogonal_vs_canonical_operator_fro": float(np.linalg.norm(orthogonal - exact, "fro")),
            "orthogonal_vs_canonical_operator_max_abs": float(np.max(np.abs(orthogonal - exact))),
            "orthogonal_vs_canonical_P_mean_relative_l2": float(np.mean(np.linalg.norm(difference, axis=1) / np.maximum(np.linalg.norm(p_exact, axis=1), 1e-12))),
            "orthogonal_vs_canonical_P_max_abs": float(np.max(np.abs(difference))),
            "canonical_projector_idempotence_fro": float(np.linalg.norm(exact @ exact - exact, "fro")),
            "canonical_projector_asymmetry_fro": float(np.linalg.norm(exact - exact.T, "fro")),
            "orthogonal_projector_idempotence_fro": float(np.linalg.norm(orthogonal @ orthogonal - orthogonal, "fro")),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_basis_sha256": source["hashes"]["basis_sha256"],
            "upstream_checkpoint_sha256": source["hashes"]["checkpoint_sha256"],
            "implementation_sha256": C.digest(Path(__file__)),
            "final_heldout_accessed": False,
        })
        print("PROJECTOR_CONSISTENCY_AUDITED", tag, flush=True)
    except Exception as error:
        C.write_json(failure_path, {
            "status": "FAIL_CLOSED", "reason": f"{type(error).__name__}: {error}",
            "implementation_sha256": C.digest(Path(__file__)),
            "final_heldout_accessed": False,
        })
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
