from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.special import softmax

from hsc_tta.models import make_token_head

from .io import sha256_file

CACHE_SCHEMA = "contextual-risk-source-cache-v1"


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_source(project_root: Path, dataset: str, seed: int, device: str):
    selected_path = project_root / "outputs/v2_joint_certified/source_models" / dataset / f"seed_{seed}" / "selected.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    model_path = Path(selected["model_path"])
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model = make_token_head(payload["architecture"], int(payload["n_classes"]))
    model.load_state_dict(payload["state_dict"])
    model.requires_grad_(False).to(device).eval()
    state_hash = _state_hash(payload["state_dict"])
    if payload.get("state_hash") and payload["state_hash"] != state_hash:
        raise RuntimeError("source checkpoint state hash mismatch")
    return model, payload, sha256_file(model_path), state_hash


def _infer(model: torch.nn.Module, tokens: np.ndarray, device: str, batch_size: int = 128) -> np.ndarray:
    logits = []
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            current = torch.as_tensor(tokens[start:start+batch_size], dtype=torch.float32, device=device)
            logits.append(model(current).float().cpu().numpy())
    return np.concatenate(logits)


def cache_source_predictions(
    project_root: str | Path,
    datasets: tuple[str, ...] = ("hmc", "eegmmidb"),
    device: str = "cuda",
) -> list[dict[str, object]]:
    root = Path(project_root)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in range(5):
            model, payload, checkpoint_hash, state_hash = load_frozen_source(root, dataset, seed, device)
            episodes = __import__("pandas").read_parquet(
                root / "data/episodes_contextual_risk" / dataset / f"seed_{seed}.parquet"
            )
            for episode in episodes.itertuples(index=False):
                target = root / "repo/outputs/contextual_risk/source_cache" / dataset / f"seed_{seed}" / f"{episode.subject_id.replace(':','_')}.npz"
                if target.exists():
                    with np.load(target, allow_pickle=False) as cached:
                        valid = (
                            str(cached["schema_version"]) == CACHE_SCHEMA
                            and str(cached["episode_hash"]) == episode.episode_hash
                            and str(cached["source_model_hash"]) == checkpoint_hash
                        )
                    if valid:
                        rows.append({"dataset": dataset, "seed": seed, "subject_id": episode.subject_id, "cache_path": str(target), "status": "reused"})
                        continue
                    raise RuntimeError(f"invalid pre-existing cache: {target}")
                h5_path = root / "data/embeddings_tokens_v2" / dataset / f"{episode.subject_id.split(':',1)[1]}.h5"
                with h5py.File(h5_path, "r") as handle:
                    ci = np.asarray(episode.context_indices, dtype=int)
                    fi = np.asarray(episode.future_indices, dtype=int)
                    context_tokens = handle["token_embeddings"][ci].astype(np.float32)
                    future_tokens = handle["token_embeddings"][fi].astype(np.float32)
                    future_labels = handle["labels"][fi].astype(np.int64)
                context_logits = _infer(model, context_tokens, device)
                future_logits = _infer(model, future_tokens, device)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".npz.part")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        context_indices=ci, future_indices=fi,
                        context_logits=context_logits.astype(np.float32),
                        future_logits=future_logits.astype(np.float32),
                        context_probabilities=softmax(context_logits.astype(np.float64), axis=1).astype(np.float32),
                        future_probabilities=softmax(future_logits.astype(np.float64), axis=1).astype(np.float32),
                        future_labels=future_labels,
                        source_model_hash=np.asarray(checkpoint_hash),
                        source_state_hash=np.asarray(state_hash),
                        episode_hash=np.asarray(episode.episode_hash),
                        schema_version=np.asarray(CACHE_SCHEMA),
                        architecture=np.asarray(payload["architecture"]),
                    )
                os.replace(temporary, target)
                rows.append({"dataset": dataset, "seed": seed, "subject_id": episode.subject_id, "cache_path": str(target), "status": "created"})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows
