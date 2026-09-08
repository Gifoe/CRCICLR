"""Runtime-only, source-session feature caching for frozen CFRF carriers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class SignalCache:
    """Materializes only caller-authorized samples, so training never loads outer labels."""
    def __init__(self, bundle: Any, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device, prepare: Any) -> None:
        self.global_indices = np.asarray(indices, dtype=np.int64)
        if len(self.global_indices) == 0:
            raise RuntimeError("empty authorized signal cache")
        self.lookup = {int(index): position for position, index in enumerate(self.global_indices.tolist())}
        chunks = [prepare(bundle, self.global_indices[start:start + 128], mean, std, device) for start in range(0, len(self.global_indices), 128)]
        self.x = torch.cat(chunks, dim=0)
        self.y = torch.as_tensor(bundle.labels(self.global_indices), dtype=torch.long, device=device)
        self.device = device

    def by_global(self, indices: np.ndarray | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        positions = [self.lookup[int(index)] for index in np.asarray(indices, dtype=np.int64).tolist()]
        position_tensor = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        return self.x.index_select(0, position_tensor), self.y.index_select(0, position_tensor)


@torch.no_grad()
def build_or_load(path: Path, eegnet: torch.nn.Module, litebn: torch.nn.Module, signal: SignalCache, extract: Any, carrier_hashes: dict[str, str]) -> dict[str, torch.Tensor]:
    """Cache logits/features only in the runtime directory; never in the Git worktree."""
    if path.is_file():
        data = torch.load(path, map_location=signal.device, weights_only=False)
        if data.get("carrier_hashes") != carrier_hashes or not torch.equal(data["global_indices"].cpu(), torch.as_tensor(signal.global_indices)):
            raise RuntimeError("feature-cache provenance mismatch")
        return {key: value.to(signal.device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in ("z_a", "z_e", "h_a", "h_e")}
    for start in range(0, signal.x.shape[0], 256):
        z_a, z_e, h_a, h_e = extract(eegnet, litebn, signal.x[start:start + 256])
        for key, value in zip(pieces, (z_a, z_e, h_a, h_e)):
            pieces[key].append(value.detach().cpu())
    data: dict[str, Any] = {key: torch.cat(value, dim=0) for key, value in pieces.items()}
    data.update({"y": signal.y.detach().cpu(), "global_indices": torch.as_tensor(signal.global_indices), "carrier_hashes": carrier_hashes})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return {key: value.to(signal.device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
