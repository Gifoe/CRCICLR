"""Leakage-audited feature access for fold-specific V5 searches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import CACHE, logit
from run_search import _aligned_context, _output_features


@dataclass
class WBCICFoldFeatures:
    output: np.ndarray
    compact: np.ndarray
    shared: dict[int, np.ndarray]

    @classmethod
    def load(cls, data) -> "WBCICFoldFeatures":
        base = _output_features(data)
        current = np.column_stack(
            [
                data.current_probability,
                logit(data.current_probability),
                np.abs(data.current_probability - 0.5),
                data.current_prediction,
            ]
        )
        output = np.column_stack([base, current]).astype(np.float32)
        compact = _aligned_context(
            "WBCIC_S3_COMPACT_EEG_CONTEXT.npy",
            "WBCIC_S3_COMPACT_EEG_CONTEXT_METADATA.parquet",
            data.trial_uid,
        )
        shared: dict[int, np.ndarray] = {}
        for fold_id in range(5):
            metadata = pd.read_parquet(
                CACHE / f"WBCIC_SHARED_FOLD_{fold_id}_EEGNET_STABLE_METADATA.parquet"
            )
            values = np.load(
                CACHE / f"WBCIC_SHARED_FOLD_{fold_id}_EEGNET_STABLE_EMBEDDINGS.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            if len(metadata) != len(values) or set(metadata.fold_representation.astype(int)) != {fold_id}:
                raise RuntimeError(f"Malformed fold representation {fold_id}")
            positions = pd.Series(np.arange(len(metadata)), index=metadata.trial_uid.astype(str))
            if positions.index.duplicated().any():
                raise RuntimeError(f"Duplicate shared trial identities in fold {fold_id}")
            indices = positions.loc[list(map(str, data.trial_uid))].to_numpy(int)
            matrix = np.asarray(values[indices], dtype=np.float32)
            if matrix.shape != (len(data.labels), 32) or not np.isfinite(matrix).all():
                raise RuntimeError(f"Invalid shared feature matrix in fold {fold_id}: {matrix.shape}")
            shared[fold_id] = matrix
        return cls(output=output, compact=np.asarray(compact, np.float32), shared=shared)

    def get(self, name: str, fold_id: int) -> np.ndarray:
        if name == "OUTPUT":
            value = self.output
        elif name == "OUTPUT_COMPACT":
            value = np.column_stack([self.output, self.compact])
        elif name == "OUTPUT_SHARED":
            value = np.column_stack([self.output, self.shared[int(fold_id)]])
        elif name == "OUTPUT_ALL":
            value = np.column_stack([self.output, self.compact, self.shared[int(fold_id)]])
        elif name == "SHARED":
            value = self.shared[int(fold_id)]
        else:
            raise ValueError(name)
        result = np.asarray(value, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError(f"Non-finite feature values in {name}, fold {fold_id}")
        return result
