"""Bounded-memory EEGNet directional cell runner; frozen estimand unchanged."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import run_directional_cell_v5 as base
from directional_core_v3 import DirectionSet
from directional_streaming_fit_v1 import fit_directions_streamed, streamed_matrix_and_pairs


class StreamedPrefixMarker:
    def __init__(self, rows, device):
        self.rows = int(rows)
        self.device = device

    def __len__(self):
        return self.rows


ORIGINAL_FACTORIZED = base.factorized_mixed_difference


def streamed_factorized(*args, **kwargs):
    kwargs["spatial_base"] = None
    kwargs["batch_size"] = 64
    return ORIGINAL_FACTORIZED(*args, **kwargs)


def streamed_fit(train_a, q, mean, **kwargs):
    return fit_directions_streamed(train_a, q, mean, direction_set_type=DirectionSet,
                                   chunk_features=8192, **kwargs)


base.factorized_mixed_difference = streamed_factorized
base.precompute_spatial_base = lambda runner, shape, activations, **kwargs: StreamedPrefixMarker(
    len(activations), runner.device)
base.fit_directions = streamed_fit
base.matrix_and_pairs = lambda *args, **kwargs: streamed_matrix_and_pairs(base, *args, **kwargs)

ORIGINAL_PRINT = print


def truthful_print(*args, **kwargs):
    values = tuple("DIRECTIONAL_V4_SPATIAL_PREFIX_STREAMED" if x == "DIRECTIONAL_V4_SPATIAL_BASE_CACHED" else x
                   for x in args)
    ORIGINAL_PRINT(*values, **kwargs)


base.print = truthful_print
ORIGINAL_WRITE_JSON = base.C.write_json


def truthful_write_json(path, value):
    if isinstance(value, dict) and str(value.get("schema", "")).startswith("PC_MECHANISM_DIRECTIONAL_"):
        value = dict(value)
        value["streamed_fit_sha256"] = base.C.digest(Path(__file__).with_name("directional_streaming_fit_v1.py"))
        if value.get("schema") == "PC_MECHANISM_DIRECTIONAL_STAGE_V4":
            value["forward_schedule"] = "EEGNET_SPATIAL_AFFINE_PREFIX_RECOMPUTED_STREAMED_BATCH64"
            value["numeric_batch_size"] = 64
            value["pca_feature_chunk_size"] = 8192
    ORIGINAL_WRITE_JSON(path, value)


base.C.write_json = truthful_write_json
base.__file__ = str(Path(__file__).resolve())
base.main()
