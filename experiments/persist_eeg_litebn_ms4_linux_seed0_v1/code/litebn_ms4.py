"""K-sample extension of the validated controlled LiteBN dual-forward primitive."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from litebn_sc_controlled import restore_torch_rng, torch_rng_state


def controlled_multi_forward(
    model: nn.Module,
    value: torch.Tensor,
    secondary_rng: dict[str, Any],
    k: int = 4,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    """Run A on live buffers and B..K on independent pre-A buffer clones.

    A consumes only the main Torch RNG stream.  All extra forwards consume one
    persistent secondary stream sequentially.  Every extra functional call
    shares the live Parameter objects but owns a distinct pre-A buffer clone.
    """
    if k < 1:
        raise ValueError("k must be at least one")
    parameters = dict(model.named_parameters())
    extra_buffers = [
        {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        for _ in range(k - 1)
    ]
    logits_a, embedding_a = model(value)
    logits, embeddings = [logits_a], [embedding_a]
    main_after_a = torch_rng_state()
    trace: list[dict[str, Any]] = []
    if k == 1:
        return logits, embeddings, secondary_rng, trace
    restore_torch_rng(secondary_rng)
    try:
        for buffers in extra_buffers:
            current_logits, current_embedding = torch.func.functional_call(
                model, (parameters, buffers), (value,), strict=True
            )
            logits.append(current_logits)
            embeddings.append(current_embedding)
            trace.append(torch_rng_state())
        secondary_after = trace[-1]
    finally:
        restore_torch_rng(main_after_a)
    return logits, embeddings, secondary_after, trace
