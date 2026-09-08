"""Pure frozen state intervention utilities for the NRRF BatchNorm audit."""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


ALPHA = 0.5


def batchnorm_buffer_keys(module: nn.Module) -> list[str]:
    """Discover allowed state keys from actual BatchNorm modules, never by heuristic."""
    keys: list[str] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            prefix = f"{name}." if name else ""
            keys.extend([prefix + "running_mean", prefix + "running_var", prefix + "num_batches_tracked"])
    expected = set(module.state_dict())
    answer = sorted(set(keys))
    if not answer or not set(answer) <= expected:
        raise RuntimeError("BatchNorm buffer discovery does not match model state")
    return answer


def frozen(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    if module.training or any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError("frozen evaluation invariant failed")
    return module


def restore_source_bn(current: dict[str, torch.Tensor], source: dict[str, torch.Tensor], allowed: list[str]) -> dict[str, torch.Tensor]:
    if set(current) != set(source):
        raise RuntimeError("source/current LiteBN state key mismatch")
    result = {key: value.detach().clone() for key, value in current.items()}
    for key in allowed:
        result[key] = source[key].detach().clone()
    return result


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a.detach().cpu(), b.detach().cpu()))


def validate_intervention(module: nn.Module, current: dict[str, torch.Tensor], source: dict[str, torch.Tensor], restored: dict[str, torch.Tensor], allowed: list[str]) -> dict[str, Any]:
    param_keys = {name for name, _ in module.named_parameters()}
    all_keys = set(current)
    if set(source) != all_keys or set(restored) != all_keys:
        raise RuntimeError("intervention state keys differ")
    changed = sorted(key for key in all_keys if not _same(current[key], restored[key]))
    parameter_identical = all(_same(current[key], restored[key]) for key in param_keys)
    only_allowed_changed = set(changed) <= set(allowed)
    restored_exact = all(_same(restored[key], source[key]) for key in allowed)
    unchanged_other = all(_same(current[key], restored[key]) for key in all_keys - set(allowed))
    affine_unchanged = all(_same(current[key], restored[key]) for key in param_keys if key.endswith(".weight") or key.endswith(".bias"))
    report = {"parameter_tensors_bitwise_identical": parameter_identical, "changed_keys": changed,
              "only_allowed_bn_buffers_differ": only_allowed_changed, "restored_buffers_equal_source": restored_exact,
              "all_non_bn_keys_bitwise_identical": unchanged_other, "bn_affine_weight_bias_remain_stage2": affine_unchanged,
              "allowed_bn_buffer_keys": allowed}
    if not all((parameter_identical, only_allowed_changed, restored_exact, unchanged_other, affine_unchanged)):
        raise RuntimeError(f"BN_BUFFER_INTERVENTION_INVALID: {report}")
    return report


def drift_rows(dataset: str, fold: int, method: str, source: dict[str, torch.Tensor], current: dict[str, torch.Tensor], allowed: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # pair the programmatically discovered keys by BatchNorm layer prefix
    prefixes = sorted({key.rsplit(".", 1)[0] for key in allowed})
    for prefix in prefixes:
        mean, var, tracked = (f"{prefix}.running_mean", f"{prefix}.running_var", f"{prefix}.num_batches_tracked")
        def norms(key: str) -> tuple[float, float]:
            delta = (current[key].float() - source[key].float()).norm().item()
            base = source[key].float().norm().item()
            return float(delta), float(delta / (base + 1e-12))
        mean_abs, mean_rel = norms(mean); var_abs, var_rel = norms(var)
        rows.append({"dataset": dataset, "fold": int(fold), "method": method, "bn_layer": prefix,
                     "running_mean_abs_L2": mean_abs, "running_mean_relative_L2": mean_rel,
                     "running_var_abs_L2": var_abs, "running_var_relative_L2": var_rel,
                     "source_num_batches_tracked": int(source[tracked].item()), "current_num_batches_tracked": int(current[tracked].item())})
    return rows


@torch.no_grad()
def fused_logits(anchor: nn.Module, expressive: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # The rule remains exactly the prior NRRF/logit-50 rule and is intentionally not configurable.
    return ALPHA * anchor(x)[0] + ALPHA * expressive(x)[0]
