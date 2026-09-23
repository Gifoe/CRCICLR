"""Frozen-stage P/C mediation primitives; no fitting or outcome selection.

Intermediate stages use orthogonal Q_l Q_l.T. At classifier input, the
explicitly amended protocol requires the frozen canonical P projector,
which is oblique. The two definitions are never silently substituted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


LONG_RANGE_STAGES: dict[str, tuple[str, str, str]] = {
    "EEGNet": ("temporal_bn", "spatial_elu_pool1", "depth_point_elu_pool2"),
    "EEGConformer": ("patch_tokenizer", "conformer_block4", "classifier_256_elu"),
}


def next_stage(runner: Any, stage: str) -> str:
    names = runner.names
    index = names.index(stage)
    if index + 1 == len(names):
        raise ValueError(f"terminal stage has no adjacent successor: {stage}")
    return names[index + 1]


def advance_one(runner: Any, stage: str, activation: torch.Tensor) -> torch.Tensor:
    """Apply exactly the frozen native modules between two audited boundaries."""
    m = runner.m
    if runner.name == "EEGNet":
        if stage == "temporal_bn":
            return m.drop1(m.pool1(F.elu(m.bn2(m.spatial(activation)))))
        if stage == "spatial_elu_pool1":
            return m.drop2(m.pool2(F.elu(m.bn3(m.point(m.depth(activation))))))
        if stage == "depth_point_elu_pool2":
            return m.embedding(activation.flatten(1))
        if stage == "embedding_64d":
            return activation
    elif runner.name == "EEGConformer":
        if stage == "patch_tokenizer":
            return m.encoder[1](m.encoder[0](activation))
        if stage == "conformer_block2":
            return m.encoder[3](m.encoder[2](activation))
        if stage == "conformer_block4":
            return m.encoder[5](m.encoder[4](activation))
        if stage == "conformer_block6":
            return m.classifier[1](m.classifier[0](activation.flatten(1)))
        if stage == "classifier_256_elu":
            return m.classifier[4](m.classifier[3](m.classifier[2](activation)))
        if stage == "classifier_32_elu":
            return m.classifier[5](activation)
    raise KeyError((runner.name, stage))


def verify_adjacent_forward(runner: Any, x: np.ndarray, *, atol: float = 1e-5) -> dict[str, float]:
    """Compare each adjacent replay to the native all-stage forward on inputs."""
    if len(x) == 0:
        raise ValueError("empty verification input")
    if runner.m.training or runner.head.training:
        raise RuntimeError("native model/head must be in eval mode")
    with torch.inference_mode():
        native, _, _ = runner.all(runner.tensor(x[: min(4, len(x))]))
        errors = {}
        for stage in runner.names[:-1]:
            following = next_stage(runner, stage)
            actual = advance_one(runner, stage, native[stage])
            target = native[following]
            if actual.shape != target.shape:
                raise RuntimeError(f"adjacent shape mismatch {stage}->{following}: {actual.shape} != {target.shape}")
            error = float((actual - target).abs().max().item())
            errors[f"{stage}->{following}"] = error
            if not torch.allclose(actual, target, rtol=1e-5, atol=atol):
                raise RuntimeError(f"adjacent native forward mismatch {stage}->{following}: {error}")
        return errors


def _project(a: torch.Tensor, q: torch.Tensor, mu: torch.Tensor,
             projector: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    x = a - mu
    p = (x @ q) @ q.T if projector is None else (x @ projector[0]) @ projector[1]
    return p, x - p


def _center(z: torch.Tensor) -> torch.Tensor:
    return z - z.mean(dim=1, keepdim=True)


def _norm(z: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(z, dim=1)


def _cosine(z: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    return (z * total).sum(dim=1) / (_norm(z) * _norm(total)).clamp_min(1e-12)


def _margin(effect: torch.Tensor, label: torch.Tensor, rival: torch.Tensor) -> torch.Tensor:
    row = torch.arange(len(effect), device=effect.device)
    return effect[row, label] - effect[row, rival]


def _one_direction(
    runner: Any,
    stage: str,
    following: str,
    next_shape: tuple[int, ...],
    base_next: torch.Tensor,
    intervention_next: torch.Tensor,
    z_base: torch.Tensor,
    z_intervention: torch.Tensor,
    q_next: torch.Tensor,
    mu_next: torch.Tensor,
    labels: torch.Tensor,
    donors: torch.Tensor,
    prefix: str,
    next_projector: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    q_next = q_next.double()
    mu_next = mu_next.double()
    if next_projector is not None:
        next_projector = (next_projector[0].double(), next_projector[1].double())
    pb, cb = _project(base_next.double(), q_next, mu_next, next_projector)
    pi, ci = _project(intervention_next.double(), q_next, mu_next, next_projector)
    if float((mu_next + pb + cb - base_next.double()).abs().max()) >= 1e-5:
        raise RuntimeError("successor-layer P+C reconstruction failed")
    p_hybrid = (mu_next + pi + cb).to(base_next.dtype).reshape((len(base_next), *next_shape))
    c_hybrid = (mu_next + pb + ci).to(base_next.dtype).reshape((len(base_next), *next_shape))
    _, zp = runner.from_stage(following, p_hybrid)
    _, zc = runner.from_stage(following, c_hybrid)
    total = _center(z_intervention - z_base)
    pmed = _center(zp - z_base)
    cmed = _center(zc - z_base)
    residual = total - pmed - cmed
    reference = z_base.clone()
    reference[torch.arange(len(reference), device=reference.device), labels] = -torch.inf
    rival = reference.argmax(dim=1)
    vectors = {"total": total, "to_P": pmed, "to_C": cmed, "nonlinear_residual": residual}
    out: dict[str, np.ndarray] = {
        f"{prefix}_to_P_rep": _norm(pi - pb).detach().cpu().numpy(),
        f"{prefix}_to_C_rep": _norm(ci - cb).detach().cpu().numpy(),
    }
    for name, effect in vectors.items():
        out[f"{prefix}_{name}_norm"] = _norm(effect).detach().cpu().numpy()
        out[f"{prefix}_{name}_true_margin"] = _margin(effect, labels, rival).detach().cpu().numpy()
        out[f"{prefix}_{name}_donor_margin"] = _margin(effect, donors, labels).detach().cpu().numpy()
        out[f"{prefix}_{name}_cosine_total"] = _cosine(effect, total).detach().cpu().numpy()
    return out


def adjacent_mediation(
    runner: Any,
    stage: str,
    recipient: np.ndarray,
    donor: np.ndarray,
    shape: tuple[int, ...],
    q: np.ndarray,
    mu: np.ndarray,
    q_next: np.ndarray,
    mu_next: np.ndarray,
    recipient_label: np.ndarray,
    donor_label: np.ndarray,
    final_projector: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Compute adjacent P/C mediated effects for fixed recipient/donor pairs.

    All arrays are already selected; this function never uses labels for
    selection, projection, fitting, or perturbation. Labels enter margins only.
    """
    if recipient.shape != donor.shape or recipient.ndim != 2:
        raise ValueError("paired flattened activations must have equal 2-D shape")
    n, d = recipient.shape
    if n == 0:
        raise ValueError("empty pairs")
    if int(np.prod(shape)) != d or len(recipient_label) != n or len(donor_label) != n:
        raise ValueError("activation shape or label length mismatch")
    if q.shape[0] != d or mu.shape != (d,):
        raise ValueError("current-layer projector shape mismatch")
    if runner.m.training or runner.head.training:
        raise RuntimeError("native model/head must be in eval mode")
    following = next_stage(runner, stage)
    if (following == "classifier_input") != (final_projector is not None):
        raise RuntimeError("classifier-input transition requires frozen canonical projector; other stages forbid it")
    device = runner.device
    ai32 = torch.from_numpy(np.ascontiguousarray(recipient, np.float32)).to(device)
    aj32 = torch.from_numpy(np.ascontiguousarray(donor, np.float32)).to(device)
    ai, aj = ai32.double(), aj32.double()
    qc = torch.from_numpy(np.ascontiguousarray(q, np.float64)).to(device)
    mc = torch.from_numpy(np.ascontiguousarray(mu, np.float64)).to(device)
    qn = torch.from_numpy(np.ascontiguousarray(q_next, np.float64)).to(device)
    mn = torch.from_numpy(np.ascontiguousarray(mu_next, np.float64)).to(device)
    final_t = None
    if final_projector is not None:
        left, right = final_projector
        final_t = (torch.from_numpy(np.ascontiguousarray(left, np.float64)).to(device),
                   torch.from_numpy(np.ascontiguousarray(right, np.float64)).to(device))
        if final_t[0].shape[0] != qn.shape[0] or final_t[1].shape[1] != qn.shape[0] or final_t[0].shape[1] != final_t[1].shape[0]:
            raise ValueError("frozen canonical final projector shape mismatch")
        operator = final_t[0] @ final_t[1]
        if not torch.allclose(operator @ operator, operator, atol=1e-4, rtol=1e-4):
            raise RuntimeError("frozen canonical final projector is not idempotent")
    labels = torch.as_tensor(recipient_label, dtype=torch.long, device=device)
    donors = torch.as_tensor(donor_label, dtype=torch.long, device=device)
    if not torch.allclose(qc.T @ qc, torch.eye(qc.shape[1], device=device, dtype=qc.dtype), atol=1e-4, rtol=1e-4):
        raise RuntimeError("current P directions are not orthonormal")
    if not torch.allclose(qn.T @ qn, torch.eye(qn.shape[1], device=device, dtype=qn.dtype), atol=1e-4, rtol=1e-4):
        raise RuntimeError("next P directions are not orthonormal")
    with torch.inference_mode():
        pi, ci = _project(ai, qc, mc)
        pj, cj = _project(aj, qc, mc)
        if float((mc + pi + ci - ai).abs().max()) >= 1e-5:
            raise RuntimeError("current-layer P+C reconstruction failed")
        base = ai32.reshape((n, *shape))
        p_swap = (mc + pj + ci).to(ai32.dtype).reshape((n, *shape))
        c_swap = (mc + pi + cj).to(ai32.dtype).reshape((n, *shape))
        base_next = advance_one(runner, stage, base)
        p_next = advance_one(runner, stage, p_swap)
        c_next = advance_one(runner, stage, c_swap)
        if qn.shape[0] != base_next[0].numel() or mn.shape != (base_next[0].numel(),):
            raise ValueError("next-layer projector shape mismatch")
        base_flat = base_next.flatten(1)
        p_flat = p_next.flatten(1)
        c_flat = c_next.flatten(1)
        _, zb = runner.from_stage(following, base_next)
        _, zp = runner.from_stage(following, p_next)
        _, zc = runner.from_stage(following, c_next)
        p_result = _one_direction(runner, stage, following, tuple(base_next.shape[1:]), base_flat, p_flat, zb, zp,
                                  qn, mn, labels, donors, "P", final_t)
        c_result = _one_direction(runner, stage, following, tuple(base_next.shape[1:]), base_flat, c_flat, zb, zc,
                                  qn, mn, labels, donors, "C", final_t)
    return {**p_result, **c_result}


def long_range_mediation(
    runner: Any,
    stage: str,
    recipient: np.ndarray,
    donor: np.ndarray,
    shape: tuple[int, ...],
    q: np.ndarray,
    mu: np.ndarray,
    q_final: np.ndarray,
    mu_final: np.ndarray,
    recipient_label: np.ndarray,
    donor_label: np.ndarray,
    final_projector: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Direct early/middle/late -> classifier-input mediation sanity check.

    Stage choices are fixed in ``LONG_RANGE_STAGES`` before outer evaluation.
    Final P/C follows the frozen canonical projector, not an orthogonalized
    surrogate. This is the user-authorized final-layer protocol amendment.
    """
    if stage not in LONG_RANGE_STAGES[runner.name]:
        raise ValueError(f"stage is not in the prespecified long-range schedule: {stage}")
    if runner.m.training or runner.head.training:
        raise RuntimeError("native model/head must be in eval mode")
    if recipient.shape != donor.shape or recipient.ndim != 2 or len(recipient) == 0:
        raise ValueError("invalid paired activations")
    n, d = recipient.shape
    if int(np.prod(shape)) != d or len(recipient_label) != n or len(donor_label) != n:
        raise ValueError("activation shape or label length mismatch")
    device = runner.device
    ai32 = torch.from_numpy(np.ascontiguousarray(recipient, np.float32)).to(device)
    aj32 = torch.from_numpy(np.ascontiguousarray(donor, np.float32)).to(device)
    ai, aj = ai32.double(), aj32.double()
    qc = torch.from_numpy(np.ascontiguousarray(q, np.float64)).to(device)
    mc = torch.from_numpy(np.ascontiguousarray(mu, np.float64)).to(device)
    qf = torch.from_numpy(np.ascontiguousarray(q_final, np.float64)).to(device)
    mf = torch.from_numpy(np.ascontiguousarray(mu_final, np.float64)).to(device)
    left, right = final_projector
    final_t = (torch.from_numpy(np.ascontiguousarray(left, np.float64)).to(device),
               torch.from_numpy(np.ascontiguousarray(right, np.float64)).to(device))
    labels = torch.as_tensor(recipient_label, dtype=torch.long, device=device)
    donors = torch.as_tensor(donor_label, dtype=torch.long, device=device)
    if qc.shape[0] != d or mc.shape != (d,):
        raise ValueError("current-layer projector shape mismatch")
    if not torch.allclose(qc.T @ qc, torch.eye(qc.shape[1], device=device, dtype=qc.dtype), atol=1e-4, rtol=1e-4):
        raise RuntimeError("current P directions are not orthonormal")
    if not torch.allclose(qf.T @ qf, torch.eye(qf.shape[1], device=device, dtype=qf.dtype), atol=1e-4, rtol=1e-4):
        raise RuntimeError("final P directions are not orthonormal")
    if final_t[0].shape[0] != qf.shape[0] or final_t[1].shape[1] != qf.shape[0] or final_t[0].shape[1] != final_t[1].shape[0]:
        raise ValueError("frozen canonical final projector shape mismatch")
    operator = final_t[0] @ final_t[1]
    if not torch.allclose(operator @ operator, operator, atol=1e-4, rtol=1e-4):
        raise RuntimeError("frozen canonical final projector is not idempotent")
    with torch.inference_mode():
        pi, ci = _project(ai, qc, mc)
        pj, cj = _project(aj, qc, mc)
        if float((mc + pi + ci - ai).abs().max()) >= 1e-5:
            raise RuntimeError("current-layer P+C reconstruction failed")
        base = ai32.reshape((n, *shape))
        p_swap = (mc + pj + ci).to(ai32.dtype).reshape((n, *shape))
        c_swap = (mc + pi + cj).to(ai32.dtype).reshape((n, *shape))
        hb, zb = runner.from_stage(stage, base)
        hp, zp = runner.from_stage(stage, p_swap)
        hc, zc = runner.from_stage(stage, c_swap)
        hb, hp, hc = hb.flatten(1), hp.flatten(1), hc.flatten(1)
        if qf.shape[0] != hb.shape[1] or mf.shape != (hb.shape[1],):
            raise ValueError("final-layer projector shape mismatch")
        p_result = _one_direction(runner, stage, "classifier_input", (hb.shape[1],),
                                  hb, hp, zb, zp, qf, mf, labels, donors, "P", final_t)
        c_result = _one_direction(runner, stage, "classifier_input", (hb.shape[1],),
                                  hb, hc, zb, zc, qf, mf, labels, donors, "C", final_t)
    return {**p_result, **c_result}
