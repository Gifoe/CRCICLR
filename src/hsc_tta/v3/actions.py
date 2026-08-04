from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.special import softmax
from torch import nn

from hsc_tta.actions import T3A
from hsc_tta.actions_v2.robust_residual import ResidualBlock

from .augmentations import NuisanceConfig, TokenNuisanceAugmenter


def _tensor_state_hash(state: dict[str, torch.Tensor], config: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class T3AConfig:
    filter_k: int = 20
    confidence_threshold: float | None = None
    prototype_interpolation: float = 1.0


class FrozenT3AAction:
    name = "official_t3a"

    def __init__(self, source_head: nn.Module, config: T3AConfig = T3AConfig(), device: str = "cuda"):
        self.source_head = source_head.to(device).eval(); self.config = config; self.device = torch.device(device)
        for parameter in self.source_head.parameters(): parameter.requires_grad_(False)
        self._initial = self.source_head.classifier.weight.detach().cpu().numpy().copy()
        self.reset_subject()

    def reset_subject(self) -> None:
        self._action = T3A(self._initial, self.config.filter_k, self.config.confidence_threshold)
        self._frozen_prototypes: np.ndarray | None = None; self._status = "reset"; self._diagnostics: dict[str, object] = {}

    def _outputs(self, tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            logits, hidden = self.source_head(tokens.to(self.device).float(), return_hidden=True)
        return logits.float().cpu().numpy(), hidden.float().cpu().numpy()

    def adapt_on_adapt(self, tokens: torch.Tensor) -> None:
        self.reset_subject()
        try:
            logits, hidden = self._outputs(tokens); initial = self._action.prototypes.copy()
            self._action.adapt(hidden, logits); rho = self.config.prototype_interpolation
            prototypes = (1 - rho) * initial + rho * self._action.prototypes
            self._action.prototypes = prototypes / np.maximum(np.linalg.norm(prototypes, axis=0, keepdims=True), 1e-12)
            shift = float(np.linalg.norm(self._action.prototypes - initial) / np.sqrt(initial.size))
            self._status = "adapted"; self._diagnostics = {"normalized_update_magnitude": shift,
                "prototype_shift": shift, "support_count": int(len(self._action.supports))}
        except Exception as error:
            self._status = f"unavailable:{type(error).__name__}:{error}"

    def freeze_state(self) -> str:
        if self._status != "adapted": raise RuntimeError(f"cannot freeze T3A: {self._status}")
        self._frozen_prototypes = self._action.prototypes.copy(); self._status = "frozen"
        return self.state_hash()

    def _predict(self, tokens: torch.Tensor) -> np.ndarray:
        if self._status != "frozen" or self._frozen_prototypes is None:
            raise RuntimeError("T3A Probe/Future prediction requires frozen state")
        _, hidden = self._outputs(tokens); return softmax(hidden @ self._frozen_prototypes, axis=1)

    def predict_probe(self, tokens: torch.Tensor) -> np.ndarray: return self._predict(tokens)
    def predict_future(self, tokens: torch.Tensor) -> np.ndarray: return self._predict(tokens)
    def state_hash(self) -> str:
        if self._frozen_prototypes is None: raise RuntimeError("T3A is not frozen")
        digest = hashlib.sha256(json.dumps(asdict(self.config), sort_keys=True).encode())
        digest.update(self._frozen_prototypes.tobytes()); return digest.hexdigest()
    def diagnostics(self) -> dict[str, object]: return {"status": self._status, **self._diagnostics}
    def failure_status(self) -> str: return self._status


@dataclass(frozen=True)
class AdapterConfig:
    steps: int = 3
    learning_rate: float = 5e-5
    source_preservation_weight: float = 0.5
    reliability_quantile: float = 0.2
    consistency_weight: float = 0.1
    parameter_weight: float = 1e-3
    collapse_weight: float = 0.5
    collapse_rho: float = 0.8
    bottleneck: int = 64
    nuisance: NuisanceConfig = NuisanceConfig()


class CorrectedResidualAdapter:
    name = "robust_residual_adapter"

    def __init__(self, source_head: nn.Module, config: AdapterConfig = AdapterConfig(), device: str = "cuda"):
        self.source_head = source_head.to(device).eval(); self.config = config; self.device = torch.device(device)
        for parameter in self.source_head.parameters(): parameter.requires_grad_(False)
        self.dim = int(source_head.classifier.in_features); self.adapter = ResidualBlock(self.dim, config.bottleneck).to(self.device)
        self._initial = copy.deepcopy(self.adapter.state_dict()); self.augmenter = TokenNuisanceAugmenter(config.nuisance)
        self.reset_subject()

    def reset_subject(self) -> None:
        self.adapter.load_state_dict(self._initial); self._frozen: dict[str, torch.Tensor] | None = None
        self._status = "reset"; self._diagnostics: dict[str, object] = {}

    def _source(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source_head(tokens.to(self.device).float(), return_hidden=True)

    @staticmethod
    def collapse_penalty(probabilities: torch.Tensor, rho: float) -> torch.Tensor:
        maximum = probabilities.mean(0).max()
        return torch.relu(maximum - rho).square()

    @staticmethod
    def symmetric_consistency(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        middle = ((left + right) / 2).clamp_min(1e-8)
        return .5 * ((left * (left.clamp_min(1e-8).log() - middle.log())).sum(1).mean() +
                     (right * (right.clamp_min(1e-8).log() - middle.log())).sum(1).mean())

    def adapt_on_adapt(self, tokens: torch.Tensor) -> None:
        self.reset_subject(); x = tokens.to(self.device).float(); c = self.config
        try:
            with torch.no_grad():
                source_logits, hidden = self._source(x); source_prob = source_logits.softmax(1)
                entropy = -(source_prob * source_prob.clamp_min(1e-8).log()).sum(1)
                confidence = source_prob.max(1).values
                reliable = (entropy <= torch.quantile(entropy, c.reliability_quantile)) & (
                    confidence >= torch.quantile(confidence, 1 - c.reliability_quantile))
            if int(reliable.sum()) < max(2, source_prob.shape[1]):
                self._status = "unavailable:insufficient_reliable_adapt"; return
            augmented_hidden = []
            with torch.no_grad():
                for augmented in self.augmenter.all(x).values():
                    _, nuisance_hidden = self._source(augmented); augmented_hidden.append(nuisance_hidden[reliable])
            optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=c.learning_rate)
            best_state = None; best = float("inf"); before = None; components = {}
            for _ in range(c.steps):
                optimizer.zero_grad(set_to_none=True); adapted = self.adapter(hidden[reliable]); logits = self.source_head.classifier(adapted)
                probabilities = logits.softmax(1); entropy_loss = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1).mean()
                source_kl = torch.nn.functional.kl_div(logits.log_softmax(1), source_prob[reliable], reduction="batchmean")
                consistency = torch.stack([self.symmetric_consistency(probabilities,
                    self.source_head.classifier(self.adapter(value)).softmax(1)) for value in augmented_hidden]).mean()
                change = sum((parameter - initial.to(self.device)).square().sum()
                             for parameter, initial in zip(self.adapter.parameters(), self._initial.values()))
                collapse = self.collapse_penalty(probabilities, c.collapse_rho)
                objective = entropy_loss + c.source_preservation_weight * source_kl + c.consistency_weight * consistency + c.parameter_weight * change + c.collapse_weight * collapse
                if before is None: before = float(objective.detach())
                if not torch.isfinite(objective): raise FloatingPointError("nonfinite adapter objective")
                objective.backward(); torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0); optimizer.step()
                if float(objective.detach()) < best:
                    best = float(objective.detach()); best_state = copy.deepcopy(self.adapter.state_dict())
                components = {"entropy_loss": float(entropy_loss.detach()), "source_kl": float(source_kl.detach()),
                              "consistency_loss": float(consistency.detach()), "collapse_penalty": float(collapse.detach())}
            assert best_state is not None; self.adapter.load_state_dict(best_state)
            update = float(sum((p - p0.to(self.device)).square().sum() for p, p0 in zip(self.adapter.parameters(), self._initial.values())).sqrt())
            normalized = update / np.sqrt(sum(p.numel() for p in self.adapter.parameters()))
            with torch.no_grad(): final = self.source_head.classifier(self.adapter(hidden)).softmax(1); max_mass = float(final.mean(0).max())
            if max_mass > .95 or normalized > .1 or best > float(before) + 1e-6:
                self.adapter.load_state_dict(self._initial); self._status = "unavailable:rollback"
            else: self._status = "adapted"
            self._diagnostics = {"objective_before": before, "objective_after": best, "normalized_update_magnitude": normalized,
                                 "class_max_mass": max_mass, "reliable_count": int(reliable.sum()), **components}
        except Exception as error:
            self.adapter.load_state_dict(self._initial); self._status = f"unavailable:{type(error).__name__}:{error}"

    def freeze_state(self) -> str:
        if self._status != "adapted": raise RuntimeError(f"cannot freeze adapter: {self._status}")
        self._frozen = {name: value.detach().cpu().clone() for name, value in self.adapter.state_dict().items()}
        self._status = "frozen"; return self.state_hash()

    def _predict(self, tokens: torch.Tensor) -> np.ndarray:
        if self._status != "frozen" or self._frozen is None:
            raise RuntimeError("adapter Probe/Future prediction requires frozen state")
        self.adapter.load_state_dict(self._frozen)
        with torch.inference_mode():
            _, hidden = self._source(tokens); return self.source_head.classifier(self.adapter(hidden)).softmax(1).cpu().numpy()

    def predict_probe(self, tokens: torch.Tensor) -> np.ndarray: return self._predict(tokens)
    def predict_future(self, tokens: torch.Tensor) -> np.ndarray: return self._predict(tokens)
    def state_hash(self) -> str:
        if self._frozen is None: raise RuntimeError("adapter is not frozen")
        return _tensor_state_hash(self._frozen, {**asdict(self.config), "nuisance": asdict(self.config.nuisance)})
    def diagnostics(self) -> dict[str, object]: return {"status": self._status, **self._diagnostics}
    def failure_status(self) -> str: return self._status


def source_probabilities(source_head: nn.Module, tokens: torch.Tensor, device: str) -> np.ndarray:
    with torch.inference_mode(): return source_head(tokens.to(device).float()).softmax(1).cpu().numpy()
