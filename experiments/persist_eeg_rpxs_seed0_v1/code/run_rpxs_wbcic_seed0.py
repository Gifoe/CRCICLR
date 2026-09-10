#!/usr/bin/env python3
"""RP-XS seed0 canonical-inner experiment.

LiteBN_RPXS uses the LiteBN-XS architecture initialized from the exact LiteBN-X
checkpoint.  X-isomorphic parameters are copied from X, the XS channel branch is
zero-effect at step 0, and all training/selection uses only frozen canonical
inner splits.  No outer-development or final held-out rows are loaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path('/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK')
H0_SCRIPT = REPO / 'experiments/persist_eeg_xs_residual_head_seed0_v1/code/run_xs_residual_head_seed0.py'
BASE_EXP = REPO / 'experiments/persist_eeg_litebn_x_singlemodel_seed0_v1'
SOURCE_RUNTIME = Path('/root/rivermind-data/litebn_x_singlemodel_seed0_runtime')
EXP = REPO / 'experiments/persist_eeg_rpxs_seed0_v1'
OUT = EXP / 'outputs_wbcic_inner'
PROTOCOL = EXP / 'protocol'
RUNTIME = Path('/root/rivermind-data/rpxs_seed0_runtime')

SEED = 0
STEPS = (0, 32, 64, 128, 256, 512)
MAX_STEPS = max(STEPS)
P_GATE_ON = 0.5
LR_NEW = 1e-4
LR_X = 3e-5
WEIGHT_DECAY = 5e-4
LAMBDA_SP = 1e-3
BETA_GATE = 1e-4
CLIP = 5.0
TOL = 1e-10
LOGIT_TOL = 1e-6
SP_EPS_PER_PARAM = 1e-6


def load_h0():
    spec = importlib.util.spec_from_file_location('rpxs_h0', H0_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import helper {H0_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_text(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=str(REPO), text=True).strip()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'cannot serialize {type(value)!r}')


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    torch.save(value, tmp)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def x_path(task: str, fold: int) -> Path:
    return SOURCE_RUNTIME / 'checkpoints' / task.lower() / f'fold{fold}_litebn_x' / 'selected_best.pt'


def is_channel_parameter(name: str) -> bool:
    return name == 'lambda_channel' or name.startswith('channel_mlp.')


class LiteBN_RPXS(nn.Module):
    """LiteBN-XS architecture with train-time batch-level residual-path dropout."""

    def __init__(self, mod, task: str):
        super().__init__()
        spec = mod.TASKS[task]
        self.core = mod.LiteBNEnhanced(int(spec['channels']), int(spec['classes']), 'LiteBN_XS')
        self.p_gate_on = P_GATE_ON
        self.force_gate_off = False
        self._last_gate_penalty = torch.tensor(0.0)
        self._last_gate_abs = torch.tensor(0.0)
        self._last_gate_rms = torch.tensor(0.0)
        self._patch_gate()

    def _patch_gate(self) -> None:
        def apply_channel_gate(value: torch.Tensor) -> torch.Tensor:
            lambda_channel = self.core.lambda_channel
            # Evaluation step0 exact replay: avoid a redundant multiply-by-one
            # path when the new channel path is still exactly zero-effect.
            if (not self.core.training) and (not self.force_gate_off) and float(lambda_channel.detach().abs().cpu()) == 0.0:
                zero = value.new_tensor(0.0)
                self._last_gate_penalty = zero
                self._last_gate_abs = zero
                self._last_gate_rms = zero
                return value
            rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
            diff_rms = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + 1e-8)
            raw = self.core.channel_mlp(torch.stack((rms, diff_rms), dim=-1)).squeeze(-1)
            residual = torch.tanh(lambda_channel) * torch.tanh(raw)
            self._last_gate_penalty = residual.square().mean()
            self._last_gate_abs = residual.abs().mean().detach()
            self._last_gate_rms = torch.sqrt(residual.square().mean()).detach()
            if self.force_gate_off:
                mask = residual.new_tensor(0.0)
            elif self.core.training:
                mask = torch.bernoulli(residual.new_tensor(self.p_gate_on))
            else:
                mask = residual.new_tensor(1.0)
            return value * (1.0 + mask * residual).unsqueeze(-1)
        self.core.apply_channel_gate = apply_channel_gate

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core(value)

    @property
    def lambda_channel(self) -> torch.Tensor:
        return self.core.lambda_channel

    @property
    def gate_penalty(self) -> torch.Tensor:
        return self._last_gate_penalty

    @property
    def last_gate_abs(self) -> float:
        return float(self._last_gate_abs.detach().cpu())

    @property
    def last_gate_rms(self) -> float:
        return float(self._last_gate_rms.detach().cpu())


def build_rpxs_from_x(mod, task: str, fold_id: int, device: torch.device) -> tuple[LiteBN_RPXS, dict[str, Any]]:
    source = x_path(task, fold_id)
    if not source.is_file():
        raise FileNotFoundError(source)
    state = torch.load(source, map_location=device, weights_only=False)
    model = LiteBN_RPXS(mod, task).to(device)
    missing, unexpected = model.core.load_state_dict(state, strict=False)
    allowed_missing = {'lambda_channel', 'channel_mlp.0.weight', 'channel_mlp.0.bias', 'channel_mlp.2.weight', 'channel_mlp.2.bias'}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f'X->RPXS load mismatch {task}/f{fold_id}: missing={missing} unexpected={unexpected}')
    with torch.no_grad():
        model.core.lambda_channel.zero_()
    provenance = {'X_checkpoint': str(source), 'X_checkpoint_sha256': sha_file(source), 'missing_new_channel_parameters': sorted(missing)}
    return model, provenance


def canonical_split(h0, task: str, fold: dict[str, Any], bundle) -> dict[str, Any]:
    return h0.canonical_split(h0.mod, task, fold, bundle)


def subject_metrics(h0, labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    rows = []
    for subject in sorted(set(map(str, subjects))):
        mask = subjects.astype(str) == subject
        rows.append(h0.mod.classification_metrics(labels[mask], logits[mask]))
    return {key: float(np.mean([row[key] for row in rows])) for key in ('BA', 'macro_F1', 'accuracy')}


def evaluate_model(h0, model: LiteBN_RPXS, bundle, cache, subjects: Iterable[str], mean: np.ndarray, std: np.ndarray,
                   gate_on: bool = True) -> dict[str, float]:
    model.eval()
    model.force_gate_off = not gate_on
    indices = bundle.indices(subjects, (int(h0.mod.TASKS[bundle.task]['future_session']),))
    labels, logits, subj, gate_abs, gate_rms = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            idx = indices[start:start + 128]
            value, y = cache.batch(idx, mean, std)
            out, _ = model(value)
            labels.append(y.cpu().numpy())
            logits.append(out.float().cpu().numpy())
            subj.extend(bundle.rows[int(i)].subject for i in idx)
            gate_abs.append(model.last_gate_abs)
            gate_rms.append(model.last_gate_rms)
    model.force_gate_off = False
    labels_np = np.concatenate(labels)
    logits_np = np.concatenate(logits)
    subjects_np = np.asarray(subj, dtype=object)
    result = subject_metrics(h0, labels_np, logits_np, subjects_np)
    result['mean_abs_channel_residual'] = float(np.mean(gate_abs)) if gate_abs else 0.0
    result['rms_channel_residual'] = float(np.mean(gate_rms)) if gate_rms else 0.0
    return result


def exact_replay_cell(h0, task: str, fold: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    fid = int(fold['fold_id'])
    bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
    split = canonical_split(h0, task, fold, bundle)
    mean, std, normalizer_meta = h0.mod.load_tensor_pair(h0.normalizer_path(task, fid))
    cache = h0.mod.RawGPUCache(bundle, device)
    x_model = h0.mod.build_model('LiteBN_X', task).to(device)
    x_source = x_path(task, fid)
    x_model.load_state_dict(torch.load(x_source, map_location=device, weights_only=False), strict=True)
    x_model.eval()
    rpxs, prov = build_rpxs_from_x(h0.mod, task, fid, device)
    rpxs.eval()
    indices = bundle.indices(fold['inner_val_subjects'], (int(h0.mod.TASKS[task]['future_session']),))
    max_diff, mismatch, labels, x_logits, r_logits, subjects = 0.0, 0, [], [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            idx = indices[start:start + 128]
            value, y = cache.batch(idx, mean, std)
            xl, _ = x_model(value)
            rl, _ = rpxs(value)
            max_diff = max(max_diff, float((xl - rl).abs().max().cpu()))
            mismatch += int((xl.argmax(1) != rl.argmax(1)).sum().cpu())
            labels.append(y.cpu().numpy()); x_logits.append(xl.cpu().numpy()); r_logits.append(rl.cpu().numpy())
            subjects.extend(bundle.rows[int(i)].subject for i in idx)
    labels_np = np.concatenate(labels)
    subjects_np = np.asarray(subjects, dtype=object)
    xm = subject_metrics(h0, labels_np, np.concatenate(x_logits), subjects_np)
    rm = subject_metrics(h0, labels_np, np.concatenate(r_logits), subjects_np)
    passed = bool(max_diff < LOGIT_TOL and mismatch == 0 and all(abs(xm[k] - rm[k]) < 1e-12 for k in ('BA', 'macro_F1', 'accuracy')))
    row = {
        'task': task, 'dataset': h0.mod.TASKS[task]['dataset'], 'fold': fid,
        'X_checkpoint_path': str(x_source), 'X_checkpoint_sha256': prov['X_checkpoint_sha256'],
        'normalizer_sha256': normalizer_meta['mean_std_sha256'],
        'max_abs_logit_difference': max_diff, 'predicted_label_mismatch_count': mismatch,
        'X_BA': xm['BA'], 'RPXS_step0_BA': rm['BA'],
        'X_macro_F1': xm['macro_F1'], 'RPXS_step0_macro_F1': rm['macro_F1'],
        'X_accuracy': xm['accuracy'], 'RPXS_step0_accuracy': rm['accuracy'], 'pass': passed,
    }
    del x_model, rpxs, cache, bundle
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return row, prov


def sp_loss_and_drift(model: LiteBN_RPXS, anchors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, float]:
    terms = []
    with torch.no_grad():
        drift_terms = []
        for name, p in model.core.named_parameters():
            if name not in anchors:
                continue
            anchor = anchors[name]
            denom = anchor.square().sum() + SP_EPS_PER_PARAM * anchor.numel()
            drift_terms.append(float(((p.detach() - anchor).square().sum() / denom).cpu()))
    for name, p in model.core.named_parameters():
        if name not in anchors:
            continue
        anchor = anchors[name]
        denom = anchor.square().sum() + SP_EPS_PER_PARAM * anchor.numel()
        terms.append((p - anchor).square().sum() / denom)
    if not terms:
        return next(model.parameters()).new_tensor(0.0), 0.0
    return torch.stack(terms).mean(), float(np.mean(drift_terms))


def batch_plan(h0, bundle, fold: dict[str, Any], task: str) -> list[np.ndarray]:
    if h0.mod.TASKS[task]['mi_protocol']:
        epochs, _ = h0.mod.mi_manifest(bundle, fold, task)
        batches = [batch for epoch in epochs for batch in epoch]
        if len(batches) < MAX_STEPS:
            reps = int(math.ceil(MAX_STEPS / max(1, len(batches))))
            batches = (batches * reps)[:MAX_STEPS]
        return batches[:MAX_STEPS]
    train_indices = bundle.indices(fold['inner_train_subjects'], h0.mod.TASKS[task]['source_sessions'])
    batches, epoch = [], 1
    while len(batches) < MAX_STEPS:
        batches.extend(h0.mod.task_epoch_batches(train_indices, task, int(fold['fold_id']), epoch))
        epoch += 1
    return batches[:MAX_STEPS]


def selected_checkpoint_path(task: str, fid: int) -> Path:
    return RUNTIME / 'checkpoints' / task.lower() / f'fold{fid}_rpxs' / 'selected_best.pt'


def record_path(task: str, fid: int) -> Path:
    return RUNTIME / 'records' / task.lower() / f'fold{fid}.json'


def train_cell(h0, task: str, fold: dict[str, Any], device: torch.device, replay_row: dict[str, Any]) -> dict[str, Any]:
    fid = int(fold['fold_id'])
    rec_path = record_path(task, fid)
    invariant = {
        'task': task, 'fold': fid, 'seed': SEED, 'max_steps': MAX_STEPS, 'steps': list(STEPS),
        'p_gate_on': P_GATE_ON, 'lr_new': LR_NEW, 'lr_x': LR_X, 'weight_decay': WEIGHT_DECAY,
        'lambda_sp': LAMBDA_SP, 'beta_gate': BETA_GATE, 'clip': CLIP,
        'X_checkpoint_sha256': replay_row['X_checkpoint_sha256'],
    }
    if rec_path.is_file():
        value = json.loads(rec_path.read_text(encoding='utf-8'))
        if value.get('invariant') == invariant:
            return value
        raise RuntimeError(f'invariant mismatch: {rec_path}')
    bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
    split = canonical_split(h0, task, fold, bundle)
    mean, std, normalizer_meta = h0.mod.load_tensor_pair(h0.normalizer_path(task, fid))
    cache = h0.mod.RawGPUCache(bundle, device)
    model, prov = build_rpxs_from_x(h0.mod, task, fid, device)
    model.train()
    anchors = {name: p.detach().clone() for name, p in model.core.named_parameters() if not is_channel_parameter(name)}
    new_params, x_params = [], []
    for name, p in model.core.named_parameters():
        (new_params if is_channel_parameter(name) else x_params).append(p)
    optimizer = torch.optim.AdamW([
        {'params': new_params, 'lr': LR_NEW, 'weight_decay': WEIGHT_DECAY},
        {'params': x_params, 'lr': LR_X, 'weight_decay': WEIGHT_DECAY},
    ])
    class_weight, class_weight_meta = h0.mod.class_weights(bundle, split['train_subjects'])
    if class_weight is not None:
        class_weight = class_weight.to(device)
    batches = batch_plan(h0, bundle, fold, task)
    y_all = torch.as_tensor(bundle.labels(np.arange(len(bundle.rows), dtype=np.int64)), dtype=torch.long, device=device)
    history = []
    best_row = None
    best_state = None

    def make_eval_row(step: int, train_metrics: dict[str, float]) -> dict[str, Any]:
        val_on = evaluate_model(h0, model, bundle, cache, fold['inner_val_subjects'], mean, std, gate_on=True)
        val_off = evaluate_model(h0, model, bundle, cache, fold['inner_val_subjects'], mean, std, gate_on=False)
        _, drift = sp_loss_and_drift(model, anchors)
        row = {
            'task': task, 'dataset': h0.mod.TASKS[task]['dataset'], 'fold': fid, 'optimizer_step': step,
            'validation_BA': val_on['BA'], 'macro_F1': val_on['macro_F1'], 'accuracy': val_on['accuracy'],
            'X_BA': replay_row['X_BA'], 'X_macro_F1': replay_row['X_macro_F1'], 'X_accuracy': replay_row['X_accuracy'],
            'delta_BA': val_on['BA'] - replay_row['X_BA'], 'delta_BA_pp': 100.0 * (val_on['BA'] - replay_row['X_BA']),
            'delta_F1': val_on['macro_F1'] - replay_row['X_macro_F1'],
            'train_loss': train_metrics.get('total_loss'), 'task_loss': train_metrics.get('task_loss'),
            'SP_penalty': train_metrics.get('sp_loss'), 'gate_penalty': train_metrics.get('gate_loss'),
            'lambda_channel': float(model.lambda_channel.detach().cpu()),
            'mean_abs_channel_residual': val_on['mean_abs_channel_residual'],
            'rms_channel_residual': val_on['rms_channel_residual'],
            'X_parameter_normalized_drift': drift,
            'gate_on_BA': val_on['BA'], 'gate_off_BA': val_off['BA'], 'gate_on_minus_off_BA': val_on['BA'] - val_off['BA'],
        }
        return row

    # Step 0 is the X checkpoint replay and a legal checkpoint.
    model.eval()
    row0 = make_eval_row(0, {'total_loss': None, 'task_loss': None, 'sp_loss': 0.0, 'gate_loss': 0.0})
    history.append(row0)
    best_row = row0
    best_state = {k: v.detach().cpu() for k, v in model.core.state_dict().items()}
    set_seed(SEED + 70_001 + fid + 101 * h0.mod.TASK_ORDER.index(task))
    for step in range(1, MAX_STEPS + 1):
        model.train()
        idx_np = batches[step - 1]
        value, _ = cache.batch(idx_np, mean, std)
        labels = y_all.index_select(0, torch.as_tensor(idx_np, dtype=torch.long, device=device))
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(value)
        task_loss = F.cross_entropy(logits, labels, weight=class_weight)
        sp, drift_train = sp_loss_and_drift(model, anchors)
        gate = model.gate_penalty
        loss = task_loss + LAMBDA_SP * sp + BETA_GATE * gate
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite RPXS loss {task}/f{fid}/step{step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        if step in STEPS:
            model.eval()
            row = make_eval_row(step, {
                'total_loss': float(loss.detach().cpu()), 'task_loss': float(task_loss.detach().cpu()),
                'sp_loss': float(sp.detach().cpu()), 'gate_loss': float(gate.detach().cpu()),
            })
            history.append(row)
            # Select only checkpoints that strictly improve BA over step0; otherwise fallback to step0.
            if row['validation_BA'] > replay_row['X_BA'] + TOL:
                if (best_row is row0 or row['validation_BA'] > best_row['validation_BA'] + TOL or
                    (abs(row['validation_BA'] - best_row['validation_BA']) <= TOL and row['macro_F1'] > best_row['macro_F1'] + TOL)):
                    best_row = row
                    best_state = {k: v.detach().cpu() for k, v in model.core.state_dict().items()}
            print(f'[RPXS {task} f{fid}] step={step} valBA={row["validation_BA"]:.6f} delta={row["delta_BA_pp"]:+.4f}pp lambda={row["lambda_channel"]:+.5f}', flush=True)
    if best_row is row0:
        selected_step = 0
    else:
        selected_step = int(best_row['optimizer_step'])
    best_path = selected_checkpoint_path(task, fid)
    torch_save(best_path, {'state_dict': best_state, 'selected_step': selected_step, 'invariant': invariant})
    delta = float(best_row['validation_BA'] - replay_row['X_BA'])
    status = 'positive' if delta > TOL else ('negative' if delta < -TOL else 'zero')
    record = {
        'invariant': invariant, 'task': task, 'dataset': h0.mod.TASKS[task]['dataset'], 'fold': fid,
        'history': history, 'selected': best_row, 'selected_step': selected_step,
        'selected_checkpoint_path': str(best_path), 'status': status,
        'X_checkpoint_path': replay_row['X_checkpoint_path'], 'X_checkpoint_sha256': replay_row['X_checkpoint_sha256'],
        'normalizer_sha256': normalizer_meta['mean_std_sha256'], 'class_weight_meta': class_weight_meta,
    }
    write_json(rec_path, record)
    del model, cache, bundle
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return record


def fold_row(record: dict[str, Any]) -> dict[str, Any]:
    s = record['selected']
    return {
        'task': record['task'], 'dataset': record['dataset'], 'fold': record['fold'],
        'X_BA': s['X_BA'], 'X_macro_F1': s['X_macro_F1'], 'X_accuracy': s['X_accuracy'],
        'selected_step': record['selected_step'], 'RPXS_BA': s['validation_BA'], 'RPXS_macro_F1': s['macro_F1'], 'RPXS_accuracy': s['accuracy'],
        'delta_BA': s['delta_BA'], 'delta_BA_pp': s['delta_BA_pp'], 'delta_F1': s['delta_F1'],
        'status': record['status'], 'lambda_channel': s['lambda_channel'],
        'mean_abs_channel_residual': s['mean_abs_channel_residual'], 'rms_channel_residual': s['rms_channel_residual'],
        'X_parameter_drift': s['X_parameter_normalized_drift'], 'gate_on_BA': s['gate_on_BA'],
        'gate_off_BA': s['gate_off_BA'], 'BA_gate_on_minus_off': s['gate_on_minus_off_BA'],
        'checkpoint_path': record['selected_checkpoint_path'], 'X_checkpoint_path': record['X_checkpoint_path'],
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in frame.groupby('task', sort=True):
        d = group['delta_BA_pp'].to_numpy(float)
        rows.append({
            'task': str(task), 'dataset': str(group['dataset'].iloc[0]), 'folds': int(len(group)),
            'mean_X_BA': float(group['X_BA'].mean()), 'mean_RPXS_BA': float(group['RPXS_BA'].mean()),
            'mean_delta_BA_pp': float(np.mean(d)), 'median_delta_BA_pp': float(np.median(d)),
            'positive_folds': int((group['delta_BA'] > TOL).sum()), 'zero_folds': int((np.abs(group['delta_BA']) <= TOL).sum()),
            'negative_folds': int((group['delta_BA'] < -TOL).sum()), 'largest_negative_delta_pp': float(np.min(d)),
            'mean_macro_F1_delta': float(group['delta_F1'].mean()), 'mean_lambda_channel': float(group['lambda_channel'].mean()),
            'mean_abs_channel_residual': float(group['mean_abs_channel_residual'].mean()),
            'mean_X_parameter_drift': float(group['X_parameter_drift'].mean()),
            'mean_BA_gate_on_minus_off': float(group['BA_gate_on_minus_off'].mean()),
        })
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame) -> str:
    task_delta = {row.task: float(row.mean_delta_BA_pp) for row in summary.itertuples(index=False)}
    task_pos = {row.task: int(row.positive_folds) for row in summary.itertuples(index=False)}
    openbmi_ok = all(task_delta[t] >= -1e-9 for t in ('OpenBMI_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP'))
    wbcic = task_delta['WBCIC_MI']
    if openbmi_ok and wbcic >= 0.4 and task_pos['WBCIC_MI'] >= 4:
        return 'RPXS_INNER_SIGNAL_FOUND_READY_FOR_OUTER_REVIEW'
    if any(task_delta[t] < -0.05 for t in ('OpenBMI_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP')) and wbcic > 0:
        return 'RPXS_HARMFUL_COADAPTATION_PERSISTS'
    if abs(wbcic) < 0.15 and sum(task_pos.values()) <= 2:
        return 'RPXS_SAFE_BUT_NO_NEW_SIGNAL'
    return 'RPXS_SAFE_BUT_NO_NEW_SIGNAL'


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    h0 = load_h0()
    h0.mod = h0.load_module()
    _, folds, split_hash = h0.mod.load_folds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    branch = git_text('branch', '--show-current')
    commit = git_text('rev-parse', 'HEAD')
    protocol = [
        '# RPXS WBCIC-only canonical-inner protocol', '',
        f'source_branch: `{branch}`', f'source_commit: `{commit}`',
        'model: LiteBN_RPXS = LiteBN_XS architecture initialized from LiteBN_X checkpoint.',
        'X-isomorphic parameters copied from X; new channel_mlp and lambda_channel are new, with lambda_channel=0 for exact step0 replay.',
        f'residual-path dropout: batch-level Bernoulli p_gate_on={P_GATE_ON} during training; inference gate ON.',
        f'optimizer: AdamW, new channel parameters lr={LR_NEW}, X-initialized parameters lr={LR_X}, weight_decay={WEIGHT_DECAY}.',
        f'L2-SP lambda={LAMBDA_SP}; gate penalty beta={BETA_GATE}; max_steps={MAX_STEPS}; eval_steps={list(STEPS)}; clip={CLIP}.',
        'selection: canonical inner-val BA; if no checkpoint strictly improves over step0 X, select step0.',
        f'canonical split sha256: `{split_hash}`',
        'scope: WBCIC_MI only; existing WBCIC fold0 record is reused and folds1-4 are resumed.',
        'OpenBMI execution is intentionally not continued in this run.',
        'OUTER_DEVELOPMENT_ACCESSED = NO', 'FINAL_HELDOUT_ACCESSED = NO', '',
    ]
    (PROTOCOL / 'RPXS_PROTOCOL.md').write_text('\n'.join(protocol), encoding='utf-8')

    replay_rows, provenance_rows = [], []
    task_order = ['WBCIC_MI']
    for task in task_order:
        for fold in folds[h0.mod.TASKS[task]['dataset']]:
            row, prov = exact_replay_cell(h0, task, fold, device)
            replay_rows.append(row)
            provenance_rows.append({'task': task, 'fold': int(fold['fold_id']), **prov})
    replay_frame = pd.DataFrame(replay_rows).sort_values(['task', 'fold']).reset_index(drop=True)
    write_csv(OUT / 'RPXS_STEP0_EXACT_X_REPLAY.csv', replay_frame)
    if len(replay_frame) != 5 or not bool(replay_frame['pass'].all()):
        terminal = 'RPXS_IMPLEMENTATION_INVALID_STOP'
        write_json(OUT / 'RPXS_METADATA.json', {'terminal': terminal, 'exact_replay_pass': False, 'OUTER_DEVELOPMENT_ACCESSED': 'NO', 'FINAL_HELDOUT_ACCESSED': 'NO'})
        print(terminal, flush=True)
        return 1
    print('RPXS_STEP0_EXACT_X_REPLAY_PASS 5/5', flush=True)
    replay_lookup = {(row['task'], int(row['fold'])): row for row in replay_rows}
    records = []
    # Smoke cell first, then the complete deterministic list. Resume avoids duplicate WBCIC/fold0 work.
    ordered = [('WBCIC_MI', folds['WBCIC'][0])]
    for task in task_order:
        for fold in folds[h0.mod.TASKS[task]['dataset']]:
            if not (task == 'WBCIC_MI' and int(fold['fold_id']) == 0):
                ordered.append((task, fold))
    for task, fold in ordered:
        fid = int(fold['fold_id'])
        record = train_cell(h0, task, fold, device, replay_lookup[(task, fid)])
        records.append(record)
    # Reload all records in canonical order for aggregation.
    records = []
    for task in task_order:
        for fold in folds[h0.mod.TASKS[task]['dataset']]:
            records.append(json.loads(record_path(task, int(fold['fold_id'])).read_text(encoding='utf-8')))
    fold_frame = pd.DataFrame([fold_row(r) for r in records]).sort_values(['task', 'fold']).reset_index(drop=True)
    trajectory_frame = pd.DataFrame([row for r in records for row in r['history']]).sort_values(['task', 'fold', 'optimizer_step']).reset_index(drop=True)
    diagnostics = fold_frame[['task', 'fold', 'gate_on_BA', 'gate_off_BA', 'BA_gate_on_minus_off', 'lambda_channel', 'mean_abs_channel_residual', 'rms_channel_residual', 'X_parameter_drift']].copy()
    summary = summarize(fold_frame)
    terminal = 'RPXS_WBCIC_ONLY_COMPLETE'
    write_csv(OUT / 'RPXS_CANONICAL_INNER_FOLD_RESULTS.csv', fold_frame)
    write_csv(OUT / 'RPXS_CANONICAL_INNER_TASK_SUMMARY.csv', summary)
    write_csv(OUT / 'RPXS_TRAINING_TRAJECTORY.csv', trajectory_frame)
    write_csv(OUT / 'RPXS_DIAGNOSTICS.csv', diagnostics)
    wbcic = summary[summary.task == 'WBCIC_MI'].iloc[0]
    metadata = {
        'experiment': 'persist_eeg_rpxs_seed0_v1', 'terminal': terminal, 'source_branch': branch, 'source_commit': commit,
        'seed': SEED, 'architecture': 'LiteBN_RPXS', 'optimizer': 'AdamW', 'lr_new_channel': LR_NEW, 'lr_x_initialized': LR_X,
        'weight_decay': WEIGHT_DECAY, 'p_gate_on': P_GATE_ON, 'lambda_SP': LAMBDA_SP, 'beta_gate': BETA_GATE,
        'max_steps': MAX_STEPS, 'eval_steps': list(STEPS), 'selection_rule': 'inner_val_BA_with_step0_fallback',
        'exact_replay_pass': True, 'exact_replay_folds': 5, 'split_sha256': split_hash,
        'wbcic_mean_delta_pp': float(wbcic.mean_delta_BA_pp), 'wbcic_positive_folds': int(wbcic.positive_folds),
        'OUTER_DEVELOPMENT_ACCESSED': 'NO', 'FINAL_HELDOUT_ACCESSED': 'NO',
    }
    write_json(OUT / 'RPXS_METADATA.json', metadata)
    report = ['# RPXS seed0 canonical-inner result', '', f'- Source branch/commit: `{branch}` / `{commit}`',
              '- Architecture: `LiteBN_RPXS` = XS architecture initialized from LiteBN-X checkpoints.',
              f'- Optimizer: AdamW; channel LR `{LR_NEW:g}`, X-initialized LR `{LR_X:g}`, weight decay `{WEIGHT_DECAY:g}`.',
              f'- Residual-path dropout: batch-level `p_gate_on={P_GATE_ON}`; L2-SP `{LAMBDA_SP:g}`; beta_gate `{BETA_GATE:g}`.',
              '- Selection: canonical inner-val BA with step0 X fallback; no outer/test access.',
              '- Step0 exact X replay: **5/5 passed**.', '',
              '| Task | mean X BA | mean RPXS BA | mean ΔBA pp | median ΔBA pp | +/0/- folds | mean ΔF1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples(index=False):
        report.append(f'| {row.task} | {row.mean_X_BA:.4f} | {row.mean_RPXS_BA:.4f} | {row.mean_delta_BA_pp:+.4f} | {row.median_delta_BA_pp:+.4f} | {row.positive_folds}/{row.zero_folds}/{row.negative_folds} | {row.mean_macro_F1_delta:+.4f} |')
    report += ['', f'WBCIC positive folds: **{int(wbcic.positive_folds)}/5**; mean ΔBA: **{float(wbcic.mean_delta_BA_pp):+.4f} pp**; median ΔBA: **{float(wbcic.median_delta_BA_pp):+.4f} pp**.',
               '', '## Diagnostics', '', '- Gate-on/off diagnostic is in `RPXS_DIAGNOSTICS.csv` and was not used for selection.',
               '- Full train trajectory is in `RPXS_TRAINING_TRAJECTORY.csv`.', '', terminal, '', '`OUTER_DEVELOPMENT_ACCESSED = NO`', '`FINAL_HELDOUT_ACCESSED = NO`', '']
    (OUT / 'RPXS_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'RPXS_COMPLETE WBCIC={float(wbcic.mean_delta_BA_pp):+.4f}pp POS={int(wbcic.positive_folds)}/5', flush=True)
    print(terminal, flush=True)
    print('OUTER_DEVELOPMENT_ACCESSED = NO', flush=True)
    print('FINAL_HELDOUT_ACCESSED = NO', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
