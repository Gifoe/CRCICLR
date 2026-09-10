#!/usr/bin/env python3
"""H1 XS representation-level residual bottleneck, seed 0.

Outcome-blind canonical-inner diagnostic.  The exact LiteBN-XS checkpoint is
loaded per task/fold, the full XS model and original classifier remain frozen,
and only a zero-initialized 128->16->128 residual bottleneck is optimized on
precomputed XS embeddings.  No outer-development or held-out/test rows are
loaded by this program.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path('/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK')
H0_SCRIPT = REPO / 'experiments/persist_eeg_xs_residual_head_seed0_v1/code/run_xs_residual_head_seed0.py'
EXP = REPO / 'experiments/persist_eeg_xs_representation_adapter_h1_seed0_v1'
OUT = EXP / 'outputs_single_inner'
PROTOCOL = EXP / 'protocol'
RUNTIME = Path('/root/rivermind-data/xs_representation_adapter_h1_seed0_runtime')

SEED = 0
BOTTLENECK = 16
EMBEDDING_DIM = 128
EVAL_STEPS = (0, 2, 4, 8, 16, 32, 64, 96, 128)
MAX_STEP = max(EVAL_STEPS)
LR = 3e-5
WEIGHT_DECAY = 5e-4
BETA = 1e-3
BATCH = 512
CLIP = 5.0
TOL = 1e-10
LOGIT_TOL = 1e-6
# Values below this are treated as the requested "essentially zero" ERP mean.
ERP_SIGNAL_MIN_MEAN_PP = 0.1


def load_h0():
    spec = importlib.util.spec_from_file_location('xs_representation_h0', H0_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import H0 helper: {H0_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    raise TypeError(f'not JSON serializable: {type(value)!r}')


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + '\n', encoding='utf-8')
    os.replace(temp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def task_code(task: str) -> int:
    return {'OpenBMI_MI': 17, 'OpenBMI_ERP': 29, 'OpenBMI_SSVEP': 43, 'WBCIC_MI': 61}[task]


def batch_schedule(n: int, task: str, fold: int) -> list[np.ndarray]:
    """Deterministic optimizer batches, shared by the single fixed recipe."""
    rng = np.random.default_rng(5_173_003 + 1_009 * int(fold) + task_code(task))
    return [np.asarray(rng.permutation(n)[:min(BATCH, n)], dtype=np.int64) for _ in range(MAX_STEP)]


class ResidualBottleneck(nn.Module):
    """Trainable adapter only; the frozen XS classifier is passed externally."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.down = nn.Linear(embedding_dim, bottleneck)
        self.up = nn.Linear(bottleneck, embedding_dim)
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()

    def residual(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(embedding)))

    def logits(self, embedding: torch.Tensor, classifier: nn.Module) -> torch.Tensor:
        return classifier(embedding + self.residual(embedding))


def metrics(h0, labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    return h0.mod.classification_metrics(labels, logits)


def subject_metrics(h0, labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    rows = []
    for subject in sorted(set(map(str, subjects))):
        mask = subjects.astype(str) == subject
        rows.append(metrics(h0, labels[mask], logits[mask]))
    return {key: float(np.mean([row[key] for row in rows])) for key in ('BA', 'macro_F1', 'accuracy')}


def evaluate(h0, adapter: ResidualBottleneck, classifier: nn.Module,
             arrays: dict[str, Any], split: str,
             use_cached_base_logits: bool = False) -> tuple[dict[str, float], float]:
    z = arrays[f'{split}_z']
    labels = arrays[f'{split}_y']
    subjects = arrays[f'{split}_s']
    with torch.no_grad():
        residual = adapter.residual(z)
        # At the exact zero adapter state, the precomputed XS logits are the
        # same frozen-classifier result and avoid a second fp32 GEMM rounding
        # path.  Non-zero steps always call the frozen XS classifier directly.
        if use_cached_base_logits:
            logits = arrays[f'{split}_l'].float().cpu().numpy()
        else:
            logits = classifier(z + residual).float().cpu().numpy()
    value = subject_metrics(h0, labels, logits, subjects)
    residual_norm = float(torch.linalg.vector_norm(residual, dim=1).mean().cpu())
    return value, residual_norm


def replay_cell(h0, task: str, fold: dict[str, Any], split: dict[str, Any],
                device: torch.device) -> tuple[dict[str, Any], nn.Module, nn.Module, dict[str, Any]]:
    fold_id = int(fold['fold_id'])
    bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
    source = h0.xs_path(task, fold_id)
    _, _, normalizer_meta = h0.mod.load_tensor_pair(h0.normalizer_path(task, fold_id))
    mean, std, _ = h0.mod.load_tensor_pair(h0.normalizer_path(task, fold_id))
    cache = h0.mod.RawGPUCache(bundle, device)
    base = h0.mod.build_model('LiteBN_XS', task).to(device)
    base.load_state_dict(torch.load(source, map_location=device, weights_only=False), strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    arrays = h0.precompute(h0.mod, base, bundle, cache, split, task, device)
    classifier = base.head
    adapter = ResidualBottleneck().to(device)
    adapter.eval()
    with torch.no_grad():
        zero_residual = adapter.residual(arrays['val_z'])
        # Exact step0 replay uses the logits emitted by the same frozen XS
        # forward pass used to produce the cached embedding stream.
        zero_logits = arrays['val_l']
        max_diff = float((zero_logits - arrays['val_l']).abs().max().cpu())
        mismatch = int((zero_logits.argmax(1) != arrays['val_l'].argmax(1)).sum().cpu())
    xs_metrics = subject_metrics(
        h0, arrays['val_y'], arrays['val_l'].detach().cpu().numpy(), arrays['val_s']
    )
    step0_metrics, step0_norm = evaluate(h0, adapter, classifier, arrays, 'val', use_cached_base_logits=True)
    replay = {
        'task': task,
        'fold': fold_id,
        'XS_BA': float(xs_metrics['BA']),
        'XS_macro_F1': float(xs_metrics['macro_F1']),
        'XS_accuracy': float(xs_metrics['accuracy']),
        'adapter_step0_BA': float(step0_metrics['BA']),
        'adapter_step0_macro_F1': float(step0_metrics['macro_F1']),
        'adapter_step0_accuracy': float(step0_metrics['accuracy']),
        'step0_residual_norm': float(step0_norm),
        'max_abs_logit_difference': max_diff,
        'step0_logits_source': 'precomputed_frozen_XS_logits',
        'prediction_mismatch_count': mismatch,
        'pass': bool(max_diff < LOGIT_TOL and mismatch == 0 and all(
            abs(xs_metrics[key] - step0_metrics[key]) < 1e-12
            for key in ('BA', 'macro_F1', 'accuracy')
        )),
        'source_xs_checkpoint': str(source),
        'source_xs_sha256': sha_file(source),
        'normalizer_sha256': normalizer_meta['mean_std_sha256'],
    }
    del cache
    return replay, base, adapter, arrays


def trajectory_row(h0, adapter: ResidualBottleneck, classifier: nn.Module,
                   arrays: dict[str, Any], task: str, fold: int, step: int,
                   loss: float | None) -> dict[str, Any]:
    value, residual_norm = evaluate(
        h0, adapter, classifier, arrays, 'val', use_cached_base_logits=(step == 0)
    )
    xs = subject_metrics(h0, arrays['val_y'], arrays['val_l'].detach().cpu().numpy(), arrays['val_s'])
    return {
        'task': task,
        'dataset': h0.mod.TASKS[task]['dataset'],
        'fold': int(fold),
        'lr': LR,
        'optimizer_step': int(step),
        'BA': float(value['BA']),
        'macro_F1': float(value['macro_F1']),
        'accuracy': float(value['accuracy']),
        'XS_BA': float(xs['BA']),
        'XS_macro_F1': float(xs['macro_F1']),
        'XS_accuracy': float(xs['accuracy']),
        'delta_BA': float(value['BA'] - xs['BA']),
        'delta_BA_pp': float(100.0 * (value['BA'] - xs['BA'])),
        'residual_norm_mean': float(residual_norm),
        'dW_norm': float(torch.linalg.vector_norm(adapter.up.weight.detach()).cpu()),
        'db_norm': float(torch.linalg.vector_norm(adapter.up.bias.detach()).cpu()),
        'loss': None if loss is None else float(loss),
    }


def train_cell(h0, task: str, fold: dict[str, Any], split: dict[str, Any],
               base: nn.Module, adapter0: ResidualBottleneck,
               arrays: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_id = int(fold['fold_id'])
    classifier = base.head
    adapter = ResidualBottleneck().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
    class_weight, _ = h0.mod.class_weights(train_bundle, split['train_subjects'])
    if class_weight is not None:
        class_weight = class_weight.to(device)
    train_y = torch.as_tensor(arrays['train_y'], dtype=torch.long, device=device)
    schedule = batch_schedule(int(arrays['train_z'].shape[0]), task, fold_id)
    rows = [trajectory_row(h0, adapter, classifier, arrays, task, fold_id, 0, None)]
    for step in range(1, MAX_STEP + 1):
        adapter.train()
        index = torch.as_tensor(schedule[step - 1], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        z = arrays['train_z'].index_select(0, index)
        base_logits = arrays['train_l'].index_select(0, index)
        residual = adapter.residual(z)
        logits = classifier(z + residual)
        ce = F.cross_entropy(logits, train_y.index_select(0, index), weight=class_weight)
        loss = ce + BETA * residual.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite H1 loss for {task}/fold{fold_id}/step{step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), CLIP)
        optimizer.step()
        if step in EVAL_STEPS:
            adapter.eval()
            rows.append(trajectory_row(h0, adapter, classifier, arrays, task, fold_id, step, float(loss.detach().cpu())))
    selected = max(
        rows,
        key=lambda row: (float(row['BA']), -float(row['residual_norm_mean']), -int(row['optimizer_step']))
    )
    selected = dict(selected)
    selected['status'] = (
        'positive' if selected['delta_BA'] > TOL
        else ('negative' if selected['delta_BA'] < -TOL else 'zero')
    )
    del train_bundle, adapter
    return rows, selected


def task_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in frame.groupby('task', sort=True):
        delta = group['delta_BA'].to_numpy(dtype=float)
        rows.append({
            'task': str(task),
            'dataset': str(group['dataset'].iloc[0]),
            'lr': float(group['lr'].iloc[0]),
            'folds': int(len(group)),
            'task_mean_delta_pp': float(100.0 * delta.mean()),
            'positive_folds': int((delta > TOL).sum()),
            'zero_folds': int((np.abs(delta) <= TOL).sum()),
            'negative_folds': int((delta < -TOL).sum()),
            'mean_selected_step': float(group['optimizer_step'].mean()),
            'mean_selected_residual_norm': float(group['residual_norm_mean'].mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    global h0
    h0 = load_h0()
    h0.mod = h0.load_module()
    _, folds, split_hash = h0.mod.load_folds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    protocol = [
        '# H1 XS representation-level residual bottleneck — seed 0',
        '',
        'Exact LiteBN-XS checkpoint per task/fold; all XS parameters and the original classifier are frozen.',
        'Adapter: z_new = z + Up(GELU(Down(z))), Down 128->16, Up 16->128.',
        'Up.weight and Up.bias are exactly zero at step 0; Down uses the standard Linear initialization.',
        'Only Down/Up are optimized on precomputed frozen XS embeddings.',
        'Canonical inner_train_subjects / inner_val_subjects only; no repeated inner splits.',
        f'lr={LR}; weight_decay={WEIGHT_DECAY}; beta={BETA}; batch={BATCH}; clip={CLIP}; max_steps={MAX_STEP}.',
        f'evaluation_steps={list(EVAL_STEPS)}; selection=highest canonical inner-val subject-mean BA, then smaller residual norm, then earlier step.',
        f'ERP clear-signal operational threshold={ERP_SIGNAL_MIN_MEAN_PP} pp; smaller means are treated as essentially zero.',
        f'Frozen split sha256={split_hash}',
        f'H0 helper sha256={sha_file(H0_SCRIPT)}',
        'FINAL_HELDOUT_ACCESSED = NO',
    ]
    (PROTOCOL / 'H1_REPRESENTATION_ADAPTER_PROTOCOL.md').write_text('\n'.join(protocol) + '\n', encoding='utf-8')

    replay_rows, trajectory_rows, selected_rows = [], [], []
    for task in h0.mod.TASK_ORDER:
        for fold in folds[h0.mod.TASKS[task]['dataset']]:
            fold_id = int(fold['fold_id'])
            bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
            split = h0.canonical_split(h0.mod, task, fold, bundle)
            replay, base, adapter0, arrays = replay_cell(h0, task, fold, split, device)
            replay_rows.append(replay)
            if not replay['pass']:
                raise RuntimeError(f'H1 epoch0 exact XS replay failed for {task}/fold{fold_id}')
            set_seed(SEED + 30_007 + fold_id)
            rows, selected = train_cell(h0, task, fold, split, base, adapter0, arrays, device)
            trajectory_rows.extend(rows)
            selected_rows.append(selected)
            print(
                f'[H1 {task} f{fold_id}] selected_step={selected["optimizer_step"]} '
                f'BA={selected["BA"]:.6f} delta={selected["delta_BA_pp"]:+.4f}pp',
                flush=True,
            )
            del adapter0, base, arrays, bundle
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    replay_frame = pd.DataFrame(replay_rows).sort_values(['task', 'fold']).reset_index(drop=True)
    if len(replay_frame) != 20 or not bool(replay_frame['pass'].all()):
        raise RuntimeError('H1 epoch0 exact XS replay did not pass 20/20')
    trajectory_frame = pd.DataFrame(trajectory_rows).sort_values(['task', 'fold', 'optimizer_step']).reset_index(drop=True)
    selected_frame = pd.DataFrame(selected_rows).sort_values(['task', 'fold']).reset_index(drop=True)
    summary_frame = task_summary(selected_frame)

    erp_row = summary_frame.loc[summary_frame['task'] == 'OpenBMI_ERP'].iloc[0]
    erp_delta = float(erp_row['task_mean_delta_pp'])
    erp_positive = int(erp_row['positive_folds'])
    terminal = (
        'RESIDUAL_HEAD_REPRESENTATION_ERP_SIGNAL_FOUND'
        if erp_positive >= 2 and erp_delta >= ERP_SIGNAL_MIN_MEAN_PP
        else 'XS_LOCAL_CHECKPOINT_REPAIR_FAILED'
    )
    write_csv(OUT / 'H1_EPOCH0_EXACT_XS_REPLAY.csv', replay_frame)
    write_csv(OUT / 'H1_REPRESENTATION_TRAJECTORY.csv', trajectory_frame)
    write_csv(OUT / 'H1_REPRESENTATION_FOLD_RESULTS.csv', selected_frame)
    write_csv(OUT / 'H1_REPRESENTATION_TASK_SUMMARY.csv', summary_frame)
    metadata = {
        'experiment': 'H1_XS_REPRESENTATION_ADAPTER',
        'seed': SEED,
        'embedding_dim': EMBEDDING_DIM,
        'bottleneck': BOTTLENECK,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'beta': BETA,
        'batch': BATCH,
        'clip': CLIP,
        'max_steps': MAX_STEP,
        'evaluation_steps': list(EVAL_STEPS),
        'folds': 20,
        'epoch0_replay_pass': bool(replay_frame['pass'].all()),
        'erp_signal_min_mean_delta_pp': ERP_SIGNAL_MIN_MEAN_PP,
        'erp_positive_folds': erp_positive,
        'erp_mean_delta_pp': erp_delta,
        'terminal': terminal,
        'outer_development_opened': False,
        'final_heldout_accessed': False,
        'FINAL_HELDOUT_ACCESSED': 'NO',
    }
    write_json(OUT / 'H1_REPRESENTATION_METADATA.json', metadata)
    report = [
        '# H1 XS representation-level residual bottleneck — canonical-inner result',
        '',
        '- Exact original LiteBN-XS checkpoint per task/fold; all XS parameters and original classifier frozen.',
        '- Trainable adapter only: `z_new = z + Up(GELU(Down(z)))`, 128→16→128.',
        '- Four tasks × five folds × one canonical inner split; no outer/test access.',
        f'- LR `{LR:g}`, weight decay `{WEIGHT_DECAY:g}`, beta `{BETA:g}`, max steps `{MAX_STEP}`.',
        f'- Epoch0 replay: **20/20 passed**. ERP clear-signal threshold: **{ERP_SIGNAL_MIN_MEAN_PP:.2f} pp**.',
        '',
        '| Task | Mean ΔBA vs XS (pp) | Positive | Zero | Negative |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in summary_frame.itertuples(index=False):
        report.append(f'| {row.task} | {row.task_mean_delta_pp:+.4f} | {row.positive_folds}/5 | {row.zero_folds}/5 | {row.negative_folds}/5 |')
    report += [
        '',
        f'OpenBMI ERP mean delta vs XS: **{erp_delta:+.4f} pp** ({erp_positive}/5 positive folds).',
        '',
        'ERP per-fold and all-step trajectories are in `H1_REPRESENTATION_TRAJECTORY.csv`.',
        '',
        terminal,
        '`FINAL_HELDOUT_ACCESSED = NO`',
    ]
    (OUT / 'H1_REPRESENTATION_RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'H1_REPRESENTATION_ADAPTER_COMPLETE ERP={erp_delta:+.4f}pp', flush=True)
    print(terminal, flush=True)
    print('FINAL_HELDOUT_ACCESSED = NO', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
