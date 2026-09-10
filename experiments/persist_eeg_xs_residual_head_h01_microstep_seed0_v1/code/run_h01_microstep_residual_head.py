#!/usr/bin/env python3
"""H0.1 micro-step residual-head diagnostic for LiteBN-XS, seed 0.

This is a separate, outcome-blind canonical-inner experiment. It imports the
frozen XS implementation and data helpers from H0, precomputes the XS
embeddings/logits once per task/fold, and trains only a zero-initialized dW/db
head. No outer-development or held-out/test rows are loaded.
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
import torch.nn.functional as F


REPO = Path('/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK')
H0_SCRIPT = REPO / 'experiments/persist_eeg_xs_residual_head_seed0_v1/code/run_xs_residual_head_seed0.py'
EXP = REPO / 'experiments/persist_eeg_xs_residual_head_h01_microstep_seed0_v1'
OUT = EXP / 'outputs_single_inner'
PROTOCOL = EXP / 'protocol'
RUNTIME = Path('/root/rivermind-data/xs_residual_head_h01_microstep_seed0_runtime')

SEED = 0
EVAL_STEPS = (0, 1, 2, 4, 8, 16, 32, 64)
CANDIDATE_LRS = (1e-5, 3e-6)
MAX_STEP = max(EVAL_STEPS)
WEIGHT_DECAY = 5e-4
BETA = 1e-3
CLIP = 5.0
BATCH = 512
TOL = 1e-10
LOGIT_TOL = 1e-6
# Operational definition for the requested "clear" ERP signal.  Values below
# this are reported as essentially zero rather than promoted to a finding.
ERP_SIGNAL_MIN_MEAN_PP = 0.1


def load_h0():
    spec = importlib.util.spec_from_file_location('xs_residual_h0', H0_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import H0 helper: {H0_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + '\n', encoding='utf-8')
    os.replace(temp, path)


def _json_default(value: Any) -> Any:
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
    """Deterministic one-batch-per-optimizer-step schedule shared by both LRs."""
    rng = np.random.default_rng(9_173_003 + 1_009 * int(fold) + task_code(task))
    schedule: list[np.ndarray] = []
    for _ in range(MAX_STEP):
        order = rng.permutation(n)
        schedule.append(np.asarray(order[:min(BATCH, n)], dtype=np.int64))
    return schedule


def metric_row(h0, head, arrays: dict[str, Any], xs_metrics: dict[str, float],
               task: str, fold: int, lr: float, step: int, loss: float | None) -> dict[str, Any]:
    val = h0.evaluate_cached(head, arrays, 'val', h0.mod)
    dwn = float(torch.linalg.vector_norm(head.dW.detach()).cpu())
    dbn = float(torch.linalg.vector_norm(head.db.detach()).cpu())
    return {
        'task': task,
        'dataset': h0.mod.TASKS[task]['dataset'],
        'fold': int(fold),
        'lr': float(lr),
        'optimizer_step': int(step),
        'BA': float(val['BA']),
        'macro_F1': float(val['macro_F1']),
        'accuracy': float(val['accuracy']),
        'XS_BA': float(xs_metrics['BA']),
        'XS_macro_F1': float(xs_metrics['macro_F1']),
        'XS_accuracy': float(xs_metrics['accuracy']),
        'delta_BA': float(val['BA'] - xs_metrics['BA']),
        'delta_BA_pp': float(100.0 * (val['BA'] - xs_metrics['BA'])),
        'dW_norm': dwn,
        'db_norm': dbn,
        'residual_head_norm': float(math.hypot(dwn, dbn)),
        'loss': None if loss is None else float(loss),
    }


def replay_cell(h0, task: str, fold: dict[str, Any], split: dict[str, Any],
                device: torch.device) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    fold_id = int(fold['fold_id'])
    bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
    source = h0.xs_path(task, fold_id)
    mean, std, normalizer_meta = h0.mod.load_tensor_pair(h0.normalizer_path(task, fold_id))
    cache = h0.mod.RawGPUCache(bundle, device)
    base = h0.mod.build_model('LiteBN_XS', task).to(device)
    base.load_state_dict(torch.load(source, map_location=device, weights_only=False), strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    arrays = h0.precompute(h0.mod, base, bundle, cache, split, task, device)
    head = h0.ResidualHead(base, int(h0.mod.TASKS[task]['classes']), int(arrays['train_z'].shape[1])).to(device)
    head.eval()
    with torch.no_grad():
        zero_logits = head(arrays['val_z'], arrays['val_l'])
        max_diff = float((zero_logits - arrays['val_l']).abs().max().cpu())
        mismatch = int((zero_logits.argmax(1) != arrays['val_l'].argmax(1)).sum().cpu())
    xs_metrics = h0.subject_mean_metrics(
        arrays['val_y'], arrays['val_l'].detach().cpu().numpy(), arrays['val_s'], h0.mod
    )
    residual_metrics = h0.evaluate_cached(head, arrays, 'val', h0.mod)
    replay = {
        'task': task,
        'fold': fold_id,
        'XS_BA': float(xs_metrics['BA']),
        'XS_macro_F1': float(xs_metrics['macro_F1']),
        'XS_accuracy': float(xs_metrics['accuracy']),
        'residual_step0_BA': float(residual_metrics['BA']),
        'residual_step0_macro_F1': float(residual_metrics['macro_F1']),
        'residual_step0_accuracy': float(residual_metrics['accuracy']),
        'max_abs_logit_difference': max_diff,
        'prediction_mismatch_count': mismatch,
        'pass': bool(max_diff < LOGIT_TOL and mismatch == 0 and all(
            abs(xs_metrics[key] - residual_metrics[key]) < 1e-12
            for key in ('BA', 'macro_F1', 'accuracy')
        )),
        'source_xs_checkpoint': str(source),
        'source_xs_sha256': sha_file(source),
        'normalizer_sha256': normalizer_meta['mean_std_sha256'],
    }
    del cache
    return replay, base, head, arrays


def train_candidate(h0, task: str, fold: dict[str, Any], split: dict[str, Any],
                     base: torch.nn.Module, arrays: dict[str, Any], lr: float,
                     device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_id = int(fold['fold_id'])
    head = h0.ResidualHead(
        base, int(h0.mod.TASKS[task]['classes']), int(arrays['train_z'].shape[1])
    ).to(device)
    for parameter in head.base.parameters():
        parameter.requires_grad_(False)
    head.base.eval()
    optimizer = torch.optim.AdamW([head.dW, head.db], lr=float(lr), weight_decay=WEIGHT_DECAY)
    train_bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
    class_weight, _ = h0.mod.class_weights(train_bundle, split['train_subjects'])
    if class_weight is not None:
        class_weight = class_weight.to(device)
    train_y = torch.as_tensor(arrays['train_y'], dtype=torch.long, device=device)
    schedule = batch_schedule(int(arrays['train_z'].shape[0]), task, fold_id)
    xs_metrics = h0.subject_mean_metrics(
        arrays['val_y'], arrays['val_l'].detach().cpu().numpy(), arrays['val_s'], h0.mod
    )
    rows = [metric_row(h0, head, arrays, xs_metrics, task, fold_id, lr, 0, None)]
    for step in range(1, MAX_STEP + 1):
        head.train()
        head.base.eval()
        index = torch.as_tensor(schedule[step - 1], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = head(
            arrays['train_z'].index_select(0, index),
            arrays['train_l'].index_select(0, index),
        )
        ce = F.cross_entropy(logits, train_y.index_select(0, index), weight=class_weight)
        reg = BETA * (head.dW.square().mean() + head.db.square().mean())
        loss = ce + reg
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss for {task}/fold{fold_id}/lr{lr}/step{step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([head.dW, head.db], CLIP)
        optimizer.step()
        if step in EVAL_STEPS:
            head.eval()
            rows.append(metric_row(h0, head, arrays, xs_metrics, task, fold_id, lr, step, float(loss.detach().cpu())))
    selected = max(
        rows,
        key=lambda row: (float(row['BA']), -float(row['residual_head_norm']), -int(row['optimizer_step']))
    )
    selected = dict(selected)
    selected['status'] = (
        'positive' if selected['delta_BA'] > TOL
        else ('negative' if selected['delta_BA'] < -TOL else 'zero')
    )
    del train_bundle, head
    return rows, selected


def select_global_lr(all_task_rows: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    summaries = []
    for lr, group in all_task_rows.groupby('lr', sort=True):
        task_means = group.groupby('task')['delta_BA'].mean()
        other = task_means.reindex(['OpenBMI_MI', 'OpenBMI_SSVEP', 'WBCIC_MI'])
        summaries.append({
            'lr': float(lr),
            'erp_mean_delta_pp': float(100.0 * task_means['OpenBMI_ERP']),
            'nonnegative_other_tasks': bool((other >= -TOL).all()),
            'mean_selected_head_norm': float(group['residual_head_norm'].mean()),
        })
    summary = pd.DataFrame(summaries)
    summary['_rank'] = summary.apply(
        lambda row: (float(row['erp_mean_delta_pp']), int(row['nonnegative_other_tasks']),
                     -float(row['mean_selected_head_norm'])),
        axis=1,
    )
    selected_lr = float(summary.sort_values('_rank', ascending=False).iloc[0]['lr'])
    return selected_lr, summary.drop(columns=['_rank'])


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
        '# H0.1_MICROSTEP_RESIDUAL_HEAD',
        '',
        'Exact LiteBN-XS checkpoint per task/fold; every XS parameter is frozen.',
        'Only zero-initialized residual dW/db is trained on precomputed frozen XS embeddings/logits.',
        'Canonical inner_train_subjects / inner_val_subjects only; no additional splits.',
        f'evaluation_steps={list(EVAL_STEPS)}; candidate_lrs={list(CANDIDATE_LRS)};',
        f'weight_decay={WEIGHT_DECAY}; beta={BETA}; batch={BATCH}; seed={SEED}.',
        'Global LR is selected from canonical inner validation only: ERP mean delta first,',
        'then non-degradation on MI/SSVEP/WBCIC, then smaller selected residual norm.',
        f'Frozen split sha256={split_hash}',
        f'H0 helper sha256={sha_file(H0_SCRIPT)}',
        'FINAL_HELDOUT_ACCESSED = NO',
    ]
    (PROTOCOL / 'H0.1_MICROSTEP_RESIDUAL_HEAD_PROTOCOL.md').write_text('\n'.join(protocol) + '\n', encoding='utf-8')

    replay_rows, all_trajectory, all_selected = [], [], []
    for task in h0.mod.TASK_ORDER:
        for fold in folds[h0.mod.TASKS[task]['dataset']]:
            fold_id = int(fold['fold_id'])
            bundle = h0.mod.build_bundle(task, fold['inner_train_subjects'] + fold['inner_val_subjects'])
            split = h0.canonical_split(h0.mod, task, fold, bundle)
            replay, base, head0, arrays = replay_cell(h0, task, fold, split, device)
            replay_rows.append(replay)
            if not replay['pass']:
                raise RuntimeError(f'epoch0 exact XS replay failed for {task}/fold{fold_id}')
            for lr in CANDIDATE_LRS:
                set_seed(SEED + 10003 + fold_id)
                rows, selected = train_candidate(h0, task, fold, split, base, arrays, lr, device)
                all_trajectory.extend(rows)
                all_selected.append(selected)
                print(
                    f'[H0.1 {task} f{fold_id} lr={lr:g}] selected_step={selected["optimizer_step"]} '
                    f'BA={selected["BA"]:.6f} delta={selected["delta_BA_pp"]:+.4f}pp',
                    flush=True,
                )
            del head0, base, arrays, bundle
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    replay_frame = pd.DataFrame(replay_rows).sort_values(['task', 'fold']).reset_index(drop=True)
    if len(replay_frame) != 20 or not bool(replay_frame['pass'].all()):
        raise RuntimeError('H0.1 epoch0 replay did not pass 20/20')
    trajectory_frame = pd.DataFrame(all_trajectory).sort_values(
        ['task', 'fold', 'lr', 'optimizer_step']
    ).reset_index(drop=True)
    selected_frame = pd.DataFrame(all_selected).sort_values(['task', 'fold', 'lr']).reset_index(drop=True)
    selected_lr, lr_summary = select_global_lr(selected_frame)
    final_frame = selected_frame[selected_frame['lr'] == selected_lr].copy().sort_values(['task', 'fold']).reset_index(drop=True)

    def task_summary(frame: pd.DataFrame, by_lr: bool = False) -> pd.DataFrame:
        rows = []
        keys = ['lr', 'task'] if by_lr else ['task']
        for key, group in frame.groupby(keys, sort=True):
            if by_lr:
                lr_value, task = float(key[0]), str(key[1])
            else:
                # pandas may return a one-tuple for groupby(['task']); keep
                # the task identifier as the canonical string in CSV/JSON.
                task = str(key[0]) if isinstance(key, tuple) else str(key)
                lr_value = float(group['lr'].iloc[0])
            delta = group['delta_BA'].to_numpy(dtype=float)
            rows.append({
                'task': task,
                'dataset': str(group['dataset'].iloc[0]),
                'lr': lr_value,
                'folds': int(len(group)),
                'task_mean_delta_pp': float(100.0 * delta.mean()),
                'positive_folds': int((delta > TOL).sum()),
                'zero_folds': int((np.abs(delta) <= TOL).sum()),
                'negative_folds': int((delta < -TOL).sum()),
                'mean_selected_step': float(group['optimizer_step'].mean()),
                'mean_selected_head_norm': float(group['residual_head_norm'].mean()),
            })
        return pd.DataFrame(rows)

    all_summary = task_summary(selected_frame, by_lr=True)
    final_summary = task_summary(final_frame)
    write_csv(OUT / 'H01_EPOCH0_EXACT_XS_REPLAY.csv', replay_frame)
    write_csv(OUT / 'H01_MICROSTEP_TRAJECTORY.csv', trajectory_frame)
    write_csv(OUT / 'H01_MICROSTEP_FOLD_RESULTS_ALL_LRS.csv', selected_frame)
    write_csv(OUT / 'H01_MICROSTEP_TASK_SUMMARY_ALL_LRS.csv', all_summary)
    write_csv(OUT / 'H01_MICROSTEP_LR_SELECTION.csv', lr_summary)
    write_csv(OUT / 'H01_MICROSTEP_SELECTED_FOLD_RESULTS.csv', final_frame)
    write_csv(OUT / 'H01_MICROSTEP_SELECTED_TASK_SUMMARY.csv', final_summary)

    erp = final_summary.loc[final_summary['task'] == 'OpenBMI_ERP', 'task_mean_delta_pp']
    erp_delta = float(erp.iloc[0]) if len(erp) else float('nan')
    erp_positive = int(final_summary.loc[final_summary['task'] == 'OpenBMI_ERP', 'positive_folds'].iloc[0]) if len(erp) else 0
    terminal = (
        'RESIDUAL_HEAD_ERP_SIGNAL_FOUND'
        if (erp_positive > 0 and erp_delta >= ERP_SIGNAL_MIN_MEAN_PP)
        else 'RESIDUAL_HEAD_ERP_SIGNAL_NOT_FOUND'
    )
    metadata = {
        'experiment': 'H0.1_MICROSTEP_RESIDUAL_HEAD',
        'seed': SEED,
        'candidate_lrs': list(CANDIDATE_LRS),
        'evaluation_steps': list(EVAL_STEPS),
        'weight_decay': WEIGHT_DECAY,
        'beta': BETA,
        'batch': BATCH,
        'folds': 20,
        'epoch0_replay_pass': bool(replay_frame['pass'].all()),
        'selected_global_lr': selected_lr,
        'erp_signal_min_mean_delta_pp': ERP_SIGNAL_MIN_MEAN_PP,
        'erp_positive_folds_selected_lr': erp_positive,
        'lr_selection': lr_summary.to_dict(orient='records'),
        'terminal': terminal,
        'outer_development_opened': False,
        'final_heldout_accessed': False,
        'FINAL_HELDOUT_ACCESSED': 'NO',
    }
    write_json(OUT / 'H01_MICROSTEP_METADATA.json', metadata)

    report = [
        '# H0.1 micro-step residual head — canonical-inner result',
        '',
        '- Exact original LiteBN-XS checkpoint per task/fold; all XS parameters frozen.',
        '- Only zero-initialized residual dW/db; XS embeddings/logits precomputed once per fold.',
        '- Four tasks × five folds × one canonical inner split; no outer/test access.',
        f'- Candidate learning rates: {list(CANDIDATE_LRS)}; evaluation steps: {list(EVAL_STEPS)}.',
        f'- Selected global LR: **{selected_lr:g}** (ERP-first inner-only rule).',
        '',
        '## Selected global-LR summary',
        '',
        '| Task | Mean delta vs XS (pp) | Positive | Zero | Negative |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in final_summary.itertuples(index=False):
        report.append(
            f'| {row.task} | {row.task_mean_delta_pp:+.4f} | '
            f'{row.positive_folds}/5 | {row.zero_folds}/5 | {row.negative_folds}/5 |'
        )
    report += [
        '',
        f'OpenBMI ERP mean delta vs XS: **{erp_delta:+.4f} pp**',
        f'Operational clear-signal threshold: **{ERP_SIGNAL_MIN_MEAN_PP:.2f} pp**; smaller means are treated as essentially zero.',
        '',
        '## ERP trajectories',
        '',
        'See H01_MICROSTEP_TRAJECTORY.csv for all five folds, both learning rates, and steps 0/1/2/4/8/16/32/64.',
        '',
        terminal,
        'FINAL_HELDOUT_ACCESSED = NO',
    ]
    (OUT / 'H01_MICROSTEP_RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'H01_MICROSTEP_RESIDUAL_HEAD_COMPLETE ERP={erp_delta:+.4f}pp LR={selected_lr:g}', flush=True)
    print(terminal, flush=True)
    print('FINAL_HELDOUT_ACCESSED = NO', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
