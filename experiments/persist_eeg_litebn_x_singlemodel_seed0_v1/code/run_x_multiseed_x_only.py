#!/usr/bin/env python3
"""Run only LiteBN-X for seed1/2; reuse verified historical LiteBN baselines."""
from __future__ import annotations
import copy, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

REPO = Path('/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK').resolve()
EXP = REPO / 'experiments/persist_eeg_litebn_x_singlemodel_seed0_v1'
CODE = EXP / 'code/litebn_x.py'
OUT = EXP / 'outputs'
PROTOCOL = EXP / 'protocol'
RUNTIME_ROOT = Path('/root/rivermind-data/litebn_x_multiseed_x_only_runtime').resolve()
HIST_TASK_RUNTIME = Path('/root/rivermind-data/openbmi_task_generality_runtime').resolve()
HIST_CARRIER_RUNTIME = Path('/root/rivermind-data/carrier_5fold_multiseed_stability_runtime').resolve()
SEEDS = (1, 2)


def load_module():
    spec = importlib.util.spec_from_file_location('litebn_x_impl_multiseed', CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import LiteBN-X implementation')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compact(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != 'history'}


def baseline_path(task: str, fold: int, seed: int) -> Path:
    if task in ('OpenBMI_ERP', 'OpenBMI_SSVEP'):
        short = 'erp' if task.endswith('ERP') else 'ssvep'
        return HIST_TASK_RUNTIME / f'{short}_fold{fold}_seed{seed}_litebn/selected_best.pt'
    short = 'openbmi' if task.startswith('OpenBMI') else 'wbcic'
    return HIST_CARRIER_RUNTIME / f'{short}_fold{fold}_seed{seed}_litebn/selected_best.pt'


def verify_baselines(mod) -> dict[str, Any]:
    rows = []
    search, folds, split_hash = mod.load_folds()
    for task in mod.TASK_ORDER:
        dataset = mod.TASKS[task]['dataset']
        for fold in folds[dataset]:
            for seed in (0, 1, 2):
                path = baseline_path(task, int(fold['fold_id']), seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                model = mod.build_model('LiteBN_BASELINE', task)
                state = torch.load(path, map_location='cpu', weights_only=False)
                model.load_state_dict(state, strict=True)
                rows.append({'task': task, 'fold': int(fold['fold_id']), 'seed': seed, 'checkpoint_path': str(path), 'checkpoint_sha256': mod.sha256_file(path), 'strict_load': True})
    return {'split_sha256': split_hash, 'records': rows, 'count': len(rows), 'source': 'historical_verified_three_seed_baselines'}


def train_seed(mod, seed: int) -> list[dict[str, Any]]:
    mod.SEED = int(seed)
    runtime = RUNTIME_ROOT / f'seed{seed}'
    mod.RUNTIME = runtime
    runtime.mkdir(parents=True, exist_ok=True)
    existing = OUT / f'SEED{seed}_X_INNERVAL.csv'
    if existing.is_file():
        frame = pd.read_csv(existing)
        if len(frame) == len(mod.TASK_ORDER) * 5 and set(frame.architecture) == {'LiteBN_X'} and all(Path(p).is_file() for p in frame.checkpoint_path):
            print(f'RESUME_SEED{seed}_TRAINING_LOG', flush=True)
            return frame.to_dict('records')
    search, folds, split_hash = mod.load_folds()
    rows: list[dict[str, Any]] = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for task in mod.TASK_ORDER:
        dataset = mod.TASKS[task]['dataset']
        for fold in folds[dataset]:
            allowed = fold['inner_train_subjects'] + fold['inner_val_subjects']
            bundle = mod.build_bundle(task, allowed)
            mean, std, norm_meta = mod.normalizer(bundle, fold['inner_train_subjects'])
            mod.save_tensor_pair(runtime / 'normalizers' / f'{task.lower()}_fold{fold["fold_id"]}.npz', mean, std, norm_meta)
            cache = mod.RawGPUCache(bundle, device)
            if mod.TASKS[task]['mi_protocol']:
                episodes, manifest_meta = mod.mi_manifest(bundle, fold, task)
                batch_info = {**manifest_meta, 'episodes': episodes}
            else:
                batch_info = {'kind': 'historical_task_full_permutation_batch64', 'manifest_sha256': None, 'steps_per_epoch': None}
            weight, weight_meta = mod.class_weights(bundle, fold['inner_train_subjects'])
            mod.set_seed(seed)
            model = mod.build_model('LiteBN_X', task).to(device)
            mod.set_seed(seed + 100_000)
            record = mod.train_one(model, 'LiteBN_X', task, fold, bundle, cache, mean, std, norm_meta, batch_info, weight, weight_meta, device)
            record['seed'] = int(seed)
            rows.append(compact(record))
            print(f'X_SEED{seed}_COMPLETE {task} fold={fold["fold_id"]} epoch={record["selected_epoch"]}', flush=True)
            del model, cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(['task', 'fold'])
    if len(frame) != len(mod.TASK_ORDER) * 5:
        raise RuntimeError(f'incomplete seed{seed} X grid: {len(frame)}')
    frame.to_csv(OUT / f'SEED{seed}_X_INNERVAL.csv', index=False)
    with (runtime / 'TRAINING_LOGS.json').open('w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, sort_keys=True, default=str)
    return frame.to_dict('records')


def evaluate_seed(mod, seed: int, records: list[dict[str, Any]]) -> pd.DataFrame:
    mod.SEED = int(seed)
    runtime = RUNTIME_ROOT / f'seed{seed}'
    mod.RUNTIME = runtime
    search, folds, split_hash = mod.load_folds()
    by_cell = {(r['task'], int(r['fold']), r['architecture']): r for r in records}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows: list[dict[str, Any]] = []
    for task in mod.TASK_ORDER:
        dataset = mod.TASKS[task]['dataset']
        for fold in folds[dataset]:
            fold_id = int(fold['fold_id'])
            outer = fold['outer_dev_subjects']
            bundle = mod.build_bundle(task, outer)
            mean, std, norm_meta = mod.load_tensor_pair(runtime / 'normalizers' / f'{task.lower()}_fold{fold_id}.npz')
            cache = mod.RawGPUCache(bundle, device)
            checkpoints = {
                'LiteBN_BASELINE': baseline_path(task, fold_id, seed),
                'LiteBN_X': Path(by_cell[(task, fold_id, 'LiteBN_X')]['checkpoint_path']),
            }
            for method, checkpoint in checkpoints.items():
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                model = mod.build_model(method, task).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
                metrics = mod.evaluate(model, bundle, cache, outer, mean, std)
                checkpoint_sha = mod.sha256_file(checkpoint)
                for subject, value in metrics.items():
                    rows.append({'task': task, 'dataset': dataset, 'fold': fold_id, 'seed': int(seed), 'subject_id': str(subject), 'method': method, **value, 'checkpoint_sha256': checkpoint_sha, 'normalizer_sha256': norm_meta['mean_std_sha256'], 'baseline_reused': method == 'LiteBN_BASELINE'})
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            del cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(['task', 'fold', 'subject_id', 'method'])
    expected = 2 * sum(len(f['outer_dev_subjects']) for ds in ('OpenBMI', 'WBCIC') for f in folds[ds])
    if len(frame) != expected:
        raise RuntimeError(f'outer cardinality seed{seed}: {len(frame)}/{expected}')
    frame.to_csv(OUT / f'SEED{seed}_X_OUTER_SUBJECT_RESULTS.csv', index=False)
    return frame


def load_seed0(mod) -> pd.DataFrame:
    path = OUT / 'OUTER_SUBJECT_RESULTS.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame['seed'] = 0
    frame['baseline_reused'] = frame['method'].eq('LiteBN_BASELINE')
    return frame


def aggregate(mod, frames: list[pd.DataFrame], baseline_prov: dict[str, Any]) -> None:
    all_frame = pd.concat(frames, ignore_index=True)
    all_frame.to_csv(OUT / 'MULTISEED_X_ONLY_OUTER_SUBJECT_RESULTS.csv', index=False)
    rows = []
    for task in mod.TASK_ORDER:
        sub = all_frame[all_frame.task == task]
        wide = sub.pivot_table(index=['seed', 'fold', 'subject_id'], columns='method', values=['BA', 'macro_F1', 'accuracy'])
        delta = (wide[('BA', 'LiteBN_X')] - wide[('BA', 'LiteBN_BASELINE')]) * 100.0
        subject_delta = delta.groupby(level='subject_id').mean()
        seed_delta = delta.groupby(level='seed').mean()
        fold_delta = delta.groupby(level='fold').mean()
        rng = np.random.default_rng(0)
        draws = rng.choice(subject_delta.to_numpy(float), size=(10000, len(subject_delta)), replace=True).mean(axis=1)
        rows.append({'task': task, 'n_subjects': int(len(subject_delta)), 'mean_delta_pp': float(subject_delta.mean()), 'median_delta_pp': float(np.median(subject_delta)), 'ci_low_pp': float(np.quantile(draws, .025)), 'ci_high_pp': float(np.quantile(draws, .975)), 'seed0_delta_pp': float(seed_delta.loc[0]), 'seed1_delta_pp': float(seed_delta.loc[1]), 'seed2_delta_pp': float(seed_delta.loc[2]), 'positive_seed_means': int((seed_delta > 0).sum()), 'positive_folds': int((fold_delta > 0).sum()), 'negative_folds': int((fold_delta < 0).sum()), 'positive_subjects': int((subject_delta > 0).sum()), 'harmed_subjects': int((subject_delta < 0).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'MULTISEED_X_ONLY_SUMMARY.csv', index=False)
    terminal = 'X_ONLY_MULTISEED_POSITIVE' if bool((summary.mean_delta_pp > 0).all()) else 'X_ONLY_MULTISEED_MIXED_OR_NEGATIVE'
    report = ['# LiteBN-X only multi-seed development result', '', 'Only LiteBN-X was trained for seeds 1 and 2. LiteBN baseline was not retrained; all three seed baseline checkpoints are reused from verified historical experiments.', '', '| Task | mean delta pp | 95% CI | seed0 | seed1 | seed2 | positive seeds |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples(index=False):
        report.append(f'| {r.task} | {r.mean_delta_pp:+.3f} | [{r.ci_low_pp:+.3f}, {r.ci_high_pp:+.3f}] | {r.seed0_delta_pp:+.3f} | {r.seed1_delta_pp:+.3f} | {r.seed2_delta_pp:+.3f} | {r.positive_seed_means}/3 |')
    report += ['', f'Terminal: `{terminal}`.', '', 'Final holdout/test data were not accessed. These are SEARCH outer-development results only.']
    (OUT / 'MULTISEED_X_ONLY_DECISION.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    amendment = {'amendment': 'X_ONLY_MULTISEED', 'requested_scope': 'Train LiteBN_X only for seeds 1 and 2; skip LiteBN baseline retraining.', 'seed_scope': [0, 1, 2], 'trained_methods': ['LiteBN_X'], 'baseline_retrained': False, 'baseline_reuse': baseline_prov, 'split_sha256': mod.load_folds()[2], 'outer_scope': 'SEARCH outer-development only', 'final_holdout_accessed': False, 'final_holdout_predictions_generated': False, 'terminal': terminal}
    (PROTOCOL / 'X_ONLY_MULTISEED_AMENDMENT.json').write_text(json.dumps(amendment, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (PROTOCOL / 'X_ONLY_MULTISEED_AMENDMENT.md').write_text('# X-only multi-seed amendment\n\nUser-directed amendment: run LiteBN-X for seeds 1 and 2 only. Do not retrain LiteBN baseline; reuse the verified historical three-seed baseline checkpoints. No final holdout/test data are accessed.\n', encoding='utf-8')
    print(f'X_ONLY_MULTISEED_DONE {terminal}', flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    stage_b = json.loads((PROTOCOL / 'STAGE_B_REVEAL.json').read_text(encoding='utf-8'))
    if stage_b.get('selected_architecture') != 'LiteBN_X' or stage_b.get('final_holdout_data_accessed') or stage_b.get('final_holdout_predictions_generated'):
        raise RuntimeError('invalid freeze/holdout state')
    baseline_prov = verify_baselines(mod)
    print('BASELINE_REUSE_VERIFIED', baseline_prov['count'], flush=True)
    seed_records = {}
    seed_frames = []
    for seed in SEEDS:
        rec = train_seed(mod, seed)
        seed_records[seed] = rec
        seed_frames.append(evaluate_seed(mod, seed, rec))
    seed0 = load_seed0(mod)
    seed_frames.insert(0, seed0)
    aggregate(mod, seed_frames, baseline_prov)


if __name__ == '__main__':
    main()
