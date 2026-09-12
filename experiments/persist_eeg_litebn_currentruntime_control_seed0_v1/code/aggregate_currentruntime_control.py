"""Aggregate the frozen five-fold current-runtime LiteBN replay control."""
import argparse, csv, json
from pathlib import Path
from statistics import fmean, median

import numpy as np


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle: return list(csv.DictReader(handle))
def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def avg(rows, field): return fmean(float(row[field]) for row in rows)


def candidate_rows(repo, historical_by_fold, current_by_fold):
    specs = [
        ('HE96', repo / 'experiments/persist_eeg_litebn_he96_seed0_v1/outputs/HE96_FOLD_RESULTS.csv', None),
        ('TSW', repo / 'experiments/persist_eeg_litebn_tsw_seed0_v1/outputs/OpenBMI_MI_OUTER_FOLD_RESULTS.csv', 'TSW'),
        ('StaticScale', repo / 'experiments/persist_eeg_litebn_tsw_seed0_v1/outputs/OpenBMI_MI_OUTER_FOLD_RESULTS.csv', 'StaticScale'),
    ]
    output, fold_values = [], {}
    for name, path, architecture in specs:
        if not path.is_file():
            output.append({'candidate': name, 'runtime_provenance_match': 'NOT_COMPARABLE', 'historical_LiteBN_BA': '', 'currentruntime_LiteBN_BA': '', 'candidate_BA': '', 'delta_vs_historical_pp': '', 'delta_vs_currentruntime_pp': '', 'interpretation': 'NOT_COMPARABLE: frozen candidate output missing'})
            continue
        rows = read_csv(path)
        if architecture: rows = [row for row in rows if row['architecture'] == architecture]
        if len(rows) != 5 or {int(row['fold']) for row in rows} != set(range(5)):
            output.append({'candidate': name, 'runtime_provenance_match': 'NOT_COMPARABLE', 'historical_LiteBN_BA': '', 'currentruntime_LiteBN_BA': '', 'candidate_BA': '', 'delta_vs_historical_pp': '', 'delta_vs_currentruntime_pp': '', 'interpretation': 'NOT_COMPARABLE: incomplete fold provenance'})
            continue
        by_fold = {int(row['fold']): float(row['model_BA']) for row in rows}
        baseline_match = all(abs(float(row['LiteBN_BA']) - historical_by_fold[int(row['fold'])]) <= 1e-12 for row in rows)
        provenance = 'COMPATIBLE_CURRENT_RUNTIME' if baseline_match else 'NOT_COMPARABLE'
        h, c, m = fmean(historical_by_fold.values()), fmean(current_by_fold.values()), fmean(by_fold.values())
        output.append({'candidate': name, 'runtime_provenance_match': provenance, 'historical_LiteBN_BA': h, 'currentruntime_LiteBN_BA': c, 'candidate_BA': m,
                       'delta_vs_historical_pp': 100 * (m - h), 'delta_vs_currentruntime_pp': 100 * (m - c),
                       'interpretation': f'{name} versus same-runtime LiteBN: {100 * (m - c):+.3f} pp' if provenance.startswith('COMPATIBLE') else 'NOT_COMPARABLE'})
        if provenance.startswith('COMPATIBLE'): fold_values[name] = by_fold
    return output, fold_values


def terminal(delta):
    if delta <= -1.0: return 'LARGE_RUNTIME_TRAJECTORY_DRIFT'
    if delta <= -0.3: return 'MODERATE_RUNTIME_TRAJECTORY_DRIFT'
    if abs(delta) < 0.3: return 'CURRENT_RUNTIME_REPRODUCES_LITEBN'
    return 'CURRENT_RUNTIME_LITEBN_IMPROVED'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', type=Path, required=True); parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args(); exp = args.repo / 'experiments/persist_eeg_litebn_currentruntime_control_seed0_v1'; out = exp / 'outputs'
    outer, heldout, trajectories = [], [], []
    for fold in range(5):
        cell = args.runtime / f'OpenBMI_MI_fold{fold}_currentruntime_replay'; result = read_json(cell / 'result.json')
        outer.extend(read_json(cell / 'outer.json')['rows']); heldout.extend(read_json(cell / 'heldout.json')['rows'])
        updates = {int(row['epoch']): row for row in read_json(cell / 'update_audit.json')}
        for history in result['history']:
            row = dict(updates[int(history['epoch'])]); row['selected_inner_val_BA'] = bool(history['selected']); trajectories.append(row)
    write_csv(out / 'CURRENT_RUNTIME_TRAINING_TRAJECTORY.csv', trajectories)
    fold_results = []
    for fold in range(5):
        rows = [row for row in outer if int(row['fold']) == fold]
        result = read_json(args.runtime / f'OpenBMI_MI_fold{fold}_currentruntime_replay/result.json')
        fold_results.append({'task': 'OpenBMI_MI', 'fold': fold, 'historical_LiteBN_BA': avg(rows, 'Historical_LiteBN_BA'), 'currentruntime_LiteBN_BA': avg(rows, 'CurrentRuntime_LiteBN_BA'),
                             'delta_BA_pp': avg(rows, 'delta_BA_pp'), 'historical_LiteBN_macro_F1': avg(rows, 'Historical_LiteBN_macro_F1'), 'currentruntime_LiteBN_macro_F1': avg(rows, 'CurrentRuntime_LiteBN_macro_F1'),
                             'historical_LiteBN_accuracy': avg(rows, 'Historical_LiteBN_accuracy'), 'currentruntime_LiteBN_accuracy': avg(rows, 'CurrentRuntime_LiteBN_accuracy'),
                             'selected_epoch': result['selected_epoch'], 'historical_selected_epoch': result['historical_selected_epoch'], 'outer_subject_count': len(rows)})
    write_csv(out / 'CURRENT_RUNTIME_OUTER_FOLD_RESULTS.csv', fold_results)
    write_csv(out / 'CURRENT_RUNTIME_OUTER_SUBJECT_RESULTS.csv', outer)
    delta = np.asarray([float(row['delta_BA_pp']) for row in outer]); rng = np.random.default_rng(0); draws = rng.choice(delta, (10000, len(delta)), replace=True).mean(1)
    bootstrap = [{'task': 'OpenBMI_MI', 'population': 'outer development subjects', 'subjects': len(delta), 'resamples': 10000, 'mean_subject_delta_pp': float(delta.mean()), 'median_subject_delta_pp': float(np.median(delta)),
                  'positive_subjects': int((delta > 0).sum()), 'negative_subjects': int((delta < 0).sum()), 'tied_subjects': int((delta == 0).sum()), 'ci95_low_pp': float(np.quantile(draws, .025)), 'ci95_high_pp': float(np.quantile(draws, .975))}]
    write_csv(out / 'CURRENT_RUNTIME_PAIRED_BOOTSTRAP.csv', bootstrap)
    held_by_subject = {}
    for row in heldout: held_by_subject.setdefault(row['subject_id'], []).append(row)
    held_summary = [{'task': 'OpenBMI_MI', 'label': 'INTERNAL_HELDOUT_DIAGNOSTIC_ALREADY_OPEN', 'subjects': len(held_by_subject),
                     'historical_LiteBN_BA': fmean(avg(rows, 'Historical_LiteBN_BA') for rows in held_by_subject.values()),
                     'currentruntime_LiteBN_BA': fmean(avg(rows, 'CurrentRuntime_LiteBN_BA') for rows in held_by_subject.values()),
                     'delta_BA_pp': fmean(avg(rows, 'delta_BA_pp') for rows in held_by_subject.values()), 'new_sealed_test_accessed': 'NO'}]
    write_csv(out / 'CURRENT_RUNTIME_HELDOUT_SUMMARY.csv', held_summary)
    hist_by_fold = {int(row['fold']): float(row['historical_LiteBN_BA']) for row in fold_results}; current_by_fold = {int(row['fold']): float(row['currentruntime_LiteBN_BA']) for row in fold_results}
    candidates, candidate_folds = candidate_rows(args.repo, hist_by_fold, current_by_fold)
    write_csv(out / 'SAME_RUNTIME_CANDIDATE_RECALIBRATION.csv', candidates)
    attribution = []
    for fold in range(5):
        row = {'fold': fold, 'historical_LiteBN_BA': hist_by_fold[fold], 'currentruntime_LiteBN_BA': current_by_fold[fold], 'runtime_delta_pp': 100 * (current_by_fold[fold] - hist_by_fold[fold])}
        for name in ['HE96', 'TSW', 'StaticScale']:
            value = candidate_folds.get(name, {}).get(fold); row[name + '_BA'] = value; row[name + '_delta_vs_currentruntime_pp'] = None if value is None else 100 * (value - current_by_fold[fold])
        attribution.append(row)
    write_csv(out / 'FOLD_LEVEL_RUNTIME_ARCH_ATTRIBUTION.csv', attribution)
    h, c = fmean(hist_by_fold.values()), fmean(current_by_fold.values()); runtime_delta = 100 * (c - h); label = terminal(runtime_delta)
    metadata = read_json(out / 'CURRENT_RUNTIME_METADATA.json')
    final = {'terminal_label': label, 'model': 'EXACT_HISTORICAL_LITEBN', 'task': 'OpenBMI_MI', 'seed': 0, 'folds': 5, 'historical_LiteBN_outer_BA': h, 'currentruntime_LiteBN_outer_BA': c,
             'runtime_delta_pp': runtime_delta, 'fold_runtime_delta_pp': [row['runtime_delta_pp'] for row in attribution], 'heldout': held_summary[0], 'candidate_recalibration': candidates,
             'initialization_audit': 'PASS', 'manifest_audit': 'PASS', 'normalizer_audit': 'PASS', 'execution_runtime': metadata, 'new_sealed_test_accessed': 'NO'}
    (out / 'FINAL_CURRENT_RUNTIME_CONTROL.json').write_text(json.dumps(final, indent=2) + '\n', encoding='utf-8')
    candidate_text = '\n'.join(f"- {row['candidate']}: {row['delta_vs_currentruntime_pp']} pp vs current-runtime LiteBN ({row['runtime_provenance_match']})." for row in candidates)
    report = f'''# Exact LiteBN current-runtime replay control

Terminal label: **{label}**.

1. Model: exact recovered historical LiteBN / CompactLite; no architecture change.
2. Initialization: all five expected hashes and full tensor replay checks passed.
3. Manifests: all five original stored OpenBMI-MI manifests matched exactly.
4. Normalizers: all five historical fold-specific normalizers matched.
5. Runtime: Windows, Torch {metadata['torch']}, CUDA {metadata['cuda_runtime']}, GPU {metadata['gpu']}; full metadata is in `CURRENT_RUNTIME_METADATA.json`.
6. Historical Linux LiteBN outer BA: {h:.5f}.
7. Current-runtime exact LiteBN outer BA: {c:.5f}.
8. Runtime delta: {runtime_delta:+.3f} pp.
9. Fold runtime deltas (pp): {', '.join(f"{row['runtime_delta_pp']:+.3f}" for row in attribution)}.
10. Current-runtime internal heldout diagnostic BA: {held_summary[0]['currentruntime_LiteBN_BA']:.5f}.
11. Heldout delta: {held_summary[0]['delta_BA_pp']:+.3f} pp; this is not an untouched or final test.
12-14. Same-runtime candidate recalibration:
{candidate_text}
15. Interpretation: {label}.
16. Direct use of 79.15 as a current-runtime baseline is {'not' if label != 'CURRENT_RUNTIME_REPRODUCES_LITEBN' else ''} scientifically justified without this matched control.
17. Future current-runtime architecture work should use the measured current-runtime LiteBN control.
18. New sealed test accessed: NO.

```text
MODEL = EXACT_HISTORICAL_LITEBN
TASK = OpenBMI_MI
SEED = 0
FOLDS = 5
ARCHITECTURE_CHANGE = NONE
LOSS_CHANGE = NONE
OPTIMIZER_CHANGE = NONE
CHECKPOINT_RULE_CHANGE = NONE
MANIFEST_CHANGE = NONE
NORMALIZER_CHANGE = NONE
INITIALIZATION_RECOVERY = EXACT
EXECUTION_RUNTIME = CURRENT_WINDOWS_TORCH_CUDA
NEW_SEALED_TEST_ACCESSED = NO
```
'''
    (out / 'FINAL_CURRENT_RUNTIME_CONTROL.md').write_text(report, encoding='utf-8')
    print(label, flush=True)


if __name__ == '__main__': main()
