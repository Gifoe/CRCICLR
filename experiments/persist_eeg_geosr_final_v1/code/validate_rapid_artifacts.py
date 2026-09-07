"""Recompute the triage decision and verify the compact published evidence.

Reads already-produced tables and locks, never EEG or outcome data loaders.
Checkpoint bytes are checked additionally when run on the experiment server.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def validate(exp, require_checkpoints=False):
    root = exp / 'rapid_triage'
    result = read(root / 'results/RAPID_TRIAGE_RESULT.json')
    pre = read(root / 'RAPID_TRIAGE_PRE_OUTCOME_LOCK.json')
    access = read(root / 'RAPID_TRIAGE_OUTCOME_ACCESS_LOCK.json')
    legal = read(root / 'DATA_LEGALITY_AUDIT.json')
    optimization = read(exp / 'RAPID_TRIAGE_EXECUTION_OPTIMIZATION_LOCK.json')
    assert sha(exp / 'RAPID_TRIAGE_PROTOCOL_AMENDMENT.json') == pre['amendment_sha256']
    assert sha(root / 'RAPID_TRIAGE_PRE_OUTCOME_LOCK.json') == access['pre_outcome_lock_sha256']
    assert sha(root / 'RAPID_TRIAGE_OUTCOME_RULE.json') == access['outcome_rule_sha256']
    assert sha(exp / 'DEVICE_CACHE_EQUIVALENCE.json') == optimization['benchmark_sha256']
    for name, digest in optimization['code_sha256'].items():
        assert sha(exp / 'code' / name) == digest, name
    for item in [pre, access, legal, result]:
        assert item['WBCIC_outer_10_opened'] is False
        assert item['OpenBMI_sealed_holdout_opened'] is False
    assert legal['outcome_labels_read_before_lock'] is False
    assert legal['outcome_labels_read_after_lock'] is True
    assert optimization['locked_at_utc'] < pre['created_at_utc'] <= access['created_at_utc']
    mp = root / 'runtime/seed-0/PREFLIGHT_MANIFEST.json'
    if not mp.is_file():
        mp = root / 'evidence/PREFLIGHT_MANIFEST.json'
    assert sha(mp) == pre['manifest_sha256']
    manifest = read(mp)
    frame = pd.read_csv(root / 'results/RAPID_TRIAGE_OUTCOME_PER_SUBJECT.csv')
    summary = pd.read_csv(root / 'results/RAPID_TRIAGE_PERFORMANCE_SUMMARY.csv')
    deltas = pd.read_csv(root / 'results/RAPID_TRIAGE_SUBJECT_DELTAS.csv')
    assert set(frame.dataset) == {'OpenBMI', 'WBCIC'}
    assert set(frame.method) == {'SUBJECT_BALANCED_ERM', 'GEOSR'}
    assert set(frame.fold) == {0} and set(frame.seed) == {0}
    assert not frame.duplicated(['dataset', 'subject_id', 'method']).any()
    assert np.isfinite(frame[['BA', 'macro_F1']].to_numpy()).all()
    checked_checkpoints = 0
    counts = {}
    clear = {}
    ba_means = {}
    for dataset in ['OpenBMI', 'WBCIC']:
        wr = root / 'workers' / f'{dataset}_fold0'
        assert sha(wr / 'PRE_OUTCOME_RAPID_TRIAGE_WORKER_LOCK.json') == pre['worker_lock_sha256'][dataset]
        marker = read(wr / 'RAPID_TRIAGE_WORKER_COMPLETE.json')
        assert marker['completed_at_utc'] <= pre['created_at_utc']
        entry = manifest[f'{dataset}/fold-0/seed-0']
        paired = {}
        for method in ['SUBJECT_BALANCED_ERM', 'GEOSR']:
            ck = entry['checkpoints'][method]
            p = Path(ck['path'])
            if p.exists():
                assert sha(p) == ck['sha256']
                checked_checkpoints += 1
            elif require_checkpoints:
                raise FileNotFoundError(p)
            z = frame[(frame.dataset == dataset) & (frame.method == method)].set_index('subject_id').sort_index()
            paired[method] = z
            s = summary[(summary.dataset == dataset) & (summary.method == method)].iloc[0]
            assert np.isclose(z.BA.mean(), s.mean_subject_BA, atol=1e-12, rtol=0)
            assert np.isclose(z.macro_F1.mean(), s.mean_macro_F1, atol=1e-12, rtol=0)
        a, b = paired['SUBJECT_BALANCED_ERM'], paired['GEOSR']
        assert a.index.equals(b.index)
        ba = (b.BA - a.BA) * 100
        f1 = (b.macro_F1 - a.macro_F1) * 100
        d = deltas[deltas.dataset == dataset].set_index('subject_id').sort_index()
        assert d.index.equals(a.index)
        np.testing.assert_allclose(ba, d.delta_BA_pp, atol=1e-10, rtol=0)
        np.testing.assert_allclose(f1, d.delta_macro_F1_pp, atol=1e-10, rtol=0)
        ba_means[dataset] = float(ba.mean())
        clear[dataset] = bool(ba.mean() >= .5 and f1.mean() >= 0 and (ba >= 0).mean() >= .5)
        assert clear[dataset] == result['dataset_decisions'][dataset]['clear_positive']
        counts[dataset] = {'positive': int((ba > 1e-10).sum()), 'tied': int((ba.abs() <= 1e-10).sum()),
                           'negative': int((ba < -1e-10).sum()), 'n': len(ba)}
    terminal = ('RAPID_TRIAGE_RESTORE_FULL_PROTOCOL' if all(clear.values()) else
                'RAPID_TRIAGE_STOP_NO_POSITIVE_DIRECTION' if all(x <= 0 for x in ba_means.values()) else
                'RAPID_TRIAGE_STOP_INCONCLUSIVE_OR_MIXED')
    assert terminal == result['terminal']
    return {'pass': True, 'terminal': terminal, 'subject_counts': counts,
            'checkpoints_verified': checked_checkpoints,
            'compact_hash_chain_verified': True, 'numeric_tables_recomputed': True,
            'raw_EEG_loaded': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--require-checkpoints', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = validate(args.experiment.resolve(), args.require_checkpoints)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
