"""Fail-closed, post-run aggregation for the HE96 first-task gate."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def mean(rows, field):
    return fmean(float(row[field]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, required=True)
    args = parser.parse_args()
    out = args.outputs
    outer = read_csv(out / 'HE96_FOLD_RESULTS.csv')
    subjects = read_csv(out / 'HE96_OUTER_SUBJECT_RESULTS.csv')
    audits = read_csv(out / 'HE96_INITIALIZATION_AUDIT.csv')
    if len(outer) != 5 or len(audits) != 5 or any(row['status'] != 'PASS' for row in audits):
        raise RuntimeError('HE96_FINALIZATION_AUDIT_FAIL')
    delta = mean(outer, 'delta_BA_pp')
    positive_folds = sum(float(row['delta_BA_pp']) > 0 for row in outer)
    if delta > 0:
        raise RuntimeError('HE96_GATE_UNEXPECTED_PASS: this finalizer is only valid for the stop path')
    heldout = [row for row in subjects if row['population'] == 'heldout']
    if not heldout:
        raise RuntimeError('HE96_INTERNAL_HELDOUT_MISSING')
    held = {
        'task': 'OpenBMI_MI',
        'label': 'internal heldout diagnostic',
        'subjects_x_folds': len(heldout),
        'LiteBN_BA': mean(heldout, 'LiteBN_BA'),
        'HE96_BA': mean(heldout, 'model_BA'),
        'delta_BA_pp': mean(heldout, 'delta_BA_pp'),
    }
    with (out / 'HE96_INTERNAL_HELDOUT_SUMMARY.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=held.keys())
        writer.writeheader()
        writer.writerow(held)
    decision = {
        'decision': 'STOP_MODEL',
        'task_gate': 'OpenBMI_MI outer development comparison',
        'LiteBN_outer_BA': mean(outer, 'LiteBN_BA'),
        'HE96_outer_BA': mean(outer, 'model_BA'),
        'delta_BA_pp': delta,
        'positive_folds': positive_folds,
        'worst_fold_delta_pp': min(float(row['delta_BA_pp']) for row in outer),
        'reason': 'HE96 did not exceed LiteBN on the first-task outer comparison.',
        'prohibited_next_steps': ['WBCIC_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP', 'seed1', 'seed2'],
        'heldout_label': 'internal heldout diagnostic; not an untouched final test',
        'initialization_audit': 'PASS: all 5 folds, eval/train x FP32/AMP, zero logits/embedding difference and preserved RNG.',
    }
    (out / 'FINAL_DECISION.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    report = f'''# LiteBN-HE96 seed0 final report

## Decision

`STOP_MODEL`. The first required gate, OpenBMI-MI outer development comparison,
failed: HE96 {decision['HE96_outer_BA']:.5f} BA vs LiteBN
{decision['LiteBN_outer_BA']:.5f} BA ({decision['delta_BA_pp']:.3f} pp), with
{positive_folds}/5 positive folds. No later task or additional seed was started.

## Initialization audit

All five folds passed exact initial-function checks in eval/train and FP32/AMP:
zero logit and embedding differences, identical predictions, zero new point2
columns, and preserved RNG state.

## Internal heldout diagnostic

This is an internal heldout diagnostic, not an untouched final test: HE96
{held['HE96_BA']:.5f} BA vs LiteBN {held['LiteBN_BA']:.5f} BA
({held['delta_BA_pp']:.3f} pp) across {held['subjects_x_folds']} subject-fold rows.
'''
    (out / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
