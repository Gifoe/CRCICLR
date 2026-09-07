"""Source-only full-epoch equivalence check for cuda versus cuda:0.

No training arithmetic or protocol constant is changed. Explicit CUDA indexing
allows FoldCache's existing device equality check to reuse its resident tensors.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_geosr as g
from benchmark_geosr_numeric import run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'outcome_labels_read': False, 'arithmetic_changed': False,
              'device_alias_equal': torch.device('cuda') == torch.device('cuda:0'),
              'datasets': {}}
    for dataset in ('OpenBMI', 'WBCIC'):
        roles, _, _ = g.ap.load_roles(dataset)
        role = roles[0]
        source = g.subj_sort(role['model_fit'])
        cache = g.FoldCache(dataset, g.subj_sort(set(source) | set(role['discovery'])), 0, 0)
        rows = cache.rows(source, g.sessions_for(dataset))
        mean, std = cache.normalizer(rows)
        weights = np.linspace(0.7, 1.3, len(rows), dtype=np.float32)
        order = g.order_for(rows, dataset, 0, 0, 'benchmark', 'benchmark', 1)
        state, _, _ = g.initial_state(cache, dataset, 0, 0, 'benchmark')
        results = {}
        for label, device in [('implicit_cuda', torch.device('cuda')),
                              ('explicit_cuda_0', torch.device('cuda:0'))]:
            run_one('optimized', cache, rows, mean, std, weights,
                    order[:2 * g.BATCH_SIZE], state, device)
            results[label] = run_one('optimized', cache, rows, mean, std,
                                     weights, order, state, device)
            print(dataset, label, results[label], flush=True)
        a, b = results['implicit_cuda'], results['explicit_cuda_0']
        results['exact_loss'] = a['loss'] == b['loss']
        results['exact_model_state'] = a['state_sha256'] == b['state_sha256']
        results['speedup'] = a['sec'] / b['sec']
        results['training_rows'] = len(rows)
        report['datasets'][dataset] = results
        report['pass'] = all(r['exact_loss'] and r['exact_model_state'] for r in report['datasets'].values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix('.part')
        tmp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        tmp.replace(args.output)
        if not report['pass']:
            raise RuntimeError('Device-index optimization failed numerical equivalence')
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    print('DEVICE_CACHE_EQUIVALENCE_PASS', flush=True)


if __name__ == '__main__':
    main()
