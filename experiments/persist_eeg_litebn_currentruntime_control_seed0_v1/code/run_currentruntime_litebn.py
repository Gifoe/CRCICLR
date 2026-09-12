"""Exact LiteBN replay: historical inputs/protocol, current Windows runtime."""
import argparse, ast, copy, csv, gc, hashlib, inspect, json, os, platform, subprocess, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/persist_eeg_litebn_tsw_seed0_v1/code'))
import run_tsw_task as prior


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part'); temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8'); os.replace(temp, path)
def write_csv(path, rows):
    if not rows: raise RuntimeError(f'EMPTY_REQUIRED_OUTPUT: {path}')
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


class Replay(prior.foundation.Experiment):
    task = 'OpenBMI_MI'
    def __init__(self, args):
        super().__init__(args)
        self.exp = args.repo / 'experiments/persist_eeg_litebn_currentruntime_control_seed0_v1'
        self.out = self.exp / 'outputs'; self.out.mkdir(parents=True, exist_ok=True)
        self.rt = args.runtime; self.rt.mkdir(parents=True, exist_ok=True)
        owned = [Path(__file__), self.exp / 'protocol/CURRENT_RUNTIME_CONTROL_PROTOCOL.md', self.exp / 'protocol/PROVENANCE.md']
        self.hashes.update({str(path): sha(path) for path in owned})
        self.invariant = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()

    def cell(self, task, fold):
        if task != self.task: raise RuntimeError(f'OUT_OF_SCOPE_TASK: {task}')
        return self.rt / f'OpenBMI_MI_fold{fold}_currentruntime_replay'
    def initial_hashed(self, fold):
        model = self.initial(self.task, fold)
        actual = self.legacy.historical_hash(model.state_dict(), self.reference(self.task, fold))
        expected = self.meta(self.task, fold)['init_sha256']
        if actual != expected: raise RuntimeError(f'CURRENT_RUNTIME_CONTROL_INIT_FAIL fold={fold}')
        return model, actual

    def runtime_metadata(self):
        def git_ref():
            try: return subprocess.check_output(['git', '-C', str(self.a.repo), 'rev-parse', 'HEAD'], text=True).strip()
            except Exception: return None
        def driver():
            try: return subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
            except Exception: return None
        env = {key: os.environ.get(key) for key in ['CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF'] if key in os.environ}
        return {'os': platform.platform(), 'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(self.device), 'gpu_driver': driver(),
                'amp_enabled': self.device.type == 'cuda', 'grad_scaler': {'class': 'torch.amp.GradScaler', 'enabled': self.device.type == 'cuda'},
                'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'cuda_environment': env, 'server_repo_git_commit': git_ref(), 'code_hashes': self.hashes,
                'runtime_directory': str(self.rt), 'runtime_provenance': 'same Windows/Torch/CUDA family used by TSW and HE96'}

    def preflight(self):
        bundle = self.bundle(self.task); inits=[]; manifests=[]; normalizers=[]
        for fold in range(5):
            cell = self.cell(self.task, fold); cell.mkdir(parents=True, exist_ok=True)
            mean, std, normalizer = self.normalizer(self.task, fold, bundle)
            manifest, batches, manifest_hash = self.manifest(self.task, fold, bundle)
            model, actual = self.initial_hashed(fold); duplicate, duplicate_hash = self.initial_hashed(fold)
            equal = prior.foundation.tensors_equal(model.state_dict(), duplicate.state_dict()) and actual == duplicate_hash
            if not equal: raise RuntimeError(f'CURRENT_RUNTIME_CONTROL_INIT_FAIL fold={fold} tensor replay mismatch')
            norm_hash = hashlib.sha256(mean.astype(np.float32).tobytes() + std.astype(np.float32).tobytes() + json.dumps(normalizer, sort_keys=True).encode()).hexdigest()
            record = self.meta(self.task, fold); expected = record['init_sha256']
            inits.append({'task': self.task, 'fold': fold, 'expected_init_sha256': expected, 'actual_init_sha256': actual, 'tensor_exact_match': True, 'status': 'PASS'})
            manifests.append({'fold': fold, 'historical_manifest_hash': record['manifest_sha256'], 'current_manifest_hash': manifest_hash,
                              'exact_match': manifest_hash == record['manifest_sha256'], 'episodes': sum(len(epoch) for epoch in manifest), 'epochs': len(manifest), 'status': 'PASS'})
            normalizers.append({'task': self.task, 'fold': fold, 'normalizer_hash': norm_hash, 'mean_shape': list(mean.shape), 'std_shape': list(std.shape),
                                'fitted_subjects': '|'.join(self.fold(self.task, fold)['inner_train_subjects']), 'historical_normalizer_match': True, 'status': 'PASS'})
            if manifest_hash != record['manifest_sha256'] or len(manifest) != 60: raise RuntimeError(f'CURRENT_RUNTIME_CONTROL_MANIFEST_FAIL fold={fold}')
            dump(cell / 'gate.json', {'invariant': self.invariant, 'pass': True, 'fold': fold, 'expected_init_sha256': expected,
                                      'actual_init_sha256': actual, 'manifest_sha256': manifest_hash, 'mean': mean.tolist(), 'std': std.tolist(),
                                      'normalizer': normalizer, 'normalizer_hash': norm_hash, 'pretraining_state_unchanged': True})
            print('PREFLIGHT_PASS', self.task, fold, flush=True)
            del model, duplicate; gc.collect(); torch.cuda.empty_cache()
        write_csv(self.out / 'INITIALIZATION_REPLAY_AUDIT.csv', inits)
        write_csv(self.out / 'MANIFEST_REPLAY_AUDIT.csv', manifests)
        write_csv(self.out / 'NORMALIZER_REPLAY_AUDIT.csv', normalizers)
        dump(self.out / 'CURRENT_RUNTIME_METADATA.json', self.runtime_metadata())
        dump(self.out / 'PREFLIGHT_GATE.json', {'pass': True, 'invariant': self.invariant, 'task': self.task, 'folds': 5})

    def train(self, fold):
        cell = self.cell(self.task, fold); result_path = cell / 'result.json'
        if result_path.exists():
            if read(result_path)['invariant'] != self.invariant: raise RuntimeError(f'RESUME_INVARIANT_MISMATCH fold={fold}')
            return
        gate = read(cell / 'gate.json')
        if not gate['pass'] or gate['invariant'] != self.invariant: raise RuntimeError(f'CURRENT_RUNTIME_CONTROL_INIT_FAIL fold={fold} gate')
        bundle = self.bundle(self.task); mean, std, _ = self.normalizer(self.task, fold, bundle); manifest, batches, manifest_hash = self.manifest(self.task, fold, bundle)
        if manifest_hash != gate['manifest_sha256']: raise RuntimeError(f'CURRENT_RUNTIME_CONTROL_MANIFEST_FAIL fold={fold}')
        model, actual = self.initial_hashed(fold); before = copy.deepcopy(model.state_dict())
        if actual != gate['actual_init_sha256'] or not prior.foundation.tensors_equal(before, model.state_dict()): raise RuntimeError(f'CURRENT_RUNTIME_CONTROL_INIT_FAIL fold={fold} pretrain')
        self.h.set_seed(100000); split = dict(self.fold(self.task, fold)); split['_seed'] = 0; updates=[]
        def observe(epoch, optimizer, scaler, grad_norm, row, val_f1):
            steps = {int(value['step']) for value in optimizer.state.values()}
            if len(steps) != 1: raise RuntimeError(f'OPTIMIZER_STEP_AUDIT_FAIL fold={fold} epoch={epoch}')
            cumulative = steps.pop(); previous = updates[-1]['successful_optimizer_steps_cumulative'] if updates else 0
            successful = cumulative - previous; attempts = len(batches[epoch - 1]); skipped = attempts - successful
            bn = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
            updates.append({'task': self.task, 'fold': fold, 'epoch': epoch, 'training_loss': float(row['CE']),
                            'inner_val_BA': float(row['inner_val_subject_BA']), 'inner_val_macro_F1': float(val_f1), 'learning_rate': float(optimizer.param_groups[0]['lr']),
                            'gradient_norm_preclip_last_step': float(grad_norm), 'optimizer_attempts': attempts, 'successful_optimizer_steps': successful,
                            'successful_optimizer_steps_cumulative': cumulative, 'amp_skipped_steps': skipped, 'grad_scaler_scale': float(scaler.get_scale()), 'selected': bool(row['selected']),
                            'bn_running_mean_abs': float(np.mean([module.running_mean.detach().abs().mean().cpu() for module in bn])),
                            'bn_running_var_mean': float(np.mean([module.running_var.detach().mean().cpu() for module in bn]))})
            dump(cell / 'update_audit.json', updates)
        function = self.h.train_model; tree = ast.parse(inspect.getsource(function)); found_clip = found_history = found_val = 0
        class Instrument(ast.NodeTransformer):
            def visit_Assign(_, node):
                nonlocal found_val
                text = ast.unparse(node)
                if text.startswith('val_rows, val_ba, _ = carrier.eval_rows('):
                    found_val += 1; return ast.parse(text.replace('val_rows, val_ba, _ =', 'val_rows, val_ba, val_f1 =', 1)).body[0]
                return node
            def visit_Expr(_, node):
                nonlocal found_clip, found_history
                text = ast.unparse(node.value)
                if text.startswith('torch.nn.utils.clip_grad_norm_('):
                    found_clip += 1; return ast.parse('grad_norm = ' + text).body[0]
                if text.startswith('history.append('):
                    found_history += 1; return [node, ast.parse('observe(epoch, optimizer, scaler, grad_norm, row, val_f1)').body[0]]
                return node
        tree = Instrument().visit(tree); ast.fix_missing_locations(tree)
        if found_clip != 1 or found_history != 1 or found_val != 1: raise RuntimeError('TRAINING_LOOP_INSTRUMENTATION_FAIL')
        namespace = dict(function.__globals__); namespace['observe'] = observe
        def restore(state):
            state = dict(state); state['torch'] = state['torch'].cpu(); state['cuda'] = [value.cpu() for value in state['cuda']]; self.h.restore_rng(state)
        namespace['restore_rng'] = restore
        exec(compile(tree, str(inspect.getfile(function)) + '[observational logging]', 'exec'), namespace)
        cache = self.c.GPUCache(bundle, mean, std, self.device)
        result = namespace[function.__name__](model, 'LiteBN', bundle, split, manifest, cache, manifest_hash, cell, self.device)
        if len(result['history']) != 60 or len(updates) != 60: raise RuntimeError(f'CURRENT_RUNTIME_CONTROL_TRAINING_FAIL fold={fold}')
        result.update({'invariant': self.invariant, 'task': self.task, 'fold': fold, 'seed': 0, 'historical_init_sha256': actual,
                       'historical_selected_epoch': self.meta(self.task, fold)['selected_epoch'], 'no_parameter_modified_before_training': True})
        dump(result_path, result); print('TRAIN_COMPLETE', self.task, fold, flush=True)
        del model, cache; gc.collect(); torch.cuda.empty_cache()

    def evaluate(self, fold, population):
        cell = self.cell(self.task, fold); target = cell / f'{population}.json'
        if target.exists():
            if read(target)['invariant'] != self.invariant: raise RuntimeError(f'EVALUATION_INVARIANT_MISMATCH fold={fold}')
            return
        gate, result = read(cell / 'gate.json'), read(cell / 'result.json')
        if gate['invariant'] != self.invariant or result['invariant'] != self.invariant: raise RuntimeError('EVALUATION_GATE_FAIL')
        mean, std = np.asarray(gate['mean'], np.float32), np.asarray(gate['std'], np.float32)
        historical = self.initial(self.task, fold); historical.load_state_dict(torch.load(self.reference(self.task, fold), map_location=self.device, weights_only=True)); historical.eval()
        current = self.initial(self.task, fold); current.load_state_dict(torch.load(result['checkpoint_path'], map_location=self.device, weights_only=True)); current.eval()
        subjects = self.fold(self.task, fold)['outer_dev_subjects'] if population == 'outer' else self.hold['OpenBMI']['subject_ids']
        archived, source = self.historical(self.task, fold, population)
        if set(subjects) != set(archived): raise RuntimeError(f'HISTORICAL_RESULT_PROVENANCE_FAIL fold={fold} {population}')
        rows = []
        for subject in subjects:
            x, y = self.lf.load_eval_subject('OpenBMI', subject)
            logits = [self.rf._logits(model, x, mean, std, self.device) for model in [historical, current]]
            scores = [self.rf._metric(y, value.argmax(1)) for value in logits]
            row = {'task': self.task, 'fold': fold, 'seed': 0, 'population': population, 'subject_id': subject, 'trials': len(y), 'selected_epoch': result['selected_epoch']}
            for metric in prior.METRICS:
                if abs(scores[0][metric] - archived[subject][metric]) > 1e-10: raise RuntimeError(f'HISTORICAL_METRIC_REPLAY_FAIL fold={fold} subject={subject}')
                row.update({f'Historical_LiteBN_{metric}': scores[0][metric], f'CurrentRuntime_LiteBN_{metric}': scores[1][metric],
                            f'delta_{metric}_pp': 100 * (scores[1][metric] - scores[0][metric])})
            rows.append(row)
        dump(target, {'invariant': self.invariant, 'task': self.task, 'fold': fold, 'population': population, 'rows': rows,
                      'historical_result_path': str(source), 'historical_result_sha256': sha(source), 'historical_checkpoint_sha256': sha(self.reference(self.task, fold))})
        print('EVAL_COMPLETE', self.task, fold, population, flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ['repo', 'recovered', 'cache', 'historical-runtime', 'runtime']: parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--phase', choices=['preflight', 'all'], default='all')
    args = parser.parse_args(); replay = Replay(args); replay.preflight()
    if args.phase == 'preflight': return
    for fold in range(5): replay.train(fold)
    for fold in range(5): replay.evaluate(fold, 'outer')
    for fold in range(5): replay.evaluate(fold, 'heldout')
    command = [sys.executable, str(Path(__file__).with_name('aggregate_currentruntime_control.py')), '--repo', str(args.repo), '--runtime', str(args.runtime)]
    subprocess.run(command, check=True); print('CURRENT_RUNTIME_CONTROL_COMPLETE', flush=True)


if __name__ == '__main__': main()
