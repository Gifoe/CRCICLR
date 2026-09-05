from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import run_geosr_rapid_triage as rapid


@pytest.mark.parametrize('interrupt_epoch', [2, 3])
def test_resume_preserves_rng_optimizer_and_best_checkpoint(tmp_path, monkeypatch, interrupt_epoch):
    g = rapid.g
    monkeypatch.setattr(g, 'MAX_EPOCHS', 4)
    monkeypatch.setattr(g, 'MIN_EPOCHS', 2)
    monkeypatch.setattr(g, 'PATIENCE', 2)
    monkeypatch.setattr(g, 'make_model', lambda cache, device: torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Dropout(.3), torch.nn.Linear(3, 2)).to(device))
    calls = []

    def train(model, cache, rows, mean, std, weights, optimizer, order, device, **kwargs):
        calls.append(1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(4, 2)).square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    monkeypatch.setattr(g, 'train_epoch', train)
    # Epoch 1 is best; patience expires exactly at the interruption boundary.
    monkeypatch.setattr(g, 'eval_rows', lambda *args: pd.DataFrame({'BA': [.5], 'NLL': [1.]}))
    cache = SimpleNamespace(meta=[0, 1, 2, 3])
    device = torch.device('cpu')
    state = g.make_model(cache, device).state_dict()
    rows = np.arange(4)

    def run(folder):
        return rapid.select_keep_best(cache, rows, rows, np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32), np.ones(4, dtype=np.float32), state,
            'OpenBMI', 'GEOSR', device, folder / 'progress.pt', folder / 'best.pt', 'test-lock')

    full = tmp_path / 'full'
    resumed = tmp_path / 'resumed'
    run(full)
    assert len(calls) == 3
    original_save = rapid.save_progress

    def interrupt_after_save(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[4] == interrupt_epoch:
            raise InterruptedError('simulated process interruption after atomic save')

    monkeypatch.setattr(rapid, 'save_progress', interrupt_after_save)
    with pytest.raises(InterruptedError):
        run(resumed)
    monkeypatch.setattr(rapid, 'save_progress', original_save)
    calls.clear()
    run(resumed)
    assert len(calls) == 3 - interrupt_epoch
    a = torch.load(full / 'best.pt', weights_only=False)
    b = torch.load(resumed / 'best.pt', weights_only=False)
    assert g.state_hash(a['model_state']) == g.state_hash(b['model_state'])
    pa = torch.load(full / 'progress.pt', weights_only=False)
    pb = torch.load(resumed / 'progress.pt', weights_only=False)
    assert g.state_hash(pa['model_state']) == g.state_hash(pb['model_state'])
    assert torch.equal(pa['torch_rng_state'], pb['torch_rng_state'])
    for key, va in pa['optimizer_state']['state'].items():
        vb = pb['optimizer_state']['state'][key]
        for field in va:
            assert torch.equal(va[field], vb[field])
    assert (resumed / 'progress.pt').is_file()
    hit = run(resumed)
    assert hit[4] is True
    assert hit[5] == rapid.file_sha(resumed / 'best.pt')
