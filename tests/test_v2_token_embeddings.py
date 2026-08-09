from __future__ import annotations

import numpy as np
import torch

from hsc_tta.backbones.adapter import CBraModInputAdapter


def test_token_window_alignment_shapes():
    adapter = CBraModInputAdapter()
    sleep = adapter.adapt("hmc", np.zeros((2, 1, 6000), np.float32), ["EEG C4-M1"], 200.0)
    assert sleep.tensor.shape == (2, 1, 30, 200)
    assert sleep.input_valid_mask.shape == (2, 1, 30)


def test_mi_channel_order_is_preserved():
    adapter = CBraModInputAdapter()
    channels = [f"ch{i:02d}" for i in range(64)]
    signal = np.zeros((1, 64, 640), np.float32)
    signal[0, 0] = np.sin(np.linspace(0, 4, 640))
    adapted = adapter.adapt("eegmmidb", signal, channels, 160.0)
    assert adapted.tensor.shape == (1, 64, 4, 200)
    assert adapted.channel_mask.tolist() == [True] * 64


def test_token_pooling_contract():
    tokens = torch.zeros(3, 64, 4, 200)
    assert tokens.mean((1, 2)).shape == (3, 200)
