from __future__ import annotations

import hashlib
import pathlib
import sys
from collections import OrderedDict

import numpy as np
import torch
from scipy.signal import resample_poly

STANDARD_1020 = [
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3",
    "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10", "PO9", "PO7", "PO5", "PO3",
    "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10", "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10",
    "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2", "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6",
    "CFC7", "CFC8", "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8", "T1", "T2",
]

BIOT_MONTAGES = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_channel(name: str) -> str:
    value = name.upper().replace("EEG ", "").replace(".", "").split("-")[0]
    return {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(value, value)


def resample_signals(signals: np.ndarray, sampling_rate: float) -> np.ndarray:
    rounded = int(round(float(sampling_rate)))
    if rounded == 200:
        return signals
    divisor = np.gcd(200, rounded)
    return resample_poly(signals, 200 // divisor, rounded // divisor, axis=-1).astype(np.float32)


def load_labram(project_root: pathlib.Path, device: torch.device) -> torch.nn.Module:
    code = project_root / "external/LaBraM"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import modeling_finetune

    model = modeling_finetune.labram_base_patch200_200(num_classes=0, init_values=0.1)
    checkpoint = torch.load(code / "checkpoints/labram-base.pth", map_location="cpu", weights_only=False)
    state = OrderedDict(
        (key[8:], value) for key, value in checkpoint["model"].items() if key.startswith("student.")
    )
    state.pop("head.weight", None)
    state.pop("head.bias", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    meaningful_missing = [key for key in missing if not key.startswith(("head.", "fc_norm."))]
    allowed_unexpected = {"mask_token", "lm_head.weight", "lm_head.bias", "norm.weight", "norm.bias"}
    if meaningful_missing or not set(unexpected) <= allowed_unexpected:
        raise RuntimeError(f"LaBraM checkpoint mismatch: missing={meaningful_missing}, unexpected={unexpected}")
    return model.eval().requires_grad_(False).to(device)


def labram_embeddings(
    model: torch.nn.Module,
    signals: np.ndarray,
    names: list[str],
    dataset: str,
    device: torch.device,
    sampling_rate: float = 200.0,
) -> torch.Tensor:
    signals = resample_signals(signals, sampling_rate)
    clean = [normalized_channel(name) for name in names]
    keep = [(index, name) for index, name in enumerate(clean) if name in STANDARD_1020]
    if not keep:
        raise RuntimeError("LaBraM has no resolvable channels")
    input_chans = torch.tensor([0] + [STANDARD_1020.index(name) + 1 for _, name in keep], device=device)
    x = torch.from_numpy(signals[:, [index for index, _ in keep]] * 1e6).float().to(device)
    with torch.inference_mode():
        if dataset == "hmc":
            outputs = []
            for start in range(0, x.shape[-1], 2000):
                part = x[..., start : start + 2000]
                if part.shape[-1] < 2000:
                    part = torch.nn.functional.pad(part, (0, 2000 - part.shape[-1]))
                part = part.reshape(part.shape[0], part.shape[1], 10, 200)
                outputs.append(model.forward_features(part, input_chans=input_chans))
            return torch.stack(outputs).mean(0)
        x = x[..., : (x.shape[-1] // 200) * 200]
        return model.forward_features(x.reshape(x.shape[0], x.shape[1], -1, 200), input_chans=input_chans)


def load_biot(project_root: pathlib.Path, device: torch.device) -> torch.nn.Module:
    code = project_root / "external/BIOT"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from model.biot import BIOTEncoder

    model = BIOTEncoder(n_channels=16)
    state = torch.load(
        code / "pretrained-models/EEG-PREST-16-channels.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def biot_input(signals: np.ndarray, names: list[str], dataset: str) -> np.ndarray:
    if dataset == "hmc":
        return signals * 1e6
    clean = [normalized_channel(name) for name in names]
    lookup = {name: index for index, name in enumerate(clean)}
    if not all(left in lookup and right in lookup for left, right in BIOT_MONTAGES):
        raise RuntimeError("BIOT 16-montage adapter lacks required electrodes")
    return np.stack(
        [signals[:, lookup[left]] - signals[:, lookup[right]] for left, right in BIOT_MONTAGES], axis=1
    ) * 1e6


def biot_embeddings(
    model: torch.nn.Module,
    signals: np.ndarray,
    names: list[str],
    dataset: str,
    device: torch.device,
    sampling_rate: float = 200.0,
) -> torch.Tensor:
    signals = resample_signals(signals, sampling_rate)
    x = torch.from_numpy(biot_input(signals, names, dataset)).float().to(device)
    with torch.inference_mode():
        if dataset != "hmc":
            return model(x)
        embeddings = []
        for channel, token_index in enumerate((10, 14)):
            spectrum = model.stft(x[:, channel : channel + 1, :])
            patch = model.patch_embedding(spectrum)
            batch, steps, _ = patch.shape
            token = model.channel_tokens(model.index[token_index]).view(1, 1, -1).repeat(batch, steps, 1)
            embeddings.append(model.positional_encoding(patch + token))
        return model.transformer(torch.cat(embeddings, dim=1)).mean(dim=1)
