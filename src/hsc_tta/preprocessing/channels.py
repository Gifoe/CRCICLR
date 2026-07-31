from __future__ import annotations

import re


def _normalize(name: str) -> str:
    x = re.sub(r"\s+", "", name.upper()).replace("_", "-")
    x = re.sub(r"^(EEG)?", "", x)
    return x.replace("M1", "A1").replace("M2", "A2")


def select_sleep_channels(channel_names: list[str], dataset: str) -> dict[str, object]:
    preferred = ["C3-M2", "C4-M1"] if dataset.lower() == "hmc" else ["C3-A2", "C4-A1"]
    normalized = {_normalize(name): name for name in channel_names}
    selected, mask = [], []
    for target in preferred:
        original = normalized.get(_normalize(target))
        selected.append(original)
        mask.append(original is not None)
    return {"selected": [x for x in selected if x is not None], "channel_mask": mask, "eligible": any(mask), "montage_note": "A1/A2 and M1/M2 are treated as nearby reference aliases, not identical montages"}

