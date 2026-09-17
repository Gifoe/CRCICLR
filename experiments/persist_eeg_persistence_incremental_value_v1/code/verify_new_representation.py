"""Metadata-only frozen checkpoint classifier-input shape verification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
P1 = ROOT.parent
sys.path.insert(0, str(ROOT / "experiments/persist_eeg_eegconformer_fbcnet_multiseed_v1/code"))
from models import build_model  # noqa: E402


def main() -> None:
    for model in ("EEGConformer", "FBCNet"):
        for task in ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"):
            cell = P1 / "eegconformer_fbcnet_runtime/cells" / model.lower() / task.lower() / "fold0_seed0"
            record = json.loads((cell / "record.json").read_text(encoding="utf-8"))
            net = build_model(model, int(record["channels"]), int(record["samples"]), int(record["classes"]))
            payload = torch.load(cell / "selected.pt", map_location="cpu", weights_only=False)
            net.load_state_dict(payload["state_dict"], strict=True)
            net.eval()
            head = net.classifier if model == "EEGConformer" else net.head
            values = []
            handle = head.register_forward_pre_hook(lambda _m, args: values.append(tuple(args[0].shape)))
            shape = ((1, 9, record["channels"], record["samples"]) if model == "FBCNet"
                     else (1, record["channels"], record["samples"]))
            with torch.inference_mode():
                net(torch.zeros(shape, dtype=torch.float32))
            handle.remove()
            if len(values) != 1:
                raise RuntimeError("head hook count mismatch")
            print(model, task, "module=" + ("classifier" if model == "EEGConformer" else "head"),
                  "shape=" + str(values[0]), "checkpoint_state_strict=PASS", flush=True)


if __name__ == "__main__":
    main()
